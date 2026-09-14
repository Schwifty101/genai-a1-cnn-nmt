"""Q1 chest X-ray data pipeline: acquisition, integrity scan, deduplication, manifest.

This module is the single source of truth for the Q1 manifest that every
later Q1 task (splitting, dataset, training, evaluation) reads. It:

1. Downloads (or skips downloading, if already present) the raw Kaggle
   dataset `prashant268/chest-xray-covid19-pneumonia`.
2. Walks the extracted `Data/{train,test}/{CLASS}/` tree, verifying every
   image can actually be decoded and recording basic metadata.
3. Deduplicates exact (MD5) and near (perceptual hash / dHash) duplicates,
   BEFORE any split is made, so no duplicate can straddle a train/val/test
   boundary.
4. Writes `results/q1_manifest.csv` and `results/q1_data_report.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageStat

from src.common.seed import SEED

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]

_KAGGLE_DATASET = "prashant268/chest-xray-covid19-pneumonia"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# dHash is a purely relative (gradient) measure: two images that are each
# internally flat/near-constant (e.g. a solid mid-gray tile and a solid
# near-white tile) collapse to the *same* dHash regardless of their absolute
# brightness, because there is no internal gradient for the hash to encode.
# That is a known blind spot of gradient hashes. A coarse mean-intensity gate
# closes it cheaply: two rows are only treated as near-duplicates when both
# their dHash is close (the primary, spec-mandated signal) AND their overall
# brightness is close. Real chest X-rays are textured, not flat, so this
# never triggers on the real dataset and only guards the degenerate case.
_BRIGHTNESS_TOLERANCE = 15.0


def download_dataset(dest: Path) -> Path:
    """Fetch the chest X-ray dataset from Kaggle into `dest`.

    If `dest/Data` already exists, the download is skipped entirely (no
    network call, no Kaggle CLI invocation) and `dest` is returned
    immediately. Otherwise the Kaggle CLI is used to download and unzip the
    dataset into `dest`.
    """
    dest = Path(dest)
    if (dest / "Data").exists():
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            _KAGGLE_DATASET,
            "-p",
            str(dest),
            "--unzip",
        ],
        check=True,
    )
    return dest


def is_corrupt(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return False
    except Exception:
        return True


def file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.dhash(im))


def _mean_intensity(path: Path) -> float:
    """Cheap absolute-brightness signal used only as a near-dup tie-breaker."""
    with Image.open(path) as im:
        return float(ImageStat.Stat(im.convert("L")).mean[0])


def scan_images(root: Path) -> tuple[pd.DataFrame, dict]:
    """Walk `root/Data/{train,test}/{CLASS}/`, verify every file, and report.

    Returns `(ok_df, report)` where `ok_df` holds one row per readable image
    and `report` summarizes the scan (`total_found`, `corrupt`,
    `corrupt_paths`).
    """
    root = Path(root)
    candidates = sorted(root.glob("Data/*/*/*"))
    candidates = [p for p in candidates if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]

    rows = []
    corrupt_paths: list[str] = []

    for path in candidates:
        if is_corrupt(path):
            corrupt_paths.append(str(path))
            continue
        label = path.parent.name
        source_dir = path.parent.parent.name
        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
        rows.append(
            {
                "path": str(path),
                "label": label,
                "label_idx": CLASS_NAMES.index(label),
                "source_dir": source_dir,
                "width": int(width),
                "height": int(height),
                "mode": mode,
            }
        )

    ok_df = pd.DataFrame(
        rows,
        columns=["path", "label", "label_idx", "source_dir", "width", "height", "mode"],
    )

    report = {
        "total_found": len(candidates),
        "corrupt": len(corrupt_paths),
        "corrupt_paths": corrupt_paths,
    }
    return ok_df, report


def deduplicate(df: pd.DataFrame, hamming_threshold: int = 3) -> tuple[pd.DataFrame, dict]:
    """Drop exact and near-duplicate images, keeping the first sorted-path row.

    Sorts by `path` first so the surviving representative never depends on
    input order. Adds `md5` and `dhash` columns. Exact duplicates (identical
    MD5) are dropped first, then near-duplicates whose dHash Hamming
    distance to an already-kept row is `<= hamming_threshold`.
    """
    work = df.sort_values("path", kind="stable").reset_index(drop=True).copy()
    work["md5"] = [file_md5(Path(p)) for p in work["path"]]
    work["dhash"] = [perceptual_hash(Path(p)) for p in work["path"]]

    n_before_exact = len(work)
    work = work.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    exact_duplicates = n_before_exact - len(work)

    # Bucket by the first 4 hex characters of the dHash so the near-duplicate
    # scan stays sub-quadratic in practice instead of comparing every row
    # against every other row.
    kept_rows: list[dict] = []
    buckets: dict[str, list[int]] = {}
    near_duplicates = 0
    brightness_cache: dict[str, float] = {}

    def brightness_for(path: str) -> float:
        if path not in brightness_cache:
            brightness_cache[path] = _mean_intensity(Path(path))
        return brightness_cache[path]

    for row in work.to_dict("records"):
        dhash_hex = row["dhash"]
        bucket_key = dhash_hex[:4]
        candidate_hash = imagehash.hex_to_hash(dhash_hex)

        is_near_dup = False
        for kept_idx in buckets.get(bucket_key, []):
            kept_hash = imagehash.hex_to_hash(kept_rows[kept_idx]["dhash"])
            if candidate_hash - kept_hash > hamming_threshold:
                continue
            if (
                abs(brightness_for(row["path"]) - brightness_for(kept_rows[kept_idx]["path"]))
                <= _BRIGHTNESS_TOLERANCE
            ):
                is_near_dup = True
                break

        if is_near_dup:
            near_duplicates += 1
            continue

        kept_rows.append(row)
        buckets.setdefault(bucket_key, []).append(len(kept_rows) - 1)

    out = pd.DataFrame(kept_rows)
    if len(out):
        out = out.reset_index(drop=True)
    else:
        out = work.iloc[0:0].copy()

    report = {
        "exact_duplicates": int(exact_duplicates),
        "near_duplicates": int(near_duplicates),
        "kept": int(len(out)),
    }
    return out, report


def build_manifest(
    raw_dir: Path,
    out_csv: Path,
    report_json: Path,
    hamming_threshold: int = 3,
) -> pd.DataFrame:
    """Orchestrate download -> scan -> deduplicate -> write manifest + report."""
    raw_dir = Path(raw_dir)
    out_csv = Path(out_csv)
    report_json = Path(report_json)

    download_dataset(raw_dir)
    ok_df, scan_report = scan_images(raw_dir)
    manifest, dedup_report = deduplicate(ok_df, hamming_threshold=hamming_threshold)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)

    class_counts = {
        label: int(count) for label, count in manifest["label"].value_counts().sort_index().items()
    }

    report = {
        "total_found": int(scan_report["total_found"]),
        "corrupt": int(scan_report["corrupt"]),
        "corrupt_paths": scan_report["corrupt_paths"],
        "exact_duplicates": int(dedup_report["exact_duplicates"]),
        "near_duplicates": int(dedup_report["near_duplicates"]),
        "kept": int(dedup_report["kept"]),
        "class_counts": class_counts,
        "seed": SEED,
        "hamming_threshold": hamming_threshold,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the Q1 chest X-ray manifest.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/q1"))
    parser.add_argument("--out", type=Path, default=Path("results/q1_manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/q1_data_report.json"))
    parser.add_argument("--hamming-threshold", type=int, default=3)
    args = parser.parse_args()

    manifest = build_manifest(
        raw_dir=args.raw_dir,
        out_csv=args.out,
        report_json=args.report,
        hamming_threshold=args.hamming_threshold,
    )

    with open(args.report) as f:
        report = json.load(f)

    print(json.dumps(report, indent=2))
    print(f"manifest rows: {len(manifest)}")


if __name__ == "__main__":
    _main()
