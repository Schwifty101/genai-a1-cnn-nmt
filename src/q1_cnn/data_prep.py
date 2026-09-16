"""Q1 chest X-ray data pipeline: acquisition, integrity scan, deduplication, manifest.

This module is the single source of truth for the Q1 manifest that every
later Q1 task (splitting, dataset, training, evaluation) reads. It:

1. Downloads (or skips downloading, if already present) the raw Kaggle
   dataset `prashant268/chest-xray-covid19-pneumonia`.
2. Walks the extracted `Data/{train,test}/{CLASS}/` tree, verifying every
   image can actually be decoded and recording basic metadata.
3. Deduplicates exact (MD5) and near (thumbnail-correlation) duplicates,
   BEFORE any split is made, so no duplicate can straddle a train/val/test
   boundary.
4. Writes `results/q1_manifest.csv` and `results/q1_data_report.json`.

Near-duplicate detection history: an earlier version of this module used
dHash Hamming distance directly as the near-duplicate criterion. A
calibration pass on the real dataset (40 flagged pairs audited, correlation
distributions compared against random unrelated pairs, dataset-wide nearest-
neighbour correlation measured at median 0.926 / p99 0.970) showed dHash-8 at
Hamming <= 3 is not a duplicate detector on this data: chest X-rays are
intrinsically near-identical under a coarse gradient hash, so the criterion
produced heavy false-positive merges (including cross-class merges). It was
replaced with the contrast-normalized-thumbnail-correlation criterion below,
validated to a corr_threshold of 0.99 (0 of the 40 originally-flagged pairs
exceeded 0.95).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.common.seed import SEED

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]

_KAGGLE_DATASET = "prashant268/chest-xray-covid19-pneumonia"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_THUMBNAIL_SIZE = (32, 32)


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


def _thumbnail_signature(path: Path) -> np.ndarray:
    """Contrast-normalized 32x32 grayscale thumbnail, L2-normalized.

    Open as L, resize to 32x32 bilinear, scale to [0,1], subtract the mean,
    divide by (std + 1e-8), flatten, then L2-normalize — so the dot product
    of two signatures *is* their correlation.
    """
    with Image.open(path) as im:
        im = im.convert("L").resize(_THUMBNAIL_SIZE, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float64) / 255.0
    arr = arr - arr.mean()
    arr = arr / (arr.std() + 1e-8)
    flat = arr.flatten()
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat = flat / norm
    return flat


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


def deduplicate(df: pd.DataFrame, corr_threshold: float = 0.99) -> tuple[pd.DataFrame, dict]:
    """Drop exact and near-duplicate images, keeping the first sorted-path row.

    Sorts by `path` first so the surviving representative never depends on
    input order. Adds `md5` and `dhash` columns. Exact duplicates (identical
    MD5) are dropped first via `drop_duplicates`.

    Near-duplicates are then found via contrast-normalized 32x32 grayscale
    thumbnail correlation (see `_thumbnail_signature`): a row is a
    near-duplicate of an already-kept row when the dot product of their
    signatures — which equals their correlation, since each signature is
    z-scored and then L2-normalized — is `>= corr_threshold`.

    `dhash` is still computed and stored (it's part of the manifest schema
    and a cheap reporting column). Near-duplicate matching itself compares
    every surviving row against every other via one vectorized correlation
    matrix (`signatures @ signatures.T`), rather than a dHash-prefix
    candidate-blocking pass: on this dataset size (~6.4k rows, 1024-d
    signatures) the dense matmul is a fraction of a second (BLAS-vectorized),
    so blocking would only cost recall with no runtime benefit — the
    dominant cost is per-image I/O, which blocking does not reduce.
    """
    work = df.sort_values("path", kind="stable").reset_index(drop=True).copy()
    work["md5"] = [file_md5(Path(p)) for p in work["path"]]
    work["dhash"] = [perceptual_hash(Path(p)) for p in work["path"]]

    n_before_exact = len(work)
    work = work.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    exact_duplicates = n_before_exact - len(work)

    n = len(work)
    if n == 0:
        report = {"exact_duplicates": int(exact_duplicates), "near_duplicates": 0, "kept": 0}
        return work.copy(), report

    signatures = np.stack([_thumbnail_signature(Path(p)) for p in work["path"]]).astype(
        np.float32
    )
    # Each signature is z-scored then L2-normalized, so this matrix's entries
    # *are* pairwise correlations.
    corr_matrix = signatures @ signatures.T

    keep_mask = np.zeros(n, dtype=bool)
    kept_indices: list[int] = []
    near_duplicates = 0

    for i in range(n):
        if kept_indices and corr_matrix[i, kept_indices].max() >= corr_threshold:
            near_duplicates += 1
            continue
        keep_mask[i] = True
        kept_indices.append(i)

    out = work.loc[keep_mask].reset_index(drop=True)

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
    corr_threshold: float = 0.99,
) -> pd.DataFrame:
    """Orchestrate download -> scan -> deduplicate -> write manifest + report."""
    raw_dir = Path(raw_dir)
    out_csv = Path(out_csv)
    report_json = Path(report_json)

    download_dataset(raw_dir)
    ok_df, scan_report = scan_images(raw_dir)
    manifest, dedup_report = deduplicate(ok_df, corr_threshold=corr_threshold)

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
        "corr_threshold": corr_threshold,
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
    parser.add_argument("--corr-threshold", type=float, default=0.99)
    args = parser.parse_args()

    manifest = build_manifest(
        raw_dir=args.raw_dir,
        out_csv=args.out,
        report_json=args.report,
        corr_threshold=args.corr_threshold,
    )

    with open(args.report) as f:
        report = json.load(f)

    print(json.dumps(report, indent=2))
    print(f"manifest rows: {len(manifest)}")


if __name__ == "__main__":
    _main()
