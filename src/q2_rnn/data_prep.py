"""Q2 English->Urdu data pipeline: acquisition, normalization, cleaning, length filtering.

This module is the single source of truth for the Q2 clean dataset that
every later Q2 task (tokenization, splits already live in `src.q2_rnn.splits`,
dataset/loaders, training) reads. It:

1. Downloads (or skips downloading, if already present) the raw Kaggle
   dataset `muhammadnoman76/translation-dataset`.
2. Loads the raw `(en, ur)` pairs from the source `.xlsx`, detecting the
   source column names case-insensitively.
3. Normalizes both sides (Urdu normalization folds Arabic-form characters to
   their Urdu forms and converts Arabic-Indic digits to ASCII, but never
   folds `آ` — alef with madda — into `ا`: that is a distinct Urdu letter).
4. Drops empty and exact-duplicate pairs.
5. Filters the long tail by a token-length percentile cap plus an
   en/ur length-ratio guard.
6. Writes the clean CSV, a JSON report, and a length-distribution figure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.plots import plot_length_hist
from src.common.seed import SEED

_KAGGLE_DATASET = "muhammadnoman76/translation-dataset"
_XLSX_NAME = "english_to_urdu_dataset.xlsx"

_EN_COLUMN_CANDIDATES = {"en", "eng", "english", "source", "src"}
_UR_COLUMN_CANDIDATES = {"ur", "urd", "urdu", "target", "tgt"}

# Character-folding map applied to Urdu text, in normalize_urdu's step 2.
# `آ` (alef with madda, U+0622) is intentionally absent: it is a distinct
# Urdu letter and folding it into `ا` would destroy information.
ARABIC_TO_URDU: dict[str, str] = {
    "ي": "ی",  # Arabic yeh -> Urdu yeh
    "ك": "ک",  # Arabic kaf -> Urdu kaf
    "ۀ": "ہ",  # heh with yeh above -> Urdu heh
    "ة": "ہ",  # teh marbuta -> Urdu heh
    "أ": "ا",  # alef with hamza above -> plain alef
    "إ": "ا",  # alef with hamza below -> plain alef
    "ى": "ی",  # alef maksura -> Urdu yeh
    # Arabic-Indic digits -> ASCII
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Extended Arabic-Indic (Urdu/Persian) digits -> ASCII
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

# Combining marks stripped when strip_diacritics=True: FATHATAN..SUKUN
# (U+064B-U+0652) plus the superscript alef (U+0670).
_DIACRITIC_RE = re.compile("[ً-ْٰ]")
_URDU_FULL_STOP = "۔"  # ۔
_URDU_COMMA = "،"  # ،
_WHITESPACE_RE = re.compile(r"\s+")
_EN_PUNCT_RE = re.compile(r'([.,!?;:"()])')


def download_dataset(dest: Path) -> Path:
    """Fetch the English-Urdu translation dataset from Kaggle into `dest`.

    If `dest/english_to_urdu_dataset.xlsx` already exists, the download is
    skipped entirely (no network call, no Kaggle CLI invocation) and that
    path is returned immediately. Otherwise the Kaggle CLI is used to
    download and unzip the dataset into `dest`.
    """
    dest = Path(dest)
    target = dest / _XLSX_NAME
    if target.exists():
        return target

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

    if target.exists():
        return target

    candidates = sorted(dest.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"no .xlsx file found in {dest} after Kaggle download")
    return candidates[0]


def load_raw(xlsx_path: Path) -> pd.DataFrame:
    """Load the raw parallel corpus, returning exactly the columns `["en", "ur"]`.

    Source column names are detected case-insensitively against a small set
    of known aliases (e.g. `eng`/`english`/`en`, `urdu`/`ur`).
    """
    df = pd.read_excel(xlsx_path)

    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    en_col = next((lower_to_actual[c] for c in _EN_COLUMN_CANDIDATES if c in lower_to_actual), None)
    ur_col = next((lower_to_actual[c] for c in _UR_COLUMN_CANDIDATES if c in lower_to_actual), None)

    if en_col is None or ur_col is None:
        raise ValueError(
            f"could not detect English/Urdu columns among {list(df.columns)}"
        )

    out = df[[en_col, ur_col]].rename(columns={en_col: "en", ur_col: "ur"})
    return out.reset_index(drop=True)


def normalize_english(text: str) -> str:
    """NFC-normalize, lowercase, split off punctuation as its own token, collapse whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = _EN_PUNCT_RE.sub(r" \1 ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_urdu(text: str, strip_diacritics: bool = True) -> str:
    """Normalize Urdu text per the rules in the task brief, applied in order:

    1. NFC normalization
    2. Character folding via `ARABIC_TO_URDU` (`آ` is deliberately excluded)
    3. Diacritic stripping (when requested)
    4. Urdu full stop / comma spaced out as their own tokens
    5. Whitespace collapse
    """
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(ARABIC_TO_URDU.get(ch, ch) for ch in text)
    if strip_diacritics:
        text = _DIACRITIC_RE.sub("", text)
    text = text.replace(_URDU_FULL_STOP, f" {_URDU_FULL_STOP} ")
    text = text.replace(_URDU_COMMA, f" {_URDU_COMMA} ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_pairs(df: pd.DataFrame, strip_diacritics: bool = True) -> tuple[pd.DataFrame, dict]:
    """Normalize both sides, drop empty rows, then drop exact-duplicate pairs.

    Adds `en_len`/`ur_len` whitespace-token-count columns to the survivors.
    """
    raw = len(df)

    work = df.copy()
    work["en"] = work["en"].fillna("").astype(str).map(normalize_english)
    work["ur"] = (
        work["ur"]
        .fillna("")
        .astype(str)
        .map(lambda t: normalize_urdu(t, strip_diacritics=strip_diacritics))
    )

    non_empty_mask = (work["en"].str.len() > 0) & (work["ur"].str.len() > 0)
    empty_dropped = int((~non_empty_mask).sum())
    work = work.loc[non_empty_mask].reset_index(drop=True)

    n_before_dupes = len(work)
    work = work.drop_duplicates(subset=["en", "ur"]).reset_index(drop=True)
    duplicates_dropped = n_before_dupes - len(work)

    work["en_len"] = work["en"].str.split().str.len()
    work["ur_len"] = work["ur"].str.split().str.len()

    report = {
        "raw": int(raw),
        "empty_dropped": int(empty_dropped),
        "duplicates_dropped": int(duplicates_dropped),
        "kept": int(len(work)),
    }
    return work, report


def filter_by_length(
    df: pd.DataFrame,
    max_percentile: float = 95.0,
    max_ratio: float = 2.5,
    min_len: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Drop the long tail (above a per-side percentile cap, or below `min_len`),
    then drop pairs whose en/ur length ratio exceeds `max_ratio`.
    """
    en_max_len = float(np.percentile(df["en_len"], max_percentile))
    ur_max_len = float(np.percentile(df["ur_len"], max_percentile))

    too_long_mask = (
        (df["en_len"] > en_max_len)
        | (df["ur_len"] > ur_max_len)
        | (df["en_len"] < min_len)
        | (df["ur_len"] < min_len)
    )
    too_long_dropped = int(too_long_mask.sum())
    stage1 = df.loc[~too_long_mask].reset_index(drop=True)

    if len(stage1) > 0:
        hi = stage1[["en_len", "ur_len"]].max(axis=1)
        lo = stage1[["en_len", "ur_len"]].min(axis=1).clip(lower=1)
        ratio = hi / lo
        bad_ratio_mask = ratio > max_ratio
    else:
        bad_ratio_mask = pd.Series([], dtype=bool)

    bad_ratio_dropped = int(bad_ratio_mask.sum())
    out = stage1.loc[~bad_ratio_mask].reset_index(drop=True)

    report = {
        "en_max_len": en_max_len,
        "ur_max_len": ur_max_len,
        "too_long_dropped": too_long_dropped,
        "bad_ratio_dropped": bad_ratio_dropped,
        "kept": int(len(out)),
    }
    return out, report


def build_dataset(
    xlsx: Path,
    out_csv: Path,
    report_json: Path,
    fig_path: Path,
    strip_diacritics: bool = True,
    max_percentile: float = 95.0,
    max_ratio: float = 2.5,
) -> pd.DataFrame:
    """Orchestrate load -> clean -> filter -> write CSV/report/figure."""
    xlsx = Path(xlsx)
    out_csv = Path(out_csv)
    report_json = Path(report_json)
    fig_path = Path(fig_path)

    raw_df = load_raw(xlsx)
    cleaned, clean_report = clean_pairs(raw_df, strip_diacritics=strip_diacritics)
    filtered, filter_report = filter_by_length(
        cleaned, max_percentile=max_percentile, max_ratio=max_ratio
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_csv, index=False)

    plot_length_hist(
        [
            {
                "title": "English token length",
                "series": {
                    "before filter": cleaned["en_len"].tolist(),
                    "after filter": filtered["en_len"].tolist(),
                },
            },
            {
                "title": "Urdu token length",
                "series": {
                    "before filter": cleaned["ur_len"].tolist(),
                    "after filter": filtered["ur_len"].tolist(),
                },
            },
        ],
        fig_path,
        suptitle="Q2 English-Urdu token length distributions",
    )

    report = {
        "clean": clean_report,
        "length_filter": filter_report,
        "kept": filter_report["kept"],
        "seed": SEED,
        "strip_diacritics": strip_diacritics,
        "max_percentile": max_percentile,
        "max_ratio": max_ratio,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    return filtered


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the Q2 English-Urdu clean dataset.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    data_cfg = cfg.get("data", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/q2"))
    clean_csv = Path(data_cfg.get("clean_csv", "results/q2_clean.csv"))
    strip_diacritics = bool(data_cfg.get("strip_diacritics", True))
    max_percentile = float(data_cfg.get("max_percentile", 95.0))
    max_ratio = float(data_cfg.get("max_ratio", 2.5))

    report_json = Path("results/q2_data_report.json")
    fig_path = Path("results/figs/q2_lengths.png")

    xlsx = download_dataset(raw_dir)

    df = build_dataset(
        xlsx=xlsx,
        out_csv=clean_csv,
        report_json=report_json,
        fig_path=fig_path,
        strip_diacritics=strip_diacritics,
        max_percentile=max_percentile,
        max_ratio=max_ratio,
    )

    with open(report_json) as f:
        report = json.load(f)

    print(json.dumps(report, indent=2))
    print(f"kept pairs: {len(df)}")


if __name__ == "__main__":
    _main()
