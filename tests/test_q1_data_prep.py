import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.q1_cnn.data_prep import (
    CLASS_NAMES,
    deduplicate,
    file_md5,
    is_corrupt,
    perceptual_hash,
    scan_images,
)


def _write_image(path, color=(120, 120, 120), size=(64, 64), noise=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        arr = np.clip(
            arr.astype(int) + rng.integers(-noise, noise + 1, arr.shape), 0, 255
        ).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def test_class_names_are_canonical_and_sorted():
    assert CLASS_NAMES == ["COVID19", "NORMAL", "PNEUMONIA"]
    assert CLASS_NAMES == sorted(CLASS_NAMES)


def test_is_corrupt_detects_truncated_file(tmp_path):
    good = _write_image(tmp_path / "good.png")
    assert is_corrupt(good) is False
    bad = tmp_path / "bad.png"
    bad.write_bytes(good.read_bytes()[:20])
    assert is_corrupt(bad) is True


def test_is_corrupt_detects_non_image(tmp_path):
    junk = tmp_path / "notes.png"
    junk.write_text("this is not an image")
    assert is_corrupt(junk) is True


def test_file_md5_is_stable_and_content_sensitive(tmp_path):
    a = _write_image(tmp_path / "a.png", color=(10, 10, 10))
    b = _write_image(tmp_path / "b.png", color=(10, 10, 10))
    c = _write_image(tmp_path / "c.png", color=(200, 200, 200))
    assert file_md5(a) == file_md5(b)
    assert file_md5(a) != file_md5(c)


def test_scan_images_walks_class_dirs_and_reports_corrupt(tmp_path):
    root = tmp_path / "raw"
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            _write_image(root / "Data" / split / cls / f"{cls}_1.png")
    broken = root / "Data" / "train" / "NORMAL" / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n truncated")

    df, report = scan_images(root)

    assert len(df) == 6
    assert report["total_found"] == 7
    assert report["corrupt"] == 1
    assert set(df["label"]) == set(CLASS_NAMES)
    assert set(df["source_dir"]) == {"train", "test"}
    assert list(df.columns) == [
        "path",
        "label",
        "label_idx",
        "source_dir",
        "width",
        "height",
        "mode",
    ]
    assert df.loc[df["label"] == "COVID19", "label_idx"].unique().tolist() == [0]


def test_deduplicate_removes_exact_duplicates(tmp_path):
    paths = [
        _write_image(tmp_path / "one.png", color=(30, 30, 30)),
        _write_image(tmp_path / "two.png", color=(30, 30, 30)),
        _write_image(tmp_path / "three.png", color=(220, 220, 220)),
    ]
    df = pd.DataFrame(
        {
            "path": [str(p) for p in paths],
            "label": ["NORMAL"] * 3,
            "label_idx": [1] * 3,
        }
    )
    out, report = deduplicate(df)
    assert report["exact_duplicates"] == 1
    assert report["kept"] == 2
    assert len(out) == 2
    assert "md5" in out.columns and "dhash" in out.columns


def test_deduplicate_removes_near_duplicates(tmp_path):
    base = _write_image(tmp_path / "base.png", color=(90, 90, 90), noise=0)
    near = _write_image(tmp_path / "near.png", color=(91, 91, 91), noise=0)
    far = _write_image(tmp_path / "far.png", color=(10, 200, 10), noise=60)
    df = pd.DataFrame(
        {
            "path": [str(base), str(near), str(far)],
            "label": ["NORMAL"] * 3,
            "label_idx": [1] * 3,
        }
    )
    out, report = deduplicate(df, hamming_threshold=3)
    assert report["near_duplicates"] >= 1
    assert len(out) < 3


def test_deduplicate_is_order_independent(tmp_path):
    paths = [
        _write_image(tmp_path / "b.png", color=(40, 40, 40)),
        _write_image(tmp_path / "a.png", color=(40, 40, 40)),
    ]
    df = pd.DataFrame(
        {"path": [str(p) for p in paths], "label": ["NORMAL"] * 2, "label_idx": [1] * 2}
    )
    out1, _ = deduplicate(df)
    out2, _ = deduplicate(df.iloc[::-1].reset_index(drop=True))
    assert out1["path"].tolist() == out2["path"].tolist()


def test_perceptual_hash_is_hex_string(tmp_path):
    p = _write_image(tmp_path / "x.png")
    h = perceptual_hash(p)
    assert isinstance(h, str)
    int(h, 16)
