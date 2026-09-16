import pandas as pd
import pytest

from src.q2_rnn.data_prep import (
    clean_pairs,
    filter_by_length,
    normalize_english,
    normalize_urdu,
)


def test_english_is_lowercased_and_whitespace_collapsed():
    assert normalize_english("  The   Sky  IS Blue ") == "the sky is blue"


def test_english_punctuation_becomes_its_own_token():
    assert normalize_english("Hello, world!") == "hello , world !"


def test_english_handles_empty_and_none_like_input():
    assert normalize_english("") == ""
    assert normalize_english("   ") == ""


def test_urdu_nfc_normalization_is_applied():
    decomposed = "آ"  # alef + madda above
    assert normalize_urdu(decomposed) == "آ"  # precomposed alef with madda


def test_urdu_arabic_yeh_is_folded_to_urdu_yeh():
    assert "ي" not in normalize_urdu("يه")
    assert "ی" in normalize_urdu("يه")


def test_urdu_arabic_kaf_is_folded_to_urdu_kaf():
    assert normalize_urdu("ك") == "ک"


def test_urdu_alef_madda_is_preserved_not_folded():
    assert normalize_urdu("آب") == "آب"


def test_urdu_diacritics_are_stripped_when_requested():
    with_marks = "کِتاب"
    assert normalize_urdu(with_marks, strip_diacritics=True) == "کتاب"
    assert "ِ" in normalize_urdu(with_marks, strip_diacritics=False)


def test_urdu_digits_are_converted_to_ascii():
    assert normalize_urdu("۱۲۳") == "123"


def test_urdu_full_stop_is_separated_as_a_token():
    out = normalize_urdu("یہ سچ ہے۔")
    assert out.endswith(" ۔")


def test_urdu_whitespace_is_collapsed():
    assert normalize_urdu("  اب   جب  ") == "اب جب"


def test_clean_pairs_drops_empty_rows():
    df = pd.DataFrame({"en": ["hello", "", "  ", "bye"], "ur": ["سلام", "x", "y", ""]})
    out, report = clean_pairs(df)
    assert report["empty_dropped"] == 3
    assert len(out) == 1


def test_clean_pairs_drops_exact_duplicate_pairs():
    df = pd.DataFrame({"en": ["hi", "hi", "bye"], "ur": ["سلام", "سلام", "خدا"]})
    out, report = clean_pairs(df)
    assert report["duplicates_dropped"] == 1
    assert len(out) == 2


def test_clean_pairs_keeps_a_pair_whose_sides_repeat_separately():
    df = pd.DataFrame({"en": ["hi", "hi"], "ur": ["سلام", "آداب"]})
    out, _ = clean_pairs(df)
    assert len(out) == 2


def test_clean_pairs_adds_token_length_columns():
    df = pd.DataFrame({"en": ["one two three"], "ur": ["ایک دو"]})
    out, _ = clean_pairs(df)
    assert out.iloc[0]["en_len"] == 3
    assert out.iloc[0]["ur_len"] == 2


def test_filter_by_length_drops_the_long_tail():
    df = pd.DataFrame(
        {
            "en": ["a b"] * 99 + ["x " * 200],
            "ur": ["ا ب"] * 99 + ["ج " * 200],
        }
    )
    df["en_len"] = df["en"].str.split().str.len()
    df["ur_len"] = df["ur"].str.split().str.len()
    out, report = filter_by_length(df, max_percentile=95.0)
    assert report["too_long_dropped"] >= 1
    assert len(out) < len(df)


def test_filter_by_length_drops_bad_ratio_pairs():
    df = pd.DataFrame({"en": ["a b c d e f g h"], "ur": ["ا"]})
    df["en_len"] = [8]
    df["ur_len"] = [1]
    out, report = filter_by_length(df, max_percentile=100.0, max_ratio=2.5)
    assert report["bad_ratio_dropped"] == 1
    assert len(out) == 0


def test_filter_by_length_keeps_reasonable_pairs():
    df = pd.DataFrame({"en": ["a b c"] * 50, "ur": ["ا ب ج"] * 50})
    df["en_len"] = 3
    df["ur_len"] = 3
    out, report = filter_by_length(df)
    assert len(out) == 50
    assert report["kept"] == 50
