import math

import pytest
import torch

from src.q2_rnn.evaluate import corpus_bleu_score, error_taxonomy
from src.q2_rnn.tokenizer import PAD
from src.q2_rnn.train import masked_loss


def test_masked_loss_ignores_padded_positions():
    torch.manual_seed(0)
    b, t, v = 2, 4, 6
    logits = torch.randn(b, t, v)
    target = torch.randint(4, v, (b, t))
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    got = masked_loss(logits, target, mask)

    flat_logits = logits[mask]
    flat_target = target[mask]
    expected = torch.nn.functional.cross_entropy(flat_logits, flat_target)
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_loss_changes_when_unmasked_content_changes():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 5)
    target = torch.tensor([[4, 4, 4]])
    mask = torch.tensor([[True, True, False]])
    a = masked_loss(logits, target, mask)
    logits[0, 0] += 5.0
    b = masked_loss(logits, target, mask)
    assert not torch.allclose(a, b)


def test_masked_loss_is_unaffected_by_masked_content():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 5)
    target = torch.tensor([[4, 4, 4]])
    mask = torch.tensor([[True, True, False]])
    a = masked_loss(logits, target, mask)
    logits[0, 2] += 100.0
    b = masked_loss(logits, target, mask)
    assert torch.allclose(a, b)


def test_masked_loss_is_a_positive_scalar():
    logits = torch.randn(2, 3, 7)
    target = torch.randint(4, 7, (2, 3))
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = masked_loss(logits, target, mask)
    assert out.ndim == 0 and out.item() > 0


def test_corpus_bleu_is_100_for_identical_text():
    refs = ["یہ ایک جملہ ہے", "دوسرا جملہ"]
    assert corpus_bleu_score(refs, refs) == pytest.approx(100.0, abs=1e-6)


def test_corpus_bleu_is_low_for_unrelated_text():
    hyps = ["کتا کتا کتا کتا"] * 2
    refs = ["یہ ایک جملہ ہے", "دوسرا جملہ"]
    assert corpus_bleu_score(hyps, refs) < 10.0


def test_error_taxonomy_detects_repetition():
    hyps = ["ایک ایک ایک ایک ایک"]
    refs = ["ایک دو تین چار پانچ"]
    src = ["one two three four five"]
    stats = error_taxonomy(hyps, refs, src)
    assert stats["repetition_rate"] > 0.5


def test_error_taxonomy_detects_unk():
    hyps = ["<unk> <unk> ایک"]
    refs = ["ایک دو تین"]
    src = ["one two three"]
    stats = error_taxonomy(hyps, refs, src)
    assert stats["unk_rate"] == pytest.approx(2 / 3)


def test_error_taxonomy_detects_empty_output():
    stats = error_taxonomy(["", "ایک"], ["ایک", "دو"], ["one", "two"])
    assert stats["empty_rate"] == pytest.approx(0.5)


def test_error_taxonomy_buckets_by_source_length():
    hyps = ["ایک"] * 4
    refs = ["ایک"] * 4
    src = ["a b c", "a b c d e f g", "a b c d e f g h i j k l", "w " * 20]
    stats = error_taxonomy(hyps, refs, src)
    assert set(stats["bleu_by_src_len_bucket"]) == {"1-5", "6-10", "11-15", "16+"}


def test_error_taxonomy_reports_n_and_length_ratio():
    stats = error_taxonomy(["ایک دو"], ["ایک دو تین چار"], ["one two three four"])
    assert stats["n"] == 1
    assert stats["mean_len_ratio"] == pytest.approx(0.5)
