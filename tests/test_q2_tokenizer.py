import pytest

from src.q2_rnn.tokenizer import (
    EOS,
    PAD,
    SOS,
    SPECIALS,
    UNK,
    Vocab,
    oov_rate,
    tokenize,
)

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat ran",
    "a bird flew",
]


def test_special_ids_are_fixed():
    assert (PAD, SOS, EOS, UNK) == (0, 1, 2, 3)
    assert SPECIALS == ["<pad>", "<sos>", "<eos>", "<unk>"]


def test_tokenize_splits_on_whitespace():
    assert tokenize("the cat  sat ") == ["the", "cat", "sat"]


def test_tokenize_empty_string_gives_empty_list():
    assert tokenize("") == []


def test_vocab_reserves_the_first_four_ids_for_specials():
    v = Vocab.build(CORPUS, min_freq=1)
    assert v.itos[:4] == SPECIALS


def test_min_freq_excludes_rare_words():
    v = Vocab.build(CORPUS, min_freq=2)
    assert "the" in v.stoi
    assert "bird" not in v.stoi


def test_min_freq_one_keeps_everything():
    v = Vocab.build(CORPUS, min_freq=1)
    assert "bird" in v.stoi


def test_encode_maps_unknown_words_to_unk():
    v = Vocab.build(CORPUS, min_freq=2)
    ids = v.encode(tokenize("the zebra sat"))
    assert ids[1] == UNK
    assert ids[0] != UNK


def test_encode_decode_roundtrip_for_known_words():
    v = Vocab.build(CORPUS, min_freq=1)
    tokens = ["the", "cat", "sat"]
    assert v.decode(v.encode(tokens)) == tokens


def test_decode_strips_specials_by_default():
    v = Vocab.build(CORPUS, min_freq=1)
    ids = [SOS] + v.encode(["the", "cat"]) + [EOS, PAD, PAD]
    assert v.decode(ids) == ["the", "cat"]


def test_decode_can_keep_specials():
    v = Vocab.build(CORPUS, min_freq=1)
    out = v.decode([SOS, EOS], strip_specials=False)
    assert out == ["<sos>", "<eos>"]


def test_vocab_is_deterministic_across_builds():
    a = Vocab.build(CORPUS, min_freq=1)
    b = Vocab.build(CORPUS, min_freq=1)
    assert a.itos == b.itos


def test_vocab_ordering_is_frequency_then_alphabetical():
    v = Vocab.build(CORPUS, min_freq=1)
    assert v.itos[4] == "the"  # the most frequent real token


def test_vocab_save_and_load_roundtrip(tmp_path):
    v = Vocab.build(CORPUS, min_freq=1)
    p = tmp_path / "vocab.json"
    v.save(p)
    loaded = Vocab.load(p)
    assert loaded.itos == v.itos
    assert len(loaded) == len(v)


def test_oov_rate_is_zero_when_everything_is_in_vocab():
    v = Vocab.build(CORPUS, min_freq=1)
    assert oov_rate(v, CORPUS) == pytest.approx(0.0)


def test_oov_rate_counts_unknown_tokens():
    v = Vocab.build(CORPUS, min_freq=1)
    assert oov_rate(v, ["zebra zebra the"]) == pytest.approx(2 / 3)
