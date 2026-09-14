import pandas as pd
import torch

from src.q2_rnn.dataset import TranslationDataset, collate_batch
from src.q2_rnn.tokenizer import EOS, PAD, SOS, Vocab

EN = ["the cat sat", "a dog ran fast", "hi"]
UR = ["بلی بیٹھی", "کتا دوڑا", "ہائی"]


def _vocabs():
    return Vocab.build(EN, min_freq=1), Vocab.build(UR, min_freq=1)


def _ds():
    src, tgt = _vocabs()
    df = pd.DataFrame({"en": EN, "ur": UR})
    return TranslationDataset(df, src, tgt), src, tgt


def test_dataset_item_is_two_id_lists_without_specials():
    ds, src, _ = _ds()
    s, t = ds[0]
    assert isinstance(s, list) and isinstance(t, list)
    assert SOS not in s and EOS not in s
    assert SOS not in t and EOS not in t


def test_dataset_length():
    ds, _, _ = _ds()
    assert len(ds) == 3


def test_collate_returns_all_required_keys():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert set(batch) == {
        "enc_in", "enc_len", "enc_mask", "dec_in", "dec_target", "dec_mask"
    }


def test_encoder_input_is_right_padded_to_the_longest_source():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["enc_in"].shape == (3, 4)
    assert batch["enc_in"][2, 1:].tolist() == [PAD, PAD, PAD]


def test_encoder_lengths_are_true_lengths():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["enc_len"].tolist() == [3, 4, 1]


def test_encoder_mask_marks_real_tokens_true():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["enc_mask"][2].tolist() == [True, False, False, False]
    assert batch["enc_mask"].sum().item() == 8


def test_decoder_input_starts_with_sos_for_every_row():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert (batch["dec_in"][:, 0] == SOS).all()


def test_decoder_target_ends_with_eos_at_the_true_length():
    ds, _, tgt = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    for row in range(3):
        true_len = int(batch["dec_mask"][row].sum())
        assert batch["dec_target"][row, true_len - 1].item() == EOS


def test_decoder_input_and_target_are_shifted_by_one():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    # dec_in[1:] equals dec_target[:-1] over the real region
    row, length = 1, int(batch["dec_mask"][1].sum())
    assert (
        batch["dec_in"][row, 1:length].tolist()
        == batch["dec_target"][row, : length - 1].tolist()
    )


def test_decoder_tensors_have_the_same_shape():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["dec_in"].shape == batch["dec_target"].shape == batch["dec_mask"].shape


def test_padding_positions_are_masked_false_in_the_decoder():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    pad_positions = batch["dec_target"] == PAD
    assert not (batch["dec_mask"] & pad_positions).any()


def test_all_tensors_are_long_or_bool():
    ds, _, _ = _ds()
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["enc_in"].dtype == torch.long
    assert batch["dec_in"].dtype == torch.long
    assert batch["dec_target"].dtype == torch.long
    assert batch["enc_mask"].dtype == torch.bool
    assert batch["dec_mask"].dtype == torch.bool


def test_single_item_batch_works():
    ds, _, _ = _ds()
    batch = collate_batch([ds[2]])
    assert batch["enc_in"].shape[0] == 1
    assert batch["dec_in"][0, 0].item() == SOS
