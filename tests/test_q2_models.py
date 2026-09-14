import inspect
import re

import torch

from src.q2_rnn import models as q2_models
from src.q2_rnn.dataset import collate_batch
from src.q2_rnn.models import Decoder, Encoder, Seq2SeqRNN, count_parameters
from src.q2_rnn.tokenizer import EOS, PAD, SOS

SRC_V, TGT_V = 50, 60


def _batch(b=4, s=7, t=5):
    src = [list(range(4, 4 + s - i)) for i in range(b)]
    tgt = [list(range(4, 4 + t - i)) for i in range(b)]
    return collate_batch(list(zip(src, tgt)))


def test_no_lstm_gru_or_attention_in_the_module_source():
    source = inspect.getsource(q2_models)
    for banned in ("nn.LSTM", "nn.GRU", "MultiheadAttention", "Transformer"):
        assert banned not in source, f"forbidden component {banned} found"


def test_encoder_uses_nn_rnn():
    enc = Encoder(SRC_V)
    assert any(isinstance(m, torch.nn.RNN) for m in enc.modules())
    assert not any(
        isinstance(m, (torch.nn.LSTM, torch.nn.GRU)) for m in enc.modules()
    )


def test_decoder_uses_nn_rnn():
    dec = Decoder(TGT_V)
    assert any(isinstance(m, torch.nn.RNN) for m in dec.modules())


def test_embeddings_use_padding_idx():
    enc = Encoder(SRC_V)
    emb = [m for m in enc.modules() if isinstance(m, torch.nn.Embedding)][0]
    assert emb.padding_idx == PAD


def test_encoder_returns_hidden_state_of_the_right_shape():
    batch = _batch()
    hidden = Encoder(SRC_V, embed_dim=16, hidden_dim=32)(
        batch["enc_in"], batch["enc_len"]
    )
    assert hidden.shape == (1, 4, 32)


def test_encoder_ignores_padding_content():
    """Changing the values sitting in padded positions must not move the state."""
    batch = _batch()
    enc = Encoder(SRC_V, embed_dim=16, hidden_dim=32)
    enc.eval()
    with torch.no_grad():
        a = enc(batch["enc_in"], batch["enc_len"])
        polluted = batch["enc_in"].clone()
        polluted[batch["enc_mask"] == False] = 7  # noqa: E712
        b = enc(polluted, batch["enc_len"])
    assert torch.allclose(a, b, atol=1e-5)


def test_decoder_output_shape():
    batch = _batch()
    hidden = Encoder(SRC_V, embed_dim=16, hidden_dim=32)(
        batch["enc_in"], batch["enc_len"]
    )
    logits, _ = Decoder(TGT_V, embed_dim=16, hidden_dim=32)(batch["dec_in"], hidden)
    assert logits.shape == (4, batch["dec_in"].shape[1], TGT_V)


def test_seq2seq_forward_shape_matches_dec_target():
    batch = _batch()
    model = Seq2SeqRNN(SRC_V, TGT_V, 16, 32, 1, 0.0)
    logits = model(batch)
    assert logits.shape[:2] == batch["dec_target"].shape
    assert logits.shape[2] == TGT_V


def test_seq2seq_is_differentiable_end_to_end():
    batch = _batch()
    model = Seq2SeqRNN(SRC_V, TGT_V, 16, 32, 1, 0.0)
    loss = torch.nn.functional.cross_entropy(
        model(batch).reshape(-1, TGT_V),
        batch["dec_target"].reshape(-1),
        ignore_index=PAD,
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )


def test_greedy_decode_shape_and_dtype():
    batch = _batch()
    model = Seq2SeqRNN(SRC_V, TGT_V, 16, 32, 1, 0.0)
    model.eval()
    with torch.no_grad():
        out = model.greedy_decode(batch["enc_in"], batch["enc_len"], max_len=9)
    assert out.shape == (4, 9)
    assert out.dtype == torch.long


def test_greedy_decode_does_not_emit_sos():
    batch = _batch()
    model = Seq2SeqRNN(SRC_V, TGT_V, 16, 32, 1, 0.0)
    model.eval()
    with torch.no_grad():
        out = model.greedy_decode(batch["enc_in"], batch["enc_len"], max_len=9)
    assert (out != SOS).all()


def test_greedy_decode_pads_after_eos():
    batch = _batch(b=2)
    model = Seq2SeqRNN(SRC_V, TGT_V, 16, 32, 1, 0.0)
    model.eval()
    with torch.no_grad():
        out = model.greedy_decode(batch["enc_in"], batch["enc_len"], max_len=12)
    for row in out.tolist():
        if EOS in row:
            after = row[row.index(EOS) + 1 :]
            assert all(tok == PAD for tok in after)


def test_parameter_count_is_positive_and_all_trainable():
    total, trainable = count_parameters(Seq2SeqRNN(SRC_V, TGT_V, 16, 32, 1, 0.0))
    assert total > 0
    assert total == trainable
