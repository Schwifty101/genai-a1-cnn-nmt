"""Vanilla RNN encoder-decoder for the Q2 English<->Urdu translation task.

Only `nn.RNN` (tanh nonlinearity) is used anywhere in this module — no
gated recurrent unit, attention module, or attention-based sequence
architecture. This is a hard constraint of the assignment (not a modeling
preference): the point of Q2 is to characterize what a *vanilla* recurrent
encoder-decoder can and cannot do on this corpus, including its failure
modes. `tests/test_q2_models.py` greps this module's source for the banned
component names.

The encoder packs its embedded input with
`nn.utils.rnn.pack_padded_sequence(..., enforce_sorted=False)` before
running the RNN so that padded positions never enter the recurrence — the
final hidden state it returns is the state at each sequence's true last
token, not one contaminated by however many PAD steps follow it. The
decoder does not pack: it consumes `dec_in` in full and the loss mask
(`masked_loss` in `src.q2_rnn.train`) is what handles its padded positions,
which is cheaper than packing a second time for no correctness benefit
(the decoder's hidden state is discarded per-batch, never read back out).
"""

from __future__ import annotations

import torch
from torch import nn

from src.q2_rnn.tokenizer import EOS, PAD, SOS, Vocab


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, enc_in, enc_len):
        """`enc_in` [B, S], `enc_len` [B] -> final hidden state [num_layers, B, hidden_dim]."""
        embedded = self.dropout(self.embedding(enc_in))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, enc_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, dec_in, hidden):
        """`dec_in` [B, T], `hidden` [num_layers, B, hidden_dim] ->
        (logits [B, T, vocab_size], next hidden state)."""
        embedded = self.dropout(self.embedding(dec_in))
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc_out(output)
        return logits, hidden


class Seq2SeqRNN(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, batch: dict) -> torch.Tensor:
        """Full teacher forcing: `dec_in` already holds the gold prefix, so the
        decoder is run once over it in full. Sampled-token mixing for
        `teacher_forcing < 1.0` is implemented in `src.q2_rnn.train`, not here."""
        hidden = self.encoder(batch["enc_in"], batch["enc_len"])
        logits, _ = self.decoder(batch["dec_in"], hidden)
        return logits

    def greedy_decode(self, enc_in, enc_len, max_len=50) -> torch.Tensor:
        """Autoregressive greedy decoding starting from SOS.

        Loops `max_len` times, each time feeding back the previous step's
        argmax as the next decoder input. Never emits SOS itself (the loop
        only ever appends predicted tokens). Once a row predicts EOS, every
        subsequent position for that row is forced to PAD rather than left
        to whatever the model would otherwise produce.
        """
        device = enc_in.device
        batch_size = enc_in.size(0)
        hidden = self.encoder(enc_in, enc_len)

        dec_input = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs = []

        for _ in range(max_len):
            logits, hidden = self.decoder(dec_input, hidden)
            # SOS is a start-of-sequence marker only, never a legitimate
            # prediction target (it never appears in `dec_target`), so it
            # is excluded from the candidates before taking argmax.
            logits = logits.clone()
            logits[:, :, SOS] = float("-inf")
            raw_next = logits.argmax(dim=-1).squeeze(1)  # [B]
            next_token = torch.where(finished, torch.full_like(raw_next, PAD), raw_next)
            outputs.append(next_token)
            finished = finished | (raw_next == EOS)
            dec_input = next_token.unsqueeze(1)

        return torch.stack(outputs, dim=1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts as plain Python ints."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "&": r"\&",
    "{": r"\{",
    "}": r"\}",
    "^": r"\^{}",
    "~": r"\~{}",
}


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


def layer_table(model: nn.Module, src_vocab: Vocab, tgt_vocab: Vocab) -> str:
    """A layer-by-layer parameter table for `model`, as a LaTeX `table` environment.

    `torchinfo.summary` expects a single forward signature it can drive
    from a synthetic `input_size`; `Seq2SeqRNN.forward` instead takes a
    batch dict built by `collate_batch`, so this walks the module tree
    directly (`named_modules`, params with `recurse=False`) rather than
    tracing a forward pass. Mirrors the format of
    `src.q1_cnn.models.export_layer_table_latex` so both studies' layer
    tables read the same way in the report.
    """
    total, _trainable = count_parameters(model)

    rows = []
    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        n_params = sum(p.numel() for p in params)
        shape = ", ".join(str(tuple(p.shape)) for p in params)
        label = name or model.__class__.__name__
        rows.append((_latex_escape(label), _latex_escape(shape), f"{n_params:,}"))

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        (
            "\\caption{Q2 vanilla RNN encoder--decoder: layer parameter shapes "
            f"(src\\_vocab={len(src_vocab)}, tgt\\_vocab={len(tgt_vocab)}).}}"
        ),
        "\\label{tab:q2seq2seqlayers}",
        "\\begin{tabular}{llr}",
        "\\hline",
        "Layer & Parameter Shapes & Parameters \\\\",
        "\\hline",
    ]
    for label, shape, n_params in rows:
        lines.append(f"{label} & {shape} & {n_params} \\\\")
    lines.append("\\hline")
    lines.append(f"Total & & {total:,} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
