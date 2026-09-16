"""PyTorch Dataset, collation, and DataLoader construction for the Q2 splits.

Reads a dataframe that already carries a `split` column (produced once by
`src.q2_rnn.splits.make_splits` and persisted to disk) — this module never
re-splits.

The shifting contract: `dec_in` and `dec_target` are the same length
`T = max_tgt_len + 1`, and position `t` of `dec_target` is what the model
must predict after consuming position `t` of `dec_in`. For a target
`[w1, w2]`: `dec_in = [SOS, w1, w2, PAD...]`, `dec_target = [w1, w2, EOS,
PAD...]`.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.seed import SEED, make_generator, seed_worker
from src.q2_rnn.tokenizer import EOS, PAD, SOS, Vocab, tokenize


class TranslationDataset(Dataset):
    """Wraps a split dataframe; `__getitem__` returns raw id lists, no specials attached.

    The collate function owns all special-token placement (SOS/EOS/PAD), so
    an item here is exactly the encoded source/target tokens.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        reverse_source: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        # Sutskever et al. (2014): reversing the source shortens the distance
        # between the first source words and the first target words, which
        # measurably helps an RNN encoder-decoder that has no attention. It
        # changes token order only -- no architectural component is added.
        self.reverse_source = reverse_source

        src_lens = self.df["en"].map(lambda t: len(tokenize(t)))
        tgt_lens = self.df["ur"].map(lambda t: len(tokenize(t)))
        assert bool((src_lens >= 1).all()) and bool((tgt_lens >= 1).all()), (
            "every row must have at least one token on both sides; "
            "Task 9's length filtering should already guarantee this"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        row = self.df.iloc[idx]
        src_ids = self.src_vocab.encode(tokenize(row["en"]))
        tgt_ids = self.tgt_vocab.encode(tokenize(row["ur"]))
        if self.reverse_source:
            src_ids = src_ids[::-1]
        return src_ids, tgt_ids


def collate_batch(batch):
    src_lists, tgt_lists = zip(*batch)
    src_lens = [len(s) for s in src_lists]
    tgt_lens = [len(t) + 1 for t in tgt_lists]  # +1 for SOS / EOS
    max_src, max_tgt = max(src_lens), max(tgt_lens)
    b = len(batch)

    enc_in = torch.full((b, max_src), PAD, dtype=torch.long)
    dec_in = torch.full((b, max_tgt), PAD, dtype=torch.long)
    dec_target = torch.full((b, max_tgt), PAD, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_lists, tgt_lists)):
        enc_in[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        dec_in[i, 0] = SOS
        dec_in[i, 1 : len(tgt) + 1] = torch.tensor(tgt, dtype=torch.long)
        dec_target[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        dec_target[i, len(tgt)] = EOS

    enc_len = torch.tensor(src_lens, dtype=torch.long)
    enc_mask = torch.arange(max_src)[None, :] < enc_len[:, None]
    dec_mask = torch.arange(max_tgt)[None, :] < torch.tensor(tgt_lens)[:, None]

    return {
        "enc_in": enc_in,
        "enc_len": enc_len,
        "enc_mask": enc_mask,
        "dec_in": dec_in,
        "dec_target": dec_target,
        "dec_mask": dec_mask,
    }


def build_loaders(
    split_df: pd.DataFrame,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    batch_size: int,
    num_workers: int,
    seed: int = SEED,
    reverse_source: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Partition `split_df` by its `split` column and build train/val/test loaders."""
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    train_ds = TranslationDataset(train_df, src_vocab, tgt_vocab, reverse_source)
    val_ds = TranslationDataset(val_df, src_vocab, tgt_vocab, reverse_source)
    test_ds = TranslationDataset(test_df, src_vocab, tgt_vocab, reverse_source)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
