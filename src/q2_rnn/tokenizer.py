"""Word-level tokenization and vocabulary for the Q2 English<->Urdu pipeline.

Special token ids are fixed and non-negotiable across the whole project:
`<pad>=0`, `<sos>=1`, `<eos>=2`, `<unk>=3`. Vocabularies must be built on the
training split only (see `_main` / Task 10 Step 7) — building them over the
full dataset would leak validation and test vocabulary into the model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


def tokenize(text: str) -> list[str]:
    """Whitespace split on already-normalized text."""
    return text.split()


class Vocab:
    """Word-level vocabulary: `itos`/`stoi` with the four specials fixed first."""

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(tok, UNK) for tok in tokens]

    def decode(self, ids: list[int], strip_specials: bool = True) -> list[str]:
        out = []
        for i in ids:
            if strip_specials and i < len(SPECIALS):
                continue
            out.append(self.itos[i])
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def build(cls, sentences: Iterable[str], min_freq: int = 2) -> "Vocab":
        """Count tokens, keep those at or above `min_freq`, order by (-count, token).

        Ordering by `(-count, token)` breaks frequency ties alphabetically so
        the result is reproducible run to run regardless of input order or
        hashing.
        """
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(tokenize(sentence))

        kept = [tok for tok, count in counter.items() if count >= min_freq]
        kept.sort(key=lambda tok: (-counter[tok], tok))

        itos = list(SPECIALS) + kept
        return cls(itos)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path) as f:
            data = json.load(f)
        return cls(data["itos"])


def oov_rate(vocab: Vocab, sentences: Iterable[str]) -> float:
    """Fraction of whitespace tokens across `sentences` that are out-of-vocabulary."""
    total = 0
    oov = 0
    for sentence in sentences:
        for tok in tokenize(sentence):
            total += 1
            if tok not in vocab.stoi:
                oov += 1
    return oov / total if total > 0 else 0.0


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Q2 word-level vocabularies from the training split only."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=Path("results/q2_split.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/vocab"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    min_freq = int(cfg.get("tokenizer", {}).get("min_freq", 2))

    df = pd.read_csv(args.split_csv, dtype={"en": str, "ur": str})
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    src_vocab = Vocab.build(train_df["en"], min_freq=min_freq)
    tgt_vocab = Vocab.build(train_df["ur"], min_freq=min_freq)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_vocab.save(args.out_dir / "src_vocab.json")
    tgt_vocab.save(args.out_dir / "tgt_vocab.json")

    val_en_oov = oov_rate(src_vocab, val_df["en"])
    val_ur_oov = oov_rate(tgt_vocab, val_df["ur"])

    report = {
        "min_freq": min_freq,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "val_en_oov_rate": val_en_oov,
        "val_ur_oov_rate": val_ur_oov,
    }
    with open(args.out_dir / "vocab_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"src (en) vocab size: {len(src_vocab)}")
    print(f"tgt (ur) vocab size: {len(tgt_vocab)}")
    print(f"val en oov rate: {val_en_oov:.4f}")
    print(f"val ur oov rate: {val_ur_oov:.4f}")


if __name__ == "__main__":
    _main()
