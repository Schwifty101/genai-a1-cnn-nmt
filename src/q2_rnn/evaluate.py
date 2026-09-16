"""Q2 test-set evaluation: greedy decoding, corpus BLEU, and an error taxonomy.

This is the only place Q2 test data is read (`run_training` in
`src.q2_rnn.train` only ever evaluates on the validation split, so
hyperparameter/config choices never see test data). `evaluate_run` loads a
trained run's checkpoint, greedy-decodes the test split, scores it with
sacreBLEU, and folds the result back into that run's `result.json` as a
`test_metrics` key (added alongside, never replacing, the training history).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import sacrebleu
import torch

from src.common.device import pick_device
from src.common.seed import SEED
from src.q2_rnn.dataset import build_loaders
from src.q2_rnn.models import Seq2SeqRNN
from src.q2_rnn.tokenizer import EOS, PAD, Vocab

_BUCKET_BOUNDS = {
    "1-5": (1, 5),
    "6-10": (6, 10),
    "11-15": (11, 15),
    "16+": (16, math.inf),
}


@torch.no_grad()
def decode_split(model, loader, tgt_vocab: Vocab, device, max_len: int = 50):
    """Greedy-decode every batch in `loader`.

    Returns `(hypotheses, references)`, one detokenized (space-joined)
    string pair per example, in loader order.

    Ids are decoded with `Vocab.decode(..., strip_specials=False)` after PAD
    and EOS are trimmed by hand (PAD/EOS ends the sequence; nothing past it
    is kept). `strip_specials=False` matters: `Vocab.decode`'s default
    (`strip_specials=True`) drops any id below 4 — which includes `<unk>`
    (id 3), not just PAD/SOS/EOS. Using the default would silently erase
    every `<unk>` prediction from the returned strings, making
    `error_taxonomy`'s `unk_rate` vacuously zero instead of reflecting what
    the model actually produced.
    """
    model.eval()
    model.to(device)

    hyps: list[str] = []
    refs: list[str] = []

    for batch in loader:
        enc_in = batch["enc_in"].to(device)
        enc_len = batch["enc_len"].to(device)
        dec_target = batch["dec_target"]
        dec_mask = batch["dec_mask"]

        pred_ids = model.greedy_decode(enc_in, enc_len, max_len=max_len)

        for i in range(pred_ids.size(0)):
            hyp_ids = []
            for tok_id in pred_ids[i].tolist():
                if tok_id in (EOS, PAD):
                    break
                hyp_ids.append(tok_id)
            hyps.append(" ".join(tgt_vocab.decode(hyp_ids, strip_specials=False)))

            ref_ids = [
                tok_id for tok_id in dec_target[i][dec_mask[i]].tolist() if tok_id != EOS
            ]
            refs.append(" ".join(tgt_vocab.decode(ref_ids, strip_specials=False)))

    return hyps, refs


def corpus_bleu_score(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score


def error_taxonomy(hyps: list[str], refs: list[str], src: list[str]) -> dict:
    """Error taxonomy over a decoded split.

    - `unk_rate`: share of hypothesis tokens (pooled across all hypotheses)
      equal to the literal string `"<unk>"`.
    - `repetition_rate`: share of hypotheses, of length >= 3 tokens, where
      some single token accounts for more than half of that hypothesis's
      tokens.
    - `empty_rate`: share of blank hypotheses.
    - `mean_len_ratio`: mean, over examples with a non-empty reference, of
      hypothesis token count / reference token count.
    - `bleu_by_src_len_bucket`: corpus BLEU computed separately over the
      examples whose source sentence falls in each of four source-length
      buckets (by whitespace token count); an empty bucket reports
      `float("nan")`, never 0.0 and never a crash.
    """
    n = len(hyps)
    if n == 0:
        return {
            "n": 0,
            "unk_rate": float("nan"),
            "repetition_rate": float("nan"),
            "empty_rate": float("nan"),
            "mean_len_ratio": float("nan"),
            "bleu_by_src_len_bucket": {name: float("nan") for name in _BUCKET_BOUNDS},
        }

    total_tokens = 0
    unk_tokens = 0
    repetitive = 0
    empty = 0
    len_ratios = []

    for hyp, ref in zip(hyps, refs):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()

        total_tokens += len(hyp_tokens)
        unk_tokens += sum(1 for tok in hyp_tokens if tok == "<unk>")

        if len(hyp_tokens) >= 3:
            most_common_count = Counter(hyp_tokens).most_common(1)[0][1]
            if most_common_count > len(hyp_tokens) / 2:
                repetitive += 1

        if hyp.strip() == "":
            empty += 1

        if len(ref_tokens) > 0:
            len_ratios.append(len(hyp_tokens) / len(ref_tokens))

    unk_rate = unk_tokens / total_tokens if total_tokens > 0 else 0.0
    repetition_rate = repetitive / n
    empty_rate = empty / n
    mean_len_ratio = sum(len_ratios) / len(len_ratios) if len_ratios else float("nan")

    bucket_hyps: dict[str, list[str]] = {name: [] for name in _BUCKET_BOUNDS}
    bucket_refs: dict[str, list[str]] = {name: [] for name in _BUCKET_BOUNDS}
    for hyp, ref, src_sent in zip(hyps, refs, src):
        src_len = len(src_sent.split())
        for name, (lo, hi) in _BUCKET_BOUNDS.items():
            if lo <= src_len <= hi:
                bucket_hyps[name].append(hyp)
                bucket_refs[name].append(ref)
                break

    bleu_by_bucket = {
        name: (
            corpus_bleu_score(bucket_hyps[name], bucket_refs[name])
            if bucket_hyps[name]
            else float("nan")
        )
        for name in _BUCKET_BOUNDS
    }

    return {
        "n": n,
        "unk_rate": unk_rate,
        "repetition_rate": repetition_rate,
        "empty_rate": empty_rate,
        "mean_len_ratio": mean_len_ratio,
        "bleu_by_src_len_bucket": bleu_by_bucket,
    }


def evaluate_run(
    run_dir: Path,
    split_df: pd.DataFrame,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device=None,
    max_len: int = 50,
) -> dict:
    """Decode `run_dir`'s checkpoint over the held-out test split and score it.

    Reads `run_dir/result.json` for the run's `config` and `ckpt_path`,
    rewrites it in place with a `test_metrics` key added (BLEU plus the
    full `error_taxonomy` breakdown), and writes `run_dir/samples.json`
    with a handful of (source, reference, hypothesis) triples for
    qualitative inspection in the report.
    """
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    stored = json.loads(result_path.read_text())
    cfg = stored["config"]

    resolved_device = pick_device() if device is None else device

    _train_loader, _val_loader, test_loader = build_loaders(
        split_df,
        src_vocab,
        tgt_vocab,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        seed=cfg.get("seed", SEED),
    )

    model = Seq2SeqRNN(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    ckpt_path = Path(stored["ckpt_path"])
    state = torch.load(ckpt_path, map_location=resolved_device, weights_only=True)
    model.load_state_dict(state)
    model.to(resolved_device)

    hyps, refs = decode_split(model, test_loader, tgt_vocab, resolved_device, max_len=max_len)

    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    src_sentences = test_df["en"].tolist()

    bleu = corpus_bleu_score(hyps, refs)
    taxonomy = error_taxonomy(hyps, refs, src_sentences)

    test_metrics = {"bleu": bleu, **taxonomy}
    stored["test_metrics"] = test_metrics
    result_path.write_text(json.dumps(stored, indent=2))

    samples = [
        {"src": s, "ref": r, "hyp": h}
        for s, r, h in list(zip(src_sentences, refs, hyps))[:20]
    ]
    (run_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))

    return test_metrics


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Q2 test-set evaluation")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=Path("results/q2_split.csv"))
    parser.add_argument("--vocab-dir", type=Path, default=Path("results/vocab"))
    parser.add_argument("--max-len", type=int, default=50)
    args = parser.parse_args(argv)

    split_df = pd.read_csv(args.split_csv, dtype={"en": str, "ur": str})
    src_vocab = Vocab.load(args.vocab_dir / "src_vocab.json")
    tgt_vocab = Vocab.load(args.vocab_dir / "tgt_vocab.json")

    metrics = evaluate_run(args.run, split_df, src_vocab, tgt_vocab, max_len=args.max_len)

    print(f"bleu={metrics['bleu']:.2f}")
    print(
        f"unk_rate={metrics['unk_rate']:.4f} "
        f"repetition_rate={metrics['repetition_rate']:.4f} "
        f"empty_rate={metrics['empty_rate']:.4f}"
    )
    print(f"mean_len_ratio={metrics['mean_len_ratio']:.4f}")
    print(f"bleu_by_src_len_bucket={metrics['bleu_by_src_len_bucket']}")

    return metrics


if __name__ == "__main__":
    main()
