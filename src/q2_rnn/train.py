"""Q2 config-driven training entry point for the vanilla RNN encoder-decoder.

Q2 does not reuse `src.common.trainer` (the Q1 classification trainer): the
translation task needs a masked cross-entropy loss over padded target
sequences, gradient-norm capture for the exploding-gradient evidence the
report needs, scheduled-sampling teacher forcing, and a different history
shape (`train_ppl`/`val_ppl`/`grad_norm` instead of accuracy/F1). So the
epoch loop below is written from scratch for this task, deliberately.

Gradient clipping here is not just training hygiene: vanilla RNNs genuinely
explode on this corpus, and `torch.nn.utils.clip_grad_norm_` returns the
*pre-clip* norm it computed before scaling gradients down. That returned
value is captured into `history["grad_norm"]` (mean over an epoch's
batches) specifically so the report can show the explosion was real and
that clipping controlled it — it is evidence, not ceremony.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.common.device import pick_device
from src.common.plots import plot_xy
from src.common.seed import SEED, set_seed
from src.q2_rnn.dataset import build_loaders
from src.q2_rnn.models import Seq2SeqRNN, count_parameters
from src.q2_rnn.tokenizer import Vocab

_PPL_CLAMP = 1e4


def masked_loss(logits, target, mask):
    flat_logits = logits.reshape(-1, logits.size(-1))[mask.reshape(-1)]
    flat_target = target.reshape(-1)[mask.reshape(-1)]
    return F.cross_entropy(flat_logits, flat_target)


def _clamped_ppl(loss: float) -> float:
    """`exp(loss)` clamped at 1e4 so a diverging epoch cannot write `inf` to JSON."""
    try:
        value = math.exp(loss)
    except OverflowError:
        value = float("inf")
    return min(value, _PPL_CLAMP)


def _forward_scheduled_sampling(model: Seq2SeqRNN, batch: dict, teacher_forcing: float):
    """Step-by-step decoding for `teacher_forcing < 1.0`.

    At each step, with probability `1 - teacher_forcing`, the model's own
    previous-step argmax is substituted for the gold token that would
    otherwise be read from `dec_in` at the next position. Used only below
    1.0; at exactly 1.0 the caller uses the fast fully-teacher-forced batched
    path (`model(batch)`) instead, since every input position is already the
    gold prefix and no per-step Python loop is needed.
    """
    dec_in = batch["dec_in"]
    batch_size, t_len = dec_in.shape
    device = dec_in.device

    hidden = model.encoder(batch["enc_in"], batch["enc_len"])
    step_input = dec_in[:, 0:1]  # SOS for every row
    logits_steps = []
    for t in range(t_len):
        step_logits, hidden = model.decoder(step_input, hidden)
        logits_steps.append(step_logits)
        if t + 1 < t_len:
            gold_next = dec_in[:, t + 1 : t + 2]
            use_model = torch.rand(batch_size, 1, device=device) >= teacher_forcing
            model_next = step_logits.argmax(dim=-1)
            step_input = torch.where(use_model, model_next, gold_next)
    return torch.cat(logits_steps, dim=1)


def _forward_batch(model: Seq2SeqRNN, batch: dict, teacher_forcing: float):
    if teacher_forcing >= 1.0:
        return model(batch)
    return _forward_scheduled_sampling(model, batch, teacher_forcing)


def _run_epoch_train(model, loader, optimizer, device, grad_clip, teacher_forcing):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_norms = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = _forward_batch(model, batch, teacher_forcing)
        loss = masked_loss(logits, batch["dec_target"], batch["dec_mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_valid = int(batch["dec_mask"].sum().item())
        total_loss += loss.item() * n_valid
        total_tokens += n_valid
        grad_norms.append(float(grad_norm))

    mean_loss = total_loss / total_tokens if total_tokens else float("nan")
    mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
    return mean_loss, mean_grad_norm


@torch.no_grad()
def _run_epoch_eval(model, loader, device):
    """Validation loss, always computed fully teacher-forced (the gold prefix
    is used regardless of the training `teacher_forcing` setting) so val_loss
    is comparable epoch to epoch and across configs."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = masked_loss(logits, batch["dec_target"], batch["dec_mask"])
        n_valid = int(batch["dec_mask"].sum().item())
        total_loss += loss.item() * n_valid
        total_tokens += n_valid
    return total_loss / total_tokens if total_tokens else float("nan")


def run_training(
    cfg: dict,
    split_df: pd.DataFrame,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    out_dir: Path,
    device=None,
) -> dict:
    """Run one full Q2 training + validation cycle and persist its artifacts.

    Writes `out_dir / "best.pt"` (checkpointed on best validation loss),
    `out_dir / "result.json"`, and `out_dir / "curves.png"`. Only the
    validation split is evaluated here — test-set decoding/BLEU/error
    analysis is `src.q2_rnn.evaluate`'s job, run later against the saved
    checkpoint.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", SEED)
    set_seed(seed)

    resolved_device = pick_device() if device is None else device

    train_loader, val_loader, _test_loader = build_loaders(
        split_df,
        src_vocab,
        tgt_vocab,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        seed=seed,
    )

    model = Seq2SeqRNN(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.to(resolved_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    ckpt_path = out_dir / "best.pt"
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = None
    best_val_ppl = None
    epochs_no_improve = 0

    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()

        train_loss, grad_norm = _run_epoch_train(
            model,
            train_loader,
            optimizer,
            resolved_device,
            cfg["grad_clip"],
            cfg["teacher_forcing"],
        )
        val_loss = _run_epoch_eval(model, val_loader, resolved_device)

        train_ppl = _clamped_ppl(train_loss)
        val_ppl = _clamped_ppl(val_loss)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_ppl"].append(float(train_ppl))
        history["val_ppl"].append(float(val_ppl))
        history["grad_norm"].append(float(grad_norm))
        history["epoch_time_s"].append(float(epoch_time))

        print(
            f"epoch {epoch}/{cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_ppl={train_ppl:.2f} val_ppl={val_ppl:.2f} "
            f"grad_norm={grad_norm:.3f} time={epoch_time:.1f}s"
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"early stopping at epoch {epoch} (no improvement for {cfg['patience']})")
                break

    train_seconds = time.time() - start

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=resolved_device, weights_only=True)
        model.load_state_dict(state)

    params_total, _params_trainable = count_parameters(model)

    result = {
        "config": cfg,
        "history": history,
        "params_total": int(params_total),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else float("nan"),
        "best_val_ppl": float(best_val_ppl) if best_val_ppl is not None else float("nan"),
        "ckpt_path": str(ckpt_path),
        "train_seconds": float(train_seconds),
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    epochs_range = list(range(1, len(history["train_loss"]) + 1))
    plot_xy(
        epochs_range,
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
        out_dir / "curves.png",
        xlabel="epoch",
        ylabel="loss (masked cross-entropy)",
        title=f"Q2 vanilla RNN training curves ({cfg.get('run_name', out_dir.name)})",
    )

    return result


def load_config(path: Path, overrides: dict) -> dict:
    """Build a flat Q2 training config dict from `configs/q2_*.yaml` plus overrides.

    `overrides` (from repeated `--set key=value`, or `--run-name`) wins over
    anything in the file.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    model_raw = raw.get("model", {}) or {}
    train_raw = raw.get("train", {}) or {}

    cfg = {
        "embed_dim": model_raw.get("embed_dim"),
        "hidden_dim": model_raw.get("hidden_dim"),
        "num_layers": model_raw.get("num_layers"),
        "dropout": model_raw.get("dropout"),
        "batch_size": train_raw.get("batch_size"),
        "lr": train_raw.get("lr"),
        "epochs": train_raw.get("epochs"),
        "patience": train_raw.get("patience"),
        "grad_clip": train_raw.get("grad_clip"),
        "teacher_forcing": train_raw.get("teacher_forcing"),
        "num_workers": train_raw.get("num_workers", 0),
        "seed": raw.get("seed", SEED),
        "run_name": raw.get("run_name", "run"),
    }
    cfg.update(overrides or {})
    return cfg


def _parse_set_overrides(pairs: list[str]) -> dict:
    """Parse repeated `--set key=value` pairs, coercing values via YAML."""
    overrides = {}
    for pair in pairs:
        key, _, raw_value = pair.partition("=")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Q2 vanilla RNN training entry point")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--split-csv", type=Path, default=Path("results/q2_split.csv"))
    parser.add_argument("--vocab-dir", type=Path, default=Path("results/vocab"))
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config value, e.g. --set epochs=2",
    )
    args = parser.parse_args(argv)

    overrides = _parse_set_overrides(args.overrides)
    if args.run_name is not None:
        overrides["run_name"] = args.run_name

    cfg = load_config(args.config, overrides)

    split_df = pd.read_csv(args.split_csv, dtype={"en": str, "ur": str})
    src_vocab = Vocab.load(args.vocab_dir / "src_vocab.json")
    tgt_vocab = Vocab.load(args.vocab_dir / "tgt_vocab.json")

    out_dir = Path("runs/q2") / cfg["run_name"]
    result = run_training(cfg, split_df, src_vocab, tgt_vocab, out_dir)

    n_epochs = len(result["history"]["epoch_time_s"])
    avg_epoch_s = sum(result["history"]["epoch_time_s"]) / n_epochs if n_epochs else 0.0
    print(f"wrote {out_dir / 'result.json'}")
    print(f"params_total={result['params_total']:,}")
    print(f"best_val_loss={result['best_val_loss']:.4f} best_val_ppl={result['best_val_ppl']:.2f}")
    print(f"avg_epoch_time_s={avg_epoch_s:.3f} over {n_epochs} epochs")

    return result


if __name__ == "__main__":
    main()
