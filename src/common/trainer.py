"""Generic supervised classification training loop shared by both studies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.common.metrics import compute_classification_metrics


@dataclass
class TrainConfig:
    epochs: int
    patience: int
    monitor: str = "val_macro_f1"
    mode: str = "max"
    ckpt_path: Path | None = None
    grad_clip: float | None = None
    l1_lambda: float = 0.0


def evaluate_model(model, loader, device, criterion=None):
    """Run `model` over `loader` in eval mode.

    Returns (y_true, y_pred, y_prob, mean_loss) as numpy arrays on CPU.
    `mean_loss` is float("nan") when `criterion is None`.
    """
    model.eval()
    all_y_true = []
    all_y_prob = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
            probs = torch.softmax(logits, dim=1)
            all_y_true.append(yb.detach().cpu())
            all_y_prob.append(probs.detach().cpu())
            n += xb.size(0)

    y_true = torch.cat(all_y_true).numpy()
    y_prob = torch.cat(all_y_prob).numpy()
    y_pred = np.argmax(y_prob, axis=1)
    mean_loss = float(total_loss / n) if criterion is not None else float("nan")
    return y_true, y_pred, y_prob, mean_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    cfg: TrainConfig,
    class_names: list[str],
    scheduler=None,
) -> dict:
    """Train `model`, evaluate every epoch, early-stop, and return a history dict."""
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time_s": [],
    }

    ckpt_path = Path(cfg.ckpt_path) if cfg.ckpt_path is not None else None
    ckpt_written = False
    best_score = None
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()

        model.train()
        running_loss = 0.0
        running_correct = 0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            if cfg.l1_lambda > 0:
                l1 = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss = loss + cfg.l1_lambda * l1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == yb).sum().item()
            n_train += batch_size

        train_loss = running_loss / n_train
        train_acc = running_correct / n_train

        y_true, y_pred, y_prob, val_loss = evaluate_model(
            model, val_loader, device, criterion
        )
        val_metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro"]["f1"]

        epoch_values = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        }
        monitored = epoch_values[cfg.monitor]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitored)
            else:
                scheduler.step()

        epoch_time = time.time() - start

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["epoch_time_s"].append(float(epoch_time))

        if best_score is None:
            improved = True
        elif cfg.mode == "max":
            improved = monitored > best_score
        else:
            improved = monitored < best_score

        if improved:
            best_score = monitored
            best_epoch = epoch
            epochs_no_improve = 0
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                ckpt_written = True
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                stopped_early = True
                break

    if ckpt_written:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

    history["best_epoch"] = best_epoch
    history["best_score"] = float(best_score) if best_score is not None else float("nan")
    history["stopped_early"] = stopped_early

    return history
