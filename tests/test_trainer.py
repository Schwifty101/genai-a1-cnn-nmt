import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.device import pick_device
from src.common.seed import set_seed
from src.common.trainer import TrainConfig, evaluate_model, train_model

CLASSES = ["a", "b", "c"]


def _tiny_problem(n=120, d=8, k=3):
    set_seed(42)
    centers = torch.randn(k, d) * 4
    y = torch.randint(0, k, (n,))
    x = centers[y] + torch.randn(n, d) * 0.3
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True), DataLoader(ds, batch_size=16)


def _model(d=8, k=3):
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, k))


def test_history_has_required_keys():
    train_loader, val_loader = _tiny_problem()
    model = _model()
    history = train_model(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        pick_device("cpu"),
        TrainConfig(epochs=3, patience=5),
        CLASSES,
    )
    for key in (
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "val_macro_f1",
        "epoch_time_s",
    ):
        assert len(history[key]) == 3, key
    assert isinstance(history["best_epoch"], int)
    assert isinstance(history["stopped_early"], bool)


def test_loss_decreases_on_a_separable_problem():
    train_loader, val_loader = _tiny_problem()
    model = _model()
    history = train_model(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        pick_device("cpu"),
        TrainConfig(epochs=8, patience=10),
        CLASSES,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_early_stopping_triggers_and_is_recorded(tmp_path):
    train_loader, val_loader = _tiny_problem()
    model = _model()
    history = train_model(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        pick_device("cpu"),
        TrainConfig(epochs=50, patience=2, ckpt_path=tmp_path / "best.pt"),
        CLASSES,
    )
    assert len(history["train_loss"]) < 50
    assert history["stopped_early"] is True


def test_checkpoint_file_is_written(tmp_path):
    train_loader, val_loader = _tiny_problem()
    model = _model()
    ckpt = tmp_path / "best.pt"
    train_model(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        pick_device("cpu"),
        TrainConfig(epochs=3, patience=5, ckpt_path=ckpt),
        CLASSES,
    )
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0


def test_evaluate_model_shapes():
    train_loader, val_loader = _tiny_problem()
    model = _model()
    y_true, y_pred, y_prob, loss = evaluate_model(
        model, val_loader, pick_device("cpu"), nn.CrossEntropyLoss()
    )
    n = len(val_loader.dataset)
    assert y_true.shape == (n,)
    assert y_pred.shape == (n,)
    assert y_prob.shape == (n, 3)
    assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-5)
    assert loss > 0


def test_l1_penalty_increases_reported_loss():
    train_loader, val_loader = _tiny_problem()
    set_seed(42)
    model_a = _model()
    h_a = train_model(
        model_a,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model_a.parameters(), lr=1e-3),
        pick_device("cpu"),
        TrainConfig(epochs=1, patience=5, l1_lambda=0.0),
        CLASSES,
    )
    set_seed(42)
    model_b = _model()
    h_b = train_model(
        model_b,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model_b.parameters(), lr=1e-3),
        pick_device("cpu"),
        TrainConfig(epochs=1, patience=5, l1_lambda=1e-2),
        CLASSES,
    )
    assert h_b["train_loss"][0] > h_a["train_loss"][0]


def test_reduce_lr_on_plateau_scheduler_does_not_crash():
    train_loader, val_loader = _tiny_problem()
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    history = train_model(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        pick_device("cpu"),
        TrainConfig(epochs=4, patience=10),
        CLASSES,
        scheduler=scheduler,
    )
    assert len(history["train_loss"]) == 4


def test_steplr_scheduler_still_steps():
    train_loader, val_loader = _tiny_problem()
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_lr = optimizer.param_groups[0]["lr"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_model(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        pick_device("cpu"),
        TrainConfig(epochs=3, patience=10),
        CLASSES,
        scheduler=scheduler,
    )
    assert optimizer.param_groups[0]["lr"] < initial_lr
