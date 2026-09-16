"""Q1 model definitions: PneumoNet (the primary model) and four pre-trained
comparison baselines (VGG16 / ResNet50, each frozen-backbone and fine-tuned).

PneumoNet is the project's primary model — trained from scratch on the Q1
chest X-ray data. VGG16 and ResNet50 exist only as comparison baselines and
must never be presented as the primary result; `MODEL_NAMES` lists
`pneumonet` first so any table built from it preserves that order.

All five models share the same input convention: 3-channel, 224x224 (batch,
C, H, W), the same head-construction convention (a fresh `nn.Sequential`
head appended to a backbone), so they are trainable under identical
conditions.
"""

from __future__ import annotations

from pathlib import Path

import torchinfo
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    VGG16_Weights,
    resnet50,
    vgg16,
)


def _block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PneumoNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _freeze(model: nn.Module) -> None:
    """Set requires_grad=False on every parameter currently in `model`."""
    for p in model.parameters():
        p.requires_grad = False


def _build_vgg16(num_classes: int, dropout: float, finetune: bool) -> nn.Module:
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    _freeze(model)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(in_features, num_classes)
    )
    if finetune:
        # Last convolutional block only; the 10x lower backbone LR is
        # applied later via optimizer parameter groups, not here.
        for p in model.features[24:].parameters():
            p.requires_grad = True
    return model


def _build_resnet50(num_classes: int, dropout: float, finetune: bool) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    _freeze(model)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    if finetune:
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model


MODEL_NAMES = [
    "pneumonet",
    "vgg16_frozen",
    "vgg16_finetune",
    "resnet50_frozen",
    "resnet50_finetune",
]


def _freeze_frozen_batchnorm(model: nn.Module) -> nn.Module:
    """Keep BatchNorm layers whose parameters are all frozen in eval mode.

    `requires_grad=False` on a BatchNorm's affine parameters blocks
    gradient updates to them, but it does NOT stop the layer from updating
    its `running_mean` / `running_var` buffers while the module is in
    `train()` mode — those buffers are updated unconditionally by the
    forward pass whenever `training=True`, regardless of `requires_grad`.
    Left alone, a "frozen" backbone would keep silently adapting its
    BatchNorm statistics to the chest X-ray distribution every epoch,
    defeating the point of freezing it (measured drift: up to ~0.15 in a
    `resnet50_frozen` `bn1.running_mean` after a single forward/backward;
    see task-6-report.md).

    This wraps `model.train()` so that, on every call the trainer makes
    (once per epoch, since `train_model` calls `model.train()` at the top
    of each epoch), any BatchNorm submodule whose parameters are *all*
    `requires_grad=False` is forced back into eval mode right after the
    normal `train()` call runs — so it always uses its stored (e.g.
    ImageNet) running statistics and produces deterministic output. A
    one-time `.eval()` call on those submodules would not survive the next
    `model.train()`, which is why the override lives on `train()` itself
    rather than being applied once at construction time. BatchNorm layers
    with at least one trainable parameter (e.g. `layer4` in a `_finetune`
    model) are left alone and train normally.
    """
    original_train = model.train

    def train(mode: bool = True):
        original_train(mode)
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params = list(m.parameters(recurse=False))
                if params and not any(p.requires_grad for p in params):
                    m.eval()
        return model

    model.train = train
    return model


def build_model(name: str, num_classes: int = 3, dropout: float = 0.3) -> nn.Module:
    """Construct one of the five Q1 models by name.

    All five consume 3-channel, 224x224 input and use the same
    head-construction convention (a fresh `nn.Sequential` head), so they
    are trainable under identical conditions. Every model returned here
    has `_freeze_frozen_batchnorm` applied, so any BatchNorm layer whose
    parameters are entirely frozen stays in eval mode across every
    `model.train()` call the trainer makes; this changes nothing for
    `pneumonet` (no frozen BatchNorm parameters) but is what actually
    makes `*_frozen` mean "frozen" for the BatchNorm-backed ResNet50
    baselines.
    """
    builders = {
        "pneumonet": lambda: PneumoNet(num_classes=num_classes, dropout=dropout),
        "vgg16_frozen": lambda: _build_vgg16(num_classes, dropout, finetune=False),
        "vgg16_finetune": lambda: _build_vgg16(num_classes, dropout, finetune=True),
        "resnet50_frozen": lambda: _build_resnet50(num_classes, dropout, finetune=False),
        "resnet50_finetune": lambda: _build_resnet50(
            num_classes, dropout, finetune=True
        ),
    }
    if name not in builders:
        raise ValueError(f"Unknown model name {name!r}; expected one of {MODEL_NAMES}")
    model = builders[name]()
    return _freeze_frozen_batchnorm(model)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts as plain Python ints."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def layer_table(model: nn.Module, input_size: tuple[int, ...]) -> str:
    """Return a torchinfo layer-by-layer summary of `model` as text."""
    summary = torchinfo.summary(
        model,
        input_size=input_size,
        verbose=0,
        col_names=("output_size", "num_params"),
    )
    return str(summary)


def _latex_escape(text: str) -> str:
    """Escape characters LaTeX treats specially, underscores included."""
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in text)


def export_layer_table_latex(
    model: nn.Module,
    input_size: tuple[int, ...],
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    """Write a LaTeX `tabular` of Layer / Output Shape / Parameters to `out_path`."""
    summary = torchinfo.summary(
        model,
        input_size=input_size,
        verbose=0,
        col_names=("output_size", "num_params"),
    )

    rows = []
    for row in summary.summary_list:
        layer_name = _latex_escape(row.get_layer_name(show_var_name=False, show_depth=True))
        output_shape = _latex_escape(str(row.output_size))
        params_str = f"{row.num_params:,}" if row.num_params else "--"
        rows.append((layer_name, output_shape, params_str))

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llr}",
        "\\hline",
        "Layer & Output Shape & Parameters \\\\",
        "\\hline",
    ]
    for layer_name, output_shape, params_str in rows:
        lines.append(f"{layer_name} & {output_shape} & {params_str} \\\\")
    lines.append("\\hline")
    lines.append(f"Total & & {summary.total_params:,} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
