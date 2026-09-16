import pytest
import torch
from torch import nn

from src.q1_cnn.models import (
    MODEL_NAMES,
    PneumoNet,
    build_model,
    count_parameters,
    layer_table,
)


def test_model_names_list():
    assert MODEL_NAMES == [
        "pneumonet",
        "vgg16_frozen",
        "vgg16_finetune",
        "resnet50_frozen",
        "resnet50_finetune",
    ]


def test_pneumonet_forward_shape_at_224():
    model = PneumoNet(num_classes=3)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)


def test_pneumonet_forward_shape_at_128():
    model = PneumoNet(num_classes=3)
    assert model(torch.randn(2, 3, 128, 128)).shape == (2, 3)


def test_pneumonet_is_size_agnostic_because_of_global_pooling():
    model = PneumoNet(num_classes=3)
    a = model(torch.randn(1, 3, 96, 96))
    b = model(torch.randn(1, 3, 160, 160))
    assert a.shape == b.shape == (1, 3)


def test_pneumonet_parameter_count_is_in_the_expected_band():
    total, trainable = count_parameters(PneumoNet())
    assert 1_000_000 < total < 3_000_000
    assert total == trainable


def test_pneumonet_dropout_is_configurable():
    model = PneumoNet(dropout=0.5)
    rates = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]
    assert rates and all(r == 0.5 for r in rates)


def test_pneumonet_uses_batchnorm():
    assert any(isinstance(m, nn.BatchNorm2d) for m in PneumoNet().modules())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_outputs_three_logits(name):
    model = build_model(name, num_classes=3)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 3)


def test_frozen_vgg_trains_only_the_head():
    total, trainable = count_parameters(build_model("vgg16_frozen"))
    assert trainable < total * 0.02


def test_finetuned_vgg_trains_more_than_the_frozen_one():
    _, frozen = count_parameters(build_model("vgg16_frozen"))
    _, tuned = count_parameters(build_model("vgg16_finetune"))
    assert tuned > frozen


def test_frozen_resnet_trains_only_the_head():
    total, trainable = count_parameters(build_model("resnet50_frozen"))
    assert trainable < total * 0.02


def test_finetuned_resnet_unfreezes_layer4():
    model = build_model("resnet50_finetune")
    assert any(p.requires_grad for p in model.layer4.parameters())


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("efficientnet_b7")


def test_frozen_resnet_batchnorm_stays_in_eval_after_train_call():
    model = build_model("resnet50_frozen")
    model.train()
    assert model.bn1.training is False


def test_frozen_resnet_running_stats_do_not_drift():
    model = build_model("resnet50_frozen")
    model.train()
    before = model.bn1.running_mean.clone()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.allclose(model.bn1.running_mean, before)


def test_finetuned_resnet_layer4_batchnorm_still_trains():
    model = build_model("resnet50_finetune")
    model.train()
    layer4_bn = next(m for m in model.layer4.modules() if isinstance(m, nn.BatchNorm2d))
    assert layer4_bn.training is True
    assert model.bn1.training is False


def test_pneumonet_batchnorm_trains_normally():
    model = build_model("pneumonet")
    model.train()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bn_modules and all(m.training is True for m in bn_modules)


def test_layer_table_mentions_shapes_and_params():
    table = layer_table(PneumoNet(), (1, 3, 224, 224))
    assert "Output Shape" in table
    assert "Param" in table
