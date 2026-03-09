"""
Model factory for CIFAR-10 training.

All models are adapted for 32x32 input resolution.
"""

import torch.nn as nn
import torchvision.models as tv_models


SUPPORTED_MODELS = ("resnet18", "resnet50", "efficientnet_b0")


def _make_resnet_cifar(base_fn):
    model = base_fn(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def create_model(name: str) -> nn.Module:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")

    if name == "resnet18":
        return _make_resnet_cifar(tv_models.resnet18)
    elif name == "resnet50":
        return _make_resnet_cifar(tv_models.resnet50)
    elif name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 10)
        return model
    raise ValueError(f"Unknown model '{name}'")
