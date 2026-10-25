import torch.nn as nn
from torchvision import models

MODES = ("scratch", "feature_extract", "finetune")


def build_model(arch: str = "resnet18", mode: str = "finetune", num_classes: int = 37):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    pretrained = mode != "scratch"

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"unknown arch: {arch}")

    # Freeze BEFORE replacing the head, so the new head stays trainable.
    if mode == "feature_extract":
        for p in model.parameters():
            p.requires_grad = False

    if arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable