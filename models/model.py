"""Model definition.

Architecture matches the final production training run exactly: a
torchvision ResNet18 backbone with ImageNet weights and its final fully
connected layer replaced by a plain ``nn.Linear`` binary classification
head. Earlier notebook experiments used deeper heads (Sequential blocks
with dropout, frozen backbones, etc.) but those were superseded — this is
the architecture the shipped checkpoint was actually trained with.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Construct the ResNet18-based deepfake classifier.

    Args:
        num_classes: Number of output classes (2: Fake/Real).
        pretrained: Whether to initialize the backbone with ImageNet
            weights. Set to ``False`` when loading a trained checkpoint
            (the weights will be overwritten immediately after).

    Returns:
        An uncompiled, untrained-mode ``nn.Module`` (call ``.to(device)``
        and ``.train()``/``.eval()`` as appropriate).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
