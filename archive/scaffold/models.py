"""Starter torchvision factory; not used by the recorded notebook experiments.

The ViT option is ViT-B/16, not the research pipeline's timm ViT-Small.
"""

from torch import nn
from torchvision import models


def create_model(name: str, num_classes: int, pretrained: bool = True):
    """Return ResNet18 or ViT-B/16 with a replacement classification head.

    ``name`` is ``resnet18`` or ``vit_b_16``; ``num_classes`` must be positive.
    ``pretrained`` selects torchvision's default weights, not a saved research
    checkpoint. Unsupported names raise ``ValueError``.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer")

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model name: {name}")
