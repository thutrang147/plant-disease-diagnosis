"""Model factory for the supported image-classification architectures."""

from torch import nn
from torchvision import models


def create_model(name: str, num_classes: int, pretrained: bool = True):
    """Create a torchvision classifier with a head for ``num_classes``."""
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
