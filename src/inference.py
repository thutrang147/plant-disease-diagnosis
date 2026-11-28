"""Single-image inference helpers shared by the demo app.

Loads project checkpoints with the same architecture and preprocessing rules as
Notebook 11 and the round-2 evaluation scripts (torchvision ResNet18 transform;
ViT-Small transform from the Lab baseline's stored ``preprocessing_config``), and
computes Grad-CAM on ResNet18 ``layer4``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"]
VIT_FALLBACK_CONFIG = {"input_size": (3, 224, 224), "interpolation": "bicubic", "mean": (0.5, 0.5, 0.5),
                       "std": (0.5, 0.5, 0.5), "crop_pct": 0.9, "crop_mode": "center"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registry() -> list[dict]:
    return json.loads((ROOT / "data/checkpoints.json").read_text())["checkpoints"]


def load_checkpoint(path: Path, vit_config: dict | None = None):
    """Return (model, transform, metadata) for a project checkpoint, in eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if list(state["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Unexpected class order in {path}")
    architecture = state["architecture"]
    if architecture == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    else:
        import timm
        model = timm.create_model(architecture, pretrained=False, num_classes=len(CLASS_NAMES))
        config = state.get("preprocessing_config") or vit_config or VIT_FALLBACK_CONFIG
        transform = timm.data.create_transform(**config, is_training=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    metadata = {k: v for k, v in state.items() if k not in ("model_state_dict", "optimizer_state_dict")}
    return model.eval(), transform, metadata


def prepare(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def predict(model: nn.Module, transform, image: Image.Image) -> np.ndarray:
    """Softmax probabilities for one RGB image."""
    with torch.inference_mode():
        return torch.softmax(model(transform(image).unsqueeze(0)), 1)[0].numpy()


def grad_cam(model: nn.Module, transform, image: Image.Image, target: int) -> np.ndarray:
    """Grad-CAM on ResNet18 layer4 for class ``target``; 224x224 map in [0, 1]."""
    if not hasattr(model, "layer4"):
        raise ValueError("Grad-CAM is implemented for ResNet18 checkpoints only")
    store = {}
    handle = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logits = model(transform(image).unsqueeze(0))
            grads = torch.autograd.grad(logits[0, target], store["a"])[0]
    finally:
        handle.remove()
    cam = F.relu((grads.mean(dim=(2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def model_view(image: Image.Image, transform=None) -> Image.Image:
    """The image region a model actually sees, recovered from its own eval transform.

    The transform is applied and its final normalization inverted, so the geometry
    (resize, crop percentage, interpolation) matches the model exactly. Without a
    transform, the ResNet18 geometry (Resize 256, CenterCrop 224) is used.
    """
    if transform is None:
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    tensor = transform(image)
    mean = getattr(transform, "mean", None)
    std = getattr(transform, "std", None)
    if mean is None:  # torchvision Compose / timm transforms: find the Normalize step
        for step in getattr(transform, "transforms", []):
            if hasattr(step, "mean") and hasattr(step, "std"):
                mean, std = step.mean, step.std
    mean = torch.as_tensor(mean if mean is not None else (0.0, 0.0, 0.0)).view(3, 1, 1)
    std = torch.as_tensor(std if std is not None else (1.0, 1.0, 1.0)).view(3, 1, 1)
    pixels = ((tensor * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, transform=None) -> Image.Image:
    import matplotlib
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(model_view(image, transform).convert("RGB"), dtype=np.float32)
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))
