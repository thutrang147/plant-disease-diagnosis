"""Single-image inference with the frozen research checkpoints.

Run from the repository root; no training, dataset download or GPU is needed:
    python -m src.inference --checkpoint models/vit_small_mixed_best.pt leaf.jpg
    python -m src.inference --checkpoint models/resnet18_lab_best.pt --strategy five_crop leaf.jpg

The loader accepts the ResNet18 and ViT-Small checkpoint dictionaries written by
Notebooks 02-08 and the controlled Notebook 15 run. It checks the canonical class
order and architecture, then loads weights strictly. Preprocessing follows the
evaluation code: torchvision ``IMAGENET1K_V1`` transforms for ResNet18 and the
saved timm configuration for ViT-Small. Mixed-domain ViT checkpoints do not store
that configuration, so the value recorded for their source checkpoint (checked in
Notebook 12) is used. Images are EXIF-transposed before RGB conversion, matching
Notebook 11.

``standard`` inference forwards the whole image through the model's evaluation
transform. ``five_crop`` reuses Notebook 14's view construction: a 448 x 448
bicubic resize, four corner crops plus a centre crop, and the mean of the five
softmax vectors. Predictions are research outputs, not agronomic diagnoses; see
the README for the laboratory-to-field performance gap.
"""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import timm
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from src.evaluate_resolution_sensitivity import CLASS_NAMES, REGIMES, prepare_views

RESNET_ARCHITECTURE = "resnet18"
VIT_ARCHITECTURE = "vit_small_patch16_224.augreg_in1k"
RESNET_PREPROCESSING_SOURCE = "torchvision.ResNet18_Weights.IMAGENET1K_V1.transforms"
# Saved by Notebook 03 from timm.data.resolve_model_data_config and asserted for
# both ViT checkpoints in Notebook 12. Used only when a checkpoint lacks it.
VIT_RECORDED_PREPROCESSING = {
    "input_size": [3, 224, 224],
    "interpolation": "bicubic",
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "crop_pct": 0.9,
    "crop_mode": "center",
}
EXPECTED_PRETRAINING = {RESNET_ARCHITECTURE: "IMAGENET1K_V1", VIT_ARCHITECTURE: "augreg_in1k"}
STRATEGIES = {"standard": REGIMES[0], "five_crop": REGIMES[3]}
CHECKPOINT_SUFFIXES = (".pt", ".pth")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass(frozen=True)
class LoadedModel:
    """A frozen model with the preprocessing it was evaluated with."""

    model: nn.Module
    architecture: str
    standard_transform: Callable
    view_normalize: Callable
    preprocessing_source: str
    device: torch.device
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """Mean class probabilities over the views of one image."""

    label_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    strategy: str
    view_count: int

    def as_dict(self) -> dict:
        return {
            "label_id": self.label_id, "class_name": self.class_name,
            "confidence": self.confidence, "probabilities": self.probabilities,
            "strategy": self.strategy, "view_count": self.view_count,
        }


def resolve_device(name: str = "cpu") -> torch.device:
    """Return ``cpu``, ``cuda`` or ``mps``; ``auto`` picks the first available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device: {name!r}")
    return torch.device(name)


def _summarize_metadata(checkpoint: dict) -> dict:
    """Keep JSON-friendly scalar provenance fields; drop weights and large tables."""
    summary = {}
    for key, value in checkpoint.items():
        if key == "model_state_dict":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
    return summary


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> LoadedModel:
    """Load a research checkpoint after validating its labels and architecture.

    Raises ``ValueError`` for unsupported files, a non-canonical class order,
    an unexpected pretraining identifier, a ViT input size other than 224, or
    weights that do not match the architecture exactly.
    """
    path = Path(path)
    device = device if isinstance(device, torch.device) else resolve_device(device)
    try:
        # weights_only refuses arbitrary pickled objects from untrusted files.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError(f"{path.name}: not a readable PyTorch checkpoint") from error
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path.name}: not a research checkpoint dictionary with model_state_dict")
    if list(checkpoint.get("class_names", [])) != CLASS_NAMES:
        raise ValueError(f"{path.name}: class order differs from {CLASS_NAMES}")

    architecture = checkpoint.get("architecture")
    if architecture not in EXPECTED_PRETRAINING:
        raise ValueError(f"{path.name}: unsupported architecture {architecture!r}")
    pretraining = checkpoint.get("pretrained_weights")
    if pretraining is not None and pretraining != EXPECTED_PRETRAINING[architecture]:
        raise ValueError(f"{path.name}: unexpected pretrained_weights {pretraining!r}")

    if architecture == RESNET_ARCHITECTURE:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        standard = ResNet18_Weights.IMAGENET1K_V1.transforms()
        mean, std = standard.mean, standard.std
        source = RESNET_PREPROCESSING_SOURCE
    else:
        config = checkpoint.get("preprocessing_config")
        source = "checkpoint preprocessing_config"
        if config is None:
            config = VIT_RECORDED_PREPROCESSING
            source = "recorded ViT-Small source preprocessing (Notebooks 03/12)"
        if tuple(config["input_size"]) != (3, 224, 224):
            raise ValueError(f"{path.name}: expected a 3 x 224 x 224 ViT input")
        model = timm.create_model(VIT_ARCHITECTURE, pretrained=False, num_classes=len(CLASS_NAMES))
        standard = timm.data.create_transform(**config, is_training=False)
        mean, std = config["mean"], config["std"]

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"{path.name}: weights do not match {architecture}") from error
    model.eval().requires_grad_(False).to(device)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return LoadedModel(model, architecture, standard, normalize, source, device,
                       _summarize_metadata(checkpoint))


def load_image(source) -> Image.Image:
    """Open a path or file-like object, apply EXIF orientation, return RGB."""
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def predict_image(loaded: LoadedModel, image: Image.Image, strategy: str = "standard") -> Prediction:
    """Classify one RGB image with ``standard`` or ``five_crop`` inference."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    views, _ = prepare_views(image, STRATEGIES[strategy], loaded.standard_transform, loaded.view_normalize)
    with torch.inference_mode():
        view_probabilities = loaded.model(views.to(loaded.device)).float().softmax(-1).cpu()
    # Research aggregation: mean of float32 softmax vectors, not mean logits.
    probabilities = view_probabilities.mean(dim=0)
    label_id = int(probabilities.argmax())
    return Prediction(
        label_id=label_id,
        class_name=CLASS_NAMES[label_id],
        confidence=float(probabilities[label_id]),
        probabilities={name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        strategy=strategy,
        view_count=int(views.shape[0]),
    )


def discover_checkpoints(directory: str | Path) -> list[Path]:
    """Return checkpoint files directly inside ``directory``, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in CHECKPOINT_SUFFIXES)


def _expand_images(paths: list[Path]) -> list[Path]:
    images = []
    for path in paths:
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir()
                                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(path)
    return images


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="Image files or folders of images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="standard")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per image")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    try:
        images = _expand_images(args.images)
    except FileNotFoundError as error:
        parser.error(f"Image not found: {error}")
    if not images:
        parser.error("No images found")

    loaded = load_checkpoint(args.checkpoint, args.device)
    if not args.json:
        print(f"Checkpoint: {args.checkpoint.name} ({loaded.architecture}); "
              f"preprocessing: {loaded.preprocessing_source}; strategy: {args.strategy}")
    for path in images:
        prediction = predict_image(loaded, load_image(path), args.strategy)
        if args.json:
            print(json.dumps({"image": str(path), **prediction.as_dict()}))
        else:
            print(f"{path}: {prediction.class_name} ({prediction.confidence:.1%})")


if __name__ == "__main__":
    main()
