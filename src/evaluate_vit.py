"""Evaluate a frozen ViT-Small checkpoint on Lab Test and Field Test.

Run locally from the root of plant-disease-diagnosis:
    python -m src.evaluate_vit --checkpoint models/vit_small_lab_best.pt

The model is never updated. The same checkpoint and its saved preprocessing
are used for both test sets. Test results must not be used for model tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.dataset import PlantDiseaseDataset

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ARCHITECTURE = "vit_small_patch16_224.augreg_in1k"


def evaluate_partition(model, manifest, transform, device, batch_size, class_names, output_dir):
    name = manifest.stem
    dataset = PlantDiseaseDataset(manifest, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, confidences = [], [], []

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    labels = list(range(len(class_names)))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        digits=4, zero_division=0,
    )
    (output_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

    predictions = dataset.metadata[["source", "row_id", "class_name", "label_id"]].copy()
    predictions["predicted_label_id"] = y_pred
    predictions["predicted_class"] = [class_names[i] for i in y_pred]
    predictions["confidence"] = confidences
    predictions["correct"] = np.array(y_true) == np.array(y_pred)
    predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False,
    )
    ax.set_title(f"ViT-Small — {name.replace('_', ' ').title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"\n{name}: n={len(dataset)}, accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}")
    print(report)
    return {"split": name, "images": len(dataset), "accuracy": accuracy, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "vit_small_lab_best.pt")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "vit_small_domain_gap")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    for filename in ("lab_test.csv", "field_test.csv", "label_map.json"):
        if not (args.splits / filename).is_file():
            parser.error(f"Split file not found: {args.splits / filename}")

    if args.device == "auto":
        chosen = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        chosen = args.device
    if chosen == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu")
    if chosen == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu")
    device = torch.device(chosen)

    label_to_id = json.loads((args.splits / "label_map.json").read_text(encoding="utf-8"))
    class_names = [name for name, i in sorted(label_to_id.items(), key=lambda pair: pair[1])]
    if sorted(label_to_id.values()) != list(range(5)):
        raise ValueError("Expected five classes with IDs 0–4")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(f"Unexpected architecture: {checkpoint.get('architecture')!r}")
    if checkpoint.get("class_names") != class_names:
        raise ValueError("Checkpoint labels do not match data/splits/label_map.json")
    if "preprocessing_config" not in checkpoint:
        raise ValueError("Missing preprocessing_config in checkpoint")

    model = timm.create_model(EXPECTED_ARCHITECTURE, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    transform = timm.data.create_transform(**checkpoint["preprocessing_config"], is_training=False)

    args.output.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    print("Checkpoint validation macro-F1:", checkpoint["best_val_macro_f1"])
    print("Preprocessing:", transform)

    results = [
        evaluate_partition(
            model, args.splits / f"{split}.csv", transform, device,
            args.batch_size, class_names, args.output,
        )
        for split in ("lab_test", "field_test")
    ]
    summary = pd.DataFrame(results)
    summary.to_csv(args.output / "domain_gap_summary.csv", index=False)
    print("\n=== Domain gap summary ===")
    print(summary.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    print(f"Macro-F1 gap (Lab - Field): {results[0]['macro_f1'] - results[1]['macro_f1']:+.4f}")
    print("Caution: Field Test contains only 47 images and some non-field-like samples.")
    print("Predictions, reports and confusion matrices saved to:", args.output)


if __name__ == "__main__":
    main()
