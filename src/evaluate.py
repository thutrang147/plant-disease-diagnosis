"""Evaluate the frozen ResNet18 lab baseline on PlantVillage and PlantDoc tests.

Run from the repository root (no training or GPU required):
    python -m src.evaluate --checkpoint models/resnet18_lab_best.pt

The model is never updated. Both test partitions are evaluated exactly once
under the same ResNet18 ImageNet preprocessing used during training.
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
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from src.dataset import PlantDiseaseDataset

ROOT = Path(__file__).resolve().parent.parent


def evaluate_partition(model, manifest_path, transform, device, batch_size, class_names, output_dir):
    partition = manifest_path.stem
    dataset = PlantDiseaseDataset(manifest_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    true_labels = []
    predicted_labels = []
    confidences = []

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            scores, predictions = probabilities.max(dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            confidences.extend(scores.cpu().tolist())

    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(predicted_labels, dtype=int)
    indices = list(range(len(class_names)))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=indices, target_names=class_names,
        zero_division=0, digits=4,
    )
    (output_dir / f"{partition}_classification_report.txt").write_text(report, encoding="utf-8")

    predictions_df = dataset.metadata[["source", "row_id", "class_name", "label_id"]].copy()
    predictions_df["predicted_label_id"] = y_pred
    predictions_df["predicted_class"] = [class_names[idx] for idx in y_pred]
    predictions_df["confidence"] = confidences
    predictions_df["correct"] = y_true == y_pred
    predictions_df.to_csv(output_dir / f"{partition}_predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30,
    )
    ax.set_title(f"ResNet18 — {partition.replace('_', ' ').title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{partition}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"\n{'=' * 70}\n{partition}: {len(y_true)} images")
    print(f"Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")
    print(report)
    return {"split": partition, "images": len(y_true), "accuracy": accuracy, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "resnet18_lab_best.pt")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "resnet18_domain_gap")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}. Download it from Google Drive first.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available; use --device cpu")
    for filename in ("lab_test.csv", "field_test.csv", "label_map.json"):
        if not (args.splits / filename).is_file():
            parser.error(f"Missing {args.splits / filename}")

    label_to_id = json.loads((args.splits / "label_map.json").read_text(encoding="utf-8"))
    class_names = [name for name, index in sorted(label_to_id.items(), key=lambda pair: pair[1])]
    if list(sorted(label_to_id.values())) != list(range(5)):
        raise ValueError("Expected five labels with IDs 0 to 4")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Checkpoint architecture is not resnet18")
    if checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint and data/splits/label_map.json have incompatible class order")
    if checkpoint.get("pretrained_weights") != "IMAGENET1K_V1":
        raise ValueError("Unexpected preprocessing weights identifier")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    preprocessing = ResNet18_Weights.IMAGENET1K_V1.transforms()

    print("Checkpoint selected using validation macro-F1:", checkpoint["best_val_macro_f1"])
    print("Device:", device)
    results = [
        evaluate_partition(model, args.splits / f"{name}.csv", preprocessing, device,
                           args.batch_size, class_names, args.output)
        for name in ("lab_test", "field_test")
    ]

    summary = pd.DataFrame(results)
    summary.to_csv(args.output / "domain_gap_summary.csv", index=False)
    gap = results[0]["macro_f1"] - results[1]["macro_f1"]
    print("\n=== Frozen checkpoint comparison ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Macro-F1 domain gap (Lab Test - Field Test): {gap:+.4f}")
    print("Field Test contains only 47 images: interpret the gap cautiously.")
    print("No test images were used to train or select this checkpoint.")
    print("Outputs:", args.output)


if __name__ == "__main__":
    main()
