"""Re-evaluate frozen ResNet18 models on a GROUPED PlantDoc Field Dev split.

Run locally from the project root (no GPU needed):
    python -m src.evaluate_field_dev_grouped

Requires:
    data/splits/{field_dev,field_adapt_train,field_test,label_map}.csv/json
    models/resnet18_lab_best.pt
    models/resnet18_aug_photometric_best.pt
    models/resnet18_aug_geometric_best.pt

The original PlantDoc Field Test is checked for row-index overlap but NEVER loaded
for inference. The Field Dev set is for model development, not final confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
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
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "baseline": "resnet18_lab_best.pt",
    "photometric": "resnet18_aug_photometric_best.pt",
    "geometric": "resnet18_aug_geometric_best.pt",
}
EXPECTED_AUGMENTATION = {
    "baseline": None,
    "photometric": "photometric",
    "geometric": "geometric",
}


def validate_manifests(splits_dir: Path) -> tuple[pd.DataFrame, list[str], str]:
    """Check partition integrity and retrieve the fixed label order."""
    needed = ["field_dev.csv", "field_adapt_train.csv", "field_test.csv", "label_map.json", "field_dev_protocol.json"]
    for filename in needed:
        if not (splits_dir / filename).is_file():
            raise FileNotFoundError(f"Missing {splits_dir / filename}")

    manifests = {
        name: pd.read_csv(splits_dir / f"{name}.csv")
        for name in ("field_dev", "field_adapt_train", "field_test")
    }
    protocol = json.loads((splits_dir / "field_dev_protocol.json").read_text(encoding="utf-8"))
    if not protocol.get("strategy", "").startswith("Exact RGB pixel SHA-256 grouping"):
        raise ValueError("This evaluation requires the NEW grouped Field Dev split; "
                         "run prepare_field_dev_grouped.py first.")
    for name, frame in manifests.items():
        if name == "field_test" and len(frame) != 47:
            raise ValueError(f"Original Field Test size changed: {len(frame)}")
        if name != "field_test" and "exact_hash" not in frame:
            raise ValueError(f"{name} lacks exact_hash; grouped split is required")
        required = {"source", "row_id", "original_class", "class_name", "label_id"}
        if required - set(frame.columns):
            raise ValueError(f"{name}: missing columns {sorted(required - set(frame.columns))}")
        if not (frame["source"] == "plantdoc").all():
            raise ValueError(f"{name}: found non-PlantDoc images")
        if frame[["row_id", "label_id"]].isnull().any().any():
            raise ValueError(f"{name}: missing row_id or label_id")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate row_id")
        if set(frame["label_id"].astype(int)) != set(range(5)):
            raise ValueError(f"{name}: expected all five class IDs")

    adapt_hashes = set(manifests["field_adapt_train"]["exact_hash"])
    dev_hashes = set(manifests["field_dev"]["exact_hash"])
    if adapt_hashes.intersection(dev_hashes):
        raise ValueError("Exact pixel hashes overlap between grouped Adapt Train and Dev")
    if (len(manifests["field_adapt_train"]) != protocol["adaptation_train_images"]
        or len(manifests["field_dev"]) != protocol["development_images"]):
        raise ValueError("Grouped split manifest counts differ from protocol")

    for left, right in (
        ("field_dev", "field_adapt_train"),
        ("field_dev", "field_test"),
        ("field_adapt_train", "field_test"),
    ):
        a = set(manifests[left]["row_id"].astype(int))
        b = set(manifests[right]["row_id"].astype(int))
        if a & b:
            raise ValueError(f"Overlapping row IDs: {left} and {right}")

    mapping = json.loads((splits_dir / "label_map.json").read_text(encoding="utf-8"))
    if sorted(mapping.values()) != list(range(5)):
        raise ValueError("Label map must contain class IDs 0..4")
    class_names = [name for name, index in sorted(mapping.items(), key=lambda item: item[1])]

    for name, frame in manifests.items():
        expected = frame["class_name"].map(mapping)
        if expected.isna().any() or not (expected.astype(int) == frame["label_id"].astype(int)).all():
            raise ValueError(f"{name}: label_id and class_name do not match label_map.json")

    file_sha256 = hashlib.sha256((splits_dir / "field_dev.csv").read_bytes()).hexdigest()
    return manifests["field_dev"], class_names, file_sha256


def load_model(checkpoint_path: Path, variant: str, class_names: list[str], device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError(f"{variant}: checkpoint architecture is not ResNet18")
    if checkpoint.get("class_names") != class_names:
        raise ValueError(f"{variant}: checkpoint label order differs from label_map.json")
    if checkpoint.get("pretrained_weights") != "IMAGENET1K_V1":
        raise ValueError(f"{variant}: unexpected preprocessing / pretrained weights identifier")
    if variant != "baseline" and checkpoint.get("augmentation") != EXPECTED_AUGMENTATION[variant]:
        raise ValueError(f"{variant}: checkpoint augmentation metadata is inconsistent")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    predictions, confidences, true_labels = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            values, ids = probs.max(dim=1)
            predictions.extend(ids.cpu().tolist())
            confidences.extend(values.cpu().tolist())
            true_labels.extend(labels.tolist())
    return np.asarray(true_labels), np.asarray(predictions), np.asarray(confidences)


def save_results(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    prediction_rows = metadata[["source", "row_id", "class_name", "label_id"]].copy()
    prediction_rows["predicted_label_id"] = y_pred
    prediction_rows["predicted_class"] = [class_names[i] for i in y_pred]
    prediction_rows["confidence"] = confidence
    prediction_rows["correct"] = y_true == y_pred
    prediction_rows.to_csv(output_dir / "predictions.csv", index=False)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
    )
    ax.set_title(f"ResNet18 {name.title()} — Field Dev (n={len(y_true)})")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    per_class = [
        {
            "experiment": name,
            "class_name": class_name,
            "support": int(support[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
        }
        for i, class_name in enumerate(class_names)
    ]
    row = {
        "experiment": name,
        "images": len(y_true),
        "correct": int((y_true == y_pred).sum()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    print(f"{name:>12}: Accuracy={accuracy:.4f}, Macro-F1={macro_f1:.4f} ")
    return row, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "field_dev_comparison_grouped")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available on this machine; use --device cpu")

    metadata, class_names, manifest_hash = validate_manifests(args.splits_dir)
    print(f"Verified grouped split: Field Dev ({len(metadata)}) and "
          "Field Adapt Train are disjoint by exact image hash; "
          "original Field Test (47) remains untouched.")
    print("Field Dev manifest SHA256:", manifest_hash)
    print("Class order:", class_names)

    # Lazily import the dataset: original images are read through Hugging Face.
    from src.dataset import PlantDiseaseDataset

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    dataset = PlantDiseaseDataset(args.splits_dir / "field_dev.csv", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, per_class_results = [], []
    for variant, file_name in MODEL_FILES.items():
        model, checkpoint = load_model(args.models_dir / file_name, variant, class_names, device)
        print(f"Checkpoint '{variant}' Lab Val Macro-F1: {checkpoint['best_val_macro_f1']:.4f}")
        y_true, y_pred, confidence = predict(model, loader, device)
        if len(y_true) != len(metadata) or not np.array_equal(y_true, metadata["label_id"].to_numpy()):
            raise ValueError(f"{variant}: output labels do not align with field_dev.csv")
        result, per_class = save_results(
            variant, y_true, y_pred, confidence,
            metadata, class_names, args.output_dir / variant
        )
        result["lab_val_macro_f1"] = float(checkpoint["best_val_macro_f1"])
        results.append(result)
        per_class_results.extend(per_class)
        del model

    summary = pd.DataFrame(results)
    summary["delta_field_dev_f1_vs_baseline"] = summary["macro_f1"] - summary.loc[
        summary["experiment"] == "baseline", "macro_f1"
    ].iloc[0]
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(per_class_results).to_csv(args.output_dir / "per_class_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["experiment"], summary["macro_f1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"ResNet18 — Grouped Field Dev (n={len(metadata)})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "field_dev_macro_f1.png", dpi=160)
    plt.close(fig)

    (args.output_dir / "evaluation_protocol.json").write_text(
        json.dumps({
            "evaluation_set": "data/splits/field_dev.csv",
            "field_dev_sha256": manifest_hash,
            "field_dev_images": len(metadata),
            "evaluated_variants": list(MODEL_FILES),
            "training_updates_performed": False,
            "original_field_test_used_for_inference": False,
            "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
            "note": "Development metrics only. Check near-duplicates and obtain an independent field test for confirmatory claims.",
        }, indent=2) + "\n", encoding="utf-8"
    )

    print("\nSummary (development only):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Saved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
