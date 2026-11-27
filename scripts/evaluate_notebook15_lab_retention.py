"""Lab-retention evaluation for the six sealed Notebook 15 checkpoints.

Notebook 15 compared Field-only and Mixed fine-tuning on Field Dev / Field Test
only. Historically, the benefit of Mixed training was source-domain retention
(Notebook 08: Lab Validation macro-F1 0.8185 Field-only vs 0.9556 Mixed), so the
controlled comparison is incomplete without it. This script performs
**inference only** on Lab Validation (1,341) and Lab Test (1,692):

* checkpoint SHA-256 values must match ``data/results/notebook15/original_evaluation_seal.json``;
* the shared Lab baseline (``models/resnet18_lab_best.pt``) is evaluated as the reference;
* ResNet18 ImageNet-1K V1 evaluation transform, float32, no gradients;
* the output directory must not exist; nothing in the imported run is modified.

Requirements: the imported run (default
``outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1``), the Lab
baseline checkpoint and the cached ``geraldmc/plantvillage-full`` dataset.
CPU is sufficient (roughly 7 × 3,033 images).

Usage from the repository root::

    python scripts/evaluate_notebook15_lab_retention.py \
        --output data/results/notebook15_lab_retention
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PlantDiseaseDataset  # noqa: E402

SEEDS = (42, 43, 44)
CONDITIONS = ("field_only", "mixed")
PARTITIONS = ("lab_val", "lab_test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    scores = []
    for k in range(5):
        tp = int(((y == k) & (p == k)).sum())
        fp = int(((y != k) & (p == k)).sum())
        fn = int(((y == k) & (p != k)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def build_model(checkpoint: Path) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.eval()


def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions.extend(model(images).argmax(dim=1).tolist())
            truth.extend(labels.tolist())
    return np.asarray(truth), np.asarray(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 15 Lab-retention inference")
    parser.add_argument("--run", type=Path,
                        default=ROOT / "outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/resnet18_lab_best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    seal = json.loads((ROOT / "data/results/notebook15/original_evaluation_seal.json").read_text())
    config = json.loads((ROOT / "data/results/notebook15/original_config.json").read_text())
    checkpoints = {("baseline", 0): args.baseline}
    for seed in SEEDS:
        for condition in CONDITIONS:
            checkpoints[(condition, seed)] = args.run / f"seed_{seed}" / f"{condition}_selected.pt"

    # Verify every checkpoint before any inference.
    hashes = {}
    for (condition, seed), path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256(path)
        expected = (config["initial_sha256"] if condition == "baseline"
                    else seal["seed_checkpoints"][str(seed)][f"{condition}_selected.pt"])
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {digest} != {expected}")
        hashes[(condition, seed)] = digest

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    loaders = {
        partition: DataLoader(
            PlantDiseaseDataset(ROOT / "data/splits" / f"{partition}.csv", transform=transform),
            batch_size=args.batch_size, shuffle=False, num_workers=0)
        for partition in PARTITIONS
    }

    rows = []
    for (condition, seed), path in checkpoints.items():
        model = build_model(path)
        for partition, loader in loaders.items():
            y, p = predict(model, loader)
            rows.append({
                "condition": condition, "seed": seed or "", "partition": partition, "n": len(y),
                "correct": int((y == p).sum()), "accuracy": float((y == p).mean()),
                "macro_f1": macro_f1(y, p), "checkpoint_sha256": hashes[(condition, seed)],
            })
            print(rows[-1], flush=True)
        del model

    output.mkdir(parents=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "lab_retention_by_checkpoint.csv", index=False)
    adapted = results[results.condition != "baseline"]
    summary = (adapted.groupby(["partition", "condition"])[["accuracy", "macro_f1"]]
               .agg(["mean", lambda s: s.std(ddof=1)]))
    summary.columns = [f"{m}_{'mean' if a == 'mean' else 'sd'}" for m, a in summary.columns]
    summary.reset_index().to_csv(output / "lab_retention_summary.csv", index=False)
    paired = adapted.pivot_table(index=["partition", "seed"], columns="condition", values="macro_f1")
    paired["mixed_minus_field_only"] = paired["mixed"] - paired["field_only"]
    paired.reset_index().to_csv(output / "lab_retention_paired.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "inference_only": True, "device": "cpu", "precision": "float32",
        "transform": "torchvision ResNet18_Weights.IMAGENET1K_V1.transforms()",
        "checkpoints_verified_against": "data/results/notebook15/original_evaluation_seal.json",
        "torch": torch.__version__, "threads": args.threads,
    }, indent=2))
    print(summary)
    print(paired)


if __name__ == "__main__":
    main()
