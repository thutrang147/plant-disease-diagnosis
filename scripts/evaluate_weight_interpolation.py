"""Weight-space interpolation between the Lab baseline and Notebook 15 Field-only checkpoints.

Question: can Lab retention comparable to Mixed rehearsal be obtained **without
Lab data during fine-tuning**, by linearly interpolating weights between the
shared Lab baseline (theta_0) and each sealed Field-only checkpoint (theta_FO)?
theta(alpha) = (1 - alpha) * theta_0 + alpha * theta_FO, in the spirit of
WiSE-FT (Wortsman et al., 2022) and PAINT (Ilharco et al., 2022). Those methods
were shown for zero-shot CLIP models; transfer to this supervised setting is the
hypothesis under test. Inference only, CPU, float32. No checkpoint is modified.

Protocol (declared in code before the first run):

* Curve: alpha in {0.0, 0.1, ..., 1.0} for seeds 42, 43, 44 on development data
  only (Lab Validation, Field Dev). Test partitions are not used for the curve.
* Two operating points, chosen from seed-mean development metrics:
  - ``alpha_field``: maximizes Field Dev macro-F1 (ties -> smaller alpha);
  - ``alpha_retention_matched``: largest alpha whose Lab Validation macro-F1 is
    at least the Notebook 15 Mixed mean (0.973961, computed from
    data/results/notebook15_lab_retention, Mixed seeds evaluated identically).
* Only the operating points (and alpha = 1, the Field-only checkpoint) are then
  evaluated on Lab Test, the historical Field Test and External v3, next to the
  sealed Mixed checkpoints.

Usage from the repository root::

    python scripts/evaluate_weight_interpolation.py --output data/results/weight_interpolation
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
from datasets import load_dataset
from PIL import ImageOps
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

SEEDS = (42, 43, 44)
ALPHAS = tuple(round(a / 10, 1) for a in range(11))
LABELS5 = [0, 1, 2, 3, 4]
DISEASE_IDS = [1, 2, 3, 4]
PLANTSEG = ("LFTTim/plantseg_disease_segmentation", "2fff5d63968d3fd0aca061e7348b0b1441c1535d")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate(theta0: dict, theta1: dict, alpha: float) -> dict:
    """Linear interpolation of floating-point tensors; integer buffers come from theta0."""
    if theta0.keys() != theta1.keys():
        raise ValueError("State dicts have different keys")
    return {k: (torch.lerp(v, theta1[k], alpha) if v.is_floating_point() else v.clone())
            for k, v in theta0.items()}


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model.eval()


def tensors_from_manifest(name: str, transform) -> tuple[torch.Tensor, np.ndarray]:
    dataset = PlantDiseaseDataset(ROOT / "data/splits" / f"{name}.csv", transform=transform)
    items = [dataset[i] for i in range(len(dataset))]
    return torch.stack([x for x, _ in items]), np.asarray([y for _, y in items])


def tensors_from_v3(transform) -> tuple[torch.Tensor, np.ndarray]:
    manifest = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    included = manifest[manifest.status.eq("included")]
    source = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    images = [transform(ImageOps.exif_transpose(source[int(r)]["image"]).convert("RGB"))
              for r in included.plantseg_row]
    return torch.stack(images), included.label_id.to_numpy()


def predict(model: nn.Module, x: torch.Tensor, batch: int = 128) -> np.ndarray:
    with torch.inference_mode():
        return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).argmax(1).numpy()


def score(partition: str, y: np.ndarray, p: np.ndarray) -> dict:
    labels = DISEASE_IDS if partition == "external_v3" else LABELS5
    return {"n": len(y), "correct": int((y == p).sum()), "accuracy": float((y == p).mean()),
            "macro_f1": macro_f1(y, p, labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/resnet18_lab_best.pt")
    parser.add_argument("--run", type=Path,
                        default=ROOT / "outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    seal = json.loads((ROOT / "data/results/notebook15/original_evaluation_seal.json").read_text())
    config = json.loads((ROOT / "data/results/notebook15/original_config.json").read_text())
    if sha256(args.baseline) != config["initial_sha256"]:
        raise ValueError("Lab baseline SHA-256 differs from the Notebook 15 configuration")
    checkpoints = {}
    for seed in SEEDS:
        for condition in ("field_only", "mixed"):
            path = args.run / f"seed_{seed}" / f"{condition}_selected.pt"
            if sha256(path) != seal["seed_checkpoints"][str(seed)][f"{condition}_selected.pt"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            checkpoints[(condition, seed)] = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    theta0 = torch.load(args.baseline, map_location="cpu", weights_only=True)["model_state_dict"]

    retention = pd.read_csv(ROOT / "data/results/notebook15_lab_retention/lab_retention_summary.csv")
    mixed_lab_val = float(retention.query("partition == 'lab_val' and condition == 'mixed'").macro_f1_mean.iloc[0])

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    dev = {name: tensors_from_manifest(name, transform) for name in ("lab_val", "field_dev")}
    model = build_model()

    curve = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            model.load_state_dict(interpolate(theta0, checkpoints[("field_only", seed)], alpha), strict=True)
            for partition, (x, y) in dev.items():
                curve.append({"seed": seed, "alpha": alpha, "partition": partition, **score(partition, y, predict(model, x))})
            print("curve", seed, alpha, flush=True)
    curve = pd.DataFrame(curve)
    mean = curve.groupby(["partition", "alpha"]).macro_f1.mean().unstack(0)
    alpha_field = float(mean.field_dev[mean.field_dev == mean.field_dev.max()].index.min())
    eligible = mean.index[mean.lab_val >= mixed_lab_val]
    alpha_matched = float(eligible.max()) if len(eligible) else None
    points = {"alpha_field": alpha_field, "alpha_retention_matched": alpha_matched, "field_only": 1.0}

    test = {name: tensors_from_manifest(name, transform) for name in ("lab_test", "field_test")}
    test["external_v3"] = tensors_from_v3(transform)
    test["field_dev"] = dev["field_dev"]
    final = []
    for seed in SEEDS:
        candidates = {name: interpolate(theta0, checkpoints[("field_only", seed)], a)
                      for name, a in points.items() if a is not None}
        candidates["mixed"] = checkpoints[("mixed", seed)]
        for name, state in candidates.items():
            model.load_state_dict(state, strict=True)
            for partition, (x, y) in test.items():
                final.append({"seed": seed, "point": name, "alpha": points.get(name, ""),
                              "partition": partition, **score(partition, y, predict(model, x))})
        print("final", seed, flush=True)
    final = pd.DataFrame(final)

    output.mkdir(parents=True)
    curve.to_csv(output / "development_curve.csv", index=False)
    final.to_csv(output / "operating_points.csv", index=False)
    (final.groupby(["partition", "point"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
     .round(4).to_csv(output / "operating_points_summary.csv"))
    (output / "protocol.json").write_text(json.dumps({
        "alphas": ALPHAS, "selection_data": ["lab_val", "field_dev"], "points": points,
        "mixed_lab_val_macro_f1_threshold": mixed_lab_val,
        "test_partitions": ["lab_test", "field_test", "external_v3"],
        "baseline_sha256": config["initial_sha256"], "torch": torch.__version__, "threads": args.threads,
        "interpolation": "torch.lerp on floating-point tensors; integer buffers from the Lab baseline",
    }, indent=2) + "\n")
    print(json.dumps(points))
    print(final.groupby(["partition", "point"])[["accuracy", "macro_f1"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
