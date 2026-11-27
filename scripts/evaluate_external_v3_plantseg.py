"""One-shot evaluation of frozen checkpoints on External v3 (screened PlantSeg).

Follows ``data/external_v3_plantseg/protocol.json``, which was committed before
this script was first run. Every checkpoint SHA-256 is checked against the
protocol or the Notebook 15 seal before inference. Inference only, CPU, float32.

Usage from the repository root (default checkpoint paths follow docs/workflow.md)::

    python scripts/evaluate_external_v3_plantseg.py --output data/results/external_v3_plantseg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from datasets import load_dataset
from PIL import ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.statistical_analysis import wilson_ci  # noqa: E402
from src.supplementary_analysis import holm, macro_f1, mcnemar_exact  # noqa: E402

V3_DIR = ROOT / "data/external_v3_plantseg"
PLANTSEG = ("LFTTim/plantseg_disease_segmentation", "2fff5d63968d3fd0aca061e7348b0b1441c1535d")
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"]
DISEASE_IDS = [1, 2, 3, 4]
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PlantSegRows(Dataset):
    def __init__(self, source, rows: list[int], labels: list[int], transform):
        self.source, self.rows, self.labels, self.transform = source, rows, labels, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = ImageOps.exif_transpose(self.source[self.rows[index]]["image"]).convert("RGB")
        return self.transform(image), self.labels[index]


def load_model(path: Path, vit_config: dict | None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if list(state["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Label order mismatch: {path}")
    if state["architecture"] == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    else:
        model = timm.create_model(state["architecture"], pretrained=False, num_classes=5)
        transform = timm.data.create_transform(**vit_config, is_training=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.eval(), transform


def summarize(y: np.ndarray, logits: np.ndarray) -> dict:
    pred5 = logits.argmax(1)
    pred4 = np.asarray(DISEASE_IDS)[logits[:, DISEASE_IDS].argmax(1)]
    k5, k4, n = int((pred5 == y).sum()), int((pred4 == y).sum()), len(y)
    lo5, hi5 = wilson_ci(k5, n)
    lo4, hi4 = wilson_ci(k4, n)
    return {"n": n, "correct": k5, "accuracy": k5 / n, "accuracy_ci_low": lo5, "accuracy_ci_high": hi5,
            "macro_f1_disease": macro_f1(y, pred5, DISEASE_IDS), "healthy_prediction_rate": float((pred5 == 0).mean()),
            "restricted_correct": k4, "restricted_accuracy": k4 / n, "restricted_ci_low": lo4,
            "restricted_ci_high": hi4, "restricted_macro_f1": macro_f1(y, pred4, DISEASE_IDS)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for mean(b) - mean(a); rows are images (a, b may be seed means)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def sign_flip_p(a: np.ndarray, b: np.ndarray, reps: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for mean(b - a) over images."""
    d = b - a
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + (null >= observed - 1e-12).sum()) / (reps + 1))


def compare(correct: dict[str, np.ndarray], mask: np.ndarray, family: str) -> list[dict]:
    rows = []
    for cid, a_key, b_key in (("P1", "resnet18_lab", "resnet18_geometric_mixed"),
                              ("P2", "vit_small_lab", "vit_small_mixed")):
        a, b = correct[a_key][mask], correct[b_key][mask]
        only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
        lo, hi = paired_bootstrap(a.astype(float), b.astype(float))
        rows.append({"family": family, "id": cid, "a": a_key, "b": b_key, "n": int(mask.sum()),
                     "accuracy_a": a.mean(), "accuracy_b": b.mean(), "difference": b.mean() - a.mean(),
                     "ci_low": lo, "ci_high": hi, "only_a_correct": only_a, "only_b_correct": only_b,
                     "p_value": mcnemar_exact(only_a, only_b), "test": "exact McNemar"})
    fo = np.mean([correct[f"nb15_field_only_s{s}"][mask] for s in SEEDS], axis=0)
    mx = np.mean([correct[f"nb15_mixed_s{s}"][mask] for s in SEEDS], axis=0)
    lo, hi = paired_bootstrap(fo, mx)
    rows.append({"family": family, "id": "P3", "a": "nb15_field_only (3 seeds)", "b": "nb15_mixed (3 seeds)",
                 "n": int(mask.sum()), "accuracy_a": fo.mean(), "accuracy_b": mx.mean(),
                 "difference": mx.mean() - fo.mean(), "ci_low": lo, "ci_high": hi,
                 "only_a_correct": "", "only_b_correct": "", "p_value": sign_flip_p(fo, mx),
                 "test": "sign-flip permutation on per-image seed-mean correctness"})
    for row, adjusted in zip(rows, holm([r["p_value"] for r in rows])):
        row["holm_p"] = adjusted
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--nb15-run", type=Path,
                        default=ROOT / "outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    protocol = json.loads((V3_DIR / "protocol.json").read_text())
    seal = json.loads((ROOT / "data/results/notebook15/original_evaluation_seal.json").read_text())
    paths = {"resnet18_lab": args.models_dir / "resnet18_lab_best.pt",
             "vit_small_lab": args.models_dir / "vit_small_lab_best.pt",
             "resnet18_geometric_mixed": args.models_dir / "resnet18_geometric_mixed_best.pt",
             "vit_small_mixed": args.models_dir / "vit_small_mixed_best.pt"}
    expected = {k: protocol["models"][k]["checkpoint_sha256"] for k in paths}
    for seed in SEEDS:
        for condition in ("field_only", "mixed"):
            key = f"nb15_{condition}_s{seed}"
            paths[key] = args.nb15_run / f"seed_{seed}" / f"{condition}_selected.pt"
            expected[key] = seal["seed_checkpoints"][str(seed)][f"{condition}_selected.pt"]
    hashes = {}
    for key, path in paths.items():
        hashes[key] = sha256(path)
        if hashes[key] != expected[key]:
            raise ValueError(f"SHA-256 mismatch for {key}: {hashes[key]} != {expected[key]}")

    manifest = pd.read_csv(V3_DIR / "manifest.csv")
    included = manifest[manifest.status.eq("included")].reset_index(drop=True)
    y = included.label_id.to_numpy()
    source = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    vit_config = torch.load(paths["vit_small_lab"], map_location="cpu", weights_only=True)["preprocessing_config"]

    predictions, metrics, correct = [], [], {}
    for key, path in paths.items():
        model, transform = load_model(path, vit_config)
        loader = DataLoader(PlantSegRows(source, included.plantseg_row.tolist(), y.tolist(), transform),
                            batch_size=32, shuffle=False, num_workers=0)
        with torch.inference_mode():
            logits = torch.cat([model(images) for images, _ in loader]).numpy()
        correct[key] = logits.argmax(1) == y
        metrics.append({"model": key, **summarize(y, logits), "checkpoint_sha256": hashes[key]})
        probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
        predictions.append(pd.DataFrame({
            "model": key, "plantseg_row": included.plantseg_row, "label_id": y,
            "predicted_label_id": logits.argmax(1),
            "restricted_predicted_label_id": np.asarray(DISEASE_IDS)[logits[:, DISEASE_IDS].argmax(1)],
            **{f"prob_{i}": probs[:, i] for i in range(5)}}))
        print(metrics[-1]["model"], metrics[-1]["correct"], "/", metrics[-1]["n"], flush=True)
        del model

    min_side = included[["width", "height"]].min(axis=1).to_numpy()
    masks = {"all": np.ones(len(y), bool), "min_side_ge_224": min_side >= 224,
             "plantseg_test_split": included.plantseg_split.eq("test").to_numpy()}
    comparisons = [row for name, mask in masks.items() for row in compare(correct, mask, name)]
    restricted = {k: (p.restricted_predicted_label_id.to_numpy() == y)
                  for k, p in zip(paths, predictions)}
    comparisons += compare(restricted, masks["all"], "disease_restricted")

    recall = pd.DataFrame([{"model": key, "class_name": CLASS_NAMES[k], "n": int((y == k).sum()),
                            "correct": int(correct[key][y == k].sum()),
                            "recall": float(correct[key][y == k].mean())}
                           for key in paths for k in DISEASE_IDS])

    output.mkdir(parents=True)
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    recall.to_csv(output / "per_class_recall.csv", index=False)
    (output / "run_record.json").write_text(json.dumps({
        "protocol_sha256": sha256(V3_DIR / "protocol.json"), "manifest_sha256": sha256(V3_DIR / "manifest.csv"),
        "plantseg": {"repository": PLANTSEG[0], "revision": PLANTSEG[1]}, "images": int(len(y)),
        "checkpoint_sha256": hashes, "device": "cpu", "precision": "float32",
        "torch": torch.__version__, "timm": timm.__version__, "threads": args.threads,
    }, indent=2) + "\n")
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
