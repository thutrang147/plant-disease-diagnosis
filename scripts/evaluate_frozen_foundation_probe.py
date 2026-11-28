"""Frozen foundation-model baseline: DINOv2 features + logistic-regression probe.

Reference point for RQ1/RQ2: how far does a modern self-supervised backbone get
with **no backbone fine-tuning**, using the same manifests as the project models?
Backbones are loaded through timm at 224 x 224 input (positional embeddings
interpolated), the CLS embedding is extracted once per image, and only a linear
head is trained (scikit-learn, CPU).

Protocol (declared in code before the first run):

* ``lab_only`` probe: trained on Lab Train; C in {0.01, 0.1, 1, 10} chosen by
  Lab Validation macro-F1.
* ``lab_plus_field`` probe: trained on Lab Train + Field Adapt Train, with
  sample weights giving both domains equal total weight; C chosen by Field Dev
  macro-F1 (the same selection data the adapted project models used).
* Both are then evaluated on Lab Test, Field Dev, the historical Field Test and
  External v3 (macro-F1 over the four disease labels there).

Usage from the repository root::

    python scripts/evaluate_frozen_foundation_probe.py --output data/results/frozen_foundation_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import timm
import torch
from datasets import load_dataset
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

PLANTSEG = ("LFTTim/plantseg_disease_segmentation", "2fff5d63968d3fd0aca061e7348b0b1441c1535d")
BACKBONES = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m", "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m"}
PARTITIONS = ("lab_train", "lab_val", "lab_test", "field_adapt_train", "field_dev", "field_test")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def extract(model, transform, get_image, count: int, batch: int = 32) -> np.ndarray:
    """Decode, transform and embed images one batch at a time (no image cache)."""
    features = []
    with torch.inference_mode():
        for start in range(0, count, batch):
            x = torch.stack([transform(get_image(i)) for i in range(start, min(start + batch, count))])
            features.append(model(x).numpy())
    return np.concatenate(features)


def partition_source(name: str):
    dataset = PlantDiseaseDataset(ROOT / "data/splits" / f"{name}.csv", transform=lambda img: img)
    return (lambda i: dataset[i][0]), dataset.metadata.label_id.to_numpy()


def v3_source():
    manifest = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    included = manifest[manifest.status.eq("included")]
    source = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    rows = included.plantseg_row.tolist()
    return (lambda i: ImageOps.exif_transpose(source[int(rows[i])]["image"]).convert("RGB"),
            included.label_id.to_numpy())


def fit(x, y, weights, c):
    scaler = StandardScaler().fit(x, sample_weight=weights)
    clf = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, clf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbones", nargs="+", default=["dinov2_vits14"], choices=sorted(BACKBONES))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    data = {name: partition_source(name) for name in PARTITIONS}
    data["external_v3"] = v3_source()
    print({k: len(v[1]) for k, v in data.items()}, flush=True)

    rows, selection = [], []
    for short, name in ((b, BACKBONES[b]) for b in args.backbones):
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).eval()
        config = timm.data.resolve_data_config({}, model=model)
        config["input_size"] = (3, 224, 224)
        transform = timm.data.create_transform(**config, is_training=False)
        feats = {k: extract(model, transform, get, len(y)) for k, (get, y) in data.items()}
        labels = {k: y for k, (_, y) in data.items()}
        print("features", short, flush=True)

        lab_w = np.ones(len(labels["lab_train"]))
        field_w = np.full(len(labels["field_adapt_train"]), len(lab_w) / len(labels["field_adapt_train"]))
        setups = {
            "lab_only": (feats["lab_train"], labels["lab_train"], lab_w, "lab_val"),
            "lab_plus_field": (np.concatenate([feats["lab_train"], feats["field_adapt_train"]]),
                               np.concatenate([labels["lab_train"], labels["field_adapt_train"]]),
                               np.concatenate([lab_w, field_w]), "field_dev"),
        }
        for probe, (x, y, w, select_on) in setups.items():
            scored = []
            for c in C_GRID:
                scaler, clf = fit(x, y, w, c)
                p = clf.predict(scaler.transform(feats[select_on]))
                scored.append((macro_f1(labels[select_on], p, range(5)), -c, c))
                selection.append({"backbone": short, "probe": probe, "C": c, "selection_partition": select_on,
                                  "selection_macro_f1": scored[-1][0]})
            best_c = max(scored)[2]
            scaler, clf = fit(x, y, w, best_c)
            for part in ("lab_test", "field_dev", "field_test", "external_v3"):
                yt, p = labels[part], clf.predict(scaler.transform(feats[part]))
                rows.append({"backbone": short, "timm_model": name, "probe": probe, "C": best_c,
                             "partition": part, "n": len(yt), "correct": int((yt == p).sum()),
                             "accuracy": float((yt == p).mean()),
                             "macro_f1": macro_f1(yt, p, [1, 2, 3, 4] if part == "external_v3" else range(5)),
                             "healthy_prediction_rate": float((p == 0).mean())})
        del model

    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "probe_results.csv", index=False)
    pd.DataFrame(selection).to_csv(output / "probe_selection.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "backbones": {b: BACKBONES[b] for b in args.backbones}, "input": "224x224, timm eval transform",
        "feature": "CLS embedding (num_classes=0)", "probe": "StandardScaler + LogisticRegression(max_iter=5000)",
        "C_grid": C_GRID, "selection": {"lab_only": "lab_val", "lab_plus_field": "field_dev"},
        "domain_weighting": "Field Adapt Train rows weighted so both domains have equal total weight",
        "torch": torch.__version__, "timm": timm.__version__, "sklearn": sklearn.__version__}, indent=2) + "\n")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
