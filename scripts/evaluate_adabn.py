"""E3: AdaBN, re-estimating ResNet18 BatchNorm statistics on field images.

Every ResNet18 checkpoint carries BatchNorm running statistics estimated on
PlantVillage (Notebook 15 froze them during adaptation). AdaBN (Li et al.,
2016) replaces them with statistics of the target domain, without labels and
without changing any weight. If part of the lab-to-field gap is a
normalization mismatch, field scores should rise.

Protocol (declared before the run): for each ResNet18 checkpoint, reset all
BatchNorm running statistics and re-estimate them as a cumulative average
(momentum None) over the 370 Field Adapt Train images (eval transform, batch
32, labels unused); then evaluate Field Dev, Field Test, External v3 and Lab
Test, next to the unchanged checkpoint. ViT-Small uses LayerNorm and is not
applicable. Exploratory; inference only, CPU.

Usage from the repository root::

    python scripts/evaluate_adabn.py --output data/results/adabn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from PIL import ImageOps
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.inference import load_checkpoint, registry, sha256  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

PLANTSEG = ("LFTTim/plantseg_disease_segmentation", "2fff5d63968d3fd0aca061e7348b0b1441c1535d")
MODELS = ("resnet18_lab", "resnet18_geometric_mixed", "nb15_field_only_s42", "nb15_mixed_s42",
          "nb15_field_only_s43", "nb15_mixed_s43", "nb15_field_only_s44", "nb15_mixed_s44")


def adapt_bn(model: nn.Module, x: torch.Tensor, batch: int = 32) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over all batches
    model.train()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            model(x[i:i + batch])
    return model.eval()


def predict(model, x, batch=64):
    with torch.inference_mode():
        return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).argmax(1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)
    entries = {e["key"]: e for e in registry()}

    _, transform, _ = load_checkpoint(ROOT / entries["resnet18_lab"]["local_path"])
    row_ids = {}

    def stack(name):
        ds = PlantDiseaseDataset(ROOT / "data/splits" / f"{name}.csv", transform=transform)
        items = [ds[i] for i in range(len(ds))]
        row_ids[name] = ds.metadata.row_id.to_numpy()
        return torch.stack([a for a, _ in items]), ds.metadata.label_id.to_numpy()
    adapt_x, _ = stack("field_adapt_train")
    sets = {name: stack(name) for name in ("field_dev", "field_test", "lab_test")}
    v3 = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    v3 = v3[v3.status.eq("included")]
    source = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    sets["external_v3"] = (torch.stack([transform(ImageOps.exif_transpose(source[int(r)]["image"]).convert("RGB"))
                                        for r in v3.plantseg_row]), v3.label_id.to_numpy())
    row_ids["external_v3"] = v3.plantseg_row.to_numpy()

    rows, per_image = [], []
    for key in MODELS:
        path = ROOT / entries[key]["local_path"]
        if sha256(path) != entries[key]["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {key}")
        for variant in ("original", "adabn_field_adapt_train"):
            model, _, _ = load_checkpoint(path)
            if variant != "original":
                adapt_bn(model, adapt_x)
            for part, (x, y) in sets.items():
                p = predict(model, x)
                per_image.append(pd.DataFrame({"model": key, "variant": variant, "partition": part,
                                               "row_id": row_ids[part], "label_id": y, "predicted_label_id": p}))
                labels = [1, 2, 3, 4] if part == "external_v3" else range(5)
                rows.append({"model": key, "variant": variant, "partition": part, "n": len(y),
                             "correct": int((p == y).sum()), "accuracy": float((p == y).mean()),
                             "macro_f1": macro_f1(y, p, labels)})
        print("done", key, flush=True)

    output.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "adabn_results.csv", index=False)
    pd.concat(per_image).to_csv(output / "predictions.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "statistics_source": "field_adapt_train (370 images, labels unused), eval transform, batch 32",
        "momentum": "None (cumulative average)", "models": MODELS, "status": "exploratory, CPU",
        "torch": torch.__version__}, indent=2) + "\n")
    print(frame.pivot_table(index=["model", "variant"], columns="partition", values="macro_f1").round(3).to_string())


if __name__ == "__main__":
    main()
