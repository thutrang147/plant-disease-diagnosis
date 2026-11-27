"""Exploratory test of the trainable-fraction confound in the CNN-vs-ViT comparison.

Historical adaptation updated ResNet18 ``layer4`` (~75% of parameters) but only
the last ViT-Small block (~8%). This script re-adapts both Lab baselines on
Field Adapt Train under one shared recipe and two strategies:

* ``last_block``: head for 2 epochs, then head + last block for 6 epochs
  (the Notebook 15 Field-only recipe; trainable fractions differ by design);
* ``lp_ft``: head for 2 epochs (linear probe), then **all** parameters for 6
  epochs (LP-FT, Kumar et al., 2022; both architectures fully trainable).

Shared settings follow ``src/controlled_field_mixed.default_config``: AdamW reset
at the phase change, head LR 1e-4, backbone LR 1e-5, weight decay 1e-4, float32,
frozen BatchNorm statistics, the same mild augmentation policy and per-row
derived seeds. Unlike the historical runs, the batch size is 16 for both
architectures. Selection: the epoch (0-8) maximizing Field Dev macro-F1,
earliest tie. The selected state is evaluated once on Lab Test, the historical
Field Test and External v3.

Status: exploratory. CPU training, Field-only condition, three seeds; it is not
a replication of the sealed Notebook 15 protocol and produces no sealed
checkpoints. Usage from the repository root::

    python scripts/explore_lpft_architecture.py --output data/results/lpft_architecture
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from datasets import load_dataset
from PIL import ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controlled_field_mixed import derived_seed  # noqa: E402
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

PLANTSEG = ("LFTTim/plantseg_disease_segmentation", "2fff5d63968d3fd0aca061e7348b0b1441c1535d")
CHECKPOINTS = {"resnet18": ("resnet18_lab_best.pt", "65d53766258f26cb505cce4c319c3f783cb0f3c6ef4608680906919731f2782d"),
               "vit_small": ("vit_small_lab_best.pt", "42c26a8aa70311d9504d55dda422f20e2ea5f0a5878f5922b9f855c052a18c2e")}
CONFIG = {"head_epochs": 2, "second_epochs": 6, "batch_size": 16, "head_lr": 1e-4,
          "backbone_lr": 1e-5, "weight_decay": 1e-4, "seeds": [42, 43, 44]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(arch: str, models_dir: Path):
    name, expected = CHECKPOINTS[arch]
    path = models_dir / name
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if arch == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        evaluation = ResNet18_Weights.IMAGENET1K_V1.transforms()
        mean, std, interp = evaluation.mean, evaluation.std, transforms.InterpolationMode.BILINEAR
    else:
        model = timm.create_model(state["architecture"], pretrained=False, num_classes=5)
        cfg = state["preprocessing_config"]
        evaluation = timm.data.create_transform(**cfg, is_training=False)
        mean, std, interp = cfg["mean"], cfg["std"], transforms.InterpolationMode.BICUBIC
    model.load_state_dict(state["model_state_dict"], strict=True)
    # Same policy as src/controlled_field_mixed.load_model.
    training = transforms.Compose([
        transforms.Resize(256, interpolation=interp),
        transforms.RandomResizedCrop(224, scale=(.85, 1.), ratio=(.90, 1.10), interpolation=interp),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=.10, contrast=.10, saturation=.10)], p=.35),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    return model, training, evaluation


def head_of(model, arch):
    return model.fc if arch == "resnet18" else model.get_classifier()


def set_phase(model, arch: str, strategy: str, phase: str) -> float:
    """Set requires_grad; return the trainable parameter fraction."""
    last = ("layer4.", "fc.") if arch == "resnet18" else (f"blocks.{len(model.blocks) - 1}.", "norm.", "fc_norm.", "head.")
    head = ("fc.",) if arch == "resnet18" else ("head.",)
    for name, p in model.named_parameters():
        if phase == "head":
            p.requires_grad_(name.startswith(head))
        elif strategy == "lp_ft":
            p.requires_grad_(True)
        else:
            p.requires_grad_(name.startswith(last))
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / total


def optimizer_for(model, arch: str, phase: str):
    head = list(head_of(model, arch).parameters())
    groups = [{"params": head, "lr": CONFIG["head_lr"]}]
    if phase != "head":
        ids = {id(p) for p in head}
        groups.append({"params": [p for p in model.parameters() if p.requires_grad and id(p) not in ids],
                       "lr": CONFIG["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])


def eval_tensors(evaluation, get, count):
    return torch.stack([evaluation(get(i)) for i in range(count)])


def predict(model, x, batch=64):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).argmax(1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    raw = {}
    for name in ("field_adapt_train", "field_dev", "field_test", "lab_test"):
        ds = PlantDiseaseDataset(ROOT / "data/splits" / f"{name}.csv", transform=lambda img: img)
        raw[name] = ((lambda d: (lambda i: d[i][0]))(ds), ds.metadata.label_id.to_numpy(),
                     ds.metadata.row_id.to_numpy())
    manifest = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    included = manifest[manifest.status.eq("included")]
    plantseg = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    rows = included.plantseg_row.tolist()
    raw["external_v3"] = (lambda i: ImageOps.exif_transpose(plantseg[int(rows[i])]["image"]).convert("RGB"),
                          included.label_id.to_numpy(), np.asarray(rows))
    # Training images: Resize(256) is the first training op, so caching it is exact.
    train_get, train_y, train_ids = raw["field_adapt_train"]

    history, results = [], []
    for arch in ("resnet18", "vit_small"):
        _, training, evaluation = load(arch, args.models_dir)
        resize = training.transforms[0]
        train_imgs = [resize(train_get(i)) for i in range(len(train_y))]
        tensors = {k: (eval_tensors(evaluation, g, len(y)), y) for k, (g, y, _) in raw.items()
                   if k != "field_adapt_train"}
        steps = transforms.Compose(training.transforms[1:])
        for strategy in ("last_block", "lp_ft"):
            for seed in CONFIG["seeds"]:
                start = time.time()
                model, _, _ = load(arch, args.models_dir)
                x_dev, y_dev = tensors["field_dev"]
                best = (macro_f1(y_dev, predict(model, x_dev), range(5)), 0, copy.deepcopy(model.state_dict()))
                history.append({"arch": arch, "strategy": strategy, "seed": seed, "epoch": 0,
                                "trainable_fraction": 0.0, "field_dev_macro_f1": best[0]})
                epoch = 0
                for phase, epochs in (("head", CONFIG["head_epochs"]), ("second", CONFIG["second_epochs"])):
                    fraction = set_phase(model, arch, strategy, phase)
                    optimizer = optimizer_for(model, arch, phase)
                    for _ in range(epochs):
                        epoch += 1
                        set_phase(model, arch, strategy, phase)
                        order = np.random.default_rng(derived_seed(seed, epoch, "field_order")).permutation(len(train_y))
                        for i in range(0, len(order), CONFIG["batch_size"]):
                            batch = order[i:i + CONFIG["batch_size"]]
                            xs = []
                            for j in batch:
                                g = torch.Generator().manual_seed(derived_seed(seed, epoch, "field", int(train_ids[j])))
                                with torch.random.fork_rng(devices=[]):
                                    torch.set_rng_state(g.get_state())
                                    xs.append(steps(train_imgs[j]))
                            loss = nn.functional.cross_entropy(model(torch.stack(xs)), torch.as_tensor(train_y[batch]))
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                        score = macro_f1(y_dev, predict(model, x_dev), range(5))
                        history.append({"arch": arch, "strategy": strategy, "seed": seed, "epoch": epoch,
                                        "trainable_fraction": fraction, "field_dev_macro_f1": score})
                        if score > best[0]:
                            best = (score, epoch, copy.deepcopy(model.state_dict()))
                model.load_state_dict(best[2])
                for part, (x, y) in tensors.items():
                    p = predict(model, x)
                    results.append({"arch": arch, "strategy": strategy, "seed": seed, "selected_epoch": best[1],
                                    "partition": part, "n": len(y), "correct": int((p == y).sum()),
                                    "accuracy": float((p == y).mean()),
                                    "macro_f1": macro_f1(y, p, [1, 2, 3, 4] if part == "external_v3" else range(5)),
                                    "healthy_prediction_rate": float((p == 0).mean())})
                print(arch, strategy, seed, "epoch", best[1], f"{time.time() - start:.0f}s", flush=True)

    output.mkdir(parents=True)
    pd.DataFrame(history).to_csv(output / "field_dev_history.csv", index=False)
    results = pd.DataFrame(results)
    results.to_csv(output / "selected_results.csv", index=False)
    (results.groupby(["partition", "arch", "strategy"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
     .round(4).to_csv(output / "summary.csv"))
    (output / "protocol.json").write_text(json.dumps({
        **CONFIG, "strategies": {
            "last_block": f"head {CONFIG['head_epochs']} ep, then head + last block {CONFIG['second_epochs']} ep",
            "lp_ft": f"head {CONFIG['head_epochs']} ep, then all parameters {CONFIG['second_epochs']} ep"},
        "selection": f"epoch 0-{CONFIG['head_epochs'] + CONFIG['second_epochs']} maximizing Field Dev macro-F1, earliest tie",
        "status": "exploratory, CPU",
        "torch": torch.__version__, "timm": timm.__version__, "threads": args.threads}, indent=2) + "\n")
    print(results.groupby(["partition", "arch", "strategy"])[["accuracy", "macro_f1"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
