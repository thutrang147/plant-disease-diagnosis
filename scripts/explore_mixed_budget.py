"""E2: Notebook 15's Mixed (Lab rehearsal) arm at the 24-epoch budget, seed 42.

The training-budget ablation showed ResNet18 Field Dev still rising at 24
epochs for Field-only training, so Notebook 15's 8-epoch Field-only vs Mixed
comparison describes under-trained models. This run is the matched-budget
counterpart of the 24-epoch Field-only last-block run in
``data/results/training_budget`` (same script family, same seed).

Recipe (declared before the run): ResNet18 Lab baseline; 2 head epochs, then
head + ``layer4`` for 22 epochs; AdamW (head 1e-4, backbone 1e-5, weight decay
1e-4), reset at the phase change; frozen BatchNorm statistics. Every step uses
16 Field Adapt Train images (each once per epoch) and 16 distinct Lab Train
images (370 per epoch), sampled with Notebook 15's ``epoch_plan`` seeds and
per-row augmentation seeds. Loss = field CE + 1.0 x lab CE, as in Notebook 15.

Built-in sanity check: the epoch-8 state must reproduce the sealed Notebook 15
Mixed seed-42 Lab Test macro-F1 (0.9553) within 0.01; otherwise the run is
flagged as not reproducing and must not be used.

Primary outcomes: Field Dev (per epoch) and Lab Test retention at the selected
epoch. Field Test and External v3 are reported once, exploratory.

Protocol amendment (committed before running seeds 43-44): the script also runs
the Field-only arm (``--condition field_only``: same loop without the Lab loss)
and seeds 43 and 44. Each seed and arm has the same epoch-8 sanity check
against its sealed Notebook 15 Lab Test macro-F1; an arm that fails is reported
and excluded, never tuned. Field-only seed 42 is re-run here as a consistency
check against ``data/results/training_budget``. Endpoints are unchanged;
External v3 is a pre-declared secondary endpoint on an already reused set.

Correction (committed before any amendment run): Notebook 15 selected epoch 7
for seed 43 (epoch 8 for seeds 42 and 44), so its sealed values belong to that
epoch. The sanity check is therefore evaluated at the Notebook 15 selected
epoch of each seed: Lab Test macro-F1 within 0.01 of the sealed value (the
declared criterion), plus Field Dev macro-F1 within 1e-3 of the per-arm value in
the README Notebook 15 table (reported, not a gate). The epoch-8 Lab Test value
is still recorded. Per-image predictions of the selected checkpoint are now
saved, so the pre-declared P3-style sign-flip on External v3 can be computed;
Mixed seed 42 is re-run for that reason and must match data/results/mixed_budget.

Usage from the repository root::

    python scripts/explore_mixed_budget.py --output data/results/mixed_budget
    python scripts/explore_mixed_budget.py --condition field_only --seed 43 --output data/results/mixed_budget_amendment/field_only_s43
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import ImageOps
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controlled_field_mixed import derived_seed  # noqa: E402
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

spec = importlib.util.spec_from_file_location("lpft", ROOT / "scripts/explore_lpft_architecture.py")
lpft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpft)

SEED, HEAD_EPOCHS, SECOND_EPOCHS, BATCH, LAMBDA_LAB = 42, 2, 22, 16, 1.0
NB15_MIXED_S42_LAB_TEST = 0.9552975465455024
SANITY_TOLERANCE = 0.01
# Sealed Notebook 15 Lab Test macro-F1 (data/results/notebook15_lab_retention/lab_retention_by_checkpoint.csv).
NB15_LAB_TEST = {("mixed", 42): 0.9552975465455024, ("mixed", 43): 0.9449250178210471, ("mixed", 44): 0.9603422930100965,
                 ("field_only", 42): 0.8116476460444467, ("field_only", 43): 0.8092055163696659,
                 ("field_only", 44): 0.8199427434570354}
NB15_SELECTED_EPOCH = {42: 8, 43: 7, 44: 8}
# Per-arm Field Dev macro-F1 at the selected epoch (README, Notebook 15 table; 4 decimals).
NB15_FIELD_DEV = {("field_only", 42): 0.4388, ("mixed", 42): 0.4390, ("field_only", 43): 0.4361,
                  ("mixed", 43): 0.4478, ("field_only", 44): 0.4147, ("mixed", 44): 0.4327}


def augmented(images, ids, idx, transform, epoch, tag):
    out = []
    for j in idx:
        g = torch.Generator().manual_seed(derived_seed(SEED, epoch, tag, int(ids[j])))
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(g.get_state())
            out.append(transform(images[j]))
    return torch.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--condition", default="mixed", choices=["mixed", "field_only"])
    args = parser.parse_args()
    global SEED
    SEED = args.seed
    lab_weight = LAMBDA_LAB if args.condition == "mixed" else 0.0
    expected = NB15_LAB_TEST[(args.condition, SEED)]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    model, training, evaluation = lpft.load("resnet18", args.models_dir)
    resize, steps = training.transforms[0], lpft.transforms.Compose(training.transforms[1:])

    def manifest(name):
        ds = PlantDiseaseDataset(ROOT / "data/splits" / f"{name}.csv", transform=lambda img: img)
        return ds, ds.metadata.label_id.to_numpy(), ds.metadata.row_id.to_numpy()

    field_ds, field_y, field_ids = manifest("field_adapt_train")
    field_imgs = [resize(field_ds[i][0]) for i in range(len(field_y))]
    lab_ds, lab_y, lab_ids = manifest("lab_train")
    lab_imgs = [resize(lab_ds[i][0]) for i in range(len(lab_y))]
    tensors = {}
    for name in ("field_dev", "field_test", "lab_test"):
        ds, y, _ = manifest(name)
        tensors[name] = (torch.stack([evaluation(ds[i][0]) for i in range(len(y))]), y)
    v3 = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    v3 = v3[v3.status.eq("included")]
    plantseg = load_dataset(*lpft.PLANTSEG[:1], split="train", revision=lpft.PLANTSEG[1])
    tensors["external_v3"] = (torch.stack([evaluation(ImageOps.exif_transpose(plantseg[int(r)]["image"]).convert("RGB"))
                                           for r in v3.plantseg_row]), v3.label_id.to_numpy())
    row_ids = {"external_v3": v3.plantseg_row.to_numpy(),
               **{name: manifest(name)[2] for name in ("field_dev", "field_test", "lab_test")}}
    print("data ready", flush=True)

    predictions = {}

    def score(part, state=None):
        x, y = tensors[part]
        p = lpft.predict(model, x)
        predictions[part] = p
        labels = [1, 2, 3, 4] if part == "external_v3" else range(5)
        return {"n": len(y), "correct": int((p == y).sum()), "accuracy": float((p == y).mean()),
                "macro_f1": macro_f1(y, p, labels)}

    start = time.time()
    dev0 = score("field_dev")["macro_f1"]
    history = [{"epoch": 0, "field_dev_macro_f1": dev0}]
    best = (dev0, 0, copy.deepcopy(model.state_dict()))
    sanity = {}
    nb15_epoch = NB15_SELECTED_EPOCH[SEED]
    epoch = 0
    for phase, epochs in (("head", HEAD_EPOCHS), ("second", SECOND_EPOCHS)):
        lpft.set_phase(model, "resnet18", "last_block", phase)
        optimizer = lpft.optimizer_for(model, "resnet18", phase)
        for _ in range(epochs):
            epoch += 1
            lpft.set_phase(model, "resnet18", "last_block", phase)
            f_order = np.random.default_rng(derived_seed(SEED, epoch, "field_order")).permutation(len(field_y))
            l_order = np.random.default_rng(derived_seed(SEED, epoch, "lab_order")).choice(len(lab_y), len(field_y), replace=False)
            for i in range(0, len(f_order), BATCH):
                fi, li = f_order[i:i + BATCH], l_order[i:i + BATCH]
                loss = nn.functional.cross_entropy(model(augmented(field_imgs, field_ids, fi, steps, epoch, "field")),
                                                   torch.as_tensor(field_y[fi]))
                if lab_weight:
                    loss = loss + lab_weight * nn.functional.cross_entropy(
                        model(augmented(lab_imgs, lab_ids, li, steps, epoch, "lab")), torch.as_tensor(lab_y[li]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            dev = score("field_dev")["macro_f1"]
            history.append({"epoch": epoch, "field_dev_macro_f1": dev})
            if epoch == 8:
                sanity["epoch8_lab_test_macro_f1"] = score("lab_test")["macro_f1"]
            if epoch == nb15_epoch:
                lab = score("lab_test")["macro_f1"]
                sanity.update({"nb15_selected_epoch": nb15_epoch, "lab_test_macro_f1": lab,
                               "nb15_sealed_lab_test_macro_f1": expected,
                               "within_tolerance": abs(lab - expected) <= SANITY_TOLERANCE,
                               "field_dev_macro_f1": dev, "nb15_field_dev_macro_f1": NB15_FIELD_DEV[(args.condition, SEED)],
                               "field_dev_within_1e-3": abs(dev - NB15_FIELD_DEV[(args.condition, SEED)]) <= 1e-3})
            if epoch == max(8, nb15_epoch):
                print("sanity", sanity, flush=True)
            if dev > best[0]:
                best = (dev, epoch, copy.deepcopy(model.state_dict()))
            print("epoch", epoch, round(dev, 4), f"{time.time() - start:.0f}s", flush=True)

    model.load_state_dict(best[2])
    results = [{"partition": part, "selected_epoch": best[1], **score(part)} for part in tensors]
    output.mkdir(parents=True)
    pd.DataFrame(history).to_csv(output / "field_dev_history.csv", index=False)
    pd.DataFrame(results).to_csv(output / "selected_results.csv", index=False)
    pd.concat([pd.DataFrame({"partition": part, "row_id": row_ids[part], "label_id": tensors[part][1],
                             "predicted_label_id": predictions[part]}) for part in tensors]
              ).to_csv(output / "predictions.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "arm": f"{args.condition}, ResNet18 last block", "condition": args.condition, "seed": SEED, "head_epochs": HEAD_EPOCHS,
        "second_epochs": SECOND_EPOCHS, "batch_field": BATCH, "batch_lab": BATCH, "lambda_lab": LAMBDA_LAB,
        "selection": f"epoch 0-{HEAD_EPOCHS + SECOND_EPOCHS} maximizing Field Dev macro-F1, earliest tie",
        "sanity_check": sanity, "counterpart": "data/results/training_budget (resnet18, last_block, seed 42)",
        "status": "exploratory, CPU", "torch": torch.__version__}, indent=2) + "\n")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
