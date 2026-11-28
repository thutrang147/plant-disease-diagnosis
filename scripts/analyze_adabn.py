"""E3 paired statistics: Lab baseline with and without AdaBN (no new inference).

Reads the per-image predictions saved by ``scripts/evaluate_adabn.py`` and
compares, image by image, ``resnet18_lab`` with its original (PlantVillage)
BatchNorm statistics against the same weights with statistics re-estimated on
Field Adapt Train. For every partition it reports the accuracy difference with
an exact McNemar test and a paired bootstrap 95% interval (10,000 resamples,
seed 0), and the macro-F1 difference (disease classes only on External v3) with
a paired bootstrap interval; McNemar p values for the Lab baseline are
Holm-adjusted over its three field partitions (Field Dev, Field Test,
External v3). For the adapted checkpoints it reports the
AdaBN-minus-original change per model. Exploratory; Field Test and External v3
are reused sets.

Usage from the repository root::

    python scripts/analyze_adabn.py --output data/results/adabn_paired
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplementary_analysis import holm, macro_f1, mcnemar_exact  # noqa: E402

REPS = 10_000


def labels_for(part: str):
    return [1, 2, 3, 4] if part == "external_v3" else list(range(5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/results/adabn/predictions.csv")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    pred = pd.read_csv(args.predictions)

    rows = []
    for (model, part), g in pred.groupby(["model", "partition"], sort=False):
        a = g[g.variant == "original"].set_index("row_id")
        b = g[g.variant != "original"].set_index("row_id").loc[a.index]
        y, pa, pb = a.label_id.to_numpy(), a.predicted_label_id.to_numpy(), b.predicted_label_id.to_numpy()
        ca, cb = pa == y, pb == y
        labels = labels_for(part)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(y), size=(REPS, len(y)))
        acc = (cb[idx].mean(1) - ca[idx].mean(1))
        f1 = [macro_f1(y[i], pb[i], labels) - macro_f1(y[i], pa[i], labels) for i in idx] if model == "resnet18_lab" else None
        row = {"model": model, "partition": part, "n": len(y), "accuracy_original": ca.mean(), "accuracy_adabn": cb.mean(),
               "accuracy_difference": cb.mean() - ca.mean(), "accuracy_ci_low": np.percentile(acc, 2.5),
               "accuracy_ci_high": np.percentile(acc, 97.5), "only_original_correct": int((ca & ~cb).sum()),
               "only_adabn_correct": int((cb & ~ca).sum()),
               "mcnemar_p": mcnemar_exact(int((ca & ~cb).sum()), int((cb & ~ca).sum())),
               "macro_f1_original": macro_f1(y, pa, labels), "macro_f1_adabn": macro_f1(y, pb, labels)}
        row["macro_f1_difference"] = row["macro_f1_adabn"] - row["macro_f1_original"]
        if f1 is not None:
            row["macro_f1_ci_low"], row["macro_f1_ci_high"] = np.percentile(f1, 2.5), np.percentile(f1, 97.5)
        rows.append(row)
    frame = pd.DataFrame(rows)
    lab = frame.model.eq("resnet18_lab") & frame.partition.isin(["field_dev", "field_test", "external_v3"])
    frame.loc[lab, "mcnemar_holm_p"] = holm(frame.loc[lab, "mcnemar_p"].tolist())
    output.mkdir(parents=True)
    frame.round(5).to_csv(output / "adabn_paired.csv", index=False)
    cols = ["model", "partition", "accuracy_difference", "accuracy_ci_low", "accuracy_ci_high", "mcnemar_p", "mcnemar_holm_p",
            "macro_f1_difference", "macro_f1_ci_low", "macro_f1_ci_high"]
    print(frame[[c for c in cols if c in frame]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
