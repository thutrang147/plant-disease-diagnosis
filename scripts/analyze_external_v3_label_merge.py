"""Label-noise sensitivity on External v3: merge Bacterial Spot and Septoria.

Thirteen of 64 cross-dataset duplicates carry different labels, and most of the
disagreement is Bacterial Spot vs Septoria Leaf Spot (two small-spot diseases
that are hard to tell apart in field photographs). This post-hoc check re-scores
the saved External v3 predictions with those two classes merged into one
"small leaf spot" class, and repeats the P1-P3 comparisons with the same tests
as the confirmatory run. If a conclusion holds under the merge, it does not
depend on the most disputed label boundary.

No new inference: only ``data/results/external_v3_plantseg/predictions.csv`` is
read. Exploratory and post-hoc (written after the confirmatory results were
known), so it is not a replacement for P1-P3.

Usage from the repository root::

    python scripts/analyze_external_v3_label_merge.py --output data/results/external_v3_label_merge
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACTERIAL_SPOT, SEPTORIA = 3, 4

spec = importlib.util.spec_from_file_location("ev3", ROOT / "scripts/evaluate_external_v3_plantseg.py")
ev3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev3)


def merge(labels: pd.Series) -> pd.Series:
    return labels.replace({SEPTORIA: BACTERIAL_SPOT})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")

    pred = pd.read_csv(ROOT / "data/results/external_v3_plantseg/predictions.csv")
    order = pred[pred.model == pred.model.iloc[0]].plantseg_row.to_numpy()
    rows, correct = [], {}
    for model, g in pred.groupby("model", sort=False):
        g = g.set_index("plantseg_row").loc[order]
        original = g.label_id.eq(g.predicted_label_id)
        merged = merge(g.label_id).eq(merge(g.predicted_label_id))
        spot_swap = (g.label_id.isin([BACTERIAL_SPOT, SEPTORIA]) & g.predicted_label_id.isin([BACTERIAL_SPOT, SEPTORIA])
                     & ~original)
        correct[model] = merged.to_numpy()
        rows.append({"model": model, "n": len(g), "accuracy_original": original.mean(), "accuracy_merged": merged.mean(),
                     "gain_pp": 100 * (merged.mean() - original.mean()), "errors_original": int((~original).sum()),
                     "errors_that_are_bs_septoria_swaps": int(spot_swap.sum())})
    accuracy = pd.DataFrame(rows)
    comparisons = pd.DataFrame(ev3.compare(correct, np.ones(len(order), dtype=bool), "bs_septoria_merged"))

    output.mkdir(parents=True)
    accuracy.round(4).to_csv(output / "merged_accuracy.csv", index=False)
    comparisons.to_csv(output / "merged_comparisons.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "merge": "Septoria Leaf Spot (4) relabelled as Bacterial Spot (3) in both labels and predictions",
        "source": "data/results/external_v3_plantseg/predictions.csv (argmax over all five classes)",
        "tests": "as in the confirmatory run: exact McNemar (P1, P2), sign-flip on per-image seed means (P3), "
                 "paired bootstrap 95% CI, Holm over the three comparisons",
        "status": "exploratory, post-hoc, no new inference"}, indent=2) + "\n")
    print(accuracy.round(3).to_string(index=False))
    print(comparisons[["id", "accuracy_a", "accuracy_b", "difference", "ci_low", "ci_high", "p_value"]].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    main()
