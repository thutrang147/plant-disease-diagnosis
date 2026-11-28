"""Error analysis of the saved External v3 predictions (no new inference).

Reads ``data/results/external_v3_plantseg/predictions.csv`` and the External v3
manifest and writes:

* accuracy by lesion-area tertile (share of image pixels marked as lesion);
* confusion matrices of the four fixed models (true disease x predicted class);
* calibration: expected calibration error (10 bins) and mean confidence minus
  accuracy (positive = over-confident);
* two figures: accuracy by lesion size, and the four confusion matrices.

Usage from the repository root::

    python scripts/analyze_external_v3_errors.py --output data/results/external_v3_error_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.statistical_analysis import compute_ece  # noqa: E402

CLASSES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"]
FIXED = {"resnet18_lab": "ResNet18 Lab", "vit_small_lab": "ViT-Small Lab",
         "resnet18_geometric_mixed": "ResNet18 Geo+Mixed", "vit_small_mixed": "ViT-Small Mixed"}
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]  # validated categorical slots 1-4 (light)
TEXT, GRID = "#1f1f1f", "#d9d9d9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")

    preds = pd.read_csv(ROOT / "data/results/external_v3_plantseg/predictions.csv")
    manifest = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    area = manifest.set_index("plantseg_row").lesion_area_fraction
    preds["lesion_area"] = preds.plantseg_row.map(area)
    preds["correct"] = preds.label_id.eq(preds.predicted_label_id)
    preds["confidence"] = preds[[f"prob_{i}" for i in range(5)]].max(axis=1)
    images = preds.drop_duplicates("plantseg_row")
    edges = np.quantile(images.lesion_area, [0, 1 / 3, 2 / 3, 1])
    labels = [f"small (<{edges[1]:.0%})", f"medium ({edges[1]:.0%}-{edges[2]:.0%})", f"large (>{edges[2]:.0%})"]
    preds["lesion_size"] = pd.cut(preds.lesion_area, edges, labels=labels, include_lowest=True)
    preds["group"] = preds.model.where(~preds.model.str.startswith("nb15_"),
                                       preds.model.str.replace(r"_s\d+$", "", regex=True))

    by_size = (preds.groupby(["group", "lesion_size"], observed=True)
               .agg(n=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")).reset_index())
    calibration = pd.DataFrame([{
        "model": m, "accuracy": g.correct.mean(), "mean_confidence": g.confidence.mean(),
        "overconfidence": g.confidence.mean() - g.correct.mean(),
        "ece_10_bins": compute_ece(g.correct.to_numpy(), g.confidence.to_numpy(), n_bins=10)}
        for m, g in preds.groupby("model")])
    confusion = {key: pd.crosstab(preds[preds.model == key].label_id.map(dict(enumerate(CLASSES))),
                                  preds[preds.model == key].predicted_label_id.map(dict(enumerate(CLASSES))))
                 .reindex(index=CLASSES[1:], columns=CLASSES, fill_value=0) for key in FIXED}

    output.mkdir(parents=True)
    by_size.to_csv(output / "accuracy_by_lesion_size.csv", index=False)
    calibration.round(4).to_csv(output / "calibration.csv", index=False)
    for key, table in confusion.items():
        table.to_csv(output / f"confusion_{key}.csv")

    # Figure 1: accuracy by lesion-size tertile, four fixed models, direct labels at line ends.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    x = np.arange(3)
    ends = []
    for (key, name), color in zip(FIXED.items(), SERIES):
        acc = by_size[by_size.group == key].set_index("lesion_size").reindex(labels).accuracy.to_numpy()
        ax.plot(x, acc, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=2, label=name)
        ends.append([acc[-1], f"{name} {acc[-1]:.2f}"])
    ends.sort()  # spread end labels so converging lines stay readable
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045)
    for y_pos, text in ends:
        ax.annotate(text, (x[-1] + 0.08, y_pos), va="center", fontsize=8.5, color=TEXT, annotation_clip=False)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.2, 2.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_xlabel("Lesion area (share of image, PlantSeg mask), tertiles", color=TEXT)
    ax.set_title(f"External v3: accuracy by lesion size (n = {len(images)})", color=TEXT, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output / "accuracy_by_lesion_size.png")
    plt.close(fig)

    # Figure 2: confusion matrices, single-hue sequential blue, counts in every cell.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=200)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cce3fd", "#519ced", "#1757ac", "#0e3d7e"])
    vmax = max(t.to_numpy().max() for t in confusion.values())
    for ax, (key, name) in zip(axes.flat, FIXED.items()):
        table = confusion[key].to_numpy()
        ax.imshow(table, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), value in np.ndenumerate(table):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 * vmax else TEXT)
        ax.set_xticks(range(5), ["H", "EB", "LB", "BS", "SLS"])
        ax.set_yticks(range(4), ["EB", "LB", "BS", "SLS"])
        ax.set_title(name, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("Predicted", fontsize=8.5)
        ax.set_ylabel("True", fontsize=8.5)
    fig.suptitle("External v3 confusion matrices (counts; rows = true disease)", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png")
    plt.close(fig)

    print(by_size.pivot(index="group", columns="lesion_size", values="accuracy").round(3).to_string())
    print(calibration.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
