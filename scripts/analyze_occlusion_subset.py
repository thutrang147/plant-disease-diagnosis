"""Put the occlusion arms on the same images (no new inference).

The lesion-reliance run (486 images) and the control run (378 images with room
for a non-overlapping same-area region) were separate. This script reports,
per model, accuracy on the SAME 378-image subset for: original, lesion removed,
random region removed (control) and lesion only; plus lesion removed on all
486 images for continuity. The 108 excluded images are the ones whose dilated
lesions are too large for a non-overlapping region, so the subset is biased
towards smaller lesions. All numbers are computed from the committed CSVs.

Usage from the repository root::

    python scripts/analyze_occlusion_subset.py --output data/results/occlusion_subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")

    reliance = pd.read_csv(ROOT / "data/results/lesion_reliance/occlusion_predictions.csv")
    reliance["correct"] = reliance.label_id.eq(reliance.predicted_label_id)
    control = pd.read_csv(ROOT / "data/results/occlusion_control/control_predictions.csv")
    subset = set(control.plantseg_row)

    rows = []
    for model, g in reliance.groupby("model"):
        acc = lambda cond, rows_=None: g[(g.condition == cond) & (g.plantseg_row.isin(rows_) if rows_ else True)].correct.mean()
        c = control[control.model == model]
        c_acc = lambda cond: c[c.condition == cond].label_id.eq(c[c.condition == cond].predicted_label_id).mean()
        rows.append({
            "model": model, "subset_images": len(subset),
            "original_378": acc("original", subset), "lesion_removed_378": acc("lesion_removed", subset),
            "control_removed_378": c_acc("control_inpainted"), "lesion_only_378": acc("lesion_only", subset),
            "original_486": acc("original"), "lesion_removed_486": acc("lesion_removed"),
            "control_minus_original_pp": 100 * (c_acc("control_inpainted") - c_acc("original")),
            "lesion_removed_consistency_check": abs(acc("lesion_removed", subset) - c_acc("lesion_inpainted")),
        })
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True)
    frame.round(4).to_csv(output / "same_subset_summary.csv", index=False)
    span = (float(frame.control_minus_original_pp.min()), float(frame.control_minus_original_pp.max()))
    (output / "ranges.json").write_text(json.dumps({
        "control_minus_original_pp_range": [round(span[0], 1), round(span[1], 1)],
        "lesion_removed_net_accuracy_loss_share_486": [
            round(float((1 - frame.lesion_removed_486 / frame.original_486).min()), 3),
            round(float((1 - frame.lesion_removed_486 / frame.original_486).max()), 3)],
        "excluded_images": 486 - len(subset), "note": "excluded images have lesions too large for a control region"}, indent=2) + "\n")
    print(frame.round(3).to_string(index=False))
    print("control minus original (pp):", [round(x, 1) for x in span])


if __name__ == "__main__":
    main()
