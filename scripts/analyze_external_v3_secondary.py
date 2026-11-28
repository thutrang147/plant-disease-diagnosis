"""Post-hoc interval estimates for the External v3 comparisons (no new inference).

The pre-registered protocol (data/external_v3_plantseg/protocol.json) names
five-way accuracy as primary and disease macro-F1 as secondary, but reported
intervals only for accuracy. This script adds, from the saved predictions:

* paired image-level bootstrap 95% intervals (10,000 resamples, seed 0) for the
  differences P1, P2 and P3 in accuracy, disease macro-F1 and balanced accuracy
  (mean recall over the four disease classes);
* the same intervals for each model's own metrics;
* for P3, the seed-level view: per-seed differences and a one-sample t-test.

Post-hoc: these are additional summaries of the locked evaluation, not new
confirmatory tests. Output goes to a separate folder so the one-shot
confirmatory results stay untouched.

Usage from the repository root::

    python scripts/analyze_external_v3_secondary.py --output data/results/external_v3_secondary_metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplementary_analysis import macro_f1  # noqa: E402

DISEASE = [1, 2, 3, 4]
SEEDS = (42, 43, 44)
REPS, SEED = 10_000, 0


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"accuracy": float((y == p).mean()), "disease_macro_f1": macro_f1(y, p, DISEASE),
            "balanced_accuracy": float(np.mean([(p[y == k] == k).mean() for k in DISEASE]))}


def boot(y: np.ndarray, preds_a: list[np.ndarray], preds_b: list[np.ndarray] | None, rng) -> dict:
    """Bootstrap the seed-mean metric (or difference b - a) over images."""
    idx = rng.integers(0, len(y), size=(REPS, len(y)))
    out = {k: [] for k in ("accuracy", "disease_macro_f1", "balanced_accuracy")}
    for i in idx:
        ma = [metrics(y[i], p[i]) for p in preds_a]
        mb = [metrics(y[i], p[i]) for p in preds_b] if preds_b else None
        for k in out:
            a = np.mean([m[k] for m in ma])
            out[k].append(np.mean([m[k] for m in mb]) - a if mb else a)
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")

    preds = pd.read_csv(ROOT / "data/results/external_v3_plantseg/predictions.csv")
    table = preds.pivot(index="plantseg_row", columns="model", values="predicted_label_id")
    y = preds.drop_duplicates("plantseg_row").set_index("plantseg_row").loc[table.index, "label_id"].to_numpy()
    p = {m: table[m].to_numpy() for m in table.columns}
    rng = np.random.default_rng(SEED)

    per_model = []
    for model, pred in p.items():
        point, ci = metrics(y, pred), boot(y, [pred], None, rng)
        for k, v in point.items():
            per_model.append({"model": model, "metric": k, "value": v, "ci_low": ci[k][0], "ci_high": ci[k][1]})

    groups = {"P1": (["resnet18_lab"], ["resnet18_geometric_mixed"]),
              "P2": (["vit_small_lab"], ["vit_small_mixed"]),
              "P3": ([f"nb15_field_only_s{s}" for s in SEEDS], [f"nb15_mixed_s{s}" for s in SEEDS])}
    diffs, seed_rows = [], []
    for cid, (a_keys, b_keys) in groups.items():
        a, b = [p[k] for k in a_keys], [p[k] for k in b_keys]
        ma = {k: np.mean([metrics(y, x)[k] for x in a]) for k in ("accuracy", "disease_macro_f1", "balanced_accuracy")}
        mb = {k: np.mean([metrics(y, x)[k] for x in b]) for k in ma}
        ci = boot(y, a, b, rng)
        for k in ma:
            diffs.append({"id": cid, "a": "+".join(a_keys), "b": "+".join(b_keys), "metric": k, "a_value": ma[k],
                          "b_value": mb[k], "difference": mb[k] - ma[k], "ci_low": ci[k][0], "ci_high": ci[k][1]})
    for k in ("accuracy", "disease_macro_f1", "balanced_accuracy"):
        d = [metrics(y, p[f"nb15_mixed_s{s}"])[k] - metrics(y, p[f"nb15_field_only_s{s}"])[k] for s in SEEDS]
        seed_rows.append({"metric": k, **{f"seed_{s}": v for s, v in zip(SEEDS, d)}, "mean": float(np.mean(d)),
                          "seed_level_t_test_p": float(stats.ttest_1samp(d, 0).pvalue)})

    output.mkdir(parents=True)
    pd.DataFrame(per_model).round(5).to_csv(output / "per_model_intervals.csv", index=False)
    pd.DataFrame(diffs).round(5).to_csv(output / "comparison_intervals.csv", index=False)
    pd.DataFrame(seed_rows).round(5).to_csv(output / "p3_seed_level.csv", index=False)
    print(pd.DataFrame(diffs).round(4).to_string(index=False))
    print(pd.DataFrame(seed_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
