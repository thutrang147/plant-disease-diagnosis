"""E2 summary: Field-only vs Mixed at the 24-epoch budget, seeds 42-44 (no inference).

Reads the six amendment runs written by ``scripts/explore_mixed_budget.py``
(``<runs-dir>/<condition>_s<seed>/``) and reports, as declared in the
amendment:

* per run: sanity check at the Notebook 15 selected epoch, selected epoch,
  Field Dev, Lab Test, Field Test and External v3 scores;
* primary endpoints (Field Dev, Lab Test retention): per-seed Mixed - Field-only
  differences, their mean and a seed-level one-sample t-test (n = 3);
* secondary endpoint, pre-declared on an already reused set: External v3 with
  the P3 machinery, i.e. per-image seed-mean correctness, paired bootstrap 95%
  interval and sign-flip permutation p for accuracy; for disease macro-F1 a
  paired bootstrap interval and a paired permutation p that swaps the two arms'
  predictions per image (all seeds together), since macro-F1 is not a per-image
  mean;
* the mean Field Dev macro-F1 over epochs 20-24 per run, a less
  selection-sensitive summary than the selected-epoch maximum;
* consistency checks: Field-only s42 against ``data/results/training_budget``
  (different script) and the Mixed s42 re-run against ``data/results/mixed_budget``.

Field Dev selected the epoch, so selected Field Dev values are optimistic for
both arms. Runs whose sanity check fails are listed but excluded from the
comparisons.

Usage from the repository root::

    python scripts/analyze_mixed_budget.py --output data/results/mixed_budget_summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplementary_analysis import macro_f1  # noqa: E402

SEEDS = (42, 43, 44)
CONDITIONS = ("field_only", "mixed")
DISEASE = [1, 2, 3, 4]
REPS = 10_000


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def swap_permutation_p(y, fo: list[np.ndarray], mx: list[np.ndarray], metric, reps=REPS, seed=0) -> float:
    """Two-sided paired permutation p for mean_seed metric(Mixed) - metric(FO), swapping arms per image."""
    fo, mx = np.stack(fo), np.stack(mx)
    observed = abs(np.mean([metric(y, b) - metric(y, a) for a, b in zip(fo, mx)]))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        swap = rng.random(len(y)) < 0.5
        a, b = np.where(swap, mx, fo), np.where(swap, fo, mx)
        hits += abs(np.mean([metric(y, bb) - metric(y, aa) for aa, bb in zip(a, b)])) >= observed - 1e-12
    return float((1 + hits) / (reps + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data/results/mixed_budget_amendment")
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    ev3 = _load("ev3", "scripts/evaluate_external_v3_plantseg.py")

    runs, preds = [], {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            run = args.runs_dir / f"{cond}_s{seed}"
            sel = pd.read_csv(run / "selected_results.csv").set_index("partition")
            sanity = json.loads((run / "protocol.json").read_text())["sanity_check"]
            row = {"condition": cond, "seed": seed, "selected_epoch": int(sel.selected_epoch.iloc[0]),
                   "sanity_epoch": sanity["nb15_selected_epoch"], "sanity_lab_test": sanity["lab_test_macro_f1"],
                   "sanity_expected": sanity["nb15_sealed_lab_test_macro_f1"], "sanity_pass": sanity["within_tolerance"],
                   "sanity_field_dev": sanity["field_dev_macro_f1"], "sanity_field_dev_expected": sanity["nb15_field_dev_macro_f1"],
                   "sanity_field_dev_match": sanity["field_dev_within_1e-3"],
                   "epoch8_lab_test": sanity.get("epoch8_lab_test_macro_f1")}
            hist = pd.read_csv(run / "field_dev_history.csv")
            row["field_dev_mean_epochs_20_24"] = hist[hist.epoch.between(20, 24)].field_dev_macro_f1.mean()
            for part in ("field_dev", "lab_test", "field_test", "external_v3"):
                row[f"{part}_accuracy"] = sel.loc[part, "accuracy"]
                row[f"{part}_macro_f1"] = sel.loc[part, "macro_f1"]
            runs.append(row)
            p = pd.read_csv(run / "predictions.csv")
            preds[(cond, seed)] = p[p.partition.eq("external_v3")].reset_index(drop=True)
    runs = pd.DataFrame(runs)
    ok = {(r.condition, r.seed) for r in runs.itertuples() if r.sanity_pass}
    seeds = [s for s in SEEDS if ("field_only", s) in ok and ("mixed", s) in ok]

    seed_rows = []
    for col in [c for c in runs.columns if c.endswith(("_accuracy", "_macro_f1", "_20_24"))]:
        d = [float(runs.query("condition == 'mixed' and seed == @s")[col].iloc[0]
                   - runs.query("condition == 'field_only' and seed == @s")[col].iloc[0]) for s in seeds]
        seed_rows.append({"metric": col, **{f"seed_{s}": v for s, v in zip(seeds, d)}, "mean": float(np.mean(d)),
                          "seeds_mixed_higher": int(sum(v > 0 for v in d)), "n_seeds": len(d),
                          "seed_level_t_test_p": float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else np.nan})

    base = preds[("mixed", seeds[0])]
    for key, frame in preds.items():
        if not np.array_equal(frame.row_id.to_numpy(), base.row_id.to_numpy()):
            raise ValueError(f"External v3 rows differ for {key}")
    y = base.label_id.to_numpy()
    fo = [preds[("field_only", s)].predicted_label_id.to_numpy() for s in seeds]
    mx = [preds[("mixed", s)].predicted_label_id.to_numpy() for s in seeds]
    fo_c, mx_c = np.mean([p == y for p in fo], axis=0), np.mean([p == y for p in mx], axis=0)
    lo, hi = ev3.paired_bootstrap(fo_c, mx_c)
    f1 = lambda yy, pp: macro_f1(yy, pp, DISEASE)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(args.reps, len(y)))
    f1_diffs = [np.mean([f1(y[i], b[i]) - f1(y[i], a[i]) for a, b in zip(fo, mx)]) for i in idx]
    v3 = pd.DataFrame([
        {"metric": "accuracy", "field_only": fo_c.mean(), "mixed": mx_c.mean(), "difference": mx_c.mean() - fo_c.mean(),
         "ci_low": lo, "ci_high": hi, "p_value": ev3.sign_flip_p(fo_c, mx_c), "n_seeds": len(seeds),
         "test": "sign-flip permutation on per-image seed-mean correctness (as P3)"},
        {"metric": "disease_macro_f1", "field_only": np.mean([f1(y, p) for p in fo]),
         "mixed": np.mean([f1(y, p) for p in mx]),
         "difference": np.mean([f1(y, b) - f1(y, a) for a, b in zip(fo, mx)]),
         "ci_low": float(np.percentile(f1_diffs, 2.5)), "ci_high": float(np.percentile(f1_diffs, 97.5)),
         "p_value": swap_permutation_p(y, fo, mx, f1, reps=args.reps), "n_seeds": len(seeds),
         "test": "paired permutation swapping arms per image (all seeds together)"}])

    checks = []
    tb = pd.read_csv(ROOT / "data/results/training_budget/selected_results.csv")
    tb = tb[(tb.arch == "resnet18") & (tb.strategy == "last_block")].set_index("partition")
    mb = pd.read_csv(ROOT / "data/results/mixed_budget/selected_results.csv").set_index("partition")
    for name, ref, cond in (("field_only_s42 vs training_budget", tb, "field_only"),
                            ("mixed_s42 re-run vs mixed_budget", mb, "mixed")):
        new = pd.read_csv(args.runs_dir / f"{cond}_s42/selected_results.csv").set_index("partition")
        for part in new.index:
            checks.append({"check": name, "partition": part, "selected_epoch_ref": int(ref.loc[part, "selected_epoch"]),
                           "selected_epoch_new": int(new.loc[part, "selected_epoch"]),
                           "correct_ref": int(ref.loc[part, "correct"]), "correct_new": int(new.loc[part, "correct"]),
                           "macro_f1_abs_diff": abs(float(ref.loc[part, "macro_f1"]) - float(new.loc[part, "macro_f1"]))})

    output.mkdir(parents=True)
    runs.round(5).to_csv(output / "per_run.csv", index=False)
    pd.DataFrame(seed_rows).round(5).to_csv(output / "seed_level_differences.csv", index=False)
    v3.round(5).to_csv(output / "external_v3_secondary.csv", index=False)
    pd.DataFrame(checks).round(6).to_csv(output / "consistency_checks.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "runs": str(args.runs_dir.relative_to(ROOT)) if args.runs_dir.is_relative_to(ROOT) else str(args.runs_dir),
        "seeds_in_comparison": seeds, "excluded_runs": sorted(f"{c}_s{s}" for c in CONDITIONS for s in SEEDS
                                                               if (c, s) not in ok),
        "bootstrap_reps": args.reps, "bootstrap_seed": 0,
        "endpoints": {"primary": ["field_dev_macro_f1", "lab_test_macro_f1"],
                      "secondary_pre_declared_on_reused_set": ["external_v3 accuracy", "external_v3 disease macro-F1"]},
        "note": "Field Dev selected the epoch, so selected Field Dev is optimistic for both arms.",
        "status": "exploratory, no new inference"}, indent=2) + "\n")
    print(runs[["condition", "seed", "selected_epoch", "sanity_pass", "sanity_field_dev_match", "field_dev_macro_f1",
                "field_dev_mean_epochs_20_24",
                "lab_test_macro_f1", "field_test_macro_f1", "external_v3_accuracy", "external_v3_macro_f1"]]
          .round(4).to_string(index=False))
    print(pd.DataFrame(seed_rows).round(4).to_string(index=False))
    print(v3.round(4).to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    main()
