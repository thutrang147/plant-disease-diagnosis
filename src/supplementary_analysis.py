"""Supplementary statistics computed only from committed prediction exports.

No model is loaded, no image is read and no inference is performed. Outputs:

* ``external_v1_by_stratum.csv`` – External v1 split into the 16 self-collected
  Healthy images and the 46 online disease images (4-class metrics on the
  disease stratum), because source and class are fully confounded.
* ``external_v1_prediction_distribution.csv`` – how often each model predicts
  each class (column totals of the confusion matrix) and per-class precision.
* ``mcnemar_holm.csv`` – exact McNemar tests (existing family plus the
  stratified external tests) with Holm–Bonferroni adjustment.
* ``notebook15_paired_summary.csv`` – per-seed discordant pairs, Field Dev at
  the shared selected epoch (parsed from the archived executed notebook) and a
  paired image-level bootstrap CI for the seed-averaged Mixed − Field-only
  difference; also a sensitivity analysis excluding Field Test row 2524, whose
  image is an exact pHash duplicate of a training image labelled Septoria.

Run from the repository root::

    python -m src.supplementary_analysis --output data/results/supplementary_statistics

The destination must not already exist.

Note on External v1: the committed per-image predictions in
``outputs/external_confirmatory_eval/`` were produced by the upstream loader
without EXIF transposition. ``docs/archive/experiment_reconciliation.md`` records that
the only label differences from the canonical EXIF run are 11 predictions on
self-collected Healthy images (EXIF orientation 6); the 46 online disease
predictions are identical. Disease-stratum results are therefore canonical;
Healthy-stratum results from these files are flagged as ``upstream_raw_rgb``
and the canonical Healthy counts (0, 5, 12, 13 of 16) are reported alongside.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"]
EXTERNAL_DIR = ROOT / "outputs" / "external_confirmatory_eval"
EXTERNAL_FILES = {
    "ResNet18 · Lab baseline": "predictions_model1_resnet18_lab_base_external_v1.csv",
    "ViT-Small · Lab baseline": "predictions_model2_vit_small_lab_base_external_v1.csv",
    "ResNet18 · Geometric + Mixed": "predictions_model3_resnet18_geom_mixed_external_v1.csv",
    "ViT-Small · Mixed": "predictions_model4_vit_small_mixed_external_v1.csv",
}
# Canonical (EXIF-corrected, Notebook 11) Healthy recognition out of 16.
CANONICAL_HEALTHY_CORRECT = {
    "ResNet18 · Lab baseline": 0,
    "ViT-Small · Lab baseline": 5,
    "ResNet18 · Geometric + Mixed": 12,
    "ViT-Small · Mixed": 13,
}
NB15_DIR = ROOT / "data" / "results" / "notebook15"
NB15_ARCHIVE = ROOT / "notebooks" / "archive" / "15_controlled_field_mixed_colab_executed.ipynb"
SUSPECT_TEST_ROW = 2524  # pHash 0 to train row 1760 (labelled Septoria); test label Bacterial Spot


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels) -> float:
    scores = []
    for k in labels:
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    return float(binomtest(min(only_a, only_b), n=n, p=0.5).pvalue)


def holm(pvalues) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def load_external() -> dict[str, pd.DataFrame]:
    frames = {}
    for model, name in EXTERNAL_FILES.items():
        frame = pd.read_csv(EXTERNAL_DIR / name).sort_values("image_id").reset_index(drop=True)
        frames[model] = frame
    ids = [tuple(f.image_id) for f in frames.values()]
    if len(set(ids)) != 1:
        raise ValueError("External prediction files are not aligned on image_id")
    return frames


def external_by_stratum(frames) -> pd.DataFrame:
    rows = []
    for model, f in frames.items():
        disease = f[f.label_id != 0]
        healthy = f[f.label_id == 0]
        y, p = disease.label_id.to_numpy(), disease.predicted_label_id.to_numpy()
        rows.append({
            "model": model,
            "stratum": "online_disease_46",
            "n": len(disease),
            "correct": int((y == p).sum()),
            "accuracy": round(float((y == p).mean()), 4),
            "macro_f1_4class": round(macro_f1(y, p, [1, 2, 3, 4]), 4),
            "predicted_healthy": int((p == 0).sum()),
            "loader": "identical_to_canonical",
        })
        rows.append({
            "model": model,
            "stratum": "self_collected_healthy_16",
            "n": len(healthy),
            "correct": CANONICAL_HEALTHY_CORRECT[model],
            "accuracy": round(CANONICAL_HEALTHY_CORRECT[model] / len(healthy), 4),
            "macro_f1_4class": np.nan,
            "predicted_healthy": CANONICAL_HEALTHY_CORRECT[model],
            "loader": "canonical_counts_from_notebook11",
        })
    return pd.DataFrame(rows)


def prediction_distribution(frames) -> pd.DataFrame:
    rows = []
    for model, f in frames.items():
        y, p = f.label_id.to_numpy(), f.predicted_label_id.to_numpy()
        for k, name in enumerate(CLASS_NAMES):
            predicted = int((p == k).sum())
            tp = int(((p == k) & (y == k)).sum())
            rows.append({
                "model": model,
                "class_name": name,
                "true_images": int((y == k).sum()),
                "predicted_as_class": predicted,
                "correct": tp,
                "precision": round(tp / predicted, 4) if predicted else np.nan,
                "recall": round(tp / int((y == k).sum()), 4),
                "loader": "upstream_raw_rgb (Healthy rows differ slightly from canonical)",
            })
    return pd.DataFrame(rows)


def stratified_mcnemar(frames) -> pd.DataFrame:
    pairs = [
        ("ResNet18 · Geometric + Mixed", "ResNet18 · Lab baseline"),
        ("ViT-Small · Mixed", "ViT-Small · Lab baseline"),
    ]
    rows = []
    for a, b in pairs:
        fa, fb = frames[a], frames[b]
        mask = (fa.label_id != 0).to_numpy()
        ca = fa.correct.astype(bool).to_numpy()[mask]
        cb = fb.correct.astype(bool).to_numpy()[mask]
        only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        rows.append({
            "family": "external_v1_online_disease_46",
            "model_a": a,
            "model_b": b,
            "n": int(mask.sum()),
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "exact_mcnemar_p": mcnemar_exact(only_a, only_b),
        })
    return pd.DataFrame(rows)


def mcnemar_family(frames) -> pd.DataFrame:
    existing = pd.read_csv(EXTERNAL_DIR / "confirmatory_paired_mcnemar_tests.csv")
    existing = existing.rename(columns={"partition": "family"})
    existing = existing[["family", "model_a", "model_b", "n", "only_a_correct", "only_b_correct", "exact_mcnemar_p"]]
    # The ResNet external pair has identical correctness in the canonical run
    # (docs/archive/experiment_reconciliation.md); other external pairs use the upstream loader.
    combined = pd.concat([existing, stratified_mcnemar(frames)], ignore_index=True)
    combined["holm_adjusted_p"] = [round(v, 6) for v in holm(combined.exact_mcnemar_p)]
    combined["exact_mcnemar_p"] = combined.exact_mcnemar_p.round(6)
    combined["significant_holm_0_05"] = combined.holm_adjusted_p < 0.05
    return combined


def parse_dev_curves() -> pd.DataFrame:
    notebook = json.loads(NB15_ARCHIVE.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"Seed (\d+) · epoch (\d+)/\d+ · Field Dev F1: Field-only ([0-9.]+), Mixed ([0-9.]+)"
    )
    rows = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", ""))
            for seed, epoch, fo, mx in pattern.findall(text):
                rows.append({"seed": int(seed), "epoch": int(epoch),
                             "dev_field_only": float(fo), "dev_mixed": float(mx)})
    curves = pd.DataFrame(rows).drop_duplicates(["seed", "epoch"])
    if curves.empty:
        raise ValueError("No Field Dev log lines found in the archived Notebook 15 execution")
    return curves


def notebook15_paired(n_boot: int = 10000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = {}
    for s in (42, 43, 44):
        for c in ("field_only", "mixed"):
            f = pd.read_csv(NB15_DIR / "prediction_exports" / f"seed_{s}_{c}_field_test.csv")
            preds[(s, c)] = f.sort_values("row_id").reset_index(drop=True)
    y = preds[(42, "field_only")].label_id.to_numpy()
    for frame in preds.values():
        if not np.array_equal(frame.label_id.to_numpy(), y):
            raise ValueError("Notebook 15 exports are not aligned")
    row_ids = preds[(42, "field_only")].row_id.to_numpy()
    curves = parse_dev_curves()

    rows = []
    for s in (42, 43, 44):
        fo, mx = preds[(s, "field_only")], preds[(s, "mixed")]
        a, b = fo.correct.astype(bool).to_numpy(), mx.correct.astype(bool).to_numpy()
        epoch = int(fo.selected_epoch.iloc[0])
        dev = curves[(curves.seed == s) & (curves.epoch == epoch)].iloc[0]
        rows.append({
            "seed": s,
            "selected_epoch": epoch,
            "test_field_only_macro_f1": round(macro_f1(y, fo.predicted_label_id.to_numpy(), range(5)), 4),
            "test_mixed_macro_f1": round(macro_f1(y, mx.predicted_label_id.to_numpy(), range(5)), 4),
            "test_only_field_only_correct": int((a & ~b).sum()),
            "test_only_mixed_correct": int((~a & b).sum()),
            "test_prediction_agreement": round(float((fo.predicted_label_id == mx.predicted_label_id).mean()), 4),
            "dev_field_only_macro_f1_at_selected_epoch": dev.dev_field_only,
            "dev_mixed_macro_f1_at_selected_epoch": dev.dev_mixed,
            "dev_mixed_minus_field_only": round(dev.dev_mixed - dev.dev_field_only, 4),
            "row_2524_predictions": f"{int(fo.loc[fo.row_id == SUSPECT_TEST_ROW, 'predicted_label_id'].iloc[0])}/"
                                    f"{int(mx.loc[mx.row_id == SUSPECT_TEST_ROW, 'predicted_label_id'].iloc[0])}",
        })
    per_seed = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    n = len(y)

    def seed_mean_diff(index, metric):
        diffs = []
        for s in (42, 43, 44):
            yt = y[index]
            pf = preds[(s, "field_only")].predicted_label_id.to_numpy()[index]
            pm = preds[(s, "mixed")].predicted_label_id.to_numpy()[index]
            if metric == "macro_f1":
                diffs.append(macro_f1(yt, pm, range(5)) - macro_f1(yt, pf, range(5)))
            else:
                diffs.append(float((pm == yt).mean() - (pf == yt).mean()))
        return float(np.mean(diffs))

    summary = []
    for label, keep in (("all_47", np.ones(n, bool)), ("excluding_row_2524", row_ids != SUSPECT_TEST_ROW)):
        idx_all = np.flatnonzero(keep)
        for metric in ("macro_f1", "accuracy"):
            point = seed_mean_diff(idx_all, metric)
            boots = [seed_mean_diff(rng.choice(idx_all, size=len(idx_all), replace=True), metric)
                     for _ in range(n_boot)]
            low, high = np.percentile(boots, [2.5, 97.5])
            summary.append({
                "test_subset": label,
                "metric": metric,
                "mixed_minus_field_only_seed_mean": round(point, 4),
                "bootstrap_95ci_low": round(float(low), 4),
                "bootstrap_95ci_high": round(float(high), 4),
                "share_bootstrap_above_zero": round(float(np.mean(np.array(boots) > 0)), 4),
                "n_bootstrap": n_boot,
                "unit": "image-level paired resampling; seed-mean difference",
            })
    return per_seed, pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")

    frames = load_external()
    strata = external_by_stratum(frames)
    distribution = prediction_distribution(frames)
    family = mcnemar_family(frames)
    per_seed, paired = notebook15_paired(n_boot=args.n_boot)

    output.mkdir(parents=True)
    strata.to_csv(output / "external_v1_by_stratum.csv", index=False)
    distribution.to_csv(output / "external_v1_prediction_distribution.csv", index=False)
    family.to_csv(output / "mcnemar_holm.csv", index=False)
    per_seed.to_csv(output / "notebook15_per_seed.csv", index=False)
    paired.to_csv(output / "notebook15_paired_bootstrap.csv", index=False)
    for name in ("strata", "family", "per_seed", "paired"):
        print(f"\n== {name}\n", locals()[name].to_string(index=False))


if __name__ == "__main__":
    main()
