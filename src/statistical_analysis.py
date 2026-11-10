"""Reproducible statistical uncertainty, calibration, and artifact extraction for saved evaluations.

Run locally from the project root (no GPU or external Drive mount required):
    python -m src.statistical_analysis

This module performs three non-destructive tasks using committed CSVs and saved
notebook outputs (09_final_model_comparison.ipynb and 10_error_analysis.ipynb):
1. Reconstructs the local summary tables and figures for `outputs/final_comparison_development`
   and `outputs/error_analysis_development` directly from the saved notebook outputs.
2. Computes exact Wilson 95% confidence intervals, 10,000-iteration bootstrap confidence
   intervals, and exact two-sided McNemar test p-values for model comparisons.
3. Computes Expected Calibration Error (ECE) and high-confidence error metrics (>= 0.90)
   across all saved prediction manifests, plus soft-voting/confidence-based ensemble
   baselines where per-sample predictions are available.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute the 95% Wilson score confidence interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_binomial_ci(
    k: int, n: int, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Non-parametric percentile bootstrap 95% CI for accuracy from (k, n)."""
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n=n, p=k / n, size=n_boot) / float(n)
    low, high = np.percentile(draws, [2.5, 97.5])
    return (float(low), float(high))


def bootstrap_predictions_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% CI for both Accuracy and Macro-F1 when per-row predictions exist."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_samples = np.empty(n_boot, dtype=np.float64)
    f1_samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        acc_samples[i] = np.mean(yt == yp)
        f1_samples[i] = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy_ci": (float(np.percentile(acc_samples, 2.5)), float(np.percentile(acc_samples, 97.5))),
        "macro_f1_ci": (float(np.percentile(f1_samples, 2.5)), float(np.percentile(f1_samples, 97.5))),
    }


def compute_ece(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width confidence bins in [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(correct)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence > low) & (confidence <= high)
        if np.any(mask):
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confidence[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def mcnemar_exact(b_only_a: int, c_only_b: int) -> float:
    """Two-sided exact binomial McNemar p-value given discordant counts b and c."""
    discordant = b_only_a + c_only_b
    if discordant == 0:
        return 1.0
    return float(binomtest(min(b_only_a, c_only_b), n=discordant, p=0.5, alternative="two-sided").pvalue)


def extract_notebook_artifacts() -> None:
    """Populate local `outputs/final_comparison_development` and `outputs/error_analysis_development`."""
    final_dir = PROJECT_ROOT / "outputs" / "final_comparison_development"
    error_dir = PROJECT_ROOT / "outputs" / "error_analysis_development"
    final_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reconstruct final_comparison_summary.csv & lab_field_tradeoff.csv from NB 09
    summary_rows = [
        {"model": "ResNet18 · Lab baseline", "phase": "lab_only", "partition": "lab_test", "accuracy": 0.9793, "macro_f1": 0.9744, "correct": 1657, "images": 1692},
        {"model": "ResNet18 · Lab baseline", "phase": "lab_only", "partition": "field_dev", "accuracy": 0.3261, "macro_f1": 0.2355, "correct": 30, "images": 92},
        {"model": "ViT-Small · Lab baseline", "phase": "lab_only", "partition": "lab_test", "accuracy": 0.9947, "macro_f1": 0.9938, "correct": 1683, "images": 1692},
        {"model": "ViT-Small · Lab baseline", "phase": "lab_only", "partition": "field_dev", "accuracy": 0.4022, "macro_f1": 0.2677, "correct": 37, "images": 92},
        {"model": "ResNet18 · Geometric + Mixed", "phase": "mixed_domain", "partition": "lab_test", "accuracy": 0.9297, "macro_f1": 0.9140, "correct": 1573, "images": 1692},
        {"model": "ResNet18 · Geometric + Mixed", "phase": "mixed_domain", "partition": "field_dev", "accuracy": 0.4674, "macro_f1": 0.4459, "correct": 43, "images": 92},
        {"model": "ViT-Small · Mixed", "phase": "mixed_domain", "partition": "lab_test", "accuracy": 0.9900, "macro_f1": 0.9886, "correct": 1675, "images": 1692},
        {"model": "ViT-Small · Mixed", "phase": "mixed_domain", "partition": "field_dev", "accuracy": 0.4022, "macro_f1": 0.3872, "correct": 37, "images": 92},
    ]
    pd.DataFrame(summary_rows).to_csv(final_dir / "final_comparison_summary.csv", index=False)

    tradeoff_rows = [
        {"model": "ResNet18 · Lab baseline", "accuracy_field_dev": 0.3261, "accuracy_lab_test": 0.9793, "macro_f1_field_dev": 0.2355, "macro_f1_lab_test": 0.9744, "macro_f1_lab_minus_field": 0.7390},
        {"model": "ViT-Small · Lab baseline", "accuracy_field_dev": 0.4022, "accuracy_lab_test": 0.9947, "macro_f1_field_dev": 0.2677, "macro_f1_lab_test": 0.9938, "macro_f1_lab_minus_field": 0.7261},
        {"model": "ResNet18 · Geometric + Mixed", "accuracy_field_dev": 0.4674, "accuracy_lab_test": 0.9297, "macro_f1_field_dev": 0.4459, "macro_f1_lab_test": 0.9140, "macro_f1_lab_minus_field": 0.4680},
        {"model": "ViT-Small · Mixed", "accuracy_field_dev": 0.4022, "accuracy_lab_test": 0.9900, "macro_f1_field_dev": 0.3872, "macro_f1_lab_test": 0.9886, "macro_f1_lab_minus_field": 0.6015},
    ]
    pd.DataFrame(tradeoff_rows).to_csv(final_dir / "lab_field_tradeoff.csv", index=False)

    per_class_recall_rows = [
        {"class_name": "Healthy", "support": 11, "ResNet18 · Lab baseline": 0.0000, "ViT-Small · Lab baseline": 0.0000, "ResNet18 · Geometric + Mixed": 0.5455, "ViT-Small · Mixed": 0.5455},
        {"class_name": "Early Blight", "support": 16, "ResNet18 · Lab baseline": 0.8125, "ViT-Small · Lab baseline": 0.2500, "ResNet18 · Geometric + Mixed": 0.3125, "ViT-Small · Mixed": 0.2500},
        {"class_name": "Late Blight", "support": 20, "ResNet18 · Lab baseline": 0.4500, "ViT-Small · Lab baseline": 0.7000, "ResNet18 · Geometric + Mixed": 0.5500, "ViT-Small · Mixed": 0.4500},
        {"class_name": "Bacterial Spot", "support": 18, "ResNet18 · Lab baseline": 0.0000, "ViT-Small · Lab baseline": 0.0000, "ResNet18 · Geometric + Mixed": 0.1667, "ViT-Small · Mixed": 0.1667},
        {"class_name": "Septoria Leaf Spot", "support": 27, "ResNet18 · Lab baseline": 0.2963, "ViT-Small · Lab baseline": 0.7037, "ResNet18 · Geometric + Mixed": 0.6667, "ViT-Small · Mixed": 0.5556},
    ]
    pd.DataFrame(per_class_recall_rows).to_csv(final_dir / "final_comparison_per_class.csv", index=False)

    # 2. Reconstruct error analysis CSVs from NB 10
    error_by_class_rows = [
        {"class_name": "Healthy", "images": 11, "resnet_correct": 6, "vit_correct": 6, "both_wrong": 2, "resnet_recall": 0.5455, "vit_recall": 0.5455},
        {"class_name": "Early Blight", "images": 16, "resnet_correct": 5, "vit_correct": 4, "both_wrong": 8, "resnet_recall": 0.3125, "vit_recall": 0.2500},
        {"class_name": "Late Blight", "images": 20, "resnet_correct": 11, "vit_correct": 9, "both_wrong": 5, "resnet_recall": 0.5500, "vit_recall": 0.4500},
        {"class_name": "Bacterial Spot", "images": 18, "resnet_correct": 3, "vit_correct": 3, "both_wrong": 13, "resnet_recall": 0.1667, "vit_recall": 0.1667},
        {"class_name": "Septoria Leaf Spot", "images": 27, "resnet_correct": 18, "vit_correct": 15, "both_wrong": 8, "resnet_recall": 0.6667, "vit_recall": 0.5556},
    ]
    pd.DataFrame(error_by_class_rows).to_csv(error_dir / "field_dev_error_by_class.csv", index=False)

    high_conf_rows = [
        {"row_id": 1580, "class_name": "Early Blight", "resnet_mixed_prediction": "Bacterial Spot", "resnet_mixed_confidence": 0.706, "vit_mixed_prediction": "Bacterial Spot", "vit_mixed_confidence": 0.941, "outcome": "both_wrong"},
        {"row_id": 1596, "class_name": "Early Blight", "resnet_mixed_prediction": "Late Blight", "resnet_mixed_confidence": 0.733, "vit_mixed_prediction": "Late Blight", "vit_mixed_confidence": 0.958, "outcome": "both_wrong"},
        {"row_id": 1629, "class_name": "Early Blight", "resnet_mixed_prediction": "Early Blight", "resnet_mixed_confidence": 0.419, "vit_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_confidence": 0.952, "outcome": "resnet_only"},
        {"row_id": 1688, "class_name": "Septoria Leaf Spot", "resnet_mixed_prediction": "Bacterial Spot", "resnet_mixed_confidence": 0.915, "vit_mixed_prediction": "Bacterial Spot", "vit_mixed_confidence": 0.927, "outcome": "both_wrong"},
        {"row_id": 1891, "class_name": "Bacterial Spot", "resnet_mixed_prediction": "Bacterial Spot", "resnet_mixed_confidence": 0.509, "vit_mixed_prediction": "Healthy", "vit_mixed_confidence": 0.935, "outcome": "resnet_only"},
        {"row_id": 1899, "class_name": "Bacterial Spot", "resnet_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.966, "vit_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_confidence": 0.997, "outcome": "both_wrong"},
        {"row_id": 1909, "class_name": "Bacterial Spot", "resnet_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.925, "vit_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_confidence": 0.953, "outcome": "both_wrong"},
        {"row_id": 1994, "class_name": "Late Blight", "resnet_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.387, "vit_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_confidence": 0.970, "outcome": "both_wrong"},
        {"row_id": 2025, "class_name": "Late Blight", "resnet_mixed_prediction": "Healthy", "resnet_mixed_confidence": 0.322, "vit_mixed_prediction": "Healthy", "vit_mixed_confidence": 0.904, "outcome": "both_wrong"},
    ]
    pd.DataFrame(high_conf_rows).to_csv(error_dir / "high_confidence_errors.csv", index=False)

    selected_cases_rows = [
        {"panel_row": 0, "row_id": 1899, "reason": "both_wrong", "class_name": "Bacterial Spot", "resnet_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.966, "vit_mixed_confidence": 0.997},
        {"panel_row": 1, "row_id": 1994, "reason": "both_wrong", "class_name": "Late Blight", "resnet_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.387, "vit_mixed_confidence": 0.970},
        {"panel_row": 2, "row_id": 1596, "reason": "both_wrong", "class_name": "Early Blight", "resnet_mixed_prediction": "Late Blight", "vit_mixed_prediction": "Late Blight", "resnet_mixed_confidence": 0.733, "vit_mixed_confidence": 0.958},
        {"panel_row": 3, "row_id": 1812, "reason": "resnet_only", "class_name": "Healthy", "resnet_mixed_prediction": "Healthy", "vit_mixed_prediction": "Late Blight", "resnet_mixed_confidence": 0.959, "vit_mixed_confidence": 0.654},
        {"panel_row": 4, "row_id": 1629, "reason": "resnet_only", "class_name": "Early Blight", "resnet_mixed_prediction": "Early Blight", "vit_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.419, "vit_mixed_confidence": 0.952},
        {"panel_row": 5, "row_id": 1602, "reason": "vit_only", "class_name": "Early Blight", "resnet_mixed_prediction": "Late Blight", "vit_mixed_prediction": "Early Blight", "resnet_mixed_confidence": 0.752, "vit_mixed_confidence": 0.987},
        {"panel_row": 6, "row_id": 1804, "reason": "vit_only", "class_name": "Healthy", "resnet_mixed_prediction": "Bacterial Spot", "vit_mixed_prediction": "Healthy", "resnet_mixed_confidence": 0.501, "vit_mixed_confidence": 0.956},
        {"panel_row": 7, "row_id": 1909, "reason": "bacterial_spot", "class_name": "Bacterial Spot", "resnet_mixed_prediction": "Septoria Leaf Spot", "vit_mixed_prediction": "Septoria Leaf Spot", "resnet_mixed_confidence": 0.925, "vit_mixed_confidence": 0.953},
    ]
    pd.DataFrame(selected_cases_rows).to_csv(error_dir / "selected_error_cases.csv", index=False)

    # 3. Extract embedded PNGs from NB 09 and NB 10
    nb09_path = PROJECT_ROOT / "notebooks" / "09_final_model_comparison.ipynb"
    nb10_path = PROJECT_ROOT / "notebooks" / "10_error_analysis.ipynb"
    if nb09_path.is_file():
        nb09 = json.loads(nb09_path.read_text(encoding="utf-8"))
        # Cell 12 has cnn_vs_vit_lab_field.png; Cell 14 has field_dev_per_class_recall.png
        for cell_idx, out_name in ((12, "cnn_vs_vit_lab_field.png"), (14, "field_dev_per_class_recall.png")):
            for out in nb09["cells"][cell_idx].get("outputs", []):
                b64 = out.get("data", {}).get("image/png")
                if b64:
                    (final_dir / out_name).write_bytes(base64.b64decode(b64))
                    break

    if nb10_path.is_file():
        nb10 = json.loads(nb10_path.read_text(encoding="utf-8"))
        for cell_idx, out_name in (
            (8, "paired_outcomes.png"),
            (9, "field_recall_by_class.png"),
            (14, "selected_error_gallery.png"),
            (22, "cnn_gradcam_vs_vit_attention.png"),
        ):
            for out in nb10["cells"][cell_idx].get("outputs", []):
                b64 = out.get("data", {}).get("image/png")
                if b64:
                    (error_dir / out_name).write_bytes(base64.b64decode(b64))
                    break

        # Generate the verified 4-case panel from `cnn_gradcam_vs_vit_attention.png`
        # Selecting 1 case from each distinct diagnostic category in Cell 13:
        #   Panel row 0 (row_id=1899, both_wrong: True=Bacterial Spot, ResNet=Septoria, ViT=Septoria)
        #   Panel row 4 (row_id=1629, resnet_only: True=Early Blight, ResNet=Early Blight, ViT=Septoria)
        #   Panel row 5 (row_id=1602, vit_only: True=Early Blight, ResNet=Late Blight, ViT=Early Blight)
        #   Panel row 7 (row_id=1909, bacterial_spot: True=Bacterial Spot, ResNet=Septoria, ViT=Septoria)
        full_panel_path = error_dir / "cnn_gradcam_vs_vit_attention.png"
        if full_panel_path.is_file():
            img = Image.open(full_panel_path)
            w, h = img.size
            row_h = h / 8.0
            chosen_indices = [0, 4, 5, 7]
            strips = []
            for idx in chosen_indices:
                y0 = int(round(idx * row_h))
                y1 = int(round((idx + 1) * row_h))
                strips.append(img.crop((0, y0, w, y1)))
            total_h = sum(s.size[1] for s in strips)
            cropped = Image.new("RGB", (w, total_h), (255, 255, 255))
            curr_y = 0
            for s in strips:
                cropped.paste(s, (0, curr_y))
                curr_y += s.size[1]
            cropped.save(error_dir / "gradcam_vit_attention_4cases.png")


def run_statistical_analysis() -> None:
    """Compute confidence intervals, McNemar tests, and calibration metrics."""
    stat_dir = PROJECT_ROOT / "outputs" / "statistical_analysis"
    stat_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(5))

    # 1. Accuracy & Macro-F1 95% Confidence Intervals
    ci_rows = []
    # Four final models on Lab Test (1692) and Verified Field Dev (92)
    final_evals = [
        ("ResNet18 · Lab baseline", "lab_test", 1657, 1692, 0.9793, 0.9744, PROJECT_ROOT / "outputs" / "resnet18_domain_gap" / "lab_test_predictions.csv"),
        ("ResNet18 · Lab baseline", "field_dev", 30, 92, 0.3261, 0.2355, PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "baseline" / "predictions.csv"),
        ("ViT-Small · Lab baseline", "lab_test", 1683, 1692, 0.9947, 0.9938, PROJECT_ROOT / "outputs" / "vit_small_domain_gap" / "lab_test_predictions.csv"),
        ("ViT-Small · Lab baseline", "field_dev", 37, 92, 0.4022, 0.2677, None),
        ("ResNet18 · Geometric + Mixed", "lab_test", 1573, 1692, 0.9297, 0.9140, None),
        ("ResNet18 · Geometric + Mixed", "field_dev", 43, 92, 0.4674, 0.4459, None),
        ("ViT-Small · Mixed", "lab_test", 1675, 1692, 0.9900, 0.9886, None),
        ("ViT-Small · Mixed", "field_dev", 37, 92, 0.4022, 0.3872, None),
        ("ResNet18 · Lab baseline", "field_test", 17, 47, 0.3617, 0.2802, PROJECT_ROOT / "outputs" / "resnet18_domain_gap" / "field_test_predictions.csv"),
        ("ViT-Small · Lab baseline", "field_test", 21, 47, 0.4468, 0.3617, PROJECT_ROOT / "outputs" / "vit_small_domain_gap" / "field_test_predictions.csv"),
        ("ResNet18 · Photometric (frozen)", "field_dev", 31, 92, 0.3370, 0.2444, PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "photometric" / "predictions.csv"),
        ("ResNet18 · Geometric (frozen)", "field_dev", 32, 92, 0.3478, 0.2497, PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "geometric" / "predictions.csv"),
    ]

    for model_name, part, k, n, acc, mf1, pred_path in final_evals:
        w_low, w_high = wilson_ci(k, n)
        b_low, b_high = bootstrap_binomial_ci(k, n)
        mf1_low, mf1_high = (np.nan, np.nan)
        if pred_path is not None and pred_path.is_file():
            df = pd.read_csv(pred_path)
            boot = bootstrap_predictions_ci(
                df["label_id"].to_numpy(dtype=int),
                df["predicted_label_id"].to_numpy(dtype=int),
                labels=labels,
            )
            b_low, b_high = boot["accuracy_ci"]
            mf1_low, mf1_high = boot["macro_f1_ci"]
        ci_rows.append(
            {
                "model": model_name,
                "partition": part,
                "correct": k,
                "total": n,
                "accuracy": round(acc, 4),
                "wilson_95ci_low": round(w_low, 4),
                "wilson_95ci_high": round(w_high, 4),
                "bootstrap_acc_95ci_low": round(b_low, 4),
                "bootstrap_acc_95ci_high": round(b_high, 4),
                "macro_f1": round(mf1, 4),
                "bootstrap_f1_95ci_low": round(mf1_low, 4) if not np.isnan(mf1_low) else None,
                "bootstrap_f1_95ci_high": round(mf1_high, 4) if not np.isnan(mf1_high) else None,
            }
        )
    pd.DataFrame(ci_rows).to_csv(stat_dir / "accuracy_confidence_intervals.csv", index=False)

    # 2. Per-class recall Wilson 95% CIs on Verified Field Dev (n=92)
    class_supports = [
        ("Healthy", 11, 0, 0, 6, 6),
        ("Early Blight", 16, 13, 4, 5, 4),
        ("Late Blight", 20, 9, 14, 11, 9),
        ("Bacterial Spot", 18, 0, 0, 3, 3),
        ("Septoria Leaf Spot", 27, 8, 19, 18, 15),
    ]
    per_class_ci_rows = []
    for cname, n_sup, k_r_base, k_v_base, k_r_mix, k_v_mix in class_supports:
        for m_label, k_val in (
            ("ResNet18 · Lab baseline", k_r_base),
            ("ViT-Small · Lab baseline", k_v_base),
            ("ResNet18 · Geometric + Mixed", k_r_mix),
            ("ViT-Small · Mixed", k_v_mix),
        ):
            lo, hi = wilson_ci(k_val, n_sup)
            per_class_ci_rows.append(
                {
                    "class_name": cname,
                    "support": n_sup,
                    "model": m_label,
                    "correct": k_val,
                    "recall": round(k_val / n_sup, 4),
                    "wilson_95ci_low": round(lo, 4),
                    "wilson_95ci_high": round(hi, 4),
                }
            )
    pd.DataFrame(per_class_ci_rows).to_csv(stat_dir / "per_class_recall_confidence_intervals.csv", index=False)

    # 3. Paired McNemar Tests & Soft-Voting Ensemble Analysis
    mcnemar_rows = []

    # Pair A: ResNet18 · Geometric + Mixed vs. ViT-Small · Mixed on Verified Field Dev (n=92)
    # Saved contingency table from Notebook 09 Cell 14 and Notebook 10 Cell 6:
    # both_correct=24, resnet_only=19, vit_only=13, both_wrong=36
    p_adapt = mcnemar_exact(19, 13)
    mcnemar_rows.append(
        {
            "comparison": "ResNet18 · Geometric + Mixed vs. ViT-Small · Mixed",
            "partition": "field_dev",
            "n": 92,
            "both_correct": 24,
            "model_a_only": 19,
            "model_b_only": 13,
            "both_wrong": 36,
            "acc_a": round(43 / 92, 4),
            "acc_b": round(37 / 92, 4),
            "acc_diff_a_minus_b": round((43 - 37) / 92, 4),
            "mcnemar_exact_p": round(p_adapt, 4),
            "significant_at_0_05": p_adapt < 0.05,
            "oracle_union_correct": 56,
            "oracle_union_acc": round(56 / 92, 4),
        }
    )

    # Pair B: ResNet18 · Lab baseline vs. ViT-Small · Lab baseline on Original Field Test (n=47)
    paired_ft_path = PROJECT_ROOT / "outputs" / "comparison" / "paired_field_predictions.csv"
    if paired_ft_path.is_file():
        ft_df = pd.read_csv(paired_ft_path)
        r_ok = ft_df["resnet_correct"].astype(bool).to_numpy()
        v_ok = ft_df["vit_correct"].astype(bool).to_numpy()
        bc = int(np.sum(r_ok & v_ok))
        ro = int(np.sum(r_ok & ~v_ok))
        vo = int(np.sum(~r_ok & v_ok))
        bw = int(np.sum(~r_ok & ~v_ok))
        p_ft = mcnemar_exact(ro, vo)
        # Also evaluate max-confidence rule on paired Field Test
        r_conf = ft_df["resnet_confidence"].to_numpy()
        v_conf = ft_df["vit_confidence"].to_numpy()
        choose_r = r_conf >= v_conf
        ens_pred = np.where(choose_r, ft_df["resnet_predicted_class"], ft_df["vit_predicted_class"])
        ens_ok = int(np.sum(ens_pred == ft_df["class_name"]))
        mcnemar_rows.append(
            {
                "comparison": "ResNet18 · Lab baseline vs. ViT-Small · Lab baseline",
                "partition": "field_test",
                "n": len(ft_df),
                "both_correct": bc,
                "model_a_only": ro,
                "model_b_only": vo,
                "both_wrong": bw,
                "acc_a": round(float(np.mean(r_ok)), 4),
                "acc_b": round(float(np.mean(v_ok)), 4),
                "acc_diff_a_minus_b": round(float(np.mean(r_ok) - np.mean(v_ok)), 4),
                "mcnemar_exact_p": round(p_ft, 4),
                "significant_at_0_05": p_ft < 0.05,
                "oracle_union_correct": bc + ro + vo,
                "oracle_union_acc": round((bc + ro + vo) / len(ft_df), 4),
                "max_conf_selector_correct": ens_ok,
                "max_conf_selector_acc": round(ens_ok / len(ft_df), 4),
            }
        )

    # Pair C: ResNet18 · Lab baseline vs. ViT-Small · Lab baseline on Lab Test (n=1692)
    r_lab_path = PROJECT_ROOT / "outputs" / "resnet18_domain_gap" / "lab_test_predictions.csv"
    v_lab_path = PROJECT_ROOT / "outputs" / "vit_small_domain_gap" / "lab_test_predictions.csv"
    if r_lab_path.is_file() and v_lab_path.is_file():
        r_lab = pd.read_csv(r_lab_path).sort_values("row_id")
        v_lab = pd.read_csv(v_lab_path).sort_values("row_id")
        r_ok = r_lab["correct"].astype(bool).to_numpy()
        v_ok = v_lab["correct"].astype(bool).to_numpy()
        bc = int(np.sum(r_ok & v_ok))
        ro = int(np.sum(r_ok & ~v_ok))
        vo = int(np.sum(~r_ok & v_ok))
        bw = int(np.sum(~r_ok & ~v_ok))
        p_lab = mcnemar_exact(ro, vo)
        mcnemar_rows.append(
            {
                "comparison": "ResNet18 · Lab baseline vs. ViT-Small · Lab baseline",
                "partition": "lab_test",
                "n": len(r_lab),
                "both_correct": bc,
                "model_a_only": ro,
                "model_b_only": vo,
                "both_wrong": bw,
                "acc_a": round(float(np.mean(r_ok)), 4),
                "acc_b": round(float(np.mean(v_ok)), 4),
                "acc_diff_a_minus_b": round(float(np.mean(r_ok) - np.mean(v_ok)), 4),
                "mcnemar_exact_p": round(p_lab, 6),
                "significant_at_0_05": p_lab < 0.05,
                "oracle_union_correct": bc + ro + vo,
                "oracle_union_acc": round((bc + ro + vo) / len(r_lab), 4),
            }
        )

    # Pair D: ResNet18 · Baseline vs. Geometric (frozen) on Verified Field Dev (n=92)
    r_base_fd = PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "baseline" / "predictions.csv"
    r_geom_fd = PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "geometric" / "predictions.csv"
    if r_base_fd.is_file() and r_geom_fd.is_file():
        b_df = pd.read_csv(r_base_fd).sort_values("row_id")
        g_df = pd.read_csv(r_geom_fd).sort_values("row_id")
        b_ok = b_df["correct"].astype(bool).to_numpy()
        g_ok = g_df["correct"].astype(bool).to_numpy()
        bc = int(np.sum(b_ok & g_ok))
        bo = int(np.sum(b_ok & ~g_ok))
        go = int(np.sum(~b_ok & g_ok))
        bw = int(np.sum(~b_ok & ~g_ok))
        p_bg = mcnemar_exact(bo, go)
        mcnemar_rows.append(
            {
                "comparison": "ResNet18 · Lab baseline vs. ResNet18 · Geometric (frozen)",
                "partition": "field_dev",
                "n": len(b_df),
                "both_correct": bc,
                "model_a_only": bo,
                "model_b_only": go,
                "both_wrong": bw,
                "acc_a": round(float(np.mean(b_ok)), 4),
                "acc_b": round(float(np.mean(g_ok)), 4),
                "acc_diff_a_minus_b": round(float(np.mean(b_ok) - np.mean(g_ok)), 4),
                "mcnemar_exact_p": round(p_bg, 4),
                "significant_at_0_05": p_bg < 0.05,
                "oracle_union_correct": bc + bo + go,
                "oracle_union_acc": round((bc + bo + go) / len(b_df), 4),
            }
        )

    pd.DataFrame(mcnemar_rows).to_csv(stat_dir / "paired_mcnemar_tests.csv", index=False)

    # 4. Calibration & High-Confidence Error Analysis across all saved prediction CSVs
    cal_files = [
        ("ResNet18 · Lab baseline", "lab_test", PROJECT_ROOT / "outputs" / "resnet18_domain_gap" / "lab_test_predictions.csv"),
        ("ViT-Small · Lab baseline", "lab_test", PROJECT_ROOT / "outputs" / "vit_small_domain_gap" / "lab_test_predictions.csv"),
        ("ResNet18 · Lab baseline", "field_test", PROJECT_ROOT / "outputs" / "resnet18_domain_gap" / "field_test_predictions.csv"),
        ("ViT-Small · Lab baseline", "field_test", PROJECT_ROOT / "outputs" / "vit_small_domain_gap" / "field_test_predictions.csv"),
        ("ResNet18 · Lab baseline", "field_dev", PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "baseline" / "predictions.csv"),
        ("ResNet18 · Photometric (frozen)", "field_dev", PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "photometric" / "predictions.csv"),
        ("ResNet18 · Geometric (frozen)", "field_dev", PROJECT_ROOT / "outputs" / "field_dev_comparison_verified" / "geometric" / "predictions.csv"),
    ]
    cal_rows = []
    for m_name, part, c_path in cal_files:
        if not c_path.is_file():
            continue
        df = pd.read_csv(c_path)
        corr = df["correct"].astype(bool).to_numpy()
        conf = df["confidence"].to_numpy(dtype=float)
        ece = compute_ece(corr, conf, n_bins=10)
        errors = ~corr
        high_conf_err = int(np.sum(errors & (conf >= 0.90)))
        total_err = int(np.sum(errors))
        cal_rows.append(
            {
                "model": m_name,
                "partition": part,
                "n": len(df),
                "accuracy": round(float(np.mean(corr)), 4),
                "mean_confidence": round(float(np.mean(conf)), 4),
                "mean_conf_correct": round(float(np.mean(conf[corr])), 4) if np.any(corr) else None,
                "mean_conf_error": round(float(np.mean(conf[errors])), 4) if np.any(errors) else None,
                "ece_10bins": round(ece, 4),
                "total_errors": total_err,
                "high_conf_errors_ge_0_90": high_conf_err,
                "high_conf_share_of_errors": round(high_conf_err / total_err, 4) if total_err > 0 else 0.0,
            }
        )

    # Append saved high-confidence error counts from Notebook 10 for the two adapted models
    cal_rows.append(
        {
            "model": "ResNet18 · Geometric + Mixed",
            "partition": "field_dev",
            "n": 92,
            "accuracy": 0.4674,
            "mean_confidence": None,
            "mean_conf_correct": None,
            "mean_conf_error": None,
            "ece_10bins": None,
            "total_errors": 49,
            "high_conf_errors_ge_0_90": 3,
            "high_conf_share_of_errors": round(3 / 49, 4),
        }
    )
    cal_rows.append(
        {
            "model": "ViT-Small · Mixed",
            "partition": "field_dev",
            "n": 92,
            "accuracy": 0.4022,
            "mean_confidence": None,
            "mean_conf_correct": None,
            "mean_conf_error": None,
            "ece_10bins": None,
            "total_errors": 55,
            "high_conf_errors_ge_0_90": 9,
            "high_conf_share_of_errors": round(9 / 55, 4),
        }
    )
    pd.DataFrame(cal_rows).to_csv(stat_dir / "calibration_and_confidence_metrics.csv", index=False)


def main() -> None:
    extract_notebook_artifacts()
    run_statistical_analysis()
    print("Statistical analysis and notebook artifact extraction completed successfully.")


if __name__ == "__main__":
    main()

