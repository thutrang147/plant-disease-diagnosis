"""Headline claims in the README are recomputed from committed result files.

Each check derives a number from a CSV or JSON file under ``data/results`` and
asserts that the README states it with the same rounding. If a result file or
the text changes without the other, the test fails, which is how the
consistency pass is kept honest after the freeze.
"""

import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/results"
README = (ROOT / "README.md").read_text() + "\n" + (ROOT / "docs/exploratory_analyses.md").read_text()
README_ONLY = (ROOT / "README.md").read_text()


def pp(x: float, signed: bool = True) -> str:
    s = f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"
    return s.replace("-", "−")


def f3(x: float, signed: bool = False) -> str:
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "−")


class ReadmeClaimsTest(unittest.TestCase):
    def assertStated(self, text: str):
        self.assertTrue(text in README, f"README does not state: {text!r}")

    def test_external_v3_accuracy_range(self):
        m = pd.read_csv(R / "external_v3_plantseg/metrics.csv")
        self.assertEqual((round(100 * m.accuracy.min()), round(100 * m.accuracy.max())), (41, 44))
        self.assertStated("scores 41–44% accuracy")

    def test_p1_p2_accuracy_and_disease_f1(self):
        conf = pd.read_csv(R / "external_v3_plantseg/comparisons.csv")
        conf = conf[conf.family == "all"].set_index("id")  # full-precision confirmatory intervals
        p1, p2 = conf.loc["P1"], conf.loc["P2"]
        self.assertStated(f"ResNet {pp(p1.difference)} pp (95% CI {pp(p1.ci_low)} to {pp(p1.ci_high)}; McNemar p = 1.0)")
        self.assertStated(f"ViT {pp(p2.difference)} pp ({pp(p2.ci_low)} to {pp(p2.ci_high)}; p = {p2.p_value:.2f})")
        self.assertLess(max(p1.ci_high, p2.ci_high), 0.055)  # "exclude gains larger than about 5 pp"
        c = pd.read_csv(R / "external_v3_secondary_metrics/comparison_intervals.csv").set_index(["id", "metric"])
        f1a, f1b = c.loc[("P1", "disease_macro_f1")], c.loc[("P2", "disease_macro_f1")]
        self.assertStated(f"ResNet {f3(f1a.difference, True)} (95% CI {f3(f1a.ci_low)} to {f3(f1a.ci_high)})")
        self.assertStated(f"ViT {f3(f1b.difference, True)} ({f3(f1b.ci_low)} to {f3(f1b.ci_high)})")

    def test_secondary_metrics(self):
        c = pd.read_csv(R / "external_v3_secondary_metrics/comparison_intervals.csv").set_index(["id", "metric"])
        b1, b2, b3 = (c.loc[(k, "balanced_accuracy")] for k in ("P1", "P2", "P3"))
        self.assertStated(f"Balanced accuracy rises for ResNet ({f3(b1.difference, True)}, {f3(b1.ci_low)} to {f3(b1.ci_high)})")
        self.assertStated(f"ViT ({f3(b2.difference, True)}, {f3(b2.ci_low)} to {f3(b2.ci_high, True)})")
        f = c.loc[("P3", "disease_macro_f1")]
        self.assertStated(f"Disease macro-F1 is {f.b_value:.3f} vs {f.a_value:.3f} ({f3(f.difference, True)}, "
                          f"95% CI {f3(f.ci_low)} to {f3(f.ci_high, True)}) and balanced accuracy {f3(b3.difference, True)} "
                          f"({f3(b3.ci_low)} to {f3(b3.ci_high, True)}")

    def test_p3_eight_epochs(self):
        c = pd.read_csv(R / "external_v3_plantseg/comparisons.csv")
        p3 = c[(c.family == "all") & (c.id == "P3")].iloc[0]
        self.assertStated(f"{pp(p3.difference)} pp (95% CI {pp(p3.ci_low)} to {pp(p3.ci_high)}; seed-level p = 0.12)")
        seed = pd.read_csv(R / "external_v3_secondary_metrics/p3_seed_level.csv").set_index("metric").loc["accuracy"]
        self.assertEqual(round(seed.seed_level_t_test_p, 2), 0.12)
        diffs = sorted(round(100 * seed[f"seed_{s}"], 1) for s in (42, 43, 44))
        self.assertEqual(diffs, [0.6, 1.6, 2.9])
        self.assertStated("(+2.9, +0.6, +1.6 pp)")
        self.assertStated("rehearsal is unlikely to cost more than 0.1 pp")

    def test_lab_retention_eight_epochs(self):
        s = pd.read_csv(R / "notebook15_lab_retention/lab_retention_summary.csv").set_index(["partition", "condition"])
        fo, mx = s.loc[("lab_test", "field_only")], s.loc[("lab_test", "mixed")]
        self.assertStated(f"Mean Lab Test macro-F1 is {fo.macro_f1_mean:.4f} ± {fo.macro_f1_sd:.4f} for Field-only and "
                          f"{mx.macro_f1_mean:.4f} ± {mx.macro_f1_sd:.4f} for Mixed")

    def test_mixed_budget_24_epochs(self):
        d = pd.read_csv(R / "mixed_budget_summary/seed_level_differences.csv").set_index("metric")
        lab = d.loc["lab_test_macro_f1"]
        vals = [lab[f"seed_{s}"] for s in (42, 43, 44)]
        self.assertEqual((round(min(vals), 2), round(max(vals), 2)), (0.11, 0.13))
        self.assertEqual(round(lab.seed_level_t_test_p, 3), 0.001)
        self.assertStated("Lab Test macro-F1 advantage of +0.11 to +0.13")
        v3 = pd.read_csv(R / "mixed_budget_summary/external_v3_secondary.csv").set_index("metric").loc["accuracy"]
        self.assertStated(f"{pp(v3.difference)} pp higher on External v3 (95% CI {pp(v3.ci_low)} to {pp(v3.ci_high)}; "
                          f"seed-level p = {d.loc['external_v3_accuracy', 'seed_level_t_test_p']:.2f})")
        self.assertStated(f"sign-flip p = {v3.p_value:.3f}")
        runs = pd.read_csv(R / "mixed_budget_summary/per_run.csv")
        self.assertTrue(runs.sanity_pass.all())
        self.assertEqual(int((~runs.sanity_field_dev_match).sum()), 1)

    def test_consistency_checks_are_exact(self):
        checks = pd.read_csv(R / "mixed_budget_summary/consistency_checks.csv")
        self.assertTrue((checks.correct_ref == checks.correct_new).all())
        self.assertTrue((checks.macro_f1_abs_diff == 0).all())

    def test_lesion_and_gradcam(self):
        ctrl = pd.read_csv(R / "occlusion_control/control_summary.csv")
        self.assertEqual((round(ctrl.extra_drop_lesion_vs_control.min(), 2), round(ctrl.extra_drop_lesion_vs_control.max(), 2)),
                         (0.12, 0.15))
        self.assertTrue(ctrl.extra_drop_ci_low.gt(0).all())
        cam = pd.read_csv(R / "lesion_reliance/gradcam_summary.csv")
        self.assertEqual((round(cam.pointing_accuracy.min(), 2), round(cam.pointing_accuracy.max(), 2)), (0.58, 0.66))
        centre = json.loads((R / "occlusion_control/protocol.json").read_text())["centre_baseline_pointing"]
        self.assertEqual(round(centre, 2), 0.74)
        self.assertStated("0.58–0.66 of images, against 0.46 by area chance but 0.74")
        ranges = json.loads((R / "occlusion_subset/ranges.json").read_text())
        lo, hi = ranges["control_minus_original_pp_range"]
        self.assertStated(f"changes accuracy by only {f'{lo:+.1f}'.replace('-', '−')} to {f'{hi:+.1f}'} pp")

    def test_calibration(self):
        cal = pd.read_csv(R / "external_v3_error_analysis/calibration.csv").set_index("model")
        lab = cal.loc["resnet18_lab"]
        self.assertStated(f"ResNet18 mean confidence is {lab.mean_confidence:.2f} at {lab.accuracy:.2f} accuracy (ECE {lab.ece_10_bins:.3f})")
        adapted = cal.drop(index=["resnet18_lab", "vit_small_lab"]).ece_10_bins
        self.assertStated(f"(ECE {adapted.min():.2f}–{adapted.max():.2f})")

    def test_adabn(self):
        d = pd.read_csv(R / "adabn_paired/adabn_paired.csv").set_index(["model", "partition"])
        fd, v3 = d.loc[("resnet18_lab", "field_dev")], d.loc[("resnet18_lab", "external_v3")]
        self.assertStated(f"macro-F1 rises from {fd.macro_f1_original:.3f} to {fd.macro_f1_adabn:.3f} "
                          f"({f3(fd.macro_f1_difference, True)}, paired bootstrap 95% CI {f3(fd.macro_f1_ci_low)} to "
                          f"{f3(fd.macro_f1_ci_high)})")
        self.assertStated(f"accuracy rises by {pp(fd.accuracy_difference, False)} pp (95% CI {pp(fd.accuracy_ci_low)} to "
                          f"{pp(fd.accuracy_ci_high)}; McNemar p = {fd.mcnemar_p:.2f}, Holm {fd.mcnemar_holm_p:.2f})")
        self.assertStated(f"accuracy gain is {pp(v3.accuracy_difference)} pp ({pp(v3.accuracy_ci_low)} to "
                          f"{pp(v3.accuracy_ci_high)}; p = {v3.mcnemar_p:.2f}, Holm {v3.mcnemar_holm_p:.2f})")
        self.assertStated(f"({f3(v3.macro_f1_difference, True)}, {f3(v3.macro_f1_ci_low)} to {f3(v3.macro_f1_ci_high)})")
        self.assertGreater(v3.macro_f1_ci_low, 0)  # condition for the "similar to supervised Mixed" clause
        adapted = d.drop(index="resnet18_lab", level="model").xs("field_dev", level="partition").macro_f1_difference
        self.assertEqual((round(adapted.min(), 2), round(adapted.max(), 2)), (-0.11, -0.09))
        raw = pd.read_csv(R / "adabn/adabn_results.csv")
        lab = raw[(raw.model == "resnet18_lab") & (raw.partition == "lab_test")].set_index("variant").macro_f1
        self.assertStated(f"lowers Lab Test macro-F1 from {lab['original']:.3f} to {lab['adabn_field_adapt_train']:.3f}")

    def test_readme_test_count(self):
        n = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests")).countTestCases()
        self.assertIn(f"{n} automated tests", README_ONLY)

    def test_boundary_sensitivity(self):
        c = pd.read_csv(R / "external_v3_label_merge/merged_comparisons.csv").set_index("id")
        p1 = c.loc["P1"]
        self.assertStated(f"P1 becomes {pp(p1.difference)} pp (95% CI {pp(p1.ci_low)} to {pp(p1.ci_high)}; "
                          f"McNemar p = {p1.p_value:.2f}, Holm {p1.holm_p:.2f})")


if __name__ == "__main__":
    unittest.main()
