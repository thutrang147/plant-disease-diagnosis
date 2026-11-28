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
README = (ROOT / "README.md").read_text()


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

    def test_p3_eight_epochs(self):
        c = pd.read_csv(R / "external_v3_plantseg/comparisons.csv")
        p3 = c[(c.family == "all") & (c.id == "P3")].iloc[0]
        self.assertStated(f"{pp(p3.difference)} pp (95% CI {pp(p3.ci_low)} to {pp(p3.ci_high)}; seed-level p = 0.12)")
        seed = pd.read_csv(R / "external_v3_secondary_metrics/p3_seed_level.csv").set_index("metric").loc["accuracy"]
        self.assertEqual(round(seed.seed_level_t_test_p, 2), 0.12)
        diffs = sorted(round(100 * seed[f"seed_{s}"], 1) for s in (42, 43, 44))
        self.assertEqual(diffs, [0.6, 1.6, 2.9])
        self.assertStated("(+0.6 to +2.9 pp)")

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

    def test_boundary_sensitivity(self):
        c = pd.read_csv(R / "external_v3_label_merge/merged_comparisons.csv").set_index("id")
        p1 = c.loc["P1"]
        self.assertStated(f"P1 becomes {pp(p1.difference)} pp (95% CI {pp(p1.ci_low)} to {pp(p1.ci_high)}; "
                          f"McNemar p = {p1.p_value:.2f}, Holm {p1.holm_p:.2f})")


if __name__ == "__main__":
    unittest.main()
