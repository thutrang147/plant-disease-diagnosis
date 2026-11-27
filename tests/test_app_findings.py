"""The demo's findings must match committed evidence and the README.

Also covers the presentation helpers: escaping and the consensus sentence.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import components  # noqa: E402
import findings  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
README_NAMES = {
    "ResNet18 · Lab baseline": "ResNet18 Baseline",
    "ViT-Small · Lab baseline": "ViT-Small Baseline",
    "ResNet18 · Geometric + Mixed": "ResNet18 Geometric + Mixed",
    "ViT-Small · Mixed": "ViT-Small Mixed",
}


def readme_row(name):
    rows = [line for line in README.splitlines() if line.startswith(f"| {name} ") and line.count("|") == 5]
    if len(rows) != 1:
        raise AssertionError(f"Expected one Main Results row for {name}, found {len(rows)}")
    return rows[0]


class FindingsDataTests(unittest.TestCase):
    def test_domain_gap_matches_readme_main_results(self):
        gap = findings.domain_gap()
        self.assertEqual(len(gap), 12)
        for model, name in README_NAMES.items():
            row = readme_row(name)
            values = gap[gap.model.eq(model)].set_index("evaluation").macro_f1
            with self.subTest(model=model):
                for evaluation in findings.EVALUATIONS:
                    self.assertIn(f"{values[evaluation]:.4f}", row)

    def test_class_tradeoff_matches_readme_counts(self):
        table = findings.class_tradeoff().set_index(["class_name", "training"])
        expected = {("Healthy", "Lab only"): (0, 16), ("Healthy", "Field-adapted"): (12, 16),
                    ("Early Blight", "Lab only"): (11, 13), ("Early Blight", "Field-adapted"): (1, 13),
                    ("Bacterial Spot", "Lab only"): (1, 9), ("Bacterial Spot", "Field-adapted"): (5, 9)}
        for key, (correct, images) in expected.items():
            with self.subTest(key=key):
                self.assertEqual((table.loc[key, "correct"], table.loc[key, "images"]), (correct, images))
                self.assertIn(f"{correct}/{images}", README)
        self.assertEqual(int(table.images.sum()), 2 * 62)

    def test_controlled_runs_match_readme_summary(self):
        runs = findings.controlled_runs().set_index(["row", "condition"])
        self.assertEqual(round(runs.loc[("Mean of 3", "Field-only"), "macro_f1"], 4), 0.4300)
        self.assertEqual(round(runs.loc[("Mean of 3", "Mixed"), "macro_f1"], 4), 0.4072)
        self.assertEqual(round(runs.loc[("Mean of 3", "Field-only"), "accuracy"], 4), 0.4610)
        self.assertEqual(round(runs.loc[("Mean of 3", "Mixed"), "accuracy"], 4), 0.4468)
        wins = sum(runs.loc[(f"Seed {s}", "Field-only"), "macro_f1"] > runs.loc[(f"Seed {s}", "Mixed"), "macro_f1"]
                   for s in (42, 43, 44))
        self.assertEqual(wins, 2)  # "Field-only scores higher in two of three seeds"

    def test_charts_compile_to_vega_lite(self):
        gap = findings.domain_gap()
        charts = [findings.domain_gap_chart(gap, arch) for arch in findings.ARCHITECTURES]
        charts += [findings.tradeoff_chart(findings.class_tradeoff()),
                   findings.controlled_chart(findings.controlled_runs())]
        for chart in charts:
            spec = chart.to_dict()
            self.assertIn("$schema", spec)


class ComponentTests(unittest.TestCase):
    def test_slip_escapes_text_and_validates_colour(self):
        body = components.slip_html(model="<b>x</b>", color="#257de0", determination="A & B",
                                    confidence=0.5, meta=['"quoted"'])
        self.assertNotIn("<b>x</b>", body)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", body)
        self.assertIn("A &amp; B", body)
        self.assertIn("width:50.0%", body)
        with self.assertRaises(ValueError):
            components.slip_html(model="m", color="red;}", determination="d", confidence=0.1, meta=[])
        self.assertIn("&lt;", components.error_slip_html("<m>", "<err>"))

    def test_consensus_sentences(self):
        self.assertEqual(components.consensus([]), "")
        self.assertEqual(components.consensus([("Lab only", "Healthy")]), "One checkpoint: Healthy.")
        self.assertEqual(components.consensus([("Lab only", "Healthy"), (None, "Healthy")]),
                         "All 2 checkpoints agree: Healthy.")
        split = components.consensus([("Lab only", "Early Blight"), ("Lab only", "Early Blight"),
                                      ("Field-adapted", "Healthy"), ("Field-adapted", "Healthy")])
        self.assertTrue(split.startswith("Lab-only checkpoints say Early Blight; "
                                         "field-adapted checkpoints say Healthy."))
        # A tie inside a regime gives no regime verdict, only the counts.
        tie = components.consensus([("Lab only", "Early Blight"), ("Lab only", "Healthy"),
                                    ("Field-adapted", "Healthy")])
        self.assertEqual(tie, "The checkpoints disagree: Healthy (2), Early Blight (1).")


if __name__ == "__main__":
    unittest.main()
