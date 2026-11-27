"""Checks for the supplementary statistics computed from committed exports."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import supplementary_analysis as sa


class HelperTests(unittest.TestCase):
    def test_holm_matches_hand_calculation(self):
        adjusted = sa.holm([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.20])

    def test_macro_f1_counts_absent_predictions_as_zero(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 0, 0, 0])
        # class 0: tp=2, fp=2 -> F1=2/3; class 1: F1=0
        self.assertAlmostEqual(sa.macro_f1(y, p, [0, 1]), (2 / 3) / 2)

    def test_mcnemar_symmetric_and_bounded(self):
        self.assertEqual(sa.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(sa.mcnemar_exact(24, 10), sa.mcnemar_exact(10, 24))
        self.assertAlmostEqual(round(sa.mcnemar_exact(24, 10), 6), 0.024307)


class EvidenceTests(unittest.TestCase):
    def test_disease_stratum_reproduces_notebook12_counts(self):
        strata = sa.external_by_stratum(sa.load_external())
        disease = strata[strata.stratum == "online_disease_46"].set_index("model").correct.to_dict()
        self.assertEqual(disease, {
            "ResNet18 · Lab baseline": 19,
            "ViT-Small · Lab baseline": 22,
            "ResNet18 · Geometric + Mixed": 21,
            "ViT-Small · Mixed": 19,
        })

    def test_notebook15_per_seed_macro_f1_matches_original_summary(self):
        per_seed, _ = sa.notebook15_paired(n_boot=20)
        self.assertEqual(per_seed.test_field_only_macro_f1.tolist(), [0.4338, 0.4214, 0.4349])
        self.assertEqual(per_seed.test_mixed_macro_f1.tolist(), [0.3970, 0.3840, 0.4405])
        self.assertEqual(per_seed.selected_epoch.tolist(), [8, 7, 8])

    def test_existing_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["supplementary_analysis", "--output", directory]):
                with self.assertRaises(FileExistsError):
                    sa.main()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
