"""Independent arithmetic and read-only export checks; no models or inference."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.controlled_field_report import load_evidence, verify_prediction_exports

ROOT = Path(__file__).resolve().parents[1]


class ReportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load_evidence(ROOT)

    def test_all_six_matrices_match_independent_sklearn_calculation(self):
        for (seed, condition), matrix in self.evidence['matrices'].items():
            true, predicted = [], []
            for i in range(5):
                for j in range(5):
                    true.extend([i] * int(matrix[i, j]))
                    predicted.extend([j] * int(matrix[i, j]))
            row = self.evidence['summary'].query('seed == @seed and condition == @condition').iloc[0]
            self.assertEqual(len(true), 47)
            self.assertAlmostEqual(row.accuracy, accuracy_score(true, predicted), places=14)
            self.assertAlmostEqual(row.macro_f1, f1_score(true, predicted, labels=range(5),
                                                        average='macro', zero_division=0), places=14)
            expected = precision_recall_fscore_support(true, predicted, labels=range(5), zero_division=0)
            classes = self.evidence['per_class'].query('seed == @seed and condition == @condition')
            for metric, values in zip(('precision', 'recall', 'f1', 'support'), expected):
                np.testing.assert_allclose(classes[metric].to_numpy(), values, atol=1e-14)

    def test_missing_exports_are_explicit_without_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'absent'
            self.assertFalse(verify_prediction_exports(ROOT, run, self.evidence)['available'])
            self.assertFalse(run.exists())

    def test_prediction_identity_check_rejects_duplicate_rows(self):
        # Synthetic fixtures allocate canonical labels to known count cells; they
        # are not recovered original predictions and are never retained as evidence.
        canonical = pd.read_csv(ROOT / 'data/splits/field_test.csv')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for (seed, condition), matrix in self.evidence['matrices'].items():
                frame = canonical.copy()
                frame['predicted_label_id'] = -1
                for label in range(5):
                    indices = frame.index[frame.label_id.eq(label)]
                    frame.loc[indices, 'predicted_label_id'] = np.repeat(np.arange(5), matrix[label])
                frame['seed'], frame['condition'] = seed, condition
                target = run / f'seed_{seed}'
                target.mkdir(exist_ok=True)
                frame.sample(frac=1, random_state=7).to_csv(
                    target / f'predictions_{condition}_field_test.csv', index=False)
            files = sorted(run.rglob('*.csv'))
            before = {p: p.read_bytes() for p in files}
            self.assertTrue(verify_prediction_exports(ROOT, run, self.evidence)['available'])
            self.assertEqual(before, {p: p.read_bytes() for p in files})
            path = run / 'seed_42/predictions_field_only_field_test.csv'
            broken = pd.read_csv(path)
            broken.loc[1] = broken.loc[0]
            broken.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                verify_prediction_exports(ROOT, run, self.evidence)


if __name__ == '__main__':
    unittest.main()
