"""Manifest validation must fail before remote datasets are opened."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from src.dataset import PlantDiseaseDataset


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "manifest.csv"
        self.row = dict(source="plantdoc", row_id=0, original_class="Tomato leaf",
                        class_name="Healthy", label_id=0)

    def test_invalid_ids_and_class_mapping_fail_before_source_loading(self):
        invalid = [("row_id", -1), ("row_id", 0.5), ("row_id", float("inf")),
                   ("row_id", "invalid"), ("label_id", 0.5), ("label_id", 5),
                   ("class_name", "Early Blight")]
        for column, value in invalid:
            with self.subTest(column=column, value=value):
                pd.DataFrame([{**self.row, column: value}]).to_csv(self.path, index=False)
                with patch("src.dataset.load_dataset") as load:
                    with self.assertRaises(ValueError):
                        PlantDiseaseDataset(self.path)
                    load.assert_not_called()

    def test_valid_row_preserves_rgb_loading_and_label(self):
        pd.DataFrame([self.row]).to_csv(self.path, index=False)
        source = [{"class_label": "Tomato leaf", "image": Image.new("L", (8, 8))}]
        with patch("src.dataset.load_dataset", return_value=source):
            dataset = PlantDiseaseDataset(self.path, transform=lambda image: image.mode)
        self.assertEqual(dataset[0], ("RGB", 0))


if __name__ == "__main__":
    unittest.main()
