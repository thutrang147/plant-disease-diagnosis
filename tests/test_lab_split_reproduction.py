"""The committed PlantVillage splits must follow from ``src/data_prep.split_lab``.

Rebuilds the pre-split metadata from the three committed Lab manifests (which
keep the original row order fields and leaf metadata) and re-runs the
StratifiedGroupKFold selection with the recorded seed. No images are needed.
"""
import json
from pathlib import Path
import unittest

import pandas as pd

from src.data_prep import SEED, split_lab

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"
PARTITIONS = ("lab_train", "lab_val", "lab_test")
METADATA_COLUMNS = ["source", "row_id", "original_class", "original_split",
                    "leaf_id", "leaf_grouped", "class_name", "label_id"]


class LabSplitReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.committed = {name: pd.read_csv(SPLITS / f"{name}.csv") for name in PARTITIONS}
        # build_metadata yields rows in Hugging Face row order; manifests are saved sorted by row_id.
        cls.metadata = (pd.concat(cls.committed.values(), ignore_index=True)[METADATA_COLUMNS]
                        .sort_values("row_id").reset_index(drop=True))
        cls.reproduced = dict(zip(PARTITIONS, split_lab(cls.metadata)))

    def test_recorded_seed_is_the_one_used(self):
        sources = json.loads((SPLITS / "dataset_sources.json").read_text())
        self.assertEqual(sources["seed"], SEED)

    def test_original_test_is_preserved_and_train_is_group_split(self):
        self.assertTrue(self.committed["lab_test"].original_split.eq("test").all())
        for name in ("lab_train", "lab_val"):
            self.assertTrue(self.committed[name].original_split.eq("train").all())
        self.assertTrue(set(self.committed["lab_train"].group_id)
                        .isdisjoint(self.committed["lab_val"].group_id))

    def test_all_lab_manifests_are_reproduced_exactly(self):
        for name in PARTITIONS:
            with self.subTest(partition=name):
                expected = self.committed[name]
                actual = self.reproduced[name].sort_values("row_id").reset_index(drop=True)
                pd.testing.assert_frame_equal(actual[expected.columns], expected)


if __name__ == "__main__":
    unittest.main()
