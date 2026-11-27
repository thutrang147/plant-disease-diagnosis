"""The committed verified PlantDoc split must follow from its own recorded protocol.

Re-runs the grouping, quarantine and split-selection code of
``src/prepare_field_dev_verified.py`` on the image fingerprints already stored in
the committed manifests, so no images or downloads are needed.

Field Test pHashes are not committed. This does not weaken the eligible-row check:
a group containing a Field Test image is quarantined, so every eligible group is
the same with or without Field Test rows. The near-Field-Test quarantine is
checked through the committed train/test candidate pairs instead of recomputed.
"""
import json
from pathlib import Path
import unittest

import pandas as pd

from src.prepare_field_dev_verified import build_groups, choose_split

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def pair_key(frame):
    return sorted(zip(frame.row_id_a, frame.row_id_b, frame.phash_distance))


class VerifiedSplitReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((SPLITS / "field_dev_protocol.json").read_text())
        cls.train = pd.read_csv(SPLITS / "field_train.csv")
        cls.adapt = pd.read_csv(SPLITS / "field_adapt_train.csv")
        cls.dev = pd.read_csv(SPLITS / "field_dev.csv")
        cls.quarantine = pd.read_csv(SPLITS / "field_quarantined_train.csv")
        cls.conflicts = pd.read_csv(SPLITS / "field_label_conflicts.csv")
        cls.candidates = pd.read_csv(SPLITS / "field_near_duplicate_candidates.csv")

        stored = pd.concat([cls.adapt, cls.dev, cls.quarantine], ignore_index=True)
        cls.stored_groups = stored.set_index("row_id").group_id
        fingerprints = stored[["row_id", "exact_hash", "phash"]]
        # Preserve the original field_train order, as prepare() does.
        rows = cls.train.merge(fingerprints, on="row_id", how="left", validate="1:1")
        rows["original_partition"] = "field_train"
        cls.grouped, cls.pairs = build_groups(rows, cls.protocol["phash_threshold"])

        group_of = cls.grouped.set_index("row_id").group_id
        train_ids = set(cls.train.row_id)
        with_test = cls.candidates[cls.candidates[["partition_a", "partition_b"]]
                                   .eq("field_test").any(axis=1)]
        linked = (set(with_test.row_id_a) | set(with_test.row_id_b)) & train_ids
        cls.near_test = cls.grouped.group_id.isin(set(group_of[sorted(linked)]))
        classes = cls.grouped.groupby("group_id").label_id.nunique()
        cls.conflict_groups = set(classes[classes > 1].index)
        cls.conflict = cls.grouped.group_id.isin(cls.conflict_groups)

    def test_manifests_store_one_fingerprint_per_original_training_row(self):
        stored = pd.concat([self.adapt, self.dev, self.quarantine])
        self.assertTrue(stored.row_id.is_unique)
        self.assertEqual(set(stored.row_id), set(self.train.row_id))
        self.assertFalse(stored[["exact_hash", "phash"]].isna().any().any())

    def test_training_near_duplicate_pairs_are_reproduced(self):
        committed = self.candidates[self.candidates.partition_a.eq("field_train")
                                    & self.candidates.partition_b.eq("field_train")]
        self.assertEqual(pair_key(self.pairs), pair_key(committed))
        self.assertTrue(self.pairs.phash_distance.le(self.protocol["phash_threshold"]).all())

    def test_quarantine_matches_protocol_and_reasons(self):
        quarantined = self.grouped[self.near_test | self.conflict]
        self.assertEqual(set(quarantined.row_id), set(self.quarantine.row_id))
        self.assertEqual(int(self.near_test.sum()), self.protocol["quarantined_near_test_rows"])
        self.assertEqual(int(self.conflict.sum()), self.protocol["quarantined_label_conflict_rows"])
        self.assertEqual(len(self.conflict_groups), self.protocol["conflicting_groups"])
        self.assertEqual(set(self.grouped[self.conflict].row_id), set(self.conflicts.row_id))

        reasons = {
            int(row_id): ("near_field_test_and_label_conflict" if near and conflict
                          else "near_field_test" if near else "label_conflict")
            for row_id, near, conflict in zip(self.grouped.row_id, self.near_test, self.conflict)
            if near or conflict
        }
        self.assertEqual(reasons, dict(zip(self.quarantine.row_id, self.quarantine.quarantine_reason)))

    def test_adapt_and_dev_manifests_are_reproduced_exactly(self):
        eligible = self.grouped[~(self.near_test | self.conflict)].reset_index(drop=True)
        self.assertTrue(eligible.group_id.eq(self.stored_groups[eligible.row_id].to_numpy()).all())
        adapt, dev = choose_split(eligible, self.protocol["dev_fraction_requested"],
                                  self.protocol["seed"], self.protocol["split_candidates"])
        columns = list(self.adapt.columns)
        pd.testing.assert_frame_equal(adapt[columns].reset_index(drop=True), self.adapt)
        pd.testing.assert_frame_equal(dev[columns].reset_index(drop=True), self.dev)
        self.assertEqual((len(adapt), len(dev)), (self.protocol["adaptation_train_images"],
                                                  self.protocol["development_images"]))


if __name__ == "__main__":
    unittest.main()
