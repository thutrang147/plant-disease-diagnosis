"""Checks for the External v3 manifest and round-2 analysis helpers (no downloads)."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/external_v3_plantseg"


def load_script(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExternalV3ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = pd.read_csv(V3 / "manifest.csv")
        cls.report = json.loads((V3 / "build_report.json").read_text())
        cls.included = cls.manifest[cls.manifest.status.eq("included")]

    def test_counts_match_report(self):
        counts = self.report["counts"]
        self.assertEqual(len(self.manifest), counts["candidates"])
        self.assertEqual(len(self.included), counts["included"])
        self.assertEqual(self.included.class_name.value_counts().to_dict(), counts["included_per_class"])

    def test_included_rows_pass_screen(self):
        self.assertTrue(self.included.min_reference_distance.gt(10).all())
        self.assertFalse(self.included.plantseg_row.duplicated().any())
        self.assertFalse(self.included.exact_hash.duplicated().any())

    def test_labels_use_canonical_mapping(self):
        mapping = json.loads((ROOT / "data/splits/label_map.json").read_text())
        self.assertTrue(self.manifest.class_name.map(mapping).eq(self.manifest.label_id).all())
        self.assertEqual(set(self.manifest.label_id), {1, 2, 3, 4})

    def test_protocol_names_models_and_comparisons(self):
        protocol = json.loads((V3 / "protocol.json").read_text())
        self.assertEqual([c["id"] for c in protocol["primary_comparisons"]], ["P1", "P2", "P3"])
        for key in ("resnet18_lab", "vit_small_lab", "resnet18_geometric_mixed", "vit_small_mixed"):
            self.assertEqual(len(protocol["models"][key]["checkpoint_sha256"]), 64)


class HelperTest(unittest.TestCase):
    def test_hamming_and_groups(self):
        build = load_script("build_external_v3_plantseg")
        refs = np.asarray([0b0, 0b1111, 0b1], dtype=np.uint64)
        self.assertEqual(build.hamming_to_all(0b1, refs).tolist(), [1, 3, 0])
        roots = build.internal_groups([0b0, 0b1, 0b11111111_00000000, 0b11], threshold=1)
        self.assertEqual(roots[0], roots[1])
        self.assertEqual(roots[1], roots[3])
        self.assertNotEqual(roots[0], roots[2])

    def test_interpolation_endpoints_and_buffers(self):
        interp = load_script("evaluate_weight_interpolation")
        theta0 = {"w": torch.zeros(3), "n": torch.tensor(5)}
        theta1 = {"w": torch.ones(3), "n": torch.tensor(9)}
        self.assertTrue(torch.equal(interp.interpolate(theta0, theta1, 0.0)["w"], theta0["w"]))
        self.assertTrue(torch.equal(interp.interpolate(theta0, theta1, 1.0)["w"], theta1["w"]))
        self.assertTrue(torch.allclose(interp.interpolate(theta0, theta1, 0.3)["w"], torch.full((3,), 0.3)))
        self.assertEqual(int(interp.interpolate(theta0, theta1, 0.7)["n"]), 5)

    def test_paired_statistics(self):
        ev = load_script("evaluate_external_v3_plantseg")
        a = np.asarray([1, 0, 1, 0, 1, 1], dtype=float)
        self.assertEqual(ev.paired_bootstrap(a, a.copy()), (0.0, 0.0))
        self.assertEqual(ev.sign_flip_p(a, a.copy()), 1.0)
        b = np.ones(40)
        self.assertLess(ev.sign_flip_p(np.zeros(40), b), 0.001)


if __name__ == "__main__":
    unittest.main()
