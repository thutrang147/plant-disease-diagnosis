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


class ReproducibilityTest(unittest.TestCase):
    def test_checkpoint_registry(self):
        registry = json.loads((ROOT / "data/checkpoints.json").read_text())
        entries = registry["checkpoints"]
        keys = [e["key"] for e in entries]
        self.assertEqual(len(keys), len(set(keys)))
        for entry in entries:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(entry["hash_source"], registry["hash_sources"])
            self.assertIn(entry["group"], {"core", "historical"})
        by_key = {e["key"]: e["sha256"] for e in entries}
        protocol = json.loads((V3 / "protocol.json").read_text())
        for key in ("resnet18_lab", "vit_small_lab", "resnet18_geometric_mixed", "vit_small_mixed"):
            self.assertEqual(by_key[key], protocol["models"][key]["checkpoint_sha256"])
        seal = json.loads((ROOT / "data/results/notebook15/original_evaluation_seal.json").read_text())
        for seed, files in seal["seed_checkpoints"].items():
            for name, digest in files.items():
                self.assertEqual(by_key[f"nb15_{name.replace('_selected.pt', '')}_s{seed}"], digest)

    def test_dataset_revisions_are_pinned(self):
        from src.dataset import DATASET_SOURCES
        self.assertRegex(DATASET_SOURCES["plantvillage"][1], r"^[0-9a-f]{40}$")
        self.assertEqual(DATASET_SOURCES["plantdoc"][1], "v0.1.0")

    def test_compare_reproduction(self):
        compare = load_script("compare_reproduction")
        import tempfile
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            frame = pd.DataFrame({"model": ["x", "y"], "correct": [1, 2], "accuracy": [0.5, 0.25]})
            frame.to_csv(Path(a) / "m.csv", index=False)
            frame.assign(accuracy=[0.5, 0.25 + 1e-6]).to_csv(Path(b) / "m.csv", index=False)
            self.assertEqual(compare.compare(Path(a), Path(b), atol=1e-4), [])
            frame.assign(correct=[1, 3]).to_csv(Path(b) / "m.csv", index=False)
            self.assertEqual(len(compare.compare(Path(a), Path(b), atol=1e-4)), 1)

    def test_reproduce_script_runs_without_pytest(self):
        # B1: requirements.txt has no pytest, so the reproduce script must use unittest.
        script = (ROOT / "scripts/reproduce_round2.sh").read_text()
        self.assertIn("python -m unittest discover -s tests", script)
        self.assertNotIn("pytest", script)
        self.assertNotIn("pytest", (ROOT / "requirements.txt").read_text())

    def test_compare_reproduction_rejects_empty_committed_dir(self):
        compare = load_script("compare_reproduction")
        import tempfile
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(len(compare.compare(Path(a), Path(b), atol=1e-4)), 1)
            # Two-row headers (pandas multi-index summaries) are compared cell by cell.
            Path(a, "s.csv").write_text(",accuracy,accuracy\n,mean,std\nx,0.5,0.01\n")
            Path(b, "s.csv").write_text(",accuracy,accuracy\n,mean,std\nx,0.50001,0.01\n")
            self.assertEqual(compare.compare(Path(a), Path(b), atol=1e-4), [])


if __name__ == "__main__":
    unittest.main()
