"""Regression checks for the optional folder evaluator; no model inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch

from src.evaluate_confirmatory_field import scan_and_audit_self_collected, evaluate_resnet_checkpoint
from src.prepare_field_dev_verified import exact_hash


class ExternalAuditTests(unittest.TestCase):
    def test_exact_audit_compares_pixels_instead_of_encoded_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            splits.mkdir()
            (splits / "label_map.json").write_text(json.dumps({"Healthy": 0}))
            photos = root / "photos/Healthy"
            photos.mkdir(parents=True)
            image = Image.new("RGB", (19, 17), (31, 70, 10))
            photo = photos / "leaf.png"
            image.save(photo)
            pixel_hash = exact_hash(image)
            self.assertNotEqual(pixel_hash, hashlib.sha256(photo.read_bytes()).hexdigest())
            # Deliberately different pHash isolates the exact-pixel comparison.
            reference = pd.DataFrame([dict(row_id=1, exact_hash=pixel_hash, phash="ffffffffffffffff")])
            for name in ("field_adapt_train", "field_dev"):
                reference.to_csv(splits / f"{name}.csv", index=False)
            frame, audit = scan_and_audit_self_collected(root / "photos", splits, phash_threshold=0)
            self.assertTrue(frame.iloc[0].contaminated)
            self.assertTrue(audit["contaminated_examples"][0]["exact_match"])
            self.assertFalse(audit["independence_established"])

    def test_missing_audit_metadata_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "label_map.json").write_text(json.dumps({"Healthy": 0}))
            with self.assertRaises(FileNotFoundError):
                scan_and_audit_self_collected(root, root)

    def test_incompatible_checkpoint_is_rejected_before_model_creation(self):
        with patch("src.evaluate_confirmatory_field.torch.load", return_value={"architecture": "vit"}), \
             patch("src.evaluate_confirmatory_field.resnet18") as model:
            with self.assertRaises(ValueError):
                evaluate_resnet_checkpoint(Path("unused.pt"), pd.DataFrame(), ["Healthy"],
                                           Path("unused"), torch.device("cpu"))
            model.assert_not_called()
