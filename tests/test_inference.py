"""Tests for src/inference.py using a randomly initialized checkpoint (no downloads)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from src.inference import CLASS_NAMES, grad_cam, load_checkpoint, overlay, predict, prepare


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "resnet.pt"
        torch.save({"architecture": "resnet18", "class_names": CLASS_NAMES,
                    "model_state_dict": model.state_dict()}, cls.path)
        cls.image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_predict_returns_distribution(self):
        model, transform, meta = load_checkpoint(self.path)
        probs = predict(model, transform, prepare(self.image))
        self.assertEqual(probs.shape, (5,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(meta["architecture"], "resnet18")

    def test_grad_cam_and_overlay(self):
        model, transform, _ = load_checkpoint(self.path)
        cam = grad_cam(model, transform, self.image, target=2)
        self.assertEqual(cam.shape, (224, 224))
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0)
        self.assertEqual(overlay(self.image, cam).size, (224, 224))

    def test_rejects_wrong_class_order(self):
        state = torch.load(self.path, weights_only=True)
        state["class_names"] = list(reversed(CLASS_NAMES))
        bad = Path(self.tmp.name) / "bad.pt"
        torch.save(state, bad)
        with self.assertRaises(ValueError):
            load_checkpoint(bad)

    def test_model_view_matches_each_transform(self):
        import timm
        from src.inference import model_view
        vit = timm.data.create_transform(input_size=(3, 224, 224), interpolation="bicubic", mean=(0.5,) * 3,
                                         std=(0.5,) * 3, crop_pct=0.9, crop_mode="center", is_training=False)
        self.assertEqual(model_view(self.image).size, (224, 224))
        view = model_view(self.image, vit)
        self.assertEqual(view.size, (224, 224))
        # Re-normalizing the recovered view reproduces the model input up to 8-bit rounding.
        x = vit(self.image)
        y = (torch.from_numpy(np.asarray(view)).permute(2, 0, 1).float() / 255 - 0.5) / 0.5
        self.assertLess(float((x - y).abs().max()), 0.02)


if __name__ == "__main__":
    unittest.main()
