"""Single-image inference must match the research preprocessing and aggregation.

Checkpoints are randomly initialized and written to a temporary directory, so no
pretrained weights, datasets or private research checkpoints are needed.
"""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src import inference
from src.evaluate_resolution_sensitivity import CLASS_NAMES, FIVE_BOXES


def resnet_state():
    torch.manual_seed(0)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model.state_dict()


def vit_state():
    torch.manual_seed(0)
    return timm.create_model(inference.VIT_ARCHITECTURE, pretrained=False, num_classes=5).state_dict()


def leaf_image(size=(300, 200)):
    # A deterministic non-uniform RGB image so crops differ from each other.
    image = Image.new("RGB", size)
    image.putdata([((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
                   for y in range(size[1]) for x in range(size[0])])
    return image


class InferenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.resnet_path = cls.root / "resnet18_lab_best.pt"
        torch.save({"model_state_dict": resnet_state(), "class_names": CLASS_NAMES,
                    "architecture": "resnet18", "pretrained_weights": "IMAGENET1K_V1",
                    "best_val_macro_f1": 0.97}, cls.resnet_path)
        # Mixed-domain checkpoints (Notebook 08) omit pretraining and preprocessing.
        cls.vit_path = cls.root / "vit_small_mixed_best.pt"
        torch.save({"model_state_dict": vit_state(), "class_names": CLASS_NAMES,
                    "architecture": inference.VIT_ARCHITECTURE,
                    "adaptation_type": "balanced_mixed_domain"}, cls.vit_path)
        cls.resnet = inference.load_checkpoint(cls.resnet_path)
        cls.vit = inference.load_checkpoint(cls.vit_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def save(self, name, **payload):
        path = self.root / name
        torch.save(payload, path)
        return path


class LoadingTests(InferenceTestCase):
    def test_resnet_uses_torchvision_evaluation_transform(self):
        self.assertEqual(self.resnet.architecture, "resnet18")
        self.assertEqual(self.resnet.preprocessing_source, inference.RESNET_PREPROCESSING_SOURCE)
        image = leaf_image()
        expected = ResNet18_Weights.IMAGENET1K_V1.transforms()(image)
        self.assertTrue(torch.equal(self.resnet.standard_transform(image), expected))
        self.assertFalse(self.resnet.model.training)
        self.assertEqual(self.resnet.metadata["best_val_macro_f1"], 0.97)

    def test_vit_without_saved_config_uses_recorded_source_preprocessing(self):
        self.assertIn("recorded", self.vit.preprocessing_source)
        expected = timm.data.create_transform(**inference.VIT_RECORDED_PREPROCESSING, is_training=False)
        image = leaf_image()
        self.assertTrue(torch.equal(self.vit.standard_transform(image), expected(image)))

    def test_recorded_vit_config_matches_timm_default_for_the_architecture(self):
        model = timm.create_model(inference.VIT_ARCHITECTURE, pretrained=False, num_classes=5)
        resolved = timm.data.resolve_model_data_config(model)
        normalized = {key: list(value) if isinstance(value, tuple) else value
                      for key, value in resolved.items()}
        self.assertEqual(normalized, inference.VIT_RECORDED_PREPROCESSING)

    def test_vit_saved_config_takes_precedence(self):
        config = dict(inference.VIT_RECORDED_PREPROCESSING, crop_pct=1.0)
        path = self.save("vit_with_config.pt", model_state_dict=vit_state(), class_names=CLASS_NAMES,
                         architecture=inference.VIT_ARCHITECTURE, pretrained_weights="augreg_in1k",
                         preprocessing_config=config)
        loaded = inference.load_checkpoint(path)
        self.assertEqual(loaded.preprocessing_source, "checkpoint preprocessing_config")
        expected = timm.data.create_transform(**config, is_training=False)
        self.assertTrue(torch.equal(loaded.standard_transform(leaf_image()), expected(leaf_image())))

    def test_invalid_checkpoints_are_rejected(self):
        state = resnet_state()
        cases = {
            "class_order.pt": dict(model_state_dict=state, class_names=CLASS_NAMES[::-1], architecture="resnet18"),
            "architecture.pt": dict(model_state_dict=state, class_names=CLASS_NAMES, architecture="vit_b_16"),
            "pretraining.pt": dict(model_state_dict=state, class_names=CLASS_NAMES, architecture="resnet18",
                                   pretrained_weights="IMAGENET1K_V2"),
            "weights.pt": dict(model_state_dict=state, class_names=CLASS_NAMES,
                               architecture=inference.VIT_ARCHITECTURE),
            "no_weights.pt": dict(class_names=CLASS_NAMES, architecture="resnet18"),
            "vit_size.pt": dict(model_state_dict=vit_state(), class_names=CLASS_NAMES,
                                architecture=inference.VIT_ARCHITECTURE,
                                preprocessing_config=dict(inference.VIT_RECORDED_PREPROCESSING,
                                                          input_size=[3, 384, 384])),
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    inference.load_checkpoint(self.save(name, **payload))
        bare = self.root / "bare_state_dict.pt"
        torch.save(state, bare)
        corrupt = self.root / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")
        for path in (bare, corrupt):
            with self.subTest(name=path.name):
                with self.assertRaises(ValueError):
                    inference.load_checkpoint(path)

    def test_discover_checkpoints_lists_only_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("b.pt", "a.pth", "notes.txt"):
                (folder / name).write_bytes(b"")
            (folder / "nested.pt").mkdir()
            self.assertEqual([p.name for p in inference.discover_checkpoints(folder)], ["a.pth", "b.pt"])
        self.assertEqual(inference.discover_checkpoints(self.root / "missing"), [])


class PredictionTests(InferenceTestCase):
    def test_standard_prediction_matches_direct_forward_pass(self):
        image = leaf_image()
        for loaded in (self.resnet, self.vit):
            with self.subTest(architecture=loaded.architecture):
                prediction = inference.predict_image(loaded, image)
                with torch.inference_mode():
                    expected = loaded.model(loaded.standard_transform(image).unsqueeze(0)).softmax(-1)[0]
                values = torch.tensor([prediction.probabilities[name] for name in CLASS_NAMES])
                self.assertTrue(torch.allclose(values, expected, atol=1e-6))
                self.assertEqual(prediction.view_count, 1)
                self.assertEqual(prediction.label_id, int(expected.argmax()))
                self.assertEqual(prediction.class_name, CLASS_NAMES[prediction.label_id])
                self.assertAlmostEqual(prediction.confidence, float(expected.max()), places=6)

    def test_five_crop_averages_softmax_over_notebook14_views(self):
        image = leaf_image()
        prediction = inference.predict_image(self.vit, image, "five_crop")
        canvas = image.resize((448, 448), Image.Resampling.BICUBIC)
        views = torch.stack([self.vit.view_normalize(canvas.crop(box)) for box in FIVE_BOXES])
        with torch.inference_mode():
            expected = self.vit.model(views).softmax(-1).mean(0)
        values = torch.tensor([prediction.probabilities[name] for name in CLASS_NAMES])
        self.assertTrue(torch.allclose(values, expected, atol=1e-6))
        self.assertEqual(prediction.view_count, 5)
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=5)

    def test_non_rgb_input_and_unknown_strategy(self):
        grey = leaf_image().convert("L")
        self.assertEqual(inference.predict_image(self.resnet, grey).view_count, 1)
        with self.assertRaises(ValueError):
            inference.predict_image(self.resnet, leaf_image(), "ten_crop")

    def test_load_image_applies_exif_orientation(self):
        image = leaf_image((40, 20))
        exif = image.getexif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        buffer.seek(0)
        loaded = inference.load_image(buffer)
        self.assertEqual((loaded.mode, loaded.size), ("RGB", (20, 40)))

    def test_cli_prints_json_predictions(self):
        image_path = self.root / "leaf.png"
        leaf_image().save(image_path)
        from contextlib import redirect_stdout
        from io import StringIO
        import json
        output = StringIO()
        with redirect_stdout(output):
            inference.main(["--checkpoint", str(self.resnet_path), "--strategy", "five_crop",
                            "--json", str(image_path)])
        record = json.loads(output.getvalue())
        self.assertEqual(record["image"], str(image_path))
        self.assertEqual(record["view_count"], 5)
        self.assertIn(record["class_name"], CLASS_NAMES)


class ExplanationTests(InferenceTestCase):
    def check_map(self, explanation, loaded, image):
        self.assertEqual(explanation.heatmap.shape, (224, 224))
        self.assertEqual(explanation.model_input.shape, (224, 224, 3))
        for array in (explanation.heatmap, explanation.model_input, explanation.overlay()):
            self.assertGreaterEqual(float(array.min()), 0.0)
            self.assertLessEqual(float(array.max()), 1.0)
        self.assertAlmostEqual(float(explanation.heatmap.max()), 1.0, places=5)
        self.assertEqual(explanation.label_id, inference.predict_image(loaded, image).label_id)

    def test_resnet_gradcam_matches_autograd_reference(self):
        image = leaf_image()
        explanation = inference.explain_image(self.resnet, image)
        self.check_map(explanation, self.resnet, image)
        self.assertIn("Grad-CAM", explanation.method)

        model, tensor, captured = self.resnet.model, self.resnet.standard_transform(image).unsqueeze(0), {}
        handle = model.layer4[-1].register_forward_hook(lambda _, __, out: captured.update(out=out))
        try:
            logits = model(tensor.requires_grad_(True))
            (gradient,) = torch.autograd.grad(logits[0, explanation.label_id], captured["out"])
        finally:
            handle.remove()
        cam = torch.relu((captured["out"].detach() * gradient.mean(dim=(2, 3), keepdim=True)).sum(1, keepdim=True))
        expected = inference._scale(cam, (224, 224))
        self.assertTrue(np.allclose(explanation.heatmap, expected, atol=1e-5))
        # The displayed input is the de-normalized ResNet preprocessing of the image.
        view = ResNet18_Weights.IMAGENET1K_V1.transforms()(image).permute(1, 2, 0).numpy()
        expected_input = np.clip(view * np.asarray(self.resnet.std) + np.asarray(self.resnet.mean), 0, 1)
        self.assertTrue(np.allclose(explanation.model_input, expected_input, atol=1e-6))

    def test_vit_attention_matches_timm_unfused_attention(self):
        image = leaf_image()
        explanation = inference.explain_image(self.vit, image)
        self.check_map(explanation, self.vit, image)
        self.assertIn("attention", explanation.method)

        attention_module, captured = self.vit.model.blocks[-1].attn, {}
        fused = attention_module.fused_attn
        attention_module.fused_attn = False
        handle = attention_module.attn_drop.register_forward_hook(
            lambda _, __, out: captured.update(attention=out.detach()))
        try:
            with torch.inference_mode():
                self.vit.model(self.vit.standard_transform(image).unsqueeze(0))
        finally:
            handle.remove()
            attention_module.fused_attn = fused
        prefix = self.vit.model.num_prefix_tokens
        grid = self.vit.model.patch_embed.grid_size
        heat = captured["attention"][0, :, 0, prefix:].mean(0).reshape(1, 1, *grid)
        self.assertTrue(np.allclose(explanation.heatmap, inference._scale(heat, (224, 224)), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
