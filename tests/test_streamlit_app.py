"""Headless checks of the Streamlit demo with randomly initialized checkpoints."""
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import timm
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from src.inference import CLASS_NAMES, VIT_ARCHITECTURE

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Streamlit is an optional app dependency.
    AppTest = None

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def png_bytes(color=(40, 140, 60)):
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


def save_resnet(path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                "architecture": "resnet18", "pretrained_weights": "IMAGENET1K_V1"}, path)


def save_vit(path):
    model = timm.create_model(VIT_ARCHITECTURE, pretrained=False, num_classes=5)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                "architecture": VIT_ARCHITECTURE}, path)


def html_bodies(node):
    return [element.value for element in node.get("html")]


def view_caption(node):
    return next(c.value for c in node.caption if "view(s)" in c.value)


def slips(node):
    return [body for body in html_bodies(node) if 'class="pd-slip' in body]


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = Path(self.temp.name) / "models"
        self.examples = Path(self.temp.name) / "examples"
        self.models.mkdir()
        environment = patch.dict(os.environ, {"PLANT_MODELS_DIR": str(self.models),
                                              "PLANT_EXAMPLES_DIR": str(self.examples)})
        environment.start()
        self.addCleanup(environment.stop)

    def run_app(self):
        app = AppTest.from_file(APP, default_timeout=180)
        app.run()
        self.assert_no_exception(app)
        return app

    def assert_no_exception(self, app):
        self.assertFalse(app.exception, [e.value for e in app.exception])

    def test_without_checkpoints_the_findings_tab_still_works(self):
        app = self.run_app()
        diagnose, findings = app.tabs
        self.assertEqual((diagnose.label, findings.label), ("Diagnose a leaf", "Research findings"))
        self.assertIn("No checkpoints found", diagnose.info[0].value)
        self.assertEqual(len(app.get("file_uploader")), 0)
        headings = [h.value for h in findings.subheader]
        self.assertEqual(headings, ["Near-perfect in the lab, below 0.5 in the field",
                                    "Adaptation helps four classes and almost erases Early Blight",
                                    "With matched training, adding lab images did not help"])
        self.assertEqual(len(findings.get("vega_lite_chart")), 4)
        self.assertEqual(len(findings.dataframe), 3)

    def test_upload_compares_every_checkpoint(self):
        save_resnet(self.models / "resnet18_lab_best.pt")
        save_resnet(self.models / "resnet18_geometric_mixed_best.pt")
        save_vit(self.models / "vit_small_mixed_best.pt")
        app = self.run_app()
        # Default featured checkpoint: best reported Field Dev macro-F1.
        self.assertEqual(app.sidebar.selectbox[0].value, "resnet18_geometric_mixed_best.pt")
        self.assertEqual([m.label for m in app.sidebar.metric], ["Lab Test", "Field Dev", "External v1"])
        self.assertEqual(app.segmented_control(key="source").options, ["Upload", "Camera"])

        app.get("file_uploader")[0].set_value(("leaf.png", png_bytes(), "image/png")).run()
        self.assert_no_exception(app)
        diagnose = app.tabs[0]
        cards = slips(diagnose)
        # One featured slip plus one per checkpoint, adapted models first.
        self.assertEqual(len(cards), 4)
        self.assertIn("pd-slip--featured", cards[0])
        order = [card.split("Det. ", 1)[1].split("</p>", 1)[0] for card in cards[1:]]
        self.assertEqual(order, ["ResNet18 · Geometric + Mixed", "ViT-Small · Mixed",
                                 "ResNet18 · Lab baseline"])
        table = diagnose.dataframe[0].value
        self.assertEqual(sorted(table["Class"]), sorted(CLASS_NAMES))
        self.assertAlmostEqual(float(table["Probability"].sum()), 1.0, places=4)
        self.assertTrue(any("pd-consensus" in body for body in html_bodies(diagnose)))
        self.assertIn("1 view(s) averaged · standard inference", view_caption(diagnose))

        app.segmented_control(key="strategy").set_value("Five-crop").run()
        self.assert_no_exception(app)
        self.assertIn("5 view(s) averaged · five-crop inference", view_caption(app.tabs[0]))

        self.assertEqual(len(app.tabs[0].get("image")), 1)
        app.toggle(key="maps").set_value(True).run()
        self.assert_no_exception(app)
        self.assertEqual(len(app.tabs[0].get("image")), 4)  # photo + one map per checkpoint
        self.assertTrue(any("not equivalent" in c.value for c in app.tabs[0].caption))

    def test_example_photos_need_no_upload(self):
        save_resnet(self.models / "resnet18_lab_best.pt")
        self.examples.mkdir()
        (self.examples / "healthy_leaf.png").write_bytes(png_bytes())
        (self.examples / "notes.txt").write_text("ignored")
        app = self.run_app()
        self.assertEqual(app.segmented_control(key="source").value, "Example photos")
        self.assertEqual(app.pills(key="example").options, ["healthy leaf"])
        cards = slips(app.tabs[0])
        self.assertEqual(len(cards), 1)  # A single checkpoint shows only the featured slip.
        app.toggle(key="maps").set_value(True).run()
        self.assert_no_exception(app)
        self.assertEqual(len(app.tabs[0].get("image")), 2)

    def test_unreadable_checkpoint_and_image_show_errors(self):
        (self.models / "broken.pt").write_bytes(b"not a checkpoint")
        save_resnet(self.models / "custom.pt")
        app = self.run_app()
        app.get("file_uploader")[0].set_value(("leaf.png", png_bytes(), "image/png")).run()
        self.assert_no_exception(app)
        cards = slips(app.tabs[0])
        self.assertTrue(any("pd-slip--error" in card and "broken.pt" in card for card in cards))

        app.sidebar.selectbox[0].set_value("broken.pt").run()
        self.assert_no_exception(app)
        self.assertIn("Could not load", app.tabs[0].error[0].value)

        app.get("file_uploader")[0].set_value(("leaf.png", b"not an image", "image/png")).run()
        self.assert_no_exception(app)
        self.assertIn("Could not read this image", app.tabs[0].error[0].value)


if __name__ == "__main__":
    unittest.main()
