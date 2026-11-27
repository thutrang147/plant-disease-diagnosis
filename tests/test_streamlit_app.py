"""Headless checks of the Streamlit demo with randomly initialized checkpoints."""
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from src.inference import CLASS_NAMES

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Streamlit is an optional app dependency.
    AppTest = None

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (64, 48), (40, 140, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def save_resnet(path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                "architecture": "resnet18", "pretrained_weights": "IMAGENET1K_V1"}, path)


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = Path(self.temp.name)
        environment = patch.dict(os.environ, {"PLANT_MODELS_DIR": str(self.models)})
        environment.start()
        self.addCleanup(environment.stop)

    def run_app(self):
        app = AppTest.from_file(APP, default_timeout=120)
        app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def test_missing_checkpoints_explain_where_to_put_them(self):
        app = self.run_app()
        self.assertIn("No checkpoints found", app.info[0].value)
        self.assertEqual(len(app.get("file_uploader")), 0)

    def test_upload_shows_prediction_and_all_class_probabilities(self):
        save_resnet(self.models / "resnet18_lab_best.pt")
        save_resnet(self.models / "resnet18_geometric_mixed_best.pt")
        app = self.run_app()
        # Default is the checkpoint with the best reported Field Dev macro-F1.
        self.assertTrue(app.sidebar.selectbox[0].value.name == "resnet18_geometric_mixed_best.pt")
        self.assertEqual([m.label for m in app.sidebar.metric], ["Lab Test", "Field Dev", "External v1"])

        app.get("file_uploader")[0].set_value(("leaf.png", png_bytes(), "image/png")).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.subheader[0].value.startswith("Most likely: "))
        table = app.dataframe[0].value
        self.assertEqual(sorted(table["Class"]), sorted(CLASS_NAMES))
        self.assertAlmostEqual(float(table["Probability"].sum()), 1.0, places=4)
        self.assertIn("1 view(s)", app.main.caption[-1].value)

        app.sidebar.radio[0].set_value("five_crop").run()
        self.assertFalse(app.exception)
        self.assertIn("5 view(s)", app.main.caption[-1].value)

    def test_unreadable_checkpoint_and_image_show_errors(self):
        (self.models / "broken.pt").write_bytes(b"not a checkpoint")
        app = self.run_app()
        self.assertIn("Could not load", app.error[0].value)

        (self.models / "broken.pt").unlink()
        save_resnet(self.models / "custom.pt")
        app = self.run_app()
        self.assertIn("No reported field results", app.sidebar.caption[0].value)
        app.get("file_uploader")[0].set_value(("leaf.png", b"not an image", "image/png")).run()
        self.assertFalse(app.exception)
        self.assertIn("Could not read this image", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
