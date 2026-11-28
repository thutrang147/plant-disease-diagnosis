"""Headless smoke test of the Streamlit demo (streamlit.testing)."""

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # streamlit is only in app/requirements.txt
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit not installed (pip install -r app/requirements.txt)")
class DemoAppTest(unittest.TestCase):
    def test_app_renders_without_exception(self):
        at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()
        self.assertEqual([e.message for e in at.exception], [])
        self.assertTrue(at.title)

    @unittest.skipUnless(os.environ.get("DEMO_SMOKE") == "1" and (ROOT / "models/resnet18_lab_best.pt").exists(),
                         "set DEMO_SMOKE=1 with checkpoints downloaded (loads test images from Hugging Face)")
    def test_example_prediction_and_gradcam(self):
        at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=600).run()
        at.radio[0].set_value("Example from the project's test sets").run()
        self.assertEqual([e.message for e in at.exception], [])
        self.assertIn("Grad-CAM class", [s.label for s in at.selectbox])
