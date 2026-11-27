"""Streamlit demo: classify one tomato leaf photo with a frozen research checkpoint.

Run from the repository root after copying checkpoints into ``models/``:
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

Set ``PLANT_MODELS_DIR`` to read checkpoints from another folder. Inference is
delegated to ``src.inference`` so the app uses the research preprocessing and
the Notebook 14 five-crop views. Nothing is trained, uploaded or saved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    CLASS_NAMES, discover_checkpoints, explain_image, load_checkpoint, load_image, predict_image,
)

MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", ROOT / "models"))
UPLOAD_TYPES = ("jpg", "jpeg", "png", "webp")
# Display heuristics for the warning, not calibrated decision thresholds.
LOW_CONFIDENCE = 0.5
LOW_MARGIN = 0.15

# Five-class macro-F1 from the README "Main Results" table (historical checkpoints).
KNOWN_CHECKPOINTS = {
    "resnet18_geometric_mixed_best.pt": ("ResNet18 · Geometric + Mixed", (0.9140, 0.4459, 0.4882)),
    "vit_small_mixed_best.pt": ("ViT-Small · Mixed", (0.9886, 0.3872, 0.4808)),
    "vit_small_lab_best.pt": ("ViT-Small · Lab baseline", (0.9938, 0.2677, 0.4157)),
    "resnet18_lab_best.pt": ("ResNet18 · Lab baseline", (0.9744, 0.2355, 0.2406)),
    "resnet18_aug_geometric_best.pt": ("ResNet18 · Geometric augmentation (Lab)", None),
    "resnet18_aug_photometric_best.pt": ("ResNet18 · Photometric augmentation (Lab)", None),
}
STRATEGY_LABELS = {
    "standard": "Standard (whole image)",
    "five_crop": "Five-crop (exploratory)",
}
STRATEGY_HELP = (
    "Standard is the evaluation protocol used for the reported results. Five-crop "
    "resizes to 448 × 448 and averages five 224 × 224 views (Notebook 14). It improved "
    "ViT-Small Mixed on External v1 (38/62 vs 32/62 correct) but changes coverage, "
    "scale and aggregation together, so treat it as exploratory."
)
# Short visual cues to help read a result; not diagnostic criteria.
CLASS_NOTES = {
    "Healthy": "None of the four modelled diseases was predicted. Other diseases, pests "
               "and nutrient problems are outside the model's five classes.",
    "Early Blight": "Typical signs: brown lesions with concentric rings, often on older leaves.",
    "Late Blight": "Typical signs: large, dark, water-soaked lesions that spread quickly.",
    "Bacterial Spot": "Typical signs: many small dark spots, sometimes with yellow halos.",
    "Septoria Leaf Spot": "Typical signs: many small round spots with pale centres and dark edges.",
}


def display_name(path: Path) -> str:
    known = KNOWN_CHECKPOINTS.get(path.name)
    return known[0] if known else path.name


def default_index(paths: list[Path]) -> int:
    """Prefer the checkpoint with the best reported Field Dev macro-F1."""
    order = list(KNOWN_CHECKPOINTS)
    ranked = sorted(range(len(paths)),
                    key=lambda i: order.index(paths[i].name) if paths[i].name in order else len(order))
    return ranked[0] if ranked else 0


@st.cache_resource(show_spinner="Loading checkpoint…")
def cached_model(path: str, modified: float):
    # ``modified`` is part of the cache key so a replaced file is reloaded.
    return load_checkpoint(path, "cpu")


def render_missing_checkpoints() -> None:
    st.info(
        f"No checkpoints found in `{MODELS_DIR}`. Checkpoints are not tracked in Git. "
        "Copy one or more `.pt` files from the project's Google Drive results "
        "(see `docs/workflow.md`, *Checkpoints and persistent Colab results*) into "
        "`models/`, or set `PLANT_MODELS_DIR`."
    )
    st.markdown("Recognized filenames:\n" + "\n".join(
        f"- `{name}` — {label}" for name, (label, _) in KNOWN_CHECKPOINTS.items()))


def render_model_context(path: Path, loaded) -> None:
    known = KNOWN_CHECKPOINTS.get(path.name)
    scores = known[1] if known else None
    if scores:
        lab, field, external = scores
        columns = st.sidebar.columns(3)
        columns[0].metric("Lab Test", f"{lab:.2f}")
        columns[1].metric("Field Dev", f"{field:.2f}")
        columns[2].metric("External v1", f"{external:.2f}")
        st.sidebar.caption("Reported five-class macro-F1 for this historical checkpoint "
                           "(README, standard inference).")
    else:
        st.sidebar.caption("No reported field results for this checkpoint in the README.")
    with st.sidebar.expander("Checkpoint details"):
        details = {"file": path.name, "architecture": loaded.architecture,
                   "preprocessing": loaded.preprocessing_source, **loaded.metadata}
        details.pop("class_names", None)
        st.json(details, expanded=True)


def render_prediction(prediction) -> None:
    st.subheader(f"Most likely: {prediction.class_name}")
    st.write(CLASS_NOTES[prediction.class_name])
    ranked = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1]
    st.metric("Model probability", f"{prediction.confidence:.1%}",
              help="Softmax probability, not a calibrated likelihood of disease.")
    if prediction.confidence < LOW_CONFIDENCE or margin < LOW_MARGIN:
        st.warning(
            f"Low-confidence result: the runner-up, {ranked[1][0]}, is {ranked[1][1]:.1%}. "
            "Try a closer, well-lit photo of a single leaf."
        )
    table = pd.DataFrame({"Class": [name for name, _ in ranked],
                          "Probability": [value for _, value in ranked]})
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", format="%.3f", min_value=0.0, max_value=1.0)},
    )
    st.caption(f"{prediction.view_count} view(s) averaged · {STRATEGY_LABELS[prediction.strategy]}")


def render_explanation(loaded, image, prediction, target) -> None:
    """Draw the map in ``target`` (under the photo) when the toggle is on."""
    if not st.toggle("Show where the model looked", key="explain",
                     help="Grad-CAM for ResNet18, last-block attention for ViT-Small (Notebook 10)."):
        return
    with st.spinner("Computing map…"):
        explanation = explain_image(loaded, image)
    note = "Computed on the standard 224 × 224 view."
    if explanation.label_id != prediction.label_id:
        note += f" On that view the model predicts {explanation.class_name}, unlike the five-crop result."
    with target:
        st.image(explanation.overlay(), width="stretch",
                 caption=f"{explanation.method} · {explanation.class_name}")
        st.caption(note + " Bright areas are weighted most by this method, but they do not show that "
                   "the model relied on a lesion; treat the map as qualitative.")


def main() -> None:
    st.set_page_config(page_title="Tomato Leaf Diagnosis", page_icon="🍅", layout="wide")
    st.title("Tomato leaf disease classifier")
    st.caption(
        "Research demo for five classes: " + ", ".join(CLASS_NAMES) + ". "
        "Models were trained mainly on laboratory images; on field photos their reported "
        "macro-F1 is 0.24–0.49 (standard inference). Do not use these predictions for "
        "crop-management decisions."
    )

    checkpoints = discover_checkpoints(MODELS_DIR)
    if not checkpoints:
        render_missing_checkpoints()
        return

    st.sidebar.header("Model")
    path = st.sidebar.selectbox("Checkpoint", checkpoints, index=default_index(checkpoints),
                                format_func=display_name)
    strategy = st.sidebar.radio("Inference", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get,
                                help=STRATEGY_HELP)
    try:
        loaded = cached_model(str(path), path.stat().st_mtime)
    except (ValueError, RuntimeError, OSError) as error:
        st.error(f"Could not load `{path.name}`: {error}")
        return
    render_model_context(path, loaded)

    upload = st.file_uploader("Upload a tomato leaf photo", type=UPLOAD_TYPES,
                              help="A close photo of one leaf works best; the image is processed in memory only.")
    if upload is None:
        st.write("Upload a photo to see the predicted class and the probability of each class.")
        return
    try:
        image = load_image(upload)
    except Exception as error:  # Pillow raises several decoder-specific errors.
        st.error(f"Could not read this image: {error}")
        return

    left, right = st.columns([1, 1])
    left.image(image, caption=f"{upload.name} · {image.width} × {image.height}", width="stretch")
    with right:
        with st.spinner("Classifying…"):
            prediction = predict_image(loaded, image, strategy)
        render_prediction(prediction)
        render_explanation(loaded, image, prediction, left)


main()
