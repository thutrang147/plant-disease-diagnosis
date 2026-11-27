"""Interactive demo: tomato leaf disease prediction with the project's checkpoints.

Run from the repository root:

    python scripts/download_checkpoints.py
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import CLASS_NAMES, grad_cam, load_checkpoint, model_view, overlay, predict, prepare, registry, sha256  # noqa: E402

DEMO_MODELS = {
    "resnet18_lab": "ResNet18 · Lab baseline",
    "vit_small_lab": "ViT-Small · Lab baseline",
    "resnet18_geometric_mixed": "ResNet18 · Geometric + Mixed (Notebook 08)",
    "vit_small_mixed": "ViT-Small · Mixed (Notebook 08)",
    "nb15_field_only_s42": "ResNet18 · Field-only, controlled (Notebook 15, seed 42)",
    "nb15_mixed_s42": "ResNet18 · Mixed, controlled (Notebook 15, seed 42)",
}
EXAMPLES = {"Field Test (PlantDoc, 47 images)": "field_test", "Lab Test (PlantVillage, 1,692 images)": "lab_test"}

st.set_page_config(page_title="Tomato leaf disease demo", layout="wide")


@st.cache_resource(show_spinner="Verifying checkpoint and loading model…")
def load(key: str):
    entry = next(e for e in registry() if e["key"] == key)
    path = ROOT / entry["local_path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"{path} does not match the registered SHA-256")
    vit_config = None
    if "vit" in key:
        lab = ROOT / next(e for e in registry() if e["key"] == "vit_small_lab")["local_path"]
        if lab.is_file():
            vit_config = load_checkpoint(lab)[2].get("preprocessing_config")
    return load_checkpoint(path, vit_config)


@st.cache_data
def measured_scores() -> pd.DataFrame:
    """Lab Test macro-F1 and External v3 accuracy per checkpoint, from committed results."""
    lab = pd.read_csv(ROOT / "data/results/lab_split_duplicates/lab_test_clean_vs_all.csv")
    lab = lab[lab.subset.eq("all")].set_index("model").macro_f1
    v3 = pd.read_csv(ROOT / "data/results/external_v3_plantseg/metrics.csv").set_index("model")
    return pd.DataFrame({"Lab Test macro-F1": lab, "External v3 accuracy": v3.accuracy,
                         "External v3 disease macro-F1": v3.macro_f1_disease})


@st.cache_resource(show_spinner="Loading dataset from Hugging Face (first use downloads it)…")
def example_dataset(split: str):
    from src.dataset import PlantDiseaseDataset
    return PlantDiseaseDataset(ROOT / "data/splits" / f"{split}.csv", transform=lambda img: img)


available = {k: v for k, v in DEMO_MODELS.items()
             if (ROOT / next(e for e in registry() if e["key"] == k)["local_path"]).is_file()}

st.title("Tomato leaf disease classification under domain shift")
st.caption("Five classes: Healthy, Early Blight, Late Blight, Bacterial Spot, Septoria Leaf Spot. "
           "Research demo, not an agronomic diagnosis tool.")

with st.expander("Read before trusting a prediction", expanded=True):
    st.markdown(
        "- On laboratory images the models score **0.81–0.99 macro-F1** (the Lab baselines 0.97–0.99; field "
        "fine-tuning without Lab rehearsal forgets most). On **486 independent field images (External v3)** every "
        "model scores only **41–44% accuracy**: field predictions are wrong more often than right. Measured scores "
        "for each model are listed below.\n"
        "- Lab baselines almost never predict **Healthy** on field photos; adapted models do, but a leaf with its "
        "lesions removed is still rarely called Healthy.\n"
        "- **Bacterial Spot** is usually confused with **Septoria Leaf Spot**, and the ResNet18 Lab baseline "
        "predicts **Early Blight** for most field images.\n"
        "- Softmax confidence is not a calibrated probability of being correct.")

if not available:
    st.error("No checkpoints found. From the repository root run `python scripts/download_checkpoints.py`.")
    st.stop()

with st.sidebar:
    st.header("Model")
    key = st.selectbox("Checkpoint", list(available), format_func=available.get)
    compare_all = st.checkbox("Compare all available models", value=True)
    st.header("Image")
    mode = st.radio("Source", ["Upload a photo", "Example from the project's test sets"])

image, truth = None, None
if mode == "Upload a photo":
    upload = st.file_uploader("Tomato leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if upload is not None:
        image = prepare(Image.open(upload))
else:
    split = EXAMPLES[st.sidebar.selectbox("Test set", list(EXAMPLES))]
    dataset = example_dataset(split)
    index = st.sidebar.number_input("Image index", 0, len(dataset) - 1, 0, step=1)
    image = prepare(dataset[int(index)][0])
    truth = CLASS_NAMES[int(dataset.metadata.label_id.iloc[int(index)])]

if image is None:
    st.info("Upload a photo or choose an example in the sidebar.")
    st.stop()

model, transform, meta = load(key)
probs = predict(model, transform, image)
top = int(probs.argmax())

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(image, width="stretch")
    if truth:
        st.markdown(f"**Ground-truth label:** {truth}")
with right:
    st.subheader(f"Prediction: {CLASS_NAMES[top]} ({probs[top]:.0%})")
    chart = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
    st.altair_chart(alt.Chart(chart).mark_bar().encode(
        x=alt.X("class:N", sort=CLASS_NAMES, title=None), y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["class", alt.Tooltip("probability:Q", format=".1%")]), width="stretch")
    if meta["architecture"] == "resnet18":
        target = st.selectbox("Grad-CAM class", CLASS_NAMES, index=top)
        cam = grad_cam(model, transform, image, CLASS_NAMES.index(target))
        st.image(overlay(image, cam, transform=transform), caption=f"Grad-CAM (layer4) for '{target}' on the 224×224 model view",
                 width="stretch")
        st.caption("On PlantSeg lesion masks, Grad-CAM peaks fall on lesions for about 0.65 of images — less often "
                   "than simply pointing at the image centre (0.74). The models do use lesion pixels, but this map "
                   "is not a reliable picture of where.")
    else:
        st.image(model_view(image, transform), caption="224×224 region this model sees (its own resize and crop)",
                 width="stretch")
        st.caption("Grad-CAM is shown for ResNet18 checkpoints only.")

if compare_all and len(available) > 1:
    st.subheader("All available models on this image")
    rows = []
    for other, label in available.items():
        m, t, _ = load(other)
        p = predict(m, t, image)
        scores = measured_scores().reindex([other]).iloc[0]
        rows.append({"model": label, "prediction": CLASS_NAMES[int(p.argmax())],
                     "confidence": f"{p.max():.0%}", "P(Healthy)": f"{p[0]:.0%}",
                     "Lab Test macro-F1 (measured)": f"{scores['Lab Test macro-F1']:.3f}",
                     "External v3 accuracy (measured)": f"{scores['External v3 accuracy']:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Disagreement between models trained on the same classes is itself a sign of domain shift.")
