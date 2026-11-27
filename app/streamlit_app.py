"""Streamlit demo: one tomato leaf photo, a determination from every research checkpoint.

Run from the repository root (so ``.streamlit/config.toml`` applies) after copying
checkpoints into ``models/``:
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

Optional environment variables: ``PLANT_MODELS_DIR`` (checkpoint folder) and
``PLANT_EXAMPLES_DIR`` (demo photos, default ``app/examples/``, which Git ignores).
Inference is delegated to ``src.inference``; the findings tab reads committed
evidence only. Nothing is trained, uploaded or saved.
"""

from __future__ import annotations

import html
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _path in (str(ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import findings  # noqa: E402
from components import ARCH_COLORS, CSS, consensus, error_slip_html, key_html, slip_html  # noqa: E402
from src.inference import (  # noqa: E402
    CLASS_NAMES, IMAGE_SUFFIXES, discover_checkpoints, explain_image, load_checkpoint, load_image,
    predict_image,
)

MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", ROOT / "models"))
EXAMPLES_DIR = Path(os.environ.get("PLANT_EXAMPLES_DIR", APP_DIR / "examples"))
UPLOAD_TYPES = ("jpg", "jpeg", "png", "webp")
# Display heuristics for the warning, not calibrated decision thresholds.
LOW_CONFIDENCE = 0.5
LOW_MARGIN = 0.15

# filename -> (label, architecture, training, (Lab Test, Field Dev, External v1) macro-F1 from README)
KNOWN_CHECKPOINTS = {
    "resnet18_geometric_mixed_best.pt": ("ResNet18 · Geometric + Mixed", "ResNet18", "Field-adapted",
                                         (0.9140, 0.4459, 0.4882)),
    "vit_small_mixed_best.pt": ("ViT-Small · Mixed", "ViT-Small", "Field-adapted", (0.9886, 0.3872, 0.4808)),
    "vit_small_lab_best.pt": ("ViT-Small · Lab baseline", "ViT-Small", "Lab only", (0.9938, 0.2677, 0.4157)),
    "resnet18_lab_best.pt": ("ResNet18 · Lab baseline", "ResNet18", "Lab only", (0.9744, 0.2355, 0.2406)),
    "resnet18_aug_geometric_best.pt": ("ResNet18 · Geometric augmentation", "ResNet18", "Lab only", None),
    "resnet18_aug_photometric_best.pt": ("ResNet18 · Photometric augmentation", "ResNet18", "Lab only", None),
}
STRATEGIES = {"Standard": "standard", "Five-crop": "five_crop"}
STRATEGY_HELP = (
    "Standard is the evaluation protocol behind the reported results. Five-crop resizes to "
    "448 × 448 and averages five 224 × 224 views (Notebook 14). It raised ViT-Small Mixed on "
    "External v1 from 32/62 to 38/62 correct but changes coverage, scale and aggregation "
    "together, so treat it as exploratory."
)
CLASS_NOTES = {  # Short visual cues to help read a result; not diagnostic criteria.
    "Healthy": "None of the four modelled diseases was predicted. Other diseases, pests and "
               "nutrient problems are outside the model's five classes.",
    "Early Blight": "Typical signs: brown lesions with concentric rings, often on older leaves.",
    "Late Blight": "Typical signs: large, dark, water-soaked lesions that spread quickly.",
    "Bacterial Spot": "Typical signs: many small dark spots, sometimes with yellow halos.",
    "Septoria Leaf Spot": "Typical signs: many small round spots with pale centres and dark edges.",
}


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    label: str
    architecture: str | None
    training: str | None
    scores: tuple[float, float, float] | None

    @property
    def modified(self) -> float:
        return self.path.stat().st_mtime


def describe(path: Path) -> Checkpoint:
    known = KNOWN_CHECKPOINTS.get(path.name)
    if known:
        return Checkpoint(path, *known)
    return Checkpoint(path, path.name, None, None, None)


def ordered(checkpoints: list[Checkpoint]) -> list[Checkpoint]:
    """Field-adapted before lab-only, ResNet18 before ViT-Small, unknown files last.

    The first entry (the default featured checkpoint) is ResNet18 Geometric + Mixed,
    which has the best reported Field Dev macro-F1.
    """
    training = {"Field-adapted": 0, "Lab only": 1}
    architecture = {"ResNet18": 0, "ViT-Small": 1}
    known = list(KNOWN_CHECKPOINTS)
    return sorted(checkpoints, key=lambda c: (
        c.training is None, training.get(c.training, 2), architecture.get(c.architecture, 2),
        known.index(c.path.name) if c.path.name in known else len(known), c.path.name))


def theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def arch_color(architecture: str | None) -> str:
    return ARCH_COLORS[theme_type()].get(architecture or "", "#7a867e")


@st.cache_resource(show_spinner=False)
def cached_model(path: str, modified: float):
    # ``modified`` is part of the cache key so a replaced file is reloaded.
    return load_checkpoint(path, "cpu")


@st.cache_data(show_spinner=False, max_entries=128)
def cached_prediction(path: str, modified: float, image_bytes: bytes, strategy: str):
    return predict_image(cached_model(path, modified), load_image(BytesIO(image_bytes)), strategy)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_explanation(path: str, modified: float, image_bytes: bytes):
    return explain_image(cached_model(path, modified), load_image(BytesIO(image_bytes)))


@st.cache_data(show_spinner=False)
def cached_findings():
    return findings.domain_gap(ROOT), findings.class_tradeoff(ROOT), findings.controlled_runs(ROOT)


def run_all(checkpoints, image_bytes, strategy):
    """Return {checkpoint: (prediction, architecture) or error message}."""
    results = {}
    for checkpoint in checkpoints:
        try:
            loaded = cached_model(str(checkpoint.path), checkpoint.modified)
            prediction = cached_prediction(str(checkpoint.path), checkpoint.modified, image_bytes, strategy)
        except (ValueError, RuntimeError, OSError) as error:
            results[checkpoint] = str(error)
            continue
        architecture = checkpoint.architecture or ("ResNet18" if loaded.architecture == "resnet18"
                                                   else "ViT-Small")
        results[checkpoint] = (prediction, architecture)
    return results


def slip_meta(checkpoint: Checkpoint, prediction) -> list[str]:
    meta = []
    if checkpoint.training:
        meta.append(checkpoint.training.lower())
    if checkpoint.scores:
        meta.append(f"F1 lab {checkpoint.scores[0]:.2f} → field {checkpoint.scores[1]:.2f}")
    ranked = sorted(prediction.probabilities.values(), reverse=True)
    if prediction.confidence < LOW_CONFIDENCE or ranked[0] - ranked[1] < LOW_MARGIN:
        meta.append("low confidence")
    return meta


# ---------------------------------------------------------------- page sections
def render_header() -> None:
    st.html('<p class="pd-eyebrow">PlantVillage → PlantDoc · five tomato leaf classes · ResNet18 and ViT-Small</p>')
    st.title("Tomato leaf diagnosis, from lab to field")
    st.html(
        '<p class="pd-lede">Every checkpoint from the study records its own determination for your photo. '
        "Models trained only on laboratory images often disagree with models adapted to field photos, "
        "which is the gap this project measures.</p>"
    )


def read_photo(examples: list[Path]) -> tuple[bytes | None, str | None]:
    sources = (["Example photos"] if examples else []) + ["Upload", "Camera"]
    source_col, strategy_col = st.columns([3, 2])
    with source_col:
        source = st.segmented_control("Photo", sources, default=sources[0], key="source") or sources[0]
    with strategy_col:
        st.segmented_control("Inference", list(STRATEGIES), default="Standard", key="strategy",
                             help=STRATEGY_HELP)
    if source == "Example photos":
        names = {path.stem.replace("_", " "): path for path in examples}
        choice = st.pills("Example", list(names), default=next(iter(names)), key="example",
                          label_visibility="collapsed") or next(iter(names))
        return names[choice].read_bytes(), names[choice].name
    if source == "Camera":
        shot = st.camera_input("Photograph one leaf, filling most of the frame")
        return (shot.getvalue(), "camera photo") if shot else (None, None)
    upload = st.file_uploader("Upload a tomato leaf photo", type=UPLOAD_TYPES,
                              help="A close photo of one leaf works best. The image stays in memory.")
    return (upload.getvalue(), upload.name) if upload else (None, None)


def render_featured(checkpoint: Checkpoint, outcome) -> None:
    if isinstance(outcome, str):
        st.error(f"Could not load `{checkpoint.path.name}`: {outcome}")
        return
    prediction, architecture = outcome
    st.html(slip_html(model=checkpoint.label, color=arch_color(architecture),
                      determination=prediction.class_name, confidence=prediction.confidence,
                      meta=slip_meta(checkpoint, prediction), featured=True))
    st.write(CLASS_NOTES[prediction.class_name])
    ranked = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
    if prediction.confidence < LOW_CONFIDENCE or ranked[0][1] - ranked[1][1] < LOW_MARGIN:
        st.warning(f"Low confidence: the runner-up, {ranked[1][0]}, has {ranked[1][1]:.1%}. "
                   "Try a closer, evenly lit photo of a single leaf.", icon=":material/warning:")
    st.dataframe(
        pd.DataFrame({"Class": [name for name, _ in ranked], "Probability": [v for _, v in ranked]}),
        hide_index=True, width="stretch",
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", help="Softmax probability, not a calibrated likelihood of disease.",
            format="%.3f", min_value=0.0, max_value=1.0)},
    )
    st.caption(f"{prediction.view_count} view(s) averaged · "
               f"{'five-crop' if prediction.strategy == 'five_crop' else 'standard'} inference")


def render_map(checkpoint: Checkpoint, image_bytes: bytes, prediction) -> None:
    try:
        explanation = cached_explanation(str(checkpoint.path), checkpoint.modified, image_bytes)
    except (ValueError, RuntimeError) as error:
        st.caption(f"Map unavailable: {error}")
        return
    method = "Grad-CAM" if "Grad-CAM" in explanation.method else "Last-block attention"
    st.image(explanation.overlay(), width="stretch", caption=f"{method} · {explanation.class_name}")
    if explanation.label_id != prediction.label_id:
        st.caption(f"On the standard view this model says {explanation.class_name}, "
                   "unlike its five-crop result.")


MAP_CAVEAT = ("Maps are computed on the standard 224 × 224 view: Grad-CAM for ResNet18 and final-block "
              "attention for ViT-Small (Notebook 10). Bright areas are weighted most by each method; they "
              "do not show that a model relied on a lesion, and the two methods are not equivalent.")


def render_comparison(results: dict, image_bytes: bytes) -> None:
    st.subheader("Every checkpoint's determination")
    determinations = [(c.training, outcome[0].class_name) for c, outcome in results.items()
                      if not isinstance(outcome, str)]
    st.html(f'<p class="pd-consensus">{html.escape(consensus(determinations))}</p>')
    show_maps = st.toggle("Show where each model looked", key="maps")
    items = list(results.items())
    per_row = 4 if len(items) % 3 else 3
    for start in range(0, len(items), per_row):
        columns = st.columns(per_row, gap="medium")
        for column, (checkpoint, outcome) in zip(columns, items[start:start + per_row]):
            with column:
                if isinstance(outcome, str):
                    st.html(error_slip_html(checkpoint.label, outcome))
                    continue
                prediction, architecture = outcome
                st.html(slip_html(model=checkpoint.label, color=arch_color(architecture),
                                  determination=prediction.class_name, confidence=prediction.confidence,
                                  meta=slip_meta(checkpoint, prediction)))
                if show_maps:
                    render_map(checkpoint, image_bytes, prediction)
    if show_maps:
        st.caption(MAP_CAVEAT)


def render_sidebar(featured: Checkpoint) -> None:
    if featured.scores:
        lab, field, external = featured.scores
        columns = st.sidebar.columns(3)
        columns[0].metric("Lab Test", f"{lab:.2f}")
        columns[1].metric("Field Dev", f"{field:.2f}")
        columns[2].metric("External v1", f"{external:.2f}")
        st.sidebar.caption("Reported five-class macro-F1 (README, standard inference).")
    else:
        st.sidebar.caption("No reported field results for this checkpoint in the README.")
    try:
        loaded = cached_model(str(featured.path), featured.modified)
    except (ValueError, RuntimeError, OSError):
        return  # The main area reports the load error.
    with st.sidebar.expander("Checkpoint details"):
        details = {"file": featured.path.name, "architecture": loaded.architecture,
                   "preprocessing": loaded.preprocessing_source, **loaded.metadata}
        details.pop("class_names", None)
        st.json(details, expanded=True)


def render_diagnose() -> None:
    checkpoints = ordered([describe(p) for p in discover_checkpoints(MODELS_DIR)])
    if not checkpoints:
        st.info(
            f"No checkpoints found in `{MODELS_DIR}`. They are not tracked in Git: copy `.pt` files "
            "from the project's Google Drive results (see `docs/workflow.md`, *Checkpoints and "
            "persistent Colab results*) into `models/`, or set `PLANT_MODELS_DIR`. "
            "The Research findings tab works without them."
        )
        st.markdown("Recognized filenames:\n" + "\n".join(
            f"- `{name}`: {label}" for name, (label, *_rest) in KNOWN_CHECKPOINTS.items()))
        return

    st.sidebar.header("Featured checkpoint")
    # Select by filename: a Checkpoint instance from an earlier rerun is a different class object.
    by_name = {c.path.name: c for c in checkpoints}
    featured = by_name[st.sidebar.selectbox("Featured checkpoint", list(by_name), key="featured",
                                            format_func=lambda name: by_name[name].label,
                                            label_visibility="collapsed")]
    render_sidebar(featured)
    examples = sorted(p for p in EXAMPLES_DIR.glob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if EXAMPLES_DIR.is_dir() else []
    image_bytes, name = read_photo(examples)
    strategy = STRATEGIES[st.session_state.get("strategy") or "Standard"]
    if not image_bytes:
        st.caption("Choose a photo to see each checkpoint's determination.")
        return
    try:
        image = load_image(BytesIO(image_bytes))
    except Exception as error:  # Pillow raises several decoder-specific errors.
        st.error(f"Could not read this image: {error}")
        return

    with st.spinner("Classifying with every checkpoint…"):
        results = run_all(checkpoints, image_bytes, strategy)

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.image(image, caption=f"{name} · {image.width} × {image.height}", width="stretch")
    with right:
        render_featured(featured, results[featured])
    if len(checkpoints) > 1:
        st.divider()
        render_comparison(results, image_bytes)
    elif not isinstance(results[featured], str) and st.toggle("Show where the model looked", key="maps"):
        with left:
            render_map(featured, image_bytes, results[featured][0])
            st.caption(MAP_CAVEAT)


def render_findings() -> None:
    gap, tradeoff, controlled = cached_findings()
    colors, ink = ARCH_COLORS[theme_type()], findings.INK[theme_type()]

    st.html('<p class="pd-eyebrow">Finding 1 · domain shift</p>')
    st.subheader("Near-perfect in the lab, below 0.5 in the field")
    st.write("Five-class macro-F1 of the four historical checkpoints. Training on some field photos "
             "recovers part of the drop, more for ResNet18 than for ViT-Small, and costs ResNet18 "
             "some lab accuracy.")
    st.html(key_html(colors["ResNet18"], [("line", "Field-adapted"), ("dashed", "Lab only")]))
    for column, architecture in zip(st.columns(2, gap="large"), findings.ARCHITECTURES):
        column.altair_chart(findings.domain_gap_chart(gap, architecture, ink), width="stretch")
    with st.expander("Data table"):
        st.dataframe(gap.pivot(index="model", columns="evaluation", values="macro_f1")
                     [findings.EVALUATIONS].round(4), width="stretch")
    st.caption(f"Source: {findings.SOURCES['domain_gap']}")

    st.divider()
    st.html('<p class="pd-eyebrow">Finding 2 · class trade-offs</p>')
    st.subheader("Adaptation helps four classes and almost erases Early Blight")
    st.write("ResNet18 on External v1 (62 images), before and after field adaptation. A better "
             "average can hide a class that got much worse.")
    st.html(key_html(colors["ResNet18"], [("filled", "Field-adapted"), ("hollow", "Lab only")]))
    st.altair_chart(findings.tradeoff_chart(tradeoff, ink), width="stretch")
    with st.expander("Data table"):
        counts = tradeoff.assign(cell=tradeoff.correct.astype(str) + " / " + tradeoff.images.astype(str))
        st.dataframe(counts.pivot(index="class_name", columns="training", values="cell")
                     .loc[findings.CLASS_NAMES, findings.TRAINING[::-1]], width="stretch")
    st.caption(f"Source: {findings.SOURCES['tradeoff']}")

    st.divider()
    st.html('<p class="pd-eyebrow">Finding 3 · controlled comparison</p>')
    st.subheader("With matched training, adding lab images did not help")
    st.write("ResNet18 fine-tuned on field photos only, or on field plus lab photos, with the same "
             "field exposure and update budget. Field-only scores higher in two of three seeds; "
             "this does not show that mixing is worse in general.")
    st.html(key_html(colors["ResNet18"], [("filled", "Field-only"), ("hollow", "Mixed")]))
    st.altair_chart(findings.controlled_chart(controlled, ink), width="stretch")
    with st.expander("Data table"):
        st.dataframe(controlled.pivot(index="row", columns="condition", values="macro_f1")
                     .loc[["Seed 42", "Seed 43", "Seed 44", "Mean of 3"]].round(4), width="stretch")
    st.caption(f"Source: {findings.SOURCES['controlled']}")


def main() -> None:
    st.set_page_config(page_title="Tomato leaf diagnosis", page_icon="🍅", layout="wide")
    st.html(CSS)
    render_header()
    diagnose_tab, findings_tab = st.tabs(["Diagnose a leaf", "Research findings"])
    with diagnose_tab:
        render_diagnose()
    with findings_tab:
        render_findings()


main()
