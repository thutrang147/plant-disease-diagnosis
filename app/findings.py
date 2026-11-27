"""Research findings for the demo, built from committed evidence only.

Nothing here needs checkpoints, images or downloads, so the findings tab works on
a fresh clone. Values come from tracked result files; the External v1 macro-F1
values are the canonical Notebook 11 figures reported in the README, whose
prediction exports are not tracked. ``tests/test_app_findings.py`` checks every
value against its source.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"]
ARCHITECTURES = ["ResNet18", "ViT-Small"]
TRAINING = ["Field-adapted", "Lab only"]
EVALUATIONS = ["Lab Test", "Field Dev", "External v1"]

# (name in final_comparison_summary.csv, architecture, training regime)
MODELS = [
    ("ResNet18 · Lab baseline", "ResNet18", "Lab only"),
    ("ViT-Small · Lab baseline", "ViT-Small", "Lab only"),
    ("ResNet18 · Geometric + Mixed", "ResNet18", "Field-adapted"),
    ("ViT-Small · Mixed", "ViT-Small", "Field-adapted"),
]
# Canonical Notebook 11 run (EXIF-corrected), as reported in the README.
EXTERNAL_V1_MACRO_F1 = {
    "ResNet18 · Lab baseline": 0.2406,
    "ViT-Small · Lab baseline": 0.4157,
    "ResNet18 · Geometric + Mixed": 0.4882,
    "ViT-Small · Mixed": 0.4808,
}
# Upstream-loader exports. For these two ResNet18 models the per-class correct
# counts equal the canonical run (docs/experiment_reconciliation.md).
TRADEOFF_FILES = {
    "Lab only": "outputs/external_confirmatory_eval/predictions_model1_resnet18_lab_base_external_v1.csv",
    "Field-adapted": "outputs/external_confirmatory_eval/predictions_model3_resnet18_geom_mixed_external_v1.csv",
}
SOURCES = {
    "domain_gap": "Lab Test and Field Dev: outputs/final_comparison_development/final_comparison_summary.csv. "
                  "External v1: canonical Notebook 11 run (README).",
    "tradeoff": "outputs/external_confirmatory_eval, ResNet18 models 1 and 3; per-class counts match the "
                "canonical Notebook 11 run.",
    "controlled": "Notebook 15 prediction exports under data/results/notebook15 (historical 47-image Field Test).",
}


def domain_gap(root: Path = ROOT) -> pd.DataFrame:
    """Macro-F1 of the four historical checkpoints on three evaluation sets."""
    summary = pd.read_csv(root / "outputs/final_comparison_development/final_comparison_summary.csv")
    rows = []
    for model, architecture, training in MODELS:
        for partition, evaluation in (("lab_test", "Lab Test"), ("field_dev", "Field Dev")):
            match = summary[summary.model.eq(model) & summary.partition.eq(partition)]
            if len(match) != 1:
                raise ValueError(f"Expected one {partition} row for {model}")
            rows.append(dict(model=model, architecture=architecture, training=training,
                             evaluation=evaluation, macro_f1=float(match.macro_f1.iloc[0]),
                             images=int(match.images.iloc[0])))
        rows.append(dict(model=model, architecture=architecture, training=training,
                         evaluation="External v1", macro_f1=EXTERNAL_V1_MACRO_F1[model], images=62))
    return pd.DataFrame(rows)


def class_tradeoff(root: Path = ROOT) -> pd.DataFrame:
    """Per-class correct predictions of the two ResNet18 checkpoints on External v1."""
    rows = []
    for training, relative in TRADEOFF_FILES.items():
        frame = pd.read_csv(root / relative)
        for label_id, group in frame.groupby("label_id"):
            correct = int(group.predicted_label_id.eq(label_id).sum())
            rows.append(dict(class_name=CLASS_NAMES[int(label_id)], training=training, correct=correct,
                             images=len(group), recall=correct / len(group)))
    table = pd.DataFrame(rows)
    table["row"] = table.class_name + " (n=" + table.images.astype(str) + ")"
    return table


def controlled_runs(root: Path = ROOT) -> pd.DataFrame:
    """Notebook 15: Field-only vs Mixed ResNet18 fine-tuning, three seeds plus mean."""
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.controlled_field_report import load_evidence

    summary = load_evidence(root)["summary"].copy()
    summary["condition"] = summary.condition.map({"field_only": "Field-only", "mixed": "Mixed"})
    summary["row"] = "Seed " + summary.seed.astype(str)
    means = (summary.groupby("condition", as_index=False)[["accuracy", "macro_f1"]].mean()
             .assign(row="Mean of 3", seed=pd.NA))
    return pd.concat([summary, means], ignore_index=True)[
        ["row", "seed", "condition", "accuracy", "macro_f1"]]


# ---------------------------------------------------------------- charts
# Text ink of each Streamlit theme (.streamlit/config.toml); Vega cannot read CSS currentColor.
INK = {"light": "#18241D", "dark": "#E6ECE5"}
# Series colors come from the Streamlit theme's chartCategoricalColors, so the
# light and dark steps are selected there; the domain fixes the order:
# slot 1 = ResNet18, slot 2 = ViT-Small, in every chart.
_ARCH_COLOR = alt.Color("architecture:N", scale=alt.Scale(domain=ARCHITECTURES), legend=None)


def domain_gap_chart(data: pd.DataFrame, architecture: str, ink: str = INK["light"]) -> alt.Chart:
    """Slope chart for one architecture: solid = field-adapted, dashed = lab only."""
    panel = data[data.architecture.eq(architecture)]
    base = alt.Chart(panel).encode(
        x=alt.X("evaluation:N", sort=EVALUATIONS, title=None,
                scale=alt.Scale(paddingOuter=0.15),
                axis=alt.Axis(labelAngle=0, domain=False, ticks=False, labelPadding=8)),
        y=alt.Y("macro_f1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".1f", tickCount=5, domain=False, ticks=False)),
        color=_ARCH_COLOR,
        detail="training:N",
        tooltip=[alt.Tooltip("model:N", title="Checkpoint"), alt.Tooltip("evaluation:N", title="Evaluated on"),
                 alt.Tooltip("macro_f1:Q", title="Macro-F1", format=".4f"),
                 alt.Tooltip("images:Q", title="Images")],
    )
    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        # The app draws a text key above the panels, so no Vega legend here.
        strokeDash=alt.StrokeDash("training:N", scale=alt.Scale(domain=TRAINING, range=[[1, 0], [5, 4]]),
                                  legend=None))
    points = base.mark_point(size=80, strokeWidth=2, filled=False).encode(
        fill=alt.condition(alt.datum.training == "Field-adapted",
                           alt.Color("architecture:N", scale=alt.Scale(domain=ARCHITECTURES)),
                           alt.value("transparent")))
    end = panel[panel.evaluation.eq("External v1")].assign(
        label=lambda f: f.training.map({"Field-adapted": "Adapted", "Lab only": "Lab only"})
        + " " + f.macro_f1.map("{:.2f}".format))
    # The higher end label sits above its point and the lower one below, so close
    # values (ViT-Small: 0.48 vs 0.42) never overlap.
    high = end.macro_f1.idxmax()
    labels = [
        alt.Chart(end.loc[[index]]).mark_text(align="right", dx=-4, dy=-12 if index == high else 14,
                                              fontSize=12, color=ink).encode(
            x=alt.X("evaluation:N", sort=EVALUATIONS), y="macro_f1:Q", text="label:N")
        for index in end.index
    ]
    return alt.layer(lines, points, *labels).properties(height=320, title=alt.Title(
        architecture, anchor="start", fontSize=14, fontWeight=600))


def tradeoff_chart(data: pd.DataFrame, ink: str = INK["light"]) -> alt.Chart:
    """Dumbbell per class: hollow = lab only, filled = field-adapted (ResNet18)."""
    order = [f"{name} (n={n})" for name, n in
             data.drop_duplicates("class_name").set_index("class_name").loc[CLASS_NAMES, "images"].items()]
    wide = data.pivot(index="row", columns="training", values="recall").reset_index()
    y = alt.Y("row:N", sort=order, title=None,
              axis=alt.Axis(domain=False, ticks=False, labelPadding=8, labelLimit=240))
    x = alt.X("recall:Q", title="Share of class images classified correctly", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", tickCount=5, domain=False, ticks=False))
    rule = alt.Chart(wide).mark_rule(strokeWidth=2, opacity=0.35).encode(
        y=y, x=alt.X("Lab only:Q", scale=alt.Scale(domain=[0, 1])), x2="Field-adapted:Q",
        color=alt.value(ink))
    dots = alt.Chart(data.assign(architecture="ResNet18")).mark_point(size=110, strokeWidth=2).encode(
        y=y, x=x, color=_ARCH_COLOR,
        fill=alt.condition(alt.datum.training == "Field-adapted",
                           alt.Color("architecture:N", scale=alt.Scale(domain=ARCHITECTURES)),
                           alt.value("transparent")),
        tooltip=[alt.Tooltip("class_name:N", title="Class"), alt.Tooltip("training:N", title="ResNet18"),
                 alt.Tooltip("correct:Q", title="Correct"), alt.Tooltip("images:Q", title="Images"),
                 alt.Tooltip("recall:Q", title="Share correct", format=".0%")],
    )
    return (rule + dots).properties(height=250)


def controlled_chart(data: pd.DataFrame, ink: str = INK["light"]) -> alt.Chart:
    """Dumbbell per seed and for the mean: Field-only (filled) vs Mixed (hollow)."""
    order = ["Seed 42", "Seed 43", "Seed 44", "Mean of 3"]
    wide = data.pivot(index="row", columns="condition", values="macro_f1").reset_index()
    y = alt.Y("row:N", sort=order, title=None, axis=alt.Axis(domain=False, ticks=False, labelPadding=8))
    scale = alt.Scale(domain=[0.36, 0.46])
    rule = alt.Chart(wide).mark_rule(strokeWidth=2, opacity=0.35).encode(
        y=y, x=alt.X("Field-only:Q", scale=scale), x2="Mixed:Q", color=alt.value(ink))
    dots = alt.Chart(data.assign(architecture="ResNet18")).mark_point(size=110, strokeWidth=2).encode(
        y=y,
        x=alt.X("macro_f1:Q", title="Macro-F1 on the historical Field Test (47 images)", scale=scale,
                axis=alt.Axis(format=".2f", tickCount=5, domain=False, ticks=False)),
        color=_ARCH_COLOR,
        fill=alt.condition(alt.datum.condition == "Field-only",
                           alt.Color("architecture:N", scale=alt.Scale(domain=ARCHITECTURES)),
                           alt.value("transparent")),
        tooltip=[alt.Tooltip("row:N", title="Run"), alt.Tooltip("condition:N", title="Fine-tuning"),
                 alt.Tooltip("macro_f1:Q", title="Macro-F1", format=".4f"),
                 alt.Tooltip("accuracy:Q", title="Accuracy", format=".4f")],
    )
    return (rule + dots).properties(height=200)
