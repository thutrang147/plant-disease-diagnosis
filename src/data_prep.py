"""Create reproducible metadata splits for five-class lab-to-field experiments.

Usage (from the repository root, on the Colab runtime):
    python src/data_prep.py

The generated CSVs contain image row indices and labels, not image pixels.
Dataset images are loaded separately when training a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "splits"
SEED = 42
VAL_FRACTION = 0.20

# Fixed label order for every CNN and ViT experiment.
CLASS_NAMES = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Bacterial Spot",
    "Septoria Leaf Spot",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

LAB_LABEL_MAP = {
    "Tomato___healthy": "Healthy",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
}

FIELD_LABEL_MAP = {
    "Tomato leaf": "Healthy",
    "Tomato Early blight leaf": "Early Blight",
    "Tomato leaf late blight": "Late Blight",
    "Tomato leaf bacterial spot": "Bacterial Spot",
    "Tomato Septoria leaf spot": "Septoria Leaf Spot",
}


def build_metadata(dataset, source: str, label_map: dict[str, str]) -> pd.DataFrame:
    """Select target classes while preserving original Hugging Face row indices."""
    available = set(dataset["class_label"])
    missing = set(label_map) - available
    if missing:
        raise ValueError(f"Missing {source} classes: {sorted(missing)}")

    metadata = pd.DataFrame({
        "source": source,
        "row_id": np.arange(len(dataset), dtype=np.int64),
        "original_class": dataset["class_label"],
        "original_split": dataset["split"],
    })
    if source == "plantvillage":
        metadata["leaf_id"] = dataset["leaf_id"]
        metadata["leaf_grouped"] = dataset["leaf_grouped"]

    metadata = metadata.loc[
        metadata["original_class"].isin(label_map)
    ].copy()
    metadata["class_name"] = metadata["original_class"].map(label_map)
    metadata["label_id"] = metadata["class_name"].map(CLASS_TO_ID).astype(int)

    unexpected = set(metadata["original_split"]) - {"train", "test"}
    if unexpected:
        raise ValueError(f"Unexpected {source} splits: {sorted(unexpected)}")
    return metadata


def make_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Group known same-leaf photos; handle ungrouped photos as singletons."""
    keys = []
    for row in frame.itertuples(index=False):
        known = bool(row.leaf_grouped) and pd.notna(row.leaf_id) and bool(str(row.leaf_id).strip())
        if known:
            keys.append(f"leaf:{row.leaf_id}")
        else:
            keys.append(f"unknown_row:{row.row_id}")
    return pd.Series(keys, index=frame.index)


def split_lab(lab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preserve original lab test and group-split only original lab train."""
    original_train = lab.loc[lab["original_split"] == "train"].copy()
    original_test = lab.loc[lab["original_split"] == "test"].copy()

    original_train["group_id"] = make_group_ids(original_train)
    original_test["group_id"] = make_group_ids(original_test)

    # Original grouped leaves should not appear in both partitions.
    known_train = set(original_train.loc[original_train["leaf_grouped"], "leaf_id"])
    known_test = set(original_test.loc[original_test["leaf_grouped"], "leaf_id"])
    if known_train & known_test:
        raise ValueError("Original Lab Train/Test share reliable leaf IDs.")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = []
    class_counts = original_train["label_id"].value_counts().reindex(range(5), fill_value=0)

    for train_pos, val_pos in splitter.split(
        original_train,
        y=original_train["label_id"],
        groups=original_train["group_id"],
    ):
        val = original_train.iloc[val_pos]
        per_class_val = val["label_id"].value_counts().reindex(range(5), fill_value=0)
        class_ratio = per_class_val / class_counts.replace(0, np.nan)
        fraction_error = abs(len(val) / len(original_train) - VAL_FRACTION)
        class_error = (class_ratio - VAL_FRACTION).abs().mean()
        candidates.append((float(fraction_error + class_error), train_pos, val_pos))

    _, train_pos, val_pos = min(candidates, key=lambda item: item[0])
    train = original_train.iloc[train_pos].copy()
    val = original_train.iloc[val_pos].copy()
    test = original_test.copy()

    train["split"] = "lab_train"
    val["split"] = "lab_val"
    test["split"] = "lab_test"

    if set(train["group_id"]) & set(val["group_id"]):
        raise AssertionError("Same-leaf groups overlap between Lab Train and Validation.")
    if set(train["row_id"]) & set(val["row_id"]):
        raise AssertionError("Lab Train and Validation share image rows.")
    if len(train) + len(val) + len(test) != len(lab):
        raise AssertionError("A lab image was lost or duplicated during splitting.")

    return train, val, test


def validate_and_save_partitions(partitions: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, frame in partitions.items():
        if frame.empty:
            raise ValueError(f"{split_name} is empty.")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"Duplicate row IDs in {split_name}.")
        if set(frame["class_name"]) != set(CLASS_NAMES):
            raise ValueError(f"Not all five classes are present in {split_name}.")
        frame.sort_values("row_id").to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)

    print("\nImage counts by split and standardized class:")
    combined = pd.concat(partitions.values(), ignore_index=True)
    print(pd.crosstab(combined["class_name"], combined["split"]).reindex(CLASS_NAMES))
    print("\nTotal images:", len(combined))
    print("Manifests saved to:", OUTPUT_DIR)

    (OUTPUT_DIR / "label_map.json").write_text(
        json.dumps(CLASS_TO_ID, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "dataset_sources.json").write_text(
        json.dumps({
            "plantvillage": {"repository": "geraldmc/plantvillage-full", "requested_revision": "default"},
            "plantdoc": {"repository": "geraldmc/plantdoc-full", "requested_revision": "v0.1.0"},
            "seed": SEED,
            "validation_strategy": "StratifiedGroupKFold: selected fold closest to 20% of original PlantVillage Train",
            "note": "Row IDs refer to the loading order of these datasets. Pin resolved dataset commit SHAs before a published experiment.",
        }, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    lab_dataset = load_dataset("geraldmc/plantvillage-full", split="train")
    field_dataset = load_dataset("geraldmc/plantdoc-full", revision="v0.1.0", split="train")

    lab = build_metadata(lab_dataset, "plantvillage", LAB_LABEL_MAP)
    field = build_metadata(field_dataset, "plantdoc", FIELD_LABEL_MAP)

    lab_train, lab_val, lab_test = split_lab(lab)
    field_train = field.loc[field["original_split"] == "train"].copy()
    field_test = field.loc[field["original_split"] == "test"].copy()
    field_train["split"] = "field_train"
    field_test["split"] = "field_test"

    partitions = {
        "lab_train": lab_train,
        "lab_val": lab_val,
        "lab_test": lab_test,
        "field_train": field_train,
        "field_test": field_test,
    }
    validate_and_save_partitions(partitions)

    print("\nImportant: field_train must NOT be used for the initial zero-shot lab-to-field baseline.")
    print("Important: review watermarked/text-overlaid PlantDoc images before final interpretation.")


if __name__ == "__main__":
    main()
