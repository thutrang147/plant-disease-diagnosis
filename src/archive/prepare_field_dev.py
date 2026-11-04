"""Create a fixed PlantDoc development split without modifying the original manifests.

Run from the repository root:
    python src/archive/prepare_field_dev.py

Uses only data/splits/field_train.csv. Never uses or modifies field_test.csv.
The outputs are metadata manifests, not image copies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = ROOT / "data" / "splits"


def prepare_field_dev(splits_dir: Path, seed: int = 42, dev_fraction: float = 0.20) -> None:
    source_path = splits_dir / "field_train.csv"
    test_path = splits_dir / "field_test.csv"
    adapt_path = splits_dir / "field_adapt_train.csv"
    dev_path = splits_dir / "field_dev.csv"
    protocol_path = splits_dir / "field_dev_protocol.json"

    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if not source_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected field_train.csv and field_test.csv in data/splits/")

    field_train = pd.read_csv(source_path)
    field_test = pd.read_csv(test_path)
    required = {"source", "row_id", "class_name", "label_id"}
    for name, frame in [("field_train", field_train), ("field_test", field_test)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
        if frame.empty or frame[["source", "row_id", "label_id"]].isna().any().any():
            raise ValueError(f"Invalid rows in {name}")
        if not (frame["source"] == "plantdoc").all():
            raise ValueError(f"Unexpected image source in {name}")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"Duplicate image row IDs in {name}")

    expected_labels = set(range(5))
    if set(field_train["label_id"]) != expected_labels or set(field_test["label_id"]) != expected_labels:
        raise ValueError("Both input manifests must contain all five label IDs")
    if set(field_train["row_id"]) & set(field_test["row_id"]):
        raise ValueError("Field train/test manifests share image row IDs")

    adaptive, development = train_test_split(
        field_train,
        test_size=dev_fraction,
        random_state=seed,
        stratify=field_train["label_id"],
    )
    adaptive = adaptive.sort_values("row_id").reset_index(drop=True)
    development = development.sort_values("row_id").reset_index(drop=True)

    if len(adaptive) + len(development) != len(field_train):
        raise AssertionError("Some field training images were lost")
    if set(adaptive["row_id"]) & set(development["row_id"]):
        raise AssertionError("Development and adaptation-train manifests overlap")
    if set(adaptive["label_id"]) != expected_labels or set(development["label_id"]) != expected_labels:
        raise AssertionError("A class is missing from an output partition")

    protocol = {
        "source_manifest": "data/splits/field_train.csv",
        "adapt_train_manifest": "data/splits/field_adapt_train.csv",
        "development_manifest": "data/splits/field_dev.csv",
        "field_test_manifest": "data/splits/field_test.csv",
        "seed": seed,
        "dev_fraction_requested": dev_fraction,
        "strategy": "Stratified random split using label_id",
        "field_test_used": False,
        "caveat": "No duplicate-image grouping is available in the input manifest. Audit exact/near-duplicates across partitions before field adaptation and confirmatory reporting.",
    }

    # Do not silently replace a previously committed experimental split.
    desired_csvs = {adapt_path: adaptive.to_csv(index=False), dev_path: development.to_csv(index=False)}
    desired_json = json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    for path, contents in [*desired_csvs.items(), (protocol_path, desired_json)]:
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise FileExistsError(f"Existing split would change: {path}. Review before overwriting.")

    for path, contents in desired_csvs.items():
        path.write_text(contents, encoding="utf-8")
    protocol_path.write_text(desired_json, encoding="utf-8")

    print("PlantDoc development split (original field_test unchanged):")
    combined = pd.concat([
        adaptive.assign(partition="field_adapt_train"),
        development.assign(partition="field_dev"),
    ])
    print(pd.crosstab(combined["class_name"], combined["partition"]))
    print(f"\nField adaptation train: {len(adaptive)} images")
    print(f"Field development:      {len(development)} images")
    print(f"Original field test:    {len(field_test)} images (untouched)")
    print(f"Saved manifests to:     {splits_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    args = parser.parse_args()
    prepare_field_dev(args.splits_dir, args.seed, args.dev_fraction)


if __name__ == "__main__":
    main()
