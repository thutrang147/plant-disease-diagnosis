"""Regenerate PlantDoc adaptation/development manifests without image-content leakage.

Run from the repository root after installing `datasets`, `pandas`, and `scikit-learn`:
    python src/archive/prepare_field_dev_grouped.py --replace-existing

- Only uses PlantDoc's original training partition for model-development splitting.
- Groups exact pixel-identical images by SHA-256 of (image size, RGB pixels).
- Checks pixel hashes of the original Field Test for contamination; any matching
  training images are quarantined, without changing the original Field Test.
- Does not train a model or select partitions based on model predictions.
- Backups previous derived manifests when --replace-existing is specified.

The resulting image counts may differ slightly from 380/96 because groups must
remain intact. Potential near-duplicates still require manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
SOURCE_DATASET = "geraldmc/plantdoc-full"
SOURCE_REVISION = "v0.1.0"


def rgb_content_hash(pil_image) -> str:
    image = pil_image.convert("RGB")
    # Deliberately identical to the check in notebooks/archive/06_field_adaptation.ipynb.
    return hashlib.sha256(f"{image.size}".encode("utf-8") + image.tobytes()).hexdigest()


def attach_hashes(frame: pd.DataFrame, dataset, partition: str) -> pd.DataFrame:
    result = frame.copy()
    hashes = []
    for row in frame.itertuples(index=False):
        sample = dataset[int(row.row_id)]
        if sample["class_label"] != row.original_class:
            raise ValueError(
                f"{partition}: row_id={row.row_id} class changed in upstream "
                "data. Check the dataset revision and manifest."
            )
        hashes.append(rgb_content_hash(sample["image"]))
    result["exact_hash"] = hashes
    return result


def choose_grouped_split(frame: pd.DataFrame, dev_fraction: float, seed: int, n_candidates: int):
    """Choose a deterministic group-disjoint candidate with balanced label counts."""
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    if len(frame) < 20:
        raise ValueError("Too few eligible images to construct a development set")

    labels = frame["label_id"].to_numpy(dtype=int)
    groups = frame["exact_hash"].to_numpy()
    target_dev = round(len(frame) * dev_fraction)
    classes = sorted(frame["label_id"].unique().tolist())
    full_counts = frame["label_id"].value_counts().to_dict()
    splitter = GroupShuffleSplit(
        n_splits=n_candidates, test_size=dev_fraction, random_state=seed
    )

    best = None
    for train_idx, dev_idx in splitter.split(frame, labels, groups=groups):
        in_train = set(labels[train_idx])
        in_dev = set(labels[dev_idx])
        if not (in_train == in_dev == set(classes)):
            continue
        observed = pd.Series(labels[dev_idx]).value_counts().to_dict()
        size_penalty = 20.0 * abs(len(dev_idx) - target_dev)
        balance_penalty = sum(
            abs(observed.get(c, 0) - full_counts[c] * dev_fraction) for c in classes
        )
        score = size_penalty + balance_penalty
        if best is None or score < best[0]:
            best = (score, train_idx, dev_idx)

    if best is None:
        raise ValueError("Could not build a five-class grouped split; examine class/group counts")
    _, train_idx, dev_idx = best
    adapt = frame.iloc[train_idx].sort_values("row_id").reset_index(drop=True)
    dev = frame.iloc[dev_idx].sort_values("row_id").reset_index(drop=True)
    assert set(adapt["exact_hash"]).isdisjoint(set(dev["exact_hash"]))
    assert set(adapt["row_id"]).isdisjoint(set(dev["row_id"]))
    return adapt, dev


def replace_derived_outputs(outputs: dict[Path, str], replace_existing: bool, splits_dir: Path):
    present = [p for p in outputs if p.exists()]
    if present and not replace_existing:
        raise FileExistsError(
            "Derived manifests already exist. Inspect changes and re-run with "
            "--replace-existing to back them up and write grouped versions: "
            + ", ".join(map(str, present))
        )
    if present:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = splits_dir / "split_backups" / stamp
        backup.mkdir(parents=True, exist_ok=False)
        for path in present:
            shutil.copy2(path, backup / path.name)
        print("Backed up previous derived manifests to:", backup)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")


def prepare(splits_dir: Path, seed: int, dev_fraction: float, n_candidates: int, replace_existing: bool):
    # Imported lazily so the partition algorithm can be tested without downloading a dataset.
    from datasets import load_dataset

    train_path = splits_dir / "field_train.csv"
    test_path = splits_dir / "field_test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Expected original field_train.csv and field_test.csv")
    original_train = pd.read_csv(train_path)
    original_test = pd.read_csv(test_path)
    required = {"source", "row_id", "original_class", "class_name", "label_id"}
    for name, frame in [("field_train", original_train), ("field_test", original_test)]:
        if required - set(frame.columns):
            raise ValueError(f"{name} missing columns: {sorted(required - set(frame.columns))}")
        if not (frame["source"] == "plantdoc").all():
            raise ValueError(f"{name} contains another source")
        if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
            raise ValueError(f"{name} has missing or duplicated row_ids")
    if not set(original_train["row_id"]).isdisjoint(set(original_test["row_id"])):
        raise ValueError("Original Field Train and Test have overlapping row_ids")

    print(f"Reading PlantDoc images ({SOURCE_DATASET}@{SOURCE_REVISION})...")
    dataset = load_dataset(SOURCE_DATASET, split="train", revision=SOURCE_REVISION)
    train = attach_hashes(original_train, dataset, "field_train")
    test = attach_hashes(original_test, dataset, "field_test")

    # A pixel-identical image with two different labels is an annotation conflict,
    # not a valid clean group. Require manual review rather than silently selecting one.
    group_sizes = train.groupby("exact_hash").size()
    repeated_groups = int((group_sizes > 1).sum())
    repeated_rows = int((group_sizes[group_sizes > 1] - 1).sum())
    print(f"Found {repeated_groups} exact-duplicate image groups "
          f"({repeated_rows} additional copies) within original Field Train.")
    label_conflicts = train.groupby("exact_hash")["label_id"].nunique()
    conflicts = set(label_conflicts[label_conflicts > 1].index)
    if conflicts:
        print(train.loc[train["exact_hash"].isin(conflicts),
                        ["row_id", "class_name", "exact_hash"]].to_string(index=False))
        raise ValueError("Identical training images have conflicting class labels")

    # Metadata-only hash audit against the original test. No model inference,
    # test label inspection, or test-based checkpoint/augmentation selection.
    test_hashes = set(test["exact_hash"])
    quarantined = train.loc[train["exact_hash"].isin(test_hashes)].copy()
    eligible = train.loc[~train["exact_hash"].isin(test_hashes)].copy()
    if len(quarantined):
        print(f"WARNING: {len(quarantined)} original field-training rows are "
              "exact pixel duplicates of Field Test images. Quarantining them.")
    if len(eligible) + len(quarantined) != len(original_train):
        raise AssertionError("Images lost unexpectedly before split")

    adapt, dev = choose_grouped_split(eligible, dev_fraction, seed, n_candidates)
    if len(adapt) + len(dev) + len(quarantined) != len(original_train):
        raise AssertionError("Split does not account for every source training row")
    assert set(adapt["exact_hash"]).isdisjoint(set(dev["exact_hash"]))
    assert set(adapt["exact_hash"]).isdisjoint(test_hashes)
    assert set(dev["exact_hash"]).isdisjoint(test_hashes)

    protocol = {
        "source_manifest": "data/splits/field_train.csv",
        "adapt_train_manifest": "data/splits/field_adapt_train.csv",
        "development_manifest": "data/splits/field_dev.csv",
        "field_test_manifest": "data/splits/field_test.csv",
        "strategy": "Exact RGB pixel SHA-256 grouping with class-balanced GroupShuffleSplit selection",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "seed": seed,
        "dev_fraction_requested": dev_fraction,
        "split_candidates": n_candidates,
        "original_field_train_images": len(original_train),
        "adaptation_train_images": len(adapt),
        "development_images": len(dev),
        "test_images_untouched": len(original_test),
        "quarantined_field_train_images_matching_test": len(quarantined),
        "exact_duplicate_groups_in_original_field_train": repeated_groups,
        "additional_exact_copies_in_original_field_train": repeated_rows,
        "exact_group_overlap_adapt_dev": 0,
        "exact_group_overlap_adapt_test": 0,
        "exact_group_overlap_dev_test": 0,
        "field_test_used_for_model_selection": False,
        "notes": "Counts may differ from the prior 380/96 split. Near-duplicates require manual review. Rerun field development inference for all source checkpoints before adaptation.",
    }
    outputs = {
        splits_dir / "field_adapt_train.csv": adapt.to_csv(index=False),
        splits_dir / "field_dev.csv": dev.to_csv(index=False),
        splits_dir / "field_dev_protocol.json": json.dumps(protocol, indent=2) + "\n",
        splits_dir / "field_quarantined_train.csv": quarantined.to_csv(index=False),
    }
    replace_derived_outputs(outputs, replace_existing, splits_dir)

    summary = pd.concat([
        adapt.assign(partition="field_adapt_train"),
        dev.assign(partition="field_dev"),
    ], ignore_index=True)
    print("\nExact duplicate groups across Field Adapt / Dev: 0")
    print(pd.crosstab(summary["class_name"], summary["partition"]))
    print(f"\nAdapt train: {len(adapt)} | Field dev: {len(dev)} | "
          f"Quarantined: {len(quarantined)} | Original field test unchanged: {len(test)}")
    print("Next: commit new CSVs/protocol, rerun 3-model Field Dev evaluation, then adapt.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--n-candidates", type=int, default=1000)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    prepare(args.splits_dir, args.seed, args.dev_fraction, args.n_candidates, args.replace_existing)


if __name__ == "__main__":
    main()
