"""Prepare PlantDoc field adaptation/dev split with near-duplicate protection.

Run locally, from the repo root:
    python src/prepare_field_dev_verified.py --replace-existing

Uses pHash Hamming distance <= 4 to create connected image groups across the
original PlantDoc field_train + field_test manifests (no model inference).
Training groups near Field Test and groups with contradictory disease labels
are quarantined, not silently relabeled. Keeps original Field Test intact.
This procedure is conservative: review candidate near-duplicate pairs manually.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
SOURCE = "geraldmc/plantdoc-full"
REVISION = "v0.1.0"
THRESHOLD = 4


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def exact_hash(img):
    rgb = img.convert("RGB")
    return hashlib.sha256(f"{rgb.size}".encode("utf8") + rgb.tobytes()).hexdigest()


def annotate(original, dataset, partition):
    df = original.copy().reset_index(drop=True)
    hashes, phashes = [], []
    for row in df.itertuples(index=False):
        record = dataset[int(row.row_id)]
        if record["class_label"] != row.original_class:
            raise ValueError(f"{partition}: source class mismatch for row_id {row.row_id}")
        img = record["image"].convert("RGB")
        hashes.append(exact_hash(img))
        phashes.append(imagehash.phash(img))
    df["exact_hash"] = hashes
    df["phash"] = [str(p) for p in phashes]
    df["original_partition"] = partition
    return df


def build_groups(all_rows, threshold):
    hashes = [imagehash.hex_to_hash(s) for s in all_rows.phash]
    uf = UnionFind(len(all_rows))
    matches = []
    for i in range(len(all_rows)):
        for j in range(i+1, len(all_rows)):
            d = hashes[i] - hashes[j]
            if d <= threshold:
                uf.union(i, j)
                matches.append({
                    "partition_a": all_rows.iloc[i].original_partition,
                    "row_id_a": int(all_rows.iloc[i].row_id),
                    "class_a": str(all_rows.iloc[i].class_name),
                    "partition_b": all_rows.iloc[j].original_partition,
                    "row_id_b": int(all_rows.iloc[j].row_id),
                    "class_b": str(all_rows.iloc[j].class_name),
                    "phash_distance": int(d),
                    "same_class": bool(all_rows.iloc[i].label_id == all_rows.iloc[j].label_id),
                })
    # Anchor each component to its smallest source row id. The original train/test
    # subsets have disjoint row IDs, so these identifiers remain reproducible.
    members = {}
    for i in range(len(all_rows)):
        members.setdefault(uf.find(i), []).append(i)
    ids = {root: f"phash_group_{min(int(all_rows.iloc[i].row_id) for i in idxs)}"
           for root, idxs in members.items()}
    result = all_rows.copy()
    result["group_id"] = [ids[uf.find(i)] for i in range(len(result))]
    return result, pd.DataFrame(matches, columns=[
        "partition_a", "row_id_a", "class_a", "partition_b", "row_id_b",
        "class_b", "phash_distance", "same_class"
    ])


def choose_split(frame, dev_fraction, seed, candidates):
    if not 0 < dev_fraction < 1:
        raise ValueError("dev-fraction must lie strictly between 0 and 1")
    labels = frame.label_id.to_numpy(dtype=int)
    groups = frame.group_id.to_numpy()
    class_set = set(labels)
    if len(class_set) != 5:
        raise ValueError(f"After quarantine, all five classes are needed; got {class_set}")
    full_count = frame.label_id.value_counts().to_dict()
    requested = round(len(frame) * dev_fraction)
    splitter = GroupShuffleSplit(n_splits=candidates, test_size=dev_fraction, random_state=seed)
    best = None
    for train_idx, dev_idx in splitter.split(frame, labels, groups):
        if set(labels[train_idx]) != class_set or set(labels[dev_idx]) != class_set:
            continue
        dev_count = pd.Series(labels[dev_idx]).value_counts().to_dict()
        score = (20 * abs(len(dev_idx) - requested) +
                 sum(abs(dev_count.get(c, 0) - full_count[c] * dev_fraction) for c in class_set))
        if best is None or score < best[0]:
            best = score, train_idx, dev_idx
    if best is None:
        raise ValueError("Could not construct a five-class pHash-grouped split")
    _, a, d = best
    train = frame.iloc[a].sort_values("row_id").reset_index(drop=True)
    dev = frame.iloc[d].sort_values("row_id").reset_index(drop=True)
    assert set(train.group_id).isdisjoint(dev.group_id)
    return train, dev


def backup_and_write(outputs, replace, splits):
    existing = [p for p in outputs if p.exists()]
    if existing and not replace:
        raise FileExistsError("Outputs already exist; specify --replace-existing to back them up")
    if existing:
        backup = splits / "split_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, backup / path.name)
        print("Old manifests backed up at:", backup)
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf8")


def prepare(splits, seed, dev_fraction, candidates, threshold, replace):
    original_train = pd.read_csv(splits / "field_train.csv")
    original_test = pd.read_csv(splits / "field_test.csv")
    required = {"source", "row_id", "original_class", "class_name", "label_id"}
    for name, df in (("field_train", original_train), ("field_test", original_test)):
        if not required.issubset(df.columns):
            raise ValueError(f"{name}: missing manifest columns: {sorted(required-set(df.columns))}")
        if not (df.source == "plantdoc").all() or df.row_id.duplicated().any():
            raise ValueError(f"{name}: unexpected source or repeated row_id")
    if set(original_train.row_id) & set(original_test.row_id):
        raise ValueError("Train and Test have overlapping row_id")

    print(f"Reading source images: {SOURCE}@{REVISION}")
    ds = load_dataset(SOURCE, revision=REVISION, split="train")
    train = annotate(original_train, ds, "field_train")
    test = annotate(original_test, ds, "field_test")
    all_rows, pair_review = build_groups(pd.concat([train, test], ignore_index=True), threshold)
    train_rows = all_rows[all_rows.original_partition == "field_train"].copy()
    test_rows = all_rows[all_rows.original_partition == "field_test"].copy()

    test_groups = set(test_rows.group_id)
    label_counts = train_rows.groupby("group_id").label_id.nunique()
    conflict_groups = set(label_counts[label_counts > 1].index)
    # Conflict groups are excluded even if someone could guess a majority label.
    near_test = train_rows.group_id.isin(test_groups)
    conflict = train_rows.group_id.isin(conflict_groups)
    quarantined = train_rows[near_test | conflict].copy()
    quarantined["quarantine_reason"] = [
        "near_field_test_and_label_conflict" if t and c else
        "near_field_test" if t else "label_conflict"
        for t, c in zip(near_test[near_test | conflict], conflict[near_test | conflict])
    ]
    eligible = train_rows[~(near_test | conflict)].copy().reset_index(drop=True)
    adapt, dev = choose_split(eligible, dev_fraction, seed, candidates)

    # All pairs with d <= threshold have been joined into the same component;
    # any edge between adapted training and dev/test would be a graph bug.
    assert set(adapt.group_id).isdisjoint(dev.group_id)
    assert set(adapt.group_id).isdisjoint(test_groups)
    assert set(dev.group_id).isdisjoint(test_groups)
    assert set(adapt.group_id).isdisjoint(conflict_groups)
    assert set(dev.group_id).isdisjoint(conflict_groups)
    assert len(adapt) + len(dev) + len(quarantined) == len(original_train)

    protocol = {
        "strategy": "Perceptual pHash connected-component grouping, Hamming distance <= threshold",
        "phash_threshold": int(threshold),
        "source_dataset": SOURCE, "source_revision": REVISION,
        "seed": seed, "dev_fraction_requested": dev_fraction,
        "split_candidates": candidates,
        "original_field_train_images": len(original_train),
        "original_field_test_images_untouched": len(original_test),
        "adaptation_train_images": len(adapt), "development_images": len(dev),
        "quarantined_total_train_images": len(quarantined),
        "quarantined_near_test_rows": int(near_test.sum()),
        "quarantined_label_conflict_rows": int(conflict.sum()),
        "conflicting_groups": len(conflict_groups),
        "near_duplicate_candidate_pairs": len(pair_review),
        "train_dev_group_overlap": 0,
        "train_test_group_overlap": 0,
        "dev_test_group_overlap": 0,
        "test_inference_or_test_based_model_selection": False,
        "caution": "Conservative pHash grouping; visually audit candidate pairs and label conflicts. Field Test was previously inspected, so obtain a truly independent test for confirmatory claims."
    }
    outputs = {
        splits / "field_adapt_train.csv": adapt.to_csv(index=False),
        splits / "field_dev.csv": dev.to_csv(index=False),
        splits / "field_quarantined_train.csv": quarantined.to_csv(index=False),
        splits / "field_dev_protocol.json": json.dumps(protocol, indent=2) + "\n",
        splits / "field_near_duplicate_candidates.csv": pair_review.to_csv(index=False),
        splits / "field_label_conflicts.csv": train_rows[train_rows.group_id.isin(conflict_groups)].to_csv(index=False),
    }
    backup_and_write(outputs, replace, splits)
    print("\nNear-duplicate candidates (all original train+test):", len(pair_review))
    print("Conflicting-label groups:", len(conflict_groups))
    print("Quarantined training images:", len(quarantined))
    print("  Near original field test:", int(near_test.sum()))
    print("  Conflicting labels:", int(conflict.sum()))
    print("  (some images may satisfy both conditions)")
    print("\nClass distribution:")
    print(pd.crosstab(
        pd.concat([adapt.assign(partition="adapt_train"), dev.assign(partition="dev")], ignore_index=True).class_name,
        pd.concat([adapt.assign(partition="adapt_train"), dev.assign(partition="dev")], ignore_index=True).partition,
    ))
    print(f"\nTrain {len(adapt)} | Dev {len(dev)} | Quarantine {len(quarantined)} | Original Test {len(test_rows)} (unchanged)")
    print("Group overlap (Train/Dev/Test): 0")
    print("Review data/splits/field_near_duplicate_candidates.csv and field_label_conflicts.csv before publication.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-dir", type=Path, default=SPLITS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.2)
    p.add_argument("--n-candidates", type=int, default=1000)
    p.add_argument("--phash-threshold", type=int, default=THRESHOLD)
    p.add_argument("--replace-existing", action="store_true")
    a = p.parse_args()
    if a.phash_threshold < 0 or a.phash_threshold > 8:
        p.error("--phash-threshold must be between 0 and 8 (conservative range)")
    prepare(a.splits_dir, a.seed, a.dev_fraction, a.n_candidates, a.phash_threshold, a.replace_existing)


if __name__ == "__main__":
    main()
