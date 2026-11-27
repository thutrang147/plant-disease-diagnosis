"""Audit cross-split near-duplicates in the PlantVillage Lab manifests.

Lab splits were grouped by PlantVillage ``leaf_id``, but 1,510 / 378 / 465
train / val / test images have no reliable leaf group and form their own group,
so same-leaf near-duplicates across partitions were not ruled out. This script:

1. fingerprints every Lab Train / Validation / Test row (raw-RGB SHA-256 and
   8x8 DCT pHash, as in ``src/evaluate_confirmatory_field.py``);
2. flags each Validation / Test image whose nearest Lab Train image is an exact
   match or within pHash distance ``THRESHOLD`` (4, the repository's rule);
3. re-evaluates the fixed and Notebook 15 checkpoints on Lab Test with and
   without the flagged images.

Inference only, CPU, float32. Usage from the repository root::

    python scripts/audit_lab_split_duplicates.py --output data/results/lab_split_duplicates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.inference import load_checkpoint, registry, sha256  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

THRESHOLD = 4
MODELS = ("resnet18_lab", "vit_small_lab", "resnet18_geometric_mixed", "vit_small_mixed",
          "nb15_field_only_s42", "nb15_mixed_s42", "nb15_field_only_s43", "nb15_mixed_s43",
          "nb15_field_only_s44", "nb15_mixed_s44")


def fingerprints(name: str) -> tuple[pd.DataFrame, PlantDiseaseDataset]:
    dataset = PlantDiseaseDataset(ROOT / "data/splits" / f"{name}.csv", transform=lambda img: img)
    exact, phash = [], []
    for i in range(len(dataset)):
        rgb = dataset[i][0].convert("RGB")
        exact.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        phash.append(int(str(imagehash.phash(rgb)), 16))
    frame = dataset.metadata.copy()
    frame["exact_hash"], frame["phash_int"] = exact, phash
    return frame, dataset


def nearest(values: np.ndarray, references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances, index = np.empty(len(values), int), np.empty(len(values), int)
    for i, value in enumerate(values):
        bits = np.unpackbits(np.bitwise_xor(references, value).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        index[i] = int(bits.argmin())
        distances[i] = int(bits[index[i]])
    return distances, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    frames, datasets = {}, {}
    for name in ("lab_train", "lab_val", "lab_test"):
        frames[name], datasets[name] = fingerprints(name)
        print("hashed", name, len(frames[name]), flush=True)
    train = frames["lab_train"]
    train_phash = train.phash_int.to_numpy(dtype=np.uint64)
    train_exact = set(train.exact_hash)

    flagged, summary = [], []
    for name in ("lab_val", "lab_test"):
        frame = frames[name]
        dist, idx = nearest(frame.phash_int.to_numpy(dtype=np.uint64), train_phash)
        frame["min_train_distance"] = dist
        frame["nearest_train_row"] = train.row_id.to_numpy()[idx]
        frame["nearest_train_class"] = train.class_name.to_numpy()[idx]
        frame["exact_in_train"] = frame.exact_hash.isin(train_exact)
        frame["flag"] = frame.exact_in_train | (frame.min_train_distance <= THRESHOLD)
        grouped = frame["leaf_grouped"].astype(bool) if "leaf_grouped" in frame else pd.Series(True, index=frame.index)
        summary.append({"partition": name, "images": len(frame), "flagged": int(frame.flag.sum()),
                        "flagged_exact": int(frame.exact_in_train.sum()),
                        "flagged_leaf_grouped": int((frame.flag & grouped).sum()),
                        "flagged_ungrouped": int((frame.flag & ~grouped).sum()),
                        "flagged_label_differs": int((frame.flag & frame.class_name.ne(frame.nearest_train_class)).sum())})
        flagged.append(frame[frame.flag][["row_id", "class_name", "leaf_grouped", "min_train_distance",
                                          "nearest_train_row", "nearest_train_class", "exact_in_train"]]
                       .assign(partition=name))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)

    test = frames["lab_test"]
    clean = ~test.flag.to_numpy()
    y = test.label_id.to_numpy()
    entries = {e["key"]: e for e in registry()}
    vit_config = None
    rows, cache = [], {}
    images = [datasets["lab_test"][i][0].convert("RGB") for i in range(len(y))]  # 256x256 PlantVillage, small
    for key in MODELS:
        path = ROOT / entries[key]["local_path"]
        if sha256(path) != entries[key]["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {key}")
        model, transform, meta = load_checkpoint(path, vit_config)
        if key == "vit_small_lab":
            vit_config = meta.get("preprocessing_config")
        kind = meta["architecture"]  # one cached tensor stack per preprocessing pipeline
        if kind not in cache:
            cache[kind] = torch.stack([transform(img) for img in images])
        x = cache[kind]
        with torch.inference_mode():
            p = torch.cat([model(x[i:i + 64]) for i in range(0, len(x), 64)]).argmax(1).numpy()
        for subset, mask in (("all", np.ones_like(clean)), ("clean", clean), ("flagged", ~clean)):
            if mask.sum():
                rows.append({"model": key, "subset": subset, "n": int(mask.sum()),
                             "correct": int((p[mask] == y[mask]).sum()), "accuracy": float((p[mask] == y[mask]).mean()),
                             "macro_f1": macro_f1(y[mask], p[mask], range(5))})
        print("evaluated", key, flush=True)

    output.mkdir(parents=True)
    pd.DataFrame(summary).to_csv(output / "duplicate_summary.csv", index=False)
    pd.concat(flagged).to_csv(output / "flagged_rows.csv", index=False)
    results = pd.DataFrame(rows)
    results.to_csv(output / "lab_test_clean_vs_all.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "fingerprints": "raw-RGB SHA-256; imagehash.phash", "threshold": THRESHOLD,
        "reference": "lab_train", "plantvillage_revision": "pinned in src/dataset.py",
        "torch": torch.__version__}, indent=2) + "\n")
    print(results.pivot_table(index="model", columns="subset", values="macro_f1").round(4).to_string())


if __name__ == "__main__":
    main()
