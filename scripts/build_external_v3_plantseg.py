"""Build External v3: a duplicate-screened PlantSeg tomato-disease evaluation set.

PlantSeg (Wei et al., arXiv:2409.04038) is an in-the-wild plant-disease
segmentation dataset with expert-reviewed lesion masks. Four of its tomato
categories map onto this project's label set (Healthy is not represented). No
model in this repository was trained or selected on PlantSeg.

Both PlantSeg and PlantDoc were assembled from web images, so the set is screened
before any evaluation:

* every candidate is fingerprinted (raw-RGB SHA-256 and 8x8 DCT pHash, the same
  fingerprints as ``src/evaluate_confirmatory_field.py``);
* references are **all** PlantDoc rows (any crop) and **all** PlantVillage tomato
  rows, not only the rows used by this repository's manifests;
* a candidate is excluded when it is an exact match or within pHash distance
  ``REFERENCE_DISTANCE`` (10, stricter than the repository's training screen of 4)
  of any reference image;
* within PlantSeg, near-duplicate groups (distance <= ``INTERNAL_DISTANCE``) keep
  their lowest row only.

Only row indices, hashes and derived statistics are written; no images. PlantSeg
is distributed under CC BY-NC-ND 4.0.

Usage from the repository root::

    python scripts/build_external_v3_plantseg.py --output data/external_v3_plantseg
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
from datasets import load_dataset
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

PLANTSEG = ("LFTTim/plantseg_disease_segmentation", "2fff5d63968d3fd0aca061e7348b0b1441c1535d")
PLANTDOC = ("geraldmc/plantdoc-full", "v0.1.0")
PLANTVILLAGE = ("geraldmc/plantvillage-full", "f808f2706d3ce50ce14652d2b6863d3cc30cf9c4")
CLASS_MAP = {
    "tomato early blight": "Early Blight",
    "tomato late blight": "Late Blight",
    "tomato bacterial leaf spot": "Bacterial Spot",
    "tomato septoria leaf spot": "Septoria Leaf Spot",
}
REFERENCE_DISTANCE = 10
INTERNAL_DISTANCE = 4


def fingerprint(image: Image.Image) -> tuple[str, int]:
    """Raw-RGB SHA-256 and 64-bit pHash, matching the repository's screening."""
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest(), int(str(imagehash.phash(rgb)), 16)


def hamming_to_all(value: int, references: np.ndarray) -> np.ndarray:
    """Hamming distances between one 64-bit hash and an array of uint64 hashes."""
    xor = np.bitwise_xor(references, np.uint64(value))
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def internal_groups(phashes: list[int], threshold: int) -> list[int]:
    """Union-find over pairs within ``threshold``; returns each row's group root."""
    parent = list(range(len(phashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    array = np.asarray(phashes, dtype=np.uint64)
    for i, value in enumerate(phashes):
        close = np.nonzero(hamming_to_all(value, array[i + 1:]) <= threshold)[0]
        for j in close + i + 1:
            a, b = find(i), find(int(j))
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(len(phashes))]


def repo_identities() -> dict[tuple[str, int], tuple[str, str]]:
    """(source, row_id) -> (manifest, class_name) for every repository manifest row."""
    identities = {}
    for path in sorted((ROOT / "data/splits").glob("*.csv")):
        frame = pd.read_csv(path)
        if {"source", "row_id", "class_name"} <= set(frame.columns):
            for source, row_id, label in frame[["source", "row_id", "class_name"]].itertuples(index=False):
                identities.setdefault((source, int(row_id)), (path.stem, label))
    return identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    label_ids = json.loads((ROOT / "data/splits/label_map.json").read_text())

    plantseg = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    candidates = []
    for index, disease in enumerate(plantseg["disease"]):
        if disease not in CLASS_MAP:
            continue
        row = plantseg[index]
        image = ImageOps.exif_transpose(row["image"])
        exact, phash = fingerprint(image)
        mask = np.asarray(row["mask"])
        if mask.shape[:2] != (image.height, image.width):
            mask = np.asarray(row["mask"].resize(image.size, Image.NEAREST))
        candidates.append({
            "plantseg_row": index, "plantseg_disease": disease, "plantseg_split": row["split"],
            "source_url": row["source_url"], "class_name": CLASS_MAP[disease],
            "label_id": label_ids[CLASS_MAP[disease]], "width": image.width, "height": image.height,
            "lesion_area_fraction": float((mask > 0).mean()), "exact_hash": exact, "phash_int": phash,
        })
    print(f"PlantSeg candidates: {len(candidates)}", flush=True)

    identities = repo_identities()
    references = []
    for name, (repository, revision), keep in (
        ("plantdoc", PLANTDOC, lambda label: True),
        ("plantvillage", PLANTVILLAGE, lambda label: label.startswith("Tomato")),
    ):
        dataset = load_dataset(repository, split="train", revision=revision)
        for row_id, label in enumerate(dataset["class_label"]):
            if keep(label):
                exact, phash = fingerprint(dataset[row_id]["image"])
                references.append((name, row_id, label, exact, phash))
        print(f"References after {name}: {len(references)}", flush=True)
    ref_phash = np.asarray([r[4] for r in references], dtype=np.uint64)
    ref_exact = {r[3]: i for i, r in enumerate(references)}

    for record in candidates:
        distances = hamming_to_all(record["phash_int"], ref_phash)
        nearest = int(np.argmin(distances))
        if record["exact_hash"] in ref_exact:
            nearest, distance = ref_exact[record["exact_hash"]], 0
        else:
            distance = int(distances[nearest])
        source, row_id, label, _, _ = references[nearest]
        manifest, repo_label = identities.get((source, row_id), ("", ""))
        record.update({
            "min_reference_distance": distance, "nearest_reference": f"{source}:{row_id}",
            "nearest_reference_label": label, "nearest_in_repo_manifest": manifest,
            "nearest_repo_class_name": repo_label,
            "reference_match": record["exact_hash"] in ref_exact or distance <= REFERENCE_DISTANCE,
        })

    eligible = [r for r in candidates if not r["reference_match"]]
    roots = internal_groups([r["phash_int"] for r in eligible], INTERNAL_DISTANCE)
    keep_roots = set()
    for record, root in zip(eligible, roots):
        record["internal_group"] = eligible[root]["plantseg_row"]
        record["internal_duplicate"] = root in keep_roots
        keep_roots.add(root)
    for record in candidates:
        if record["reference_match"]:
            record["status"] = "excluded_reference_match"
        elif record["internal_duplicate"]:
            record["status"] = "excluded_internal_duplicate"
        else:
            record["status"] = "included"
        record["phash"] = f"{record.pop('phash_int'):016x}"

    frame = pd.DataFrame(candidates).sort_values("plantseg_row")
    output.mkdir(parents=True)
    columns = ["plantseg_row", "plantseg_disease", "plantseg_split", "class_name", "label_id", "status",
               "width", "height", "lesion_area_fraction", "exact_hash", "phash", "min_reference_distance",
               "nearest_reference", "nearest_in_repo_manifest"]
    frame[columns].to_csv(output / "manifest.csv", index=False)

    # Images shared with repository manifests carry two independent labels.
    shared = frame[frame.nearest_in_repo_manifest.ne("") & frame.min_reference_distance.le(INTERNAL_DISTANCE)]
    audit = shared.assign(labels_agree=shared.class_name.eq(shared.nearest_repo_class_name))
    audit[["plantseg_row", "class_name", "nearest_reference", "nearest_in_repo_manifest",
           "nearest_repo_class_name", "min_reference_distance", "labels_agree", "source_url"]].rename(
        columns={"class_name": "plantseg_class_name"}).to_csv(output / "shared_image_label_check.csv", index=False)

    included = frame[frame.status.eq("included")]
    report = {
        "plantseg": {"repository": PLANTSEG[0], "revision": PLANTSEG[1], "license": "CC BY-NC-ND 4.0",
                     "citation": "Wei et al., PlantSeg, arXiv:2409.04038"},
        "references": {"plantdoc": {"repository": PLANTDOC[0], "revision": PLANTDOC[1], "rows": "all"},
                       "plantvillage": {"repository": PLANTVILLAGE[0], "revision": PLANTVILLAGE[1],
                                        "rows": "class_label starts with 'Tomato'"},
                       "reference_images": len(references)},
        "rules": {"fingerprints": "raw-RGB SHA-256 after EXIF transpose; imagehash.phash (8x8 DCT)",
                  "reference_exclusion": f"exact match or pHash distance <= {REFERENCE_DISTANCE}",
                  "internal_deduplication": f"pHash distance <= {INTERNAL_DISTANCE}; keep lowest plantseg_row"},
        "counts": {"candidates": len(frame), **frame.status.value_counts().to_dict(),
                   "included_per_class": included.class_name.value_counts().to_dict(),
                   "excluded_matches_in_repo_manifests": int(
                       (frame.reference_match & frame.nearest_in_repo_manifest.ne("")).sum()),
                   "included_min_side_lt_224": int((included[["width", "height"]].min(axis=1) < 224).sum())},
        "shared_image_label_check": {"images": len(audit), "labels_agree": int(audit.labels_agree.sum())},
        "scope": ("Screens known public references only. Does not establish independence from pretraining "
                  "data, other web copies, or the correctness of PlantSeg labels."),
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, default=int) + "\n")
    print(json.dumps(report["counts"], indent=2, default=int))


if __name__ == "__main__":
    sys.exit(main())
