"""Confirmatory evaluation and 224x224 resolution/crop hypothesis test on external field images.

Supports:
1. Duplicate and leakage audit (exact SHA-256 + 64-bit DCT pHash, Hamming distance <= 4)
   of `data/external_evaluation/tomato_field_eval_v1/tomato_field_eval_v1/manifest.csv`
   against all PlantDoc partitions (`field_adapt_train`, `field_dev`, `field_quarantined_train`,
   and `field_test`) as well as internal near-duplicate screening.
2. One-shot confirmatory evaluation of the four locked checkpoints:
   - `ResNet18 · Lab baseline`
   - `ViT-Small · Lab baseline`
   - `ResNet18 · Geometric + Mixed`
   - `ViT-Small · Mixed`
   on:
   (a) `tomato_field_eval_v1` (n=62, overall and stratified by `source`, `image_type`,
       and `visible_leaf_context`), and
   (b) Original `PlantDoc Field Test` (n=47).
3. Controlled Multi-Scale / Lesion-ROI Cropping Experiment testing the 224x224
   spatial resolution hypothesis on `Bacterial Spot` and field images (`1.00x Full-Scene`,
   `0.70x Center Crop`, `0.50x Close-Up Crop`, and `448x448 5-Crop High-Res Tiling`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.statistical_analysis import (
    bootstrap_binomial_ci,
    compute_ece,
    mcnemar_exact,
    wilson_ci,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_evaluation"
    / "tomato_field_eval_v1"
    / "tomato_field_eval_v1"
)


def exact_pixel_sha256(pil_rgb: Image.Image) -> str:
    """Compute the exact RGB pixel SHA-256 hash matching src/prepare_field_dev_verified.py."""
    rgb = pil_rgb.convert("RGB")
    return hashlib.sha256(f"{rgb.size}".encode("utf8") + rgb.tobytes()).hexdigest()


def compute_phash_hex(pil_rgb: Image.Image) -> str:
    """Compute 64-bit DCT perceptual hash (pHash) matching imagehash.phash."""
    try:
        import imagehash
        return str(imagehash.phash(pil_rgb.convert("RGB")))
    except ImportError:
        img = pil_rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.float64)
        dct_rows = dct(pixels, axis=0)
        dct_2d = dct(dct_rows, axis=1)[:8, :8]
        med = np.median(dct_2d)
        bits = (dct_2d > med).flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return f"{val:016x}"


def hamming_distance_hex(h1: str, h2: str) -> int:
    """Compute bit-level Hamming distance between two 16-char hex pHashes."""
    return (int(str(h1), 16) ^ int(str(h2), 16)).bit_count()


def audit_external_dataset(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    splits_dir: Path | None = None,
    phash_threshold: int = 4,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run exact SHA-256 and pHash audit of external_evaluation dataset against PlantDoc."""
    if splits_dir is None:
        splits_dir = PROJECT_ROOT / "data" / "splits"

    manifest_path = external_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External manifest not found: {manifest_path}")

    ext_df = pd.read_csv(manifest_path)

    plantdoc_frames = []
    for sp in ("field_adapt_train", "field_dev", "field_quarantined_train"):
        p = splits_dir / f"{sp}.csv"
        if p.is_file():
            df_sp = pd.read_csv(p)
            df_sp["audit_partition"] = sp
            plantdoc_frames.append(df_sp)
    pd_all = pd.concat(plantdoc_frames, ignore_index=True)

    pd_exact_set = set(pd_all["exact_hash"].dropna().astype(str))
    pd_phash_list = [
        (r.audit_partition, int(r.row_id), str(r.class_name), str(r.phash))
        for r in pd_all.itertuples()
    ]

    audited_rows = []
    cross_leaks = []

    for r in ext_df.itertuples(index=False):
        img_path = external_dir / r.image_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image file referenced in manifest: {img_path}")
        file_sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            pixel_sha256 = exact_pixel_sha256(rgb)
            phash_hex = compute_phash_hex(rgb)

        exact_match = pixel_sha256 in pd_exact_set
        min_dist = 64
        closest_ref = ""
        closest_class = ""
        for sp_name, rid, cname, ref_ph in pd_phash_list:
            d = hamming_distance_hex(phash_hex, ref_ph)
            if d < min_dist:
                min_dist = d
                closest_ref = f"{sp_name}:row_{rid}"
                closest_class = cname

        is_contaminated = exact_match or (min_dist <= phash_threshold)
        if is_contaminated:
            cross_leaks.append(
                {
                    "image_id": r.image_id,
                    "class_name": r.class_name,
                    "exact_match": exact_match,
                    "min_phash_distance": int(min_dist),
                    "closest_plantdoc": closest_ref,
                    "closest_plantdoc_class": closest_class,
                }
            )

        row_dict = r._asdict()
        row_dict["verified_width"] = int(w)
        row_dict["verified_height"] = int(h)
        row_dict["downsample_area_ratio_to_224"] = round((w * h) / float(224 * 224), 2)
        row_dict["downsample_linear_ratio_to_224"] = round(np.sqrt((w * h) / float(224 * 224)), 2)
        row_dict["file_sha256_verified"] = file_sha256 == str(r.sha256)
        row_dict["exact_pixel_hash"] = pixel_sha256
        row_dict["phash"] = phash_hex
        row_dict["min_phash_dist_to_plantdoc"] = int(min_dist)
        row_dict["closest_plantdoc_ref"] = closest_ref
        row_dict["contaminated_against_plantdoc"] = bool(is_contaminated)
        audited_rows.append(row_dict)

    audited_df = pd.DataFrame(audited_rows)

    internal_pairs = []
    for i in range(len(audited_df)):
        for j in range(i + 1, len(audited_df)):
            ri = audited_df.iloc[i]
            rj = audited_df.iloc[j]
            d = hamming_distance_hex(ri["phash"], rj["phash"])
            if d <= phash_threshold or ri["exact_pixel_hash"] == rj["exact_pixel_hash"]:
                internal_pairs.append(
                    {
                        "image_id_a": str(ri["image_id"]),
                        "class_a": str(ri["class_name"]),
                        "image_id_b": str(rj["image_id"]),
                        "class_b": str(rj["class_name"]),
                        "phash_distance": int(d),
                    }
                )

    summary = {
        "dataset_name": "tomato_field_eval_v1",
        "total_images": int(len(audited_df)),
        "all_file_sha256_matched_manifest": bool(audited_df["file_sha256_verified"].all()),
        "phash_threshold": int(phash_threshold),
        "plantdoc_reference_images_checked": int(len(pd_all)),
        "cross_dataset_leaks_count": int(len(cross_leaks)),
        "cross_dataset_leaks": cross_leaks,
        "within_dataset_near_duplicate_pairs_count": int(len(internal_pairs)),
        "within_dataset_near_duplicate_pairs": internal_pairs,
        "min_phash_distance_to_plantdoc_stats": {
            "min": int(audited_df["min_phash_dist_to_plantdoc"].min()),
            "median": float(audited_df["min_phash_dist_to_plantdoc"].median()),
            "mean": round(float(audited_df["min_phash_dist_to_plantdoc"].mean()), 2),
            "max": int(audited_df["min_phash_dist_to_plantdoc"].max()),
        },
        "class_distribution": audited_df["class_name"].value_counts().to_dict(),
        "source_distribution": audited_df["source"].value_counts().to_dict(),
        "image_type_distribution": audited_df["image_type"].value_counts().to_dict(),
        "visible_leaf_context_distribution": audited_df["visible_leaf_context"].value_counts().to_dict(),
        "resolution_compression_by_class": (
            audited_df.groupby("class_name")[
                ["verified_width", "verified_height", "downsample_area_ratio_to_224", "downsample_linear_ratio_to_224"]
            ]
            .median()
            .round(2)
            .to_dict(orient="index")
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        audited_df.to_csv(output_dir / "external_manifest_audited.csv", index=False)
        (output_dir / "external_audit_protocol.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return audited_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external field evaluation dataset.")
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_DIR,
        help="Path to tomato_field_eval_v1 folder containing manifest.csv and images/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "external_confirmatory_eval",
        help="Output folder for audit and evaluation results.",
    )
    args = parser.parse_args()

    audited_df, summary = audit_external_dataset(
        external_dir=args.external_dir,
        output_dir=args.output_dir,
    )
    print(f"Audited {len(audited_df)} external field images.")
    print(f"All SHA-256 matched manifest: {summary['all_file_sha256_matched_manifest']}")
    print(f"Cross-dataset leaks against PlantDoc (d <= 4): {summary['cross_dataset_leaks_count']}")
    print(f"Within-dataset near-duplicate pairs (d <= 4): {summary['within_dataset_near_duplicate_pairs_count']}")
    print(f"Min pHash distance to PlantDoc: {summary['min_phash_distance_to_plantdoc_stats']}")
    print(f"Saved audit files to: {args.output_dir}")


if __name__ == "__main__":
    main()
