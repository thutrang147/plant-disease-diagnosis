"""Control arm for the lesion-occlusion analysis (review-panel request).

The occlusion result (inpainting expert-marked lesions lowers accuracy) could be
caused by the edit itself rather than by losing the lesion. This script repeats
the same Telea inpainting on a **random non-lesion region of the same area**:

* the dilated lesion mask is translated by a random offset (wrapping around the
  image) until it overlaps the original dilated lesion by at most 10% of its
  area (up to 200 tries per image, seed fixed per image); if no such offset is
  found, the image is reported as ``control_unavailable`` and skipped;
* every model is evaluated on original, lesion-inpainted and control-inpainted
  images; the quantity of interest is the extra accuracy loss of lesion
  inpainting over control inpainting, with a paired bootstrap interval over
  images (10,000 resamples, seed 0);
* for ResNet18 Grad-CAM, a **centre baseline** is added: the share of images
  whose 224x224 centre pixel falls in the tolerance region (Zhang et al., 2018).

Inference only, CPU. Usage from the repository root::

    python scripts/evaluate_occlusion_control.py --output data/results/occlusion_control
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_external_v3_plantseg import PLANTSEG, SEEDS, load_model, sha256  # noqa: E402
from evaluate_lesion_reliance import DILATE, TOLERANCE, crop_mask, dilate  # noqa: E402

MAX_OVERLAP = 0.10


def control_mask(grown: np.ndarray, seed: int) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    h, w = grown.shape
    area = grown.sum()
    for _ in range(200):
        shifted = np.roll(grown, (int(rng.integers(0, h)), int(rng.integers(0, w))), axis=(0, 1))
        if (shifted & grown).sum() <= MAX_OVERLAP * area:
            return shifted
    return None


def inpaint(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.inpaint(rgb, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--nb15-run", type=Path,
                        default=ROOT / "outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    torch.set_num_threads(args.threads)

    protocol = json.loads((ROOT / "data/external_v3_plantseg/protocol.json").read_text())
    seal = json.loads((ROOT / "data/results/notebook15/original_evaluation_seal.json").read_text())
    paths = {"resnet18_lab": args.models_dir / "resnet18_lab_best.pt",
             "resnet18_geometric_mixed": args.models_dir / "resnet18_geometric_mixed_best.pt",
             "vit_small_lab": args.models_dir / "vit_small_lab_best.pt",
             "vit_small_mixed": args.models_dir / "vit_small_mixed_best.pt"}
    expected = {k: protocol["models"][k]["checkpoint_sha256"] for k in paths}
    for seed in SEEDS:
        for condition in ("field_only", "mixed"):
            key = f"nb15_{condition}_s{seed}"
            paths[key] = args.nb15_run / f"seed_{seed}" / f"{condition}_selected.pt"
            expected[key] = seal["seed_checkpoints"][str(seed)][f"{condition}_selected.pt"]
    for key, path in paths.items():
        if sha256(path) != expected[key]:
            raise ValueError(f"SHA-256 mismatch for {key}")

    manifest = pd.read_csv(ROOT / "data/external_v3_plantseg/manifest.csv")
    included = manifest[manifest.status.eq("included")].reset_index(drop=True)
    source = load_dataset(PLANTSEG[0], split="train", revision=PLANTSEG[1])
    samples, unavailable, resized, centre_hits = [], 0, 0, []
    for row in included.itertuples():
        record = source[int(row.plantseg_row)]
        image = ImageOps.exif_transpose(record["image"]).convert("RGB")
        mask_img = record["mask"]
        if mask_img.size != image.size:
            resized += 1
            mask_img = mask_img.resize(image.size, Image.NEAREST)
        mask = np.asarray(mask_img) > 0
        cropped = crop_mask(mask)
        if cropped.any():
            centre_hits.append(bool(dilate(cropped, TOLERANCE)[112, 112]))
        grown = dilate(mask, DILATE)
        ctrl = control_mask(grown, seed=int(row.plantseg_row))
        if ctrl is None:
            unavailable += 1
            continue
        rgb = np.asarray(image)
        samples.append((int(row.plantseg_row), int(row.label_id),
                        {"original": image, "lesion_inpainted": inpaint(rgb, grown), "control_inpainted": inpaint(rgb, ctrl)},
                        float(grown.mean())))
    print("prepared", len(samples), "control unavailable", unavailable, "masks resized", resized, flush=True)

    vit_config = torch.load(paths["vit_small_lab"], map_location="cpu", weights_only=True)["preprocessing_config"]
    rows = []
    for key, path in paths.items():
        model, transform = load_model(path, vit_config)
        for cond in ("original", "lesion_inpainted", "control_inpainted"):
            x = torch.stack([transform(s[2][cond]) for s in samples])
            with torch.inference_mode():
                pred = torch.cat([model(x[i:i + 64]) for i in range(0, len(x), 64)]).argmax(1).numpy()
            for (row_id, label, _, area), p in zip(samples, pred):
                rows.append({"model": key, "condition": cond, "plantseg_row": row_id, "label_id": label,
                             "predicted_label_id": int(p), "dilated_lesion_share": area})
        print("done", key, flush=True)
        del model

    frame = pd.DataFrame(rows)
    frame["correct"] = frame.label_id.eq(frame.predicted_label_id)
    wide = frame.pivot_table(index=["model", "plantseg_row"], columns="condition", values="correct", aggfunc="first").astype(float)
    rng = np.random.default_rng(0)
    summary = []
    for model, g in wide.groupby(level="model"):
        lesion_drop = (g.original - g.lesion_inpainted).to_numpy()
        control_drop = (g.original - g.control_inpainted).to_numpy()
        extra = lesion_drop - control_drop
        idx = rng.integers(0, len(extra), size=(10_000, len(extra)))
        lo, hi = np.percentile(extra[idx].mean(axis=1), [2.5, 97.5])
        summary.append({"model": model, "images": len(g), "accuracy_original": g.original.mean(),
                        "accuracy_lesion_inpainted": g.lesion_inpainted.mean(),
                        "accuracy_control_inpainted": g.control_inpainted.mean(),
                        "extra_drop_lesion_vs_control": extra.mean(), "extra_drop_ci_low": lo, "extra_drop_ci_high": hi})
    summary = pd.DataFrame(summary)

    output.mkdir(parents=True)
    frame.to_csv(output / "control_predictions.csv", index=False)
    summary.round(4).to_csv(output / "control_summary.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "control": "same-area random translation of the dilated lesion mask, overlap <= 10%, Telea inpainting radius 5",
        "dilate_px": DILATE, "images_with_control": len(samples), "control_unavailable": unavailable,
        "masks_resized_to_image": resized, "centre_baseline_pointing": float(np.mean(centre_hits)),
        "centre_baseline_images": len(centre_hits), "bootstrap": "10,000 image resamples, seed 0",
        "status": "exploratory control requested by the review panel", "torch": torch.__version__}, indent=2) + "\n")
    print(summary.round(3).to_string(index=False))
    print("centre baseline pointing:", round(float(np.mean(centre_hits)), 3))


if __name__ == "__main__":
    main()
