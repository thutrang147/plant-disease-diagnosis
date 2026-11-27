"""Quantify whether frozen classifiers rely on lesion pixels, using PlantSeg masks.

Exploratory analysis on External v3 (screened PlantSeg tomato disease images
with expert-reviewed lesion masks). It replaces the qualitative Grad-CAM reading
of Notebook 10 with two measurements:

1. **Lesion occlusion.** Each image is evaluated as (a) original, (b) lesion
   removed: mask dilated by ``DILATE`` px and filled by OpenCV Telea inpainting,
   and (c) lesion only: pixels outside the dilated mask set to the ImageNet mean
   colour. A classifier that uses lesions should lose accuracy and move towards
   *Healthy* in (b). Both edits are out-of-distribution, so changes are evidence
   about reliance, not clean causal effects.
2. **Grad-CAM localization (ResNet18 models).** Grad-CAM on ``layer4`` for the
   true class, mapped through the same Resize(256)/CenterCrop(224) as the input.
   Pointing game: the CAM maximum falls inside the lesion mask dilated by
   ``TOLERANCE`` px (Zhang et al., 2018). Energy ratio: CAM mass inside the mask
   divided by the mask's area share (1.0 = chance).

Inference only, CPU, float32. Checkpoints are hash-checked as in
``scripts/evaluate_external_v3_plantseg.py``.

Usage from the repository root::

    python scripts/evaluate_lesion_reliance.py --output data/results/lesion_reliance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_external_v3_plantseg import (  # noqa: E402
    PLANTSEG, SEEDS, load_model, sha256,
)

DILATE = 5
TOLERANCE = 15
IMAGENET_MEAN_RGB = (124, 116, 104)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def conditions(image: Image.Image, mask: np.ndarray) -> dict[str, Image.Image]:
    rgb = np.asarray(image)
    grown = dilate(mask, DILATE)
    removed = cv2.inpaint(rgb, grown.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    only = rgb.copy()
    only[~grown] = IMAGENET_MEAN_RGB
    return {"original": image, "lesion_removed": Image.fromarray(removed), "lesion_only": Image.fromarray(only)}


def crop_mask(mask: np.ndarray) -> np.ndarray:
    """Apply ResNet18 IMAGENET1K_V1 geometry (Resize 256 shorter side, CenterCrop 224) to a mask."""
    resized = TF.resize(Image.fromarray(mask.astype(np.uint8) * 255), 256,
                        interpolation=TF.InterpolationMode.NEAREST)
    return np.asarray(TF.center_crop(resized, 224)) > 0


def grad_cam(model, x: torch.Tensor, target: int) -> np.ndarray:
    store = {}
    handle = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    x = x.unsqueeze(0).requires_grad_(False)
    with torch.enable_grad():
        logits = model(x)
        activation = store["a"]
        grads = torch.autograd.grad(logits[0, target], activation)[0]
    handle.remove()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().numpy()


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
    vit_config = torch.load(paths["vit_small_lab"], map_location="cpu", weights_only=True)["preprocessing_config"]

    samples = []
    for row in included.itertuples():
        record = source[int(row.plantseg_row)]
        image = ImageOps.exif_transpose(record["image"]).convert("RGB")
        mask = record["mask"]
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        mask = np.asarray(mask) > 0
        samples.append((row.plantseg_row, int(row.label_id), conditions(image, mask), mask))
    print("prepared", len(samples), flush=True)

    occlusion, localization = [], []
    for key, path in paths.items():
        model, transform = load_model(path, vit_config)
        for cond in ("original", "lesion_removed", "lesion_only"):
            x = torch.stack([transform(s[2][cond]) for s in samples])
            with torch.inference_mode():
                pred = torch.cat([model(x[i:i + 64]) for i in range(0, len(x), 64)]).argmax(1).numpy()
            for (row, label, _, _), p in zip(samples, pred):
                occlusion.append({"model": key, "condition": cond, "plantseg_row": row,
                                  "label_id": label, "predicted_label_id": int(p)})
        if key.startswith("vit"):
            del model
            continue
        model.requires_grad_(True)
        for row, label, images, mask in samples:
            cropped = crop_mask(mask)
            if not cropped.any():
                continue
            cam = grad_cam(model, transform(images["original"]), label)
            peak = np.unravel_index(np.argmax(cam), cam.shape)
            total = cam.sum()
            localization.append({
                "model": key, "plantseg_row": row, "label_id": label,
                "pointing_hit": bool(dilate(cropped, TOLERANCE)[peak]),
                "tolerance_area_share": float(dilate(cropped, TOLERANCE).mean()),
                "energy_in_mask": float(cam[cropped].sum() / total) if total > 0 else float("nan"),
                "mask_area_share": float(cropped.mean())})
        print("done", key, flush=True)
        del model

    occlusion = pd.DataFrame(occlusion)
    localization = pd.DataFrame(localization)
    occlusion["correct"] = occlusion.label_id.eq(occlusion.predicted_label_id)
    occlusion["predicted_healthy"] = occlusion.predicted_label_id.eq(0)
    wide = occlusion.pivot_table(index=["model", "plantseg_row"], columns="condition",
                                 values="correct", aggfunc="first")
    flips = (wide.original & ~wide.lesion_removed).groupby("model").mean().rename("lost_when_lesion_removed")
    occ_summary = (occlusion.groupby(["model", "condition"])[["correct", "predicted_healthy"]].mean()
                   .unstack("condition"))
    occ_summary.columns = [f"{m}_{c}" for m, c in occ_summary.columns]
    occ_summary = occ_summary.join(flips)
    localization["energy_ratio"] = localization.energy_in_mask / localization.mask_area_share
    loc_summary = localization.groupby("model").agg(
        images=("pointing_hit", "size"), pointing_accuracy=("pointing_hit", "mean"),
        pointing_chance=("tolerance_area_share", "mean"), mean_energy_in_mask=("energy_in_mask", "mean"),
        mean_mask_area_share=("mask_area_share", "mean"), median_energy_ratio=("energy_ratio", "median"))

    output.mkdir(parents=True)
    occlusion.to_csv(output / "occlusion_predictions.csv", index=False)
    localization.to_csv(output / "gradcam_localization.csv", index=False)
    occ_summary.round(4).to_csv(output / "occlusion_summary.csv")
    loc_summary.round(4).to_csv(output / "gradcam_summary.csv")
    (output / "protocol.json").write_text(json.dumps({
        "dilate_px": DILATE, "pointing_tolerance_px": TOLERANCE, "inpainting": "cv2.INPAINT_TELEA radius 5",
        "lesion_only_fill_rgb": IMAGENET_MEAN_RGB, "cam_layer": "resnet18.layer4, target = true class",
        "images": len(samples), "status": "exploratory", "opencv": cv2.__version__, "torch": torch.__version__,
        "timm": timm.__version__}, indent=2) + "\n")
    print(occ_summary.round(3).to_string())
    print(loc_summary.round(3).to_string())


if __name__ == "__main__":
    main()
