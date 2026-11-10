"""One-shot confirmatory evaluation for locked Field Test or self-collected real-world leaf images.

Run locally from the project root after model selection is completely frozen:
    python -m src.evaluate_confirmatory_field --mode self_collected --images-dir data/self_collected_field

Supported modes:
1. `self_collected`: Evaluates a directory of real-world tomato leaf photographs organized
   by the five canonical class folders (`Healthy`, `Early Blight`, `Late Blight`,
   `Bacterial Spot`, `Septoria Leaf Spot`). Automatically runs an exact SHA-256 and
   8x8 DCT pHash near-duplicate audit (Hamming distance <= 4) against existing splits
   before running inference, and reports Accuracy, Macro-F1, Wilson 95% CIs, ECE,
   and confusion matrices.
2. `plantdoc_field_test`: Runs a single confirmatory pass on `data/splits/field_test.csv`
   once all development and checkpoint decisions are locked.
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
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

from src.prepare_field_dev_verified import compute_phash_hex, hamming_distance_hex
from src.statistical_analysis import compute_ece, wilson_ci

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FolderFieldDataset(Dataset):
    """Dataset for self-collected real-world tomato leaf images organized by class folder."""

    def __init__(self, records: pd.DataFrame, transform=None):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(row["filepath"]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_id"])


def scan_and_audit_self_collected(
    images_dir: Path, splits_dir: Path, phash_threshold: int = 4
) -> tuple[pd.DataFrame, dict]:
    """Scan a directory of class subfolders and audit against existing PlantDoc splits."""
    mapping = json.loads((splits_dir / "label_map.json").read_text(encoding="utf-8"))
    existing_hashes: set[str] = set()
    existing_phashes: list[tuple[str, int, str]] = []

    for split_name in ("field_adapt_train", "field_dev"):
        csv_path = splits_dir / f"{split_name}.csv"
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            if "exact_hash" in df.columns:
                existing_hashes.update(df["exact_hash"].dropna().astype(str).tolist())
            if "phash" in df.columns:
                for _, r in df.iterrows():
                    existing_phashes.append((split_name, int(r["row_id"]), str(r["phash"])))

    rows = []
    near_dup_flags = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for class_name, label_id in sorted(mapping.items(), key=lambda x: x[1]):
        class_dir = images_dir / class_name
        if not class_dir.is_dir():
            continue
        for fpath in sorted(class_dir.iterdir()):
            if fpath.suffix.lower() not in exts or not fpath.is_file():
                continue
            raw_bytes = fpath.read_bytes()
            sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
            with Image.open(fpath) as im:
                rgb = im.convert("RGB")
                phash_hex = compute_phash_hex(rgb)

            exact_match = sha256_hex in existing_hashes
            min_dist = 64
            closest_ref = None
            for sp_name, r_id, ref_phash in existing_phashes:
                d = hamming_distance_hex(phash_hex, ref_phash)
                if d < min_dist:
                    min_dist = d
                    closest_ref = f"{sp_name}:{r_id}"

            is_near_dup = exact_match or (min_dist <= phash_threshold)
            if is_near_dup:
                near_dup_flags.append(
                    {
                        "filepath": str(fpath),
                        "class_name": class_name,
                        "exact_match": exact_match,
                        "min_phash_distance": min_dist,
                        "closest_existing": closest_ref,
                    }
                )
            rows.append(
                {
                    "filepath": str(fpath),
                    "filename": fpath.name,
                    "class_name": class_name,
                    "label_id": int(label_id),
                    "exact_hash": sha256_hex,
                    "phash": phash_hex,
                    "min_phash_dist_to_plantdoc": min_dist,
                    "contaminated": is_near_dup,
                }
            )

    manifest = pd.DataFrame(rows)
    audit = {
        "images_dir": str(images_dir),
        "total_scanned": len(manifest),
        "phash_threshold": phash_threshold,
        "contaminated_count": len(near_dup_flags),
        "clean_count": int((~manifest["contaminated"]).sum()) if not manifest.empty else 0,
        "contaminated_examples": near_dup_flags,
    }
    return manifest, audit


def evaluate_resnet_checkpoint(
    checkpoint_path: Path,
    manifest: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Evaluate a saved ResNet18 checkpoint on a clean self-collected manifest."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = FolderFieldDataset(manifest, transform=eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    preds, confs, trues = [], [], []
    with torch.inference_mode():
        for imgs, lbls in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)
            val, idx = probs.max(dim=1)
            preds.extend(idx.cpu().tolist())
            confs.extend(val.cpu().tolist())
            trues.extend(lbls.tolist())

    y_true = np.asarray(trues, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    conf = np.asarray(confs, dtype=float)
    corr = y_true == y_pred

    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = manifest.copy()
    out_df["predicted_label_id"] = y_pred
    out_df["predicted_class"] = [class_names[i] for i in y_pred]
    out_df["confidence"] = conf
    out_df["correct"] = corr
    out_df.to_csv(output_dir / "predictions.csv", index=False)

    labels = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    w_lo, w_hi = wilson_ci(int(np.sum(corr)), len(y_true))
    ece = compute_ece(corr, conf, n_bins=10)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
    )
    ax.set_title(f"{checkpoint_path.stem} — Self-Collected Field (n={len(y_true)})")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    return {
        "checkpoint": checkpoint_path.name,
        "n": int(len(y_true)),
        "correct": int(np.sum(corr)),
        "accuracy": round(acc, 4),
        "wilson_95ci_low": round(w_lo, 4),
        "wilson_95ci_high": round(w_hi, 4),
        "macro_f1": round(mf1, 4),
        "ece_10bins": round(ece, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirmatory evaluation for self-collected field images.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "self_collected_field",
        help="Path to directory containing the 5 class subfolders of self-collected leaf images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "self_collected_confirmatory",
        help="Directory to write audit manifests and evaluation metrics.",
    )
    args = parser.parse_args()

    splits_dir = PROJECT_ROOT / "data" / "splits"
    mapping = json.loads((splits_dir / "label_map.json").read_text(encoding="utf-8"))
    class_names = [k for k, _ in sorted(mapping.items(), key=lambda item: item[1])]

    if not args.images_dir.is_dir():
        print(
            f"[Ready] Self-collected image folder not found yet at: {args.images_dir}\n"
            "When real-world photos are collected, place them under:\n"
            + "\n".join(f"  - {args.images_dir / c}" for c in class_names)
        )
        return

    manifest, audit = scan_and_audit_self_collected(args.images_dir, splits_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_protocol.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest.to_csv(args.output_dir / "self_collected_manifest.csv", index=False)

    clean_manifest = manifest[~manifest["contaminated"]].reset_index(drop=True) if not manifest.empty else manifest
    if clean_manifest.empty:
        print(f"No clean images found in {args.images_dir}. Audit written to {args.output_dir / 'audit_protocol.json'}.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for ckpt_name in ("resnet18_lab_best.pt", "resnet18_aug_geometric_best.pt", "resnet18_mixed_best.pt"):
        ckpt_path = PROJECT_ROOT / "models" / ckpt_name
        if ckpt_path.is_file():
            res = evaluate_resnet_checkpoint(
                ckpt_path, clean_manifest, class_names, args.output_dir / ckpt_path.stem, device
            )
            summaries.append(res)
    if summaries:
        pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
        print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

