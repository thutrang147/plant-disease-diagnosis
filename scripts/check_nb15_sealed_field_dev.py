"""Re-score the six sealed Notebook 15 checkpoints on Field Dev (inference only).

The E2 amendment compares each CPU re-run at the Notebook 15 selected epoch
with the per-arm Field Dev values in the README. This script evaluates the
sealed checkpoints themselves, so a mismatch can be attributed either to the
README value or to the re-run. Checkpoint SHA-256 values are checked against
the Notebook 15 seal first.

Usage from the repository root::

    python scripts/check_nb15_sealed_field_dev.py --output data/results/nb15_sealed_field_dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import PlantDiseaseDataset  # noqa: E402
from src.inference import sha256  # noqa: E402
from src.supplementary_analysis import macro_f1  # noqa: E402

spec = importlib.util.spec_from_file_location("lpft", ROOT / "scripts/explore_lpft_architecture.py")
lpft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpft)
RUN = ROOT / "outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output}")
    seal = json.loads((ROOT / "data/results/notebook15/original_evaluation_seal.json").read_text())
    model, _, evaluation = lpft.load("resnet18", ROOT / "models")
    ds = PlantDiseaseDataset(ROOT / "data/splits/field_dev.csv", transform=evaluation)
    x, y = torch.stack([ds[i][0] for i in range(len(ds))]), ds.metadata.label_id.to_numpy()
    rows = []
    for seed, files in seal["seed_checkpoints"].items():
        for name, digest in files.items():
            path = args.run / f"seed_{seed}" / name
            if sha256(path) != digest:
                raise ValueError(f"SHA-256 mismatch for {path}")
            state = torch.load(path, map_location="cpu", weights_only=False)
            state = state.get("model_state_dict", state.get("state_dict", state))
            model.load_state_dict(state)
            p = lpft.predict(model, x)
            rows.append({"seed": int(seed), "condition": name.replace("_selected.pt", ""), "n": len(y),
                         "correct": int((p == y).sum()), "macro_f1": macro_f1(y, p, range(5)), "sha256": digest})
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "sealed_field_dev.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="sha256").round(5).to_string(index=False))


if __name__ == "__main__":
    main()
