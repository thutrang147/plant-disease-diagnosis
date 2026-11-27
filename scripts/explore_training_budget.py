"""Training-budget ablation: does Field Dev keep rising beyond the historical budget?

Historical adaptation runs used 2 head + 6 second-phase epochs and selected their
final or near-final epoch while Field Dev was still rising (REPO_REVIEW #8).
This reruns ``scripts/explore_lpft_architecture.py`` unchanged except for the
budget: 2 head + 22 second-phase epochs (24 in total), seed 42 only, both
architectures and both strategies. The per-epoch Field Dev history answers the
plateau question; the selected epoch is evaluated once as before.

Status: exploratory, CPU, one seed. Usage from the repository root::

    python scripts/explore_training_budget.py --output data/results/training_budget
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lpft", ROOT / "scripts/explore_lpft_architecture.py")
lpft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpft)

lpft.CONFIG.update({"second_epochs": 22, "seeds": [42]})

if __name__ == "__main__":
    lpft.main()
