"""Compare a fresh re-run directory with the committed results directory.

Every CSV present in the committed directory must exist in the re-run.
Integer and text columns must match exactly; floating-point columns within
``--atol``. Prints one line per file and exits non-zero on any difference.

Usage::

    python scripts/compare_reproduction.py data/results/external_v3_plantseg reproduction/external_v3_plantseg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(committed: Path, rerun: Path, atol: float) -> list[str]:
    problems = []
    for path in sorted(committed.glob("*.csv")):
        other = rerun / path.name
        if not other.is_file():
            problems.append(f"{path.name}: missing in re-run")
            continue
        a, b = pd.read_csv(path), pd.read_csv(other)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            problems.append(f"{path.name}: shape or columns differ ({a.shape} vs {b.shape})")
            continue
        for column in a.columns:
            x, y = a[column], b[column]
            if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
                xf, yf = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
                if not np.allclose(xf, yf, atol=atol, equal_nan=True):
                    worst = float(np.nanmax(np.abs(xf - yf)))
                    problems.append(f"{path.name}:{column}: max abs difference {worst:.3g}")
            elif not x.astype(str).equals(y.astype(str)):
                problems.append(f"{path.name}:{column}: {int((x.astype(str) != y.astype(str)).sum())} values differ")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("committed", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    problems = compare(args.committed, args.rerun, args.atol)
    print(f"{args.committed} vs {args.rerun}: " + ("identical within tolerance" if not problems else f"{len(problems)} difference(s)"))
    for problem in problems:
        print("  " + problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
