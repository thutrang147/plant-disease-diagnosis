"""Compare a fresh re-run directory with the committed results directory.

Every CSV in the committed directory must exist in the re-run with the same
shape. Files are compared cell by cell as raw text (so multi-row headers are
handled): cells that parse as numbers must agree within ``--atol``; all other
cells must match exactly. A committed directory without CSV files is an error.
Prints one line per difference and exits non-zero on any difference.

Usage::

    python scripts/compare_reproduction.py data/results/external_v3_plantseg reproduction/external_v3_plantseg
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def compare_file(a_path: Path, b_path: Path, atol: float) -> list[str]:
    with a_path.open(newline="") as fa, b_path.open(newline="") as fb:
        a, b = list(csv.reader(fa)), list(csv.reader(fb))
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return [f"{a_path.name}: shape differs ({len(a)} vs {len(b)} rows)"]
    problems, worst = [], 0.0
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        for c, (x, y) in enumerate(zip(row_a, row_b)):
            nx, ny = _number(x), _number(y)
            if nx is not None and ny is not None:
                if not (math.isclose(nx, ny, abs_tol=atol) or (math.isnan(nx) and math.isnan(ny))):
                    worst = max(worst, abs(nx - ny))
                    problems.append(f"{a_path.name}: row {r} col {c}: {x} vs {y}")
            elif x != y:
                problems.append(f"{a_path.name}: row {r} col {c}: '{x}' vs '{y}'")
    if len(problems) > 5:
        problems = problems[:5] + [f"{a_path.name}: ... {len(problems) - 5} more (max abs numeric difference {worst:.3g})"]
    return problems


def compare(committed: Path, rerun: Path, atol: float) -> list[str]:
    files = sorted(committed.glob("*.csv"))
    if not files:
        return [f"{committed}: no committed CSV files to compare"]
    problems = []
    for path in files:
        other = rerun / path.name
        problems += [f"{path.name}: missing in re-run"] if not other.is_file() else compare_file(path, other, atol)
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
