"""Read-only submission checks; no training, inference, downloads or result writes.

Run from any directory: python /path/to/repository/scripts/audit_repository.py
Use --include-local-results to also inspect ignored local prediction exports.
Original Colab checkpoints require the separate audit_notebook15_results.py.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import nbformat
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLASSES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def audit_notebooks():
    transformer = TransformerManager()
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    require([int(p.name[:2]) for p in notebooks] == list(range(1, 16)), "Canonical sequence must be 01–15")
    rows = []
    archives = sorted((ROOT / "notebooks/archive").glob("*.ipynb"))
    for path in notebooks + archives:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_count, error_count = 0, 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            code_count += 1
            tree = ast.parse(transformer.transform_cell(cell.source))
            compile(tree, f"{path.name}:cell{index}", "exec")
            # Check local from-imports without executing notebook setup/training.
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        require(alias.name == "*" or hasattr(module, alias.name),
                                f"{path.name}:cell{index}: missing import {node.module}.{alias.name}")
            error_count += sum(o.output_type == "error" for o in cell.get("outputs", []))
        if path in notebooks:
            require(error_count == 0, f"{path.name}: saved error output")
        rows.append({"notebook": str(path.relative_to(ROOT)), "code_cells": code_count,
                     "historical_error_outputs": error_count})
    return rows


def audit_links(tracked):
    missing_local = []
    for relative in tracked:
        path = ROOT / relative
        if path.suffix == ".md":
            content = path.read_text()
        elif path.suffix == ".ipynb":
            content = "\n".join(c.source for c in nbformat.read(path, as_version=4).cells
                                if c.cell_type == "markdown")
        else:
            continue
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"\w+://|#|mailto:", target):
                continue
            destination = (path.parent / unquote(target.split("#")[0])).resolve()
            if destination.exists():
                continue
            try:
                name = str(destination.relative_to(ROOT))
            except ValueError:
                raise ValueError(f"Broken repository link: {relative}: {target}")
            ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", name]).returncode == 0
            require(ignored, f"Broken repository link: {relative}: {target}")
            missing_local.append({"document": relative, "local_dependency": name})
    return missing_local


def audit_splits():
    mapping = json.loads((ROOT / "data/splits/label_map.json").read_text())
    require(mapping == dict(zip(CLASSES, range(5))), "Canonical class mapping differs")
    sizes = {"lab_train": 5365, "lab_val": 1341, "lab_test": 1692, "field_train": 476,
             "field_adapt_train": 370, "field_dev": 92, "field_quarantined_train": 14, "field_test": 47}
    frames = {name: pd.read_csv(ROOT / f"data/splits/{name}.csv") for name in sizes}
    for name, frame in frames.items():
        require(len(frame) == sizes[name], f"{name}: count differs")
        require(not frame.duplicated(["source", "row_id"]).any(), f"{name}: duplicate image identity")
        require(frame.class_name.map(mapping).eq(frame.label_id).all(), f"{name}: label mapping differs")
    field_parts = ["field_adapt_train", "field_dev", "field_quarantined_train"]
    require(set(pd.concat([frames[n] for n in field_parts]).row_id) == set(frames["field_train"].row_id),
            "Adapt/Dev/quarantine do not partition original training rows")
    for family in [["lab_train", "lab_val", "lab_test"], field_parts + ["field_test"]]:
        for left, right in itertools.combinations(family, 2):
            for key in ["row_id", "group_id", "exact_hash"]:
                if key in frames[left] and key in frames[right]:
                    require(set(frames[left][key]).isdisjoint(frames[right][key]),
                            f"{left}/{right}: overlapping {key}")
    from src.evaluate_field_dev_verified import validate_manifests
    validate_manifests(ROOT / "data/splits")
    return sizes


def audit_predictions(paths):
    records = []
    for path in sorted(paths):
        if path.suffix != ".csv" or "predict" not in path.name:
            continue
        frame = pd.read_csv(path)
        if not {"label_id", "predicted_label_id"} <= set(frame):
            continue
        for column in ["label_id", "predicted_label_id"]:
            require(frame[column].isin(range(5)).all(), f"{path}: invalid {column}")
        if "correct" in frame:
            flags = frame.correct.astype(str).str.lower().map({"true": True, "false": False})
            require(flags.notna().all() and flags.eq(frame.label_id.eq(frame.predicted_label_id)).all(),
                    f"{path}: incorrect correctness flags")
        columns = [f"probability_label_{i}" for i in range(5)]
        if set(columns) <= set(frame):
            values = frame[columns].to_numpy(float)
            require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
                    and np.allclose(values.sum(1), 1, atol=1e-6), f"{path}: invalid probabilities")
            require(np.array_equal(values.argmax(1), frame.predicted_label_id), f"{path}: argmax mismatch")
            if "confidence" in frame:
                require(np.allclose(values.max(1), frame.confidence, atol=1e-6), f"{path}: confidence mismatch")
        keys = [k for k in ["protocol", "partition", "regime", "model_key", "seed", "condition"] if k in frame]
        groups = frame.groupby(keys) if keys else [(None, frame)]
        for key, group in groups:
            # Different protocols/regimes remain separate; no pooling repeated seeds.
            records.append({"path": str(path.relative_to(ROOT)), "group": str(key), "images": len(group),
                            "accuracy": accuracy_score(group.label_id, group.predicted_label_id),
                            "macro_f1": f1_score(group.label_id, group.predicted_label_id, labels=range(5),
                                                 average="macro", zero_division=0)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local-results", action="store_true")
    args = parser.parse_args()
    tracked = [p for p in git("ls-files", "-z").split("\0") if p]
    modules = sorted((ROOT / "src").rglob("*.py"))
    python_files = modules + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "app").glob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    for path in modules:
        importlib.import_module(".".join(path.relative_to(ROOT).with_suffix("").parts))
    notebooks = audit_notebooks()
    dependencies = audit_links(tracked)
    sizes = audit_splits()
    paths = {ROOT / p for p in tracked if p.startswith(("outputs/", "data/results/"))}
    if args.include_local_results:
        paths.update((ROOT / "outputs").rglob("*.csv"))
    predictions = audit_predictions(paths)
    from src.controlled_field_report import load_evidence
    evidence = load_evidence(ROOT)
    means = evidence["summary"].groupby("condition").macro_f1.agg(["mean", "std"])
    print(json.dumps({"status": "passed", "python_files_parsed": len(python_files),
                      "src_modules_imported": len(modules), "notebooks": notebooks,
                      "missing_ignored_local_links": dependencies, "split_sizes": sizes,
                      "prediction_files_checked": len({r["path"] for r in predictions}),
                      "prediction_groups_recalculated": len(predictions),
                      "controlled_macro_f1": means.to_dict("index"),
                      "scope": "Static notebook checks, local imports, split metadata and saved predictions; no training/inference",
                      "limitations": "Archived error outputs are historical evidence. Optional app/Colab runtimes, image-content independence, and missing private artifacts are not certified."}, indent=2))


if __name__ == "__main__":
    main()
