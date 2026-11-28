"""Download project checkpoints from the public Drive folder and verify SHA-256.

Reads ``data/checkpoints.json``. Each file is streamed to a temporary file,
hashed, and moved into place only if the hash matches; existing files with the
correct hash are skipped. Standard library only.

Usage from the repository root::

    python scripts/download_checkpoints.py                # core: 4 fixed models + 6 Notebook 15 checkpoints
    python scripts/download_checkpoints.py --group all    # also historical augmentation / adaptation runs
    python scripts/download_checkpoints.py --verify-only  # hash local files, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select(entries: list[dict], group: str) -> list[dict]:
    return entries if group == "all" else [e for e in entries if e["group"] == group]


def download(entry: dict, root: Path) -> str:
    target = root / entry["local_path"]
    if target.is_file() and sha256(target) == entry["sha256"]:
        return "ok (already present)"
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=target.parent, prefix=".download-")
    try:
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(URL.format(id=entry["drive_id"]), timeout=120) as response:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
        digest = sha256(Path(tmp))
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {entry['key']}: got {digest}")
        os.replace(tmp, target)
        return "downloaded and verified"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=["core", "historical", "all"], default="core")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory that local_path entries are relative to")
    args = parser.parse_args()
    registry = json.loads((ROOT / "data/checkpoints.json").read_text())
    failures = 0
    for entry in select(registry["checkpoints"], args.group):
        target = args.root / entry["local_path"]
        try:
            if args.verify_only:
                status = ("ok" if target.is_file() and sha256(target) == entry["sha256"]
                          else "missing" if not target.is_file() else "HASH MISMATCH")
            else:
                status = download(entry, args.root)
        except Exception as error:  # report every file, then fail
            status = f"FAILED: {error}"
        failures += not status.startswith(("ok", "downloaded"))
        print(f"{entry['key']:<28} {status}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
