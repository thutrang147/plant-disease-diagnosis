"""Pre-registration order and protocol integrity (see docs/provenance_timeline.md)."""

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = [("8762743", "cfc4752"), ("828bbfe", "4a4c8ff"), ("0d47ea8", "7c1a280"), ("7c1a280", "64d749b"),
         ("64d749b", "6a6faca")]


def history_available() -> bool:
    """True if every commit in ORDER is present (a depth-1 snapshot has none of them)."""
    for pair in ORDER:
        for sha in pair:
            out = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, capture_output=True)
            if out.returncode != 0:
                return False
    return True


class ProvenanceTest(unittest.TestCase):
    def test_protocol_hash_matches_run_record(self):
        record = json.loads((ROOT / "data/results/external_v3_plantseg/run_record.json").read_text())
        digest = hashlib.sha256((ROOT / "data/external_v3_plantseg/protocol.json").read_bytes()).hexdigest()
        self.assertEqual(record["protocol_sha256"], digest)

    @unittest.skipUnless(history_available(), "needs the git history that contains these commits")
    def test_declarations_precede_results(self):
        table = (ROOT / "docs/provenance_timeline.md").read_text()
        for before, after in ORDER:
            subprocess.run(["git", "merge-base", "--is-ancestor", before, after], cwd=ROOT, check=True)
            self.assertIn(f"`{before}`", table)
            self.assertIn(f"`{after}`", table)
