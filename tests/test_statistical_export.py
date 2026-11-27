"""Statistical reproduction must never overwrite an existing destination."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import statistical_analysis


class ExportSafetyTests(unittest.TestCase):
    def test_existing_destination_is_refused_before_any_export(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "evidence.csv"
            sentinel.write_bytes(b"original evidence\n")
            with patch("sys.argv", ["statistical_analysis", "--output", directory]), \
                 patch.object(statistical_analysis, "extract_notebook_artifacts") as export:
                with self.assertRaises(FileExistsError):
                    statistical_analysis.main()
                export.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"original evidence\n")
