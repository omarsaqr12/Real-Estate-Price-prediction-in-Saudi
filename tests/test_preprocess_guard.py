"""Verify preprocessing refuses absent data without creating an empty DB."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "preprocess_data.py"


class PreprocessGuardTests(unittest.TestCase):
    def test_missing_database_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, str(SCRIPT)], cwd=root,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Expected existing training database", result.stderr)
            self.assertFalse((root / "PandA.db").exists())
            self.assertFalse((root / "database.db").exists())


if __name__ == "__main__":
    unittest.main()
