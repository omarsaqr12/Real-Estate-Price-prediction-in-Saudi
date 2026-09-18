"""No model weights, dataset or TensorFlow required for these contract tests."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.scripts.check_assets import REQUIRED, missing_assets


class AssetChecks(unittest.TestCase):
    def test_missing_assets_are_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(missing_assets(root, "serve"), list(REQUIRED["serve"]))
            for name in REQUIRED["serve"]:
                (root / name).touch()
            self.assertEqual(missing_assets(root, "serve"), [])

    def test_train_needs_database_not_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(missing_assets(root, "train"), ["PandA.db"])
            (root / "PandA.db").touch()
            self.assertEqual(missing_assets(root, "train"), [])

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown mode"):
            missing_assets(Path("."), "deploy")

    def test_cli_reports_blocked_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = subprocess.run([sys.executable, "src/scripts/check_assets.py", "--mode", "serve", "--root", directory],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("price_prediction_model.keras", proc.stdout)


if __name__ == "__main__":
    unittest.main()
