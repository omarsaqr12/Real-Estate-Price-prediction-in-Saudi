"""Source-only checks that can run from a clean checkout.

Missing model and database artifacts are reported by src/scripts/check_assets.py;
their absence must not fail source-only tests.
"""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_required_mapping_files_are_valid(self):
        for name in ("category_mapping", "city_mapping", "district_mapping"):
            path = ROOT / "src" / "data" / (name + ".json")
            with self.subTest(file=name):
                self.assertTrue(path.is_file())
                with path.open(encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), dict)

    def test_documented_entry_points_exist(self):
        for path in ("src/web/server.py", "src/web/client.py", "src/scripts/train_model.py"):
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
