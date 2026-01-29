import unittest
from pathlib import Path


class TestSparkIntrospectorNoLegacy(unittest.TestCase):
    def test_worker_has_no_legacy_kind(self):
        worker_path = (
            Path(__file__).resolve().parents[1]
            / "services"
            / "orion-spark-introspector"
            / "app"
            / "worker.py"
        )
        data = worker_path.read_text(encoding="utf-8")
        self.assertNotIn("spark.introspection", data)


if __name__ == "__main__":
    unittest.main()
