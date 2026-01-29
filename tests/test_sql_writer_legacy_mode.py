import os
import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-sql-writer"
sys.path.insert(0, str(SERVICE_ROOT))

from app.worker import _legacy_action  # noqa: E402
from app.spark_contract_metrics import LEGACY_KINDS  # noqa: E402
from app.settings import Settings  # noqa: E402


class TestSqlWriterLegacyMode(unittest.TestCase):
    def test_legacy_action_accept(self):
        kind = next(iter(LEGACY_KINDS))
        self.assertEqual(_legacy_action(kind, "accept", LEGACY_KINDS), "accept")

    def test_legacy_action_warn(self):
        kind = next(iter(LEGACY_KINDS))
        self.assertEqual(_legacy_action(kind, "warn", LEGACY_KINDS), "warn")

    def test_legacy_action_drop(self):
        kind = next(iter(LEGACY_KINDS))
        self.assertEqual(_legacy_action(kind, "drop", LEGACY_KINDS), "drop")

    def test_legacy_action_noop(self):
        self.assertEqual(_legacy_action("spark.telemetry", "warn", LEGACY_KINDS), "noop")

    def test_snapshot_channel_flag(self):
        prior = os.environ.get("SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL")
        try:
            os.environ["SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL"] = "false"
            settings = Settings()
            self.assertNotIn("orion:spark:state:snapshot", settings.effective_subscribe_channels)

            os.environ["SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL"] = "true"
            settings = Settings()
            self.assertIn("orion:spark:state:snapshot", settings.effective_subscribe_channels)
        finally:
            if prior is None:
                os.environ.pop("SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL", None)
            else:
                os.environ["SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL"] = prior


if __name__ == "__main__":
    unittest.main()
