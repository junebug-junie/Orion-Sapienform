import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-sql-writer"
sys.path.insert(0, str(SERVICE_ROOT))

from app.worker import _map_spark_to_telemetry_row  # noqa: E402


class TestSqlWriterSparkMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.snapshot = {
            "source_service": "spark-introspector",
            "producer_boot_id": "boot-123",
            "seq": 1,
            "snapshot_ts": self.now.isoformat(),
            "phi": {"coherence": 0.7, "novelty": 0.2},
            "valence": 0.9,
            "metadata": {"extra": "keep"},
        }

    def test_snapshot_maps_phi_from_coherence(self):
        row = _map_spark_to_telemetry_row(
            "spark.state.snapshot.v1",
            self.snapshot,
            envelope_correlation_id="corr-1",
            envelope_id="env-1",
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["phi"], 0.7)
        self.assertEqual(row["novelty"], 0.2)
        self.assertEqual(row["correlation_id"], "corr-1")
        self.assertIn("spark_state_snapshot", row["metadata_"])
        self.assertEqual(row["metadata_"]["extra"], "keep")

    def test_telemetry_uses_snapshot_when_phi_missing(self):
        payload = {
            "correlation_id": "corr-2",
            "timestamp": self.now.isoformat(),
            "metadata": {"spark_state_snapshot": self.snapshot},
        }
        row = _map_spark_to_telemetry_row(
            "spark.telemetry",
            payload,
            envelope_correlation_id="corr-override",
            envelope_id="env-2",
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["phi"], 0.7)
        self.assertEqual(row["novelty"], 0.2)
        self.assertEqual(row["correlation_id"], "corr-2")


if __name__ == "__main__":
    unittest.main()
