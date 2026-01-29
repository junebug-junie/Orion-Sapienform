import unittest
from datetime import datetime, timezone

from orion.normalizers.spark import (
    normalize_spark_state_snapshot,
    normalize_spark_telemetry,
)
from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload


class TestSparkNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.snapshot_dict = {
            "source_service": "spark-introspector",
            "producer_boot_id": "boot-123",
            "seq": 42,
            "snapshot_ts": self.now.isoformat(),
            "phi": {"coherence": 0.3, "novelty": 0.2},
        }

    def test_normalize_canonical_snapshot_dict(self):
        snap = normalize_spark_state_snapshot(self.snapshot_dict, now=self.now)
        self.assertIsInstance(snap, SparkStateSnapshotV1)
        self.assertEqual(snap.source_service, "spark-introspector")
        self.assertEqual(snap.seq, 42)
        self.assertEqual(snap.snapshot_ts.tzinfo, timezone.utc)
        self.assertEqual(snap.phi.get("coherence"), 0.3)

    def test_normalize_telemetry_with_nested_snapshot_in_metadata(self):
        telemetry_dict = {
            "correlation_id": "corr-123",
            "timestamp": self.now.isoformat(),
            "metadata": {"spark_state_snapshot": self.snapshot_dict, "extra": "keep"},
        }
        telemetry = normalize_spark_telemetry(telemetry_dict, now=self.now)
        self.assertIsInstance(telemetry, SparkTelemetryPayload)
        self.assertIsNotNone(telemetry.state_snapshot)
        self.assertIsInstance(telemetry.state_snapshot, SparkStateSnapshotV1)
        self.assertEqual(telemetry.metadata.get("extra"), "keep")

    def test_normalize_partial_payload_does_not_crash(self):
        payloads = [
            {"phi": 0.9},
            {"correlation_id": "corr-only"},
            {"snapshot_ts": "2026-01-01T00:00:00Z"},
        ]
        for payload in payloads:
            self.assertIsNone(normalize_spark_state_snapshot(payload, now=self.now))
            self.assertIsNone(normalize_spark_telemetry(payload, now=self.now))


if __name__ == "__main__":
    unittest.main()
