import unittest
from datetime import datetime, timezone

from orion.schemas.telemetry.spark_ack import SparkStateSnapshotAckV1


class TestSparkStateSnapshotAckV1(unittest.TestCase):
    def test_ack_serialization(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ack = SparkStateSnapshotAckV1(
            ok=True,
            received_ts=now,
            snapshot_seq=42,
            snapshot_ts=now,
            note="ok",
            source_service="state-service",
            source_node="athena",
        )
        payload = ack.model_dump(mode="json")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["snapshot_seq"], 42)
        self.assertEqual(_coerce_iso(payload["snapshot_ts"]), now)
        self.assertEqual(_coerce_iso(payload["received_ts"]), now)


def _coerce_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


if __name__ == "__main__":
    unittest.main()
