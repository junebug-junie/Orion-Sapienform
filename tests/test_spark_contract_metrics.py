import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-sql-writer"
sys.path.insert(0, str(SERVICE_ROOT))

from app.spark_contract_metrics import (  # noqa: E402
    SparkContractMetrics,
    LEGACY_KINDS,
    CANONICAL_KINDS,
)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *args):
        self.messages.append((msg, args))


class TestSparkContractMetrics(unittest.TestCase):
    def test_observe_counts(self):
        metrics = SparkContractMetrics(emit_interval_sec=999)
        canonical_kind = next(iter(CANONICAL_KINDS))
        legacy_kind = next(iter(LEGACY_KINDS))
        other_kind = "spark.unknown.kind"

        metrics.observe(canonical_kind)
        metrics.observe(legacy_kind)
        metrics.observe(other_kind)

        self.assertEqual(metrics.counts_total, 3)
        self.assertEqual(metrics.canonical_count, 1)
        self.assertEqual(metrics.legacy_count, 1)
        self.assertEqual(metrics.other_spark_count, 1)
        self.assertEqual(metrics.counts_by_kind[canonical_kind], 1)
        self.assertEqual(metrics.counts_by_kind[legacy_kind], 1)
        self.assertEqual(metrics.counts_by_kind[other_kind], 1)

    def test_maybe_emit_interval(self):
        metrics = SparkContractMetrics(emit_interval_sec=10.0)
        logger = FakeLogger()

        metrics.observe("spark.telemetry")
        metrics.maybe_emit(logger, node="node-a", service="sql-writer", now_ts=100.0)
        self.assertEqual(len(logger.messages), 1)

        metrics.maybe_emit(logger, node="node-a", service="sql-writer", now_ts=105.0)
        self.assertEqual(len(logger.messages), 1)

        metrics.maybe_emit(logger, node="node-a", service="sql-writer", now_ts=111.0)
        self.assertEqual(len(logger.messages), 2)


if __name__ == "__main__":
    unittest.main()
