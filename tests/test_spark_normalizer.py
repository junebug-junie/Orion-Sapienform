import importlib.util
import sys
import types
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


class TestSparkTelemetryChatMeta(unittest.TestCase):
    def test_turn_effect_in_spark_meta_minimal(self):
        repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
        worker_path = repo_root / "services" / "orion-sql-writer" / "app" / "worker.py"
        package_name = "orion_sql_writer"
        app_package_name = f"{package_name}.app"
        if "app" not in sys.modules:
            sys.modules["app"] = types.ModuleType("app")
        if "app.settings" not in sys.modules:
            settings_mod = types.ModuleType("app.settings")
            settings_mod.settings = object()
            sys.modules["app.settings"] = settings_mod
        if "app.db" not in sys.modules:
            db_mod = types.ModuleType("app.db")
            db_mod.get_session = lambda: None
            db_mod.remove_session = lambda: None
            sys.modules["app.db"] = db_mod
        if "app.models" not in sys.modules:
            models_mod = types.ModuleType("app.models")
            for name in (
                "BiometricsTelemetry",
                "BiometricsSummarySQL",
                "BiometricsInductionSQL",
                "ChatHistoryLogSQL",
                "ChatMessageSQL",
                "CollapseEnrichment",
                "CollapseMirror",
                "Dream",
                "SparkIntrospectionLogSQL",
                "SparkTelemetrySQL",
                "BusFallbackLog",
                "CognitionTraceSQL",
                "MetacognitionTickSQL",
                "MetacogTriggerSQL",
            ):
                setattr(models_mod, name, type(name, (), {}))
            sys.modules["app.models"] = models_mod
        if "app.spark_contract_metrics" not in sys.modules:
            metrics_mod = types.ModuleType("app.spark_contract_metrics")
            metrics_mod.SparkContractMetrics = type("SparkContractMetrics", (), {})
            metrics_mod.LEGACY_KINDS = set()
            sys.modules["app.spark_contract_metrics"] = metrics_mod
        if package_name not in sys.modules:
            pkg = types.ModuleType(package_name)
            pkg.__path__ = [str(worker_path.parent.parent)]
            sys.modules[package_name] = pkg
        if app_package_name not in sys.modules:
            pkg = types.ModuleType(app_package_name)
            pkg.__path__ = [str(worker_path.parent)]
            sys.modules[app_package_name] = pkg
        spec = importlib.util.spec_from_file_location(f"{app_package_name}.worker", worker_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _spark_meta_minimal = module._spark_meta_minimal

        row = {
            "phi": 0.1,
            "novelty": 0.2,
            "trace_mode": "chat",
            "trace_verb": "reply",
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "stimulus_summary": "summary",
            "node": "node-1",
            "metadata": {
                "turn_effect": {"user": {"valence": 0.1}},
                "turn_effect_summary": "user: v+0.10",
            },
        }
        meta = _spark_meta_minimal(row)
        self.assertEqual(meta.get("turn_effect"), {"user": {"valence": 0.1}})
        self.assertEqual(meta.get("turn_effect_summary"), "user: v+0.10")


if __name__ == "__main__":
    unittest.main()
