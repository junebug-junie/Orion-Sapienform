import sys
import unittest
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_state_journaler() -> type:
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "services" / "orion-state-journaler" / "app"
    package = "orion_state_journaler_app"

    settings_spec = spec_from_file_location(f"{package}.settings", app_dir / "settings.py")
    if settings_spec is None or settings_spec.loader is None:
        raise RuntimeError("Unable to load state journaler settings module")
    settings_module = module_from_spec(settings_spec)
    sys.modules[f"{package}.settings"] = settings_module
    settings_spec.loader.exec_module(settings_module)

    service_spec = spec_from_file_location(f"{package}.service", app_dir / "service.py")
    if service_spec is None or service_spec.loader is None:
        raise RuntimeError("Unable to load state journaler service module")
    service_module = module_from_spec(service_spec)
    sys.modules[f"{package}.service"] = service_module
    service_spec.loader.exec_module(service_module)
    return service_module.StateJournaler


class TestStateJournalerSemantics(unittest.TestCase):
    def test_coherence_present(self) -> None:
        journaler = _load_state_journaler()()
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        journaler._add_spark_snapshot(
            {
                "source_service": "spark-introspector",
                "source_node": "athena",
                "producer_boot_id": "boot",
                "seq": 1,
                "snapshot_ts": now.isoformat(),
                "phi": {"coherence": 0.42, "novelty": 0.33},
                "valence": 0.6,
                "arousal": 0.2,
                "dominance": 0.1,
                "vector_present": False,
                "metadata": {},
            }
        )
        self.assertEqual(journaler.spark_events[0]["coherence"], 0.42)
        roll = journaler._compute_rollup(now, 60)
        self.assertEqual(roll["avg_coherence"], 0.42)

    def test_coherence_missing_never_uses_arousal(self) -> None:
        journaler = _load_state_journaler()()
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        journaler._add_spark_snapshot(
            {
                "source_service": "spark-introspector",
                "source_node": "athena",
                "producer_boot_id": "boot",
                "seq": 2,
                "snapshot_ts": now.isoformat(),
                "phi": {"novelty": 0.12},
                "valence": 0.7,
                "arousal": 0.99,
                "dominance": 0.2,
                "vector_present": False,
                "metadata": {},
            }
        )
        self.assertIsNone(journaler.spark_events[0]["coherence"])
        roll = journaler._compute_rollup(now, 60)
        self.assertEqual(roll["avg_coherence"], 0.0)


if __name__ == "__main__":
    unittest.main()
