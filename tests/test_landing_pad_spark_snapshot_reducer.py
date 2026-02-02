import asyncio
import sys
import unittest
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pad import PadEventV1


def _load_snapshot_reducer():
    root = Path(__file__).resolve().parents[1]
    reducers_path = root / "services" / "orion-landing-pad" / "app" / "reducers" / "stubs.py"
    package = "orion_landing_pad_reducers"
    spec = spec_from_file_location(f"{package}.stubs", reducers_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load landing pad reducer module")
    module = module_from_spec(spec)
    sys.modules[f"{package}.stubs"] = module
    spec.loader.exec_module(module)
    return module.snapshot_reducer


class TestLandingPadSparkSnapshotReducer(unittest.TestCase):
    def test_snapshot_reducer_ignores_extra_keys(self) -> None:
        snapshot_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        payload = {
            "source_service": "spark-introspector",
            "source_node": "athena",
            "producer_boot_id": "boot",
            "seq": 7,
            "snapshot_ts": snapshot_ts.isoformat(),
            "phi": {"coherence": 0.5, "novelty": 0.33},
            "valence": 0.6,
            "arousal": 0.2,
            "dominance": 0.1,
            "vector_present": False,
            "metadata": {},
            "junk_key": "ignored",
        }
        env = BaseEnvelope(
            kind="spark.state.snapshot.v1",
            source=ServiceRef(name="spark-introspector", node="athena"),
            payload=payload,
        )
        snapshot_reducer = _load_snapshot_reducer()
        event = asyncio.run(snapshot_reducer(env, channel="orion:spark:state:snapshot"))
        self.assertIsInstance(event, PadEventV1)
        self.assertEqual(event.subject, "athena")
        self.assertEqual(event.ts_ms, int(snapshot_ts.timestamp() * 1000))
        self.assertIn("junk_key", event.payload)

    def test_snapshot_reducer_handles_parse_failure(self) -> None:
        payload = {"source_node": "atlas", "junk_key": "ignored"}
        env = BaseEnvelope(
            kind="spark.state.snapshot.v1",
            source=ServiceRef(name="spark-introspector", node="athena"),
            payload=payload,
        )
        snapshot_reducer = _load_snapshot_reducer()
        event = asyncio.run(snapshot_reducer(env, channel="orion:spark:state:snapshot"))
        self.assertIsInstance(event, PadEventV1)
        self.assertEqual(event.subject, "atlas")


if __name__ == "__main__":
    unittest.main()
