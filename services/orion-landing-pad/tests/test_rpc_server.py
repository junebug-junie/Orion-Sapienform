from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
PACKAGE_NAME = "orion_landing_pad"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"

if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

spec = spec_from_file_location(f"{APP_PACKAGE_NAME}.rpc.server", APP_DIR / "rpc" / "server.py")
server_mod = module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = server_mod
spec.loader.exec_module(server_mod)

stats_spec = spec_from_file_location(f"{APP_PACKAGE_NAME}.observability.stats", APP_DIR / "observability" / "stats.py")
stats_mod = module_from_spec(stats_spec)
assert stats_spec and stats_spec.loader
sys.modules[stats_spec.name] = stats_mod
stats_spec.loader.exec_module(stats_mod)


class _DummyStore:
    async def get_latest_frame(self):
        return None

    async def get_frames(self, limit: int = 10):
        return []

    async def get_salient_events(self, limit: int = 20):
        return []

    async def get_latest_tensor(self):
        return None


class _DummyBus:
    pass


def test_get_stats_returns_tracker_snapshot():
    tracker = stats_mod.PadStatsTracker(tick_seconds=15)
    tracker.increment_ingested()
    tracker.increment_frames_built()
    tracker.increment_rpc_requests()
    tracker.increment_dropped(reason="queue_full")
    tracker.set_queue_depth(3)
    tracker.record_salient(0.82)

    server = server_mod.PadRpcServer(
        bus=_DummyBus(),
        store=_DummyStore(),
        settings=SimpleNamespace(app_name="pad", service_version="1.0", node_name="node-a"),
        stats=tracker,
    )

    result = __import__("asyncio").run(server._get_stats({}))

    assert result["stats"]["ingested"] == 1
    assert result["stats"]["frames_built"] == 1
    assert result["stats"]["rpc_requests"] == 1
    assert result["stats"]["queue_depth"] == 3
    assert result["stats"]["last_salience"] == 0.82
    assert result["stats"]["dropped_by_reason"]["queue_full"] == 1


def test_handler_mapping_includes_get_stats():
    tracker = stats_mod.PadStatsTracker(tick_seconds=15)
    server = server_mod.PadRpcServer(
        bus=_DummyBus(),
        store=_DummyStore(),
        settings=SimpleNamespace(app_name="pad", service_version="1.0", node_name="node-a"),
        stats=tracker,
    )

    handler = server._handler_for("get_stats")

    assert handler is not None
    assert handler.__name__ == "_get_stats"
