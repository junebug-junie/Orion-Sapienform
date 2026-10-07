"""memory_distill runs on the 27B on gpu1 only (Juniper's choice, review 2026-10-02).

Never the gpu2 seat (a different model may be loaded there) and never the lent chat card.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from orion.gpu_pool.config import load_pool_config

HERE = Path(__file__).parent


def test_route_class_is_dedicated_and_only_the_gpu1_agent_role():
    cfg = load_pool_config()
    route = cfg.routes["memory_distill"]
    assert (route.work_class, route.priority) == ("memory_distill", "system")
    cls = cfg.classes["memory_distill"]
    assert list(cls.roles) == ["agent"]
    assert list(cfg.roles["agent"].cards) == ["gpu1"]


def test_route_view_never_places_it_on_chat_or_gpu2_in_any_golden_state():
    spec = importlib.util.spec_from_file_location("rv", HERE / "test_route_view.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    golden = json.loads((HERE / "fixtures_routes_compat_golden.json").read_text())
    for case, g in golden.items():
        view = rv.build_route_view(None, rv.CFG) if g["state"] is None else rv.build_route_view(
            {**g["state"], "config": rv._config_payload()})
        entry = next(e for e in view["routes"] if e["id"] == "memory_distill")
        assert entry.get("role") in (None, "agent"), (case, entry.get("role"))
        assert entry["served_by"] == "circe-worker-agent", case
