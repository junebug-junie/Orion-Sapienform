"""The organ-structure eval must be able to fail: a blob body and a split pair are caught."""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from orion.body_atlas.build import build_atlas

_SPEC = importlib.util.spec_from_file_location(
    "run_organ_structure_eval", Path(__file__).parents[1] / "evals" / "run_organ_structure_eval.py")
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)
NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def test_a_fully_connected_body_fails_on_q_count_and_blob_share():
    names = [f"orion-s{i}" for i in range(8)]
    catalog = [{"name": f"orion:x:{a}:{b}", "producer_services": [a], "consumer_services": [b]}
               for a in names for b in names if a != b]
    failures = _EVAL.evaluate(build_atlas(catalog, None, now=NOW))
    assert any("modularity" in f for f in failures)
    assert any("organs outside" in f for f in failures)
    assert any("largest organ" in f for f in failures)


def test_the_frozen_real_bus_graph_passes():
    catalog, observed, now, body = _EVAL._fixture_inputs(_EVAL.FIXTURE)
    assert body is None
    assert _EVAL.evaluate(build_atlas(catalog, observed, now=now)) == []


def test_the_frozen_whole_body_passes_and_keeps_its_new_kinds():
    catalog, observed, now, body = _EVAL._fixture_inputs(_EVAL.WHOLE_BODY_FIXTURE)
    atlas = build_atlas(catalog, observed, now=now, body=body)
    assert _EVAL.evaluate(atlas) == []
    kinds = {p.kind for p in atlas.parts if p.status != "transient"}
    assert {"host", "sensor", "store", "process"} <= kinds
    in_organs = {m for o in atlas.organs for m in o.members}
    assert not any(m.startswith("part:host:") for m in in_organs)
