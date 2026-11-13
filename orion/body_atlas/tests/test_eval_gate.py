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


def test_the_real_store_fixture_moves_code_named_stores_into_their_writers_organs():
    """Live 2026-10-30: 8 Chroma/FalkorDB stores were lone organs. With code evidence the three
    that code names join their services; the five no code names stay alone and are listed."""
    catalog, observed, now, body = _EVAL._fixture_inputs(_EVAL.STORES_FIXTURE)
    atlas = build_atlas(catalog, observed, now=now, body=body)
    assert _EVAL.evaluate(atlas) == []
    organ = {m: o for o in atlas.organs for m in o.members}
    for store, svc in (("chroma:orion_social_room", "orion-vector-host"),
                       ("chroma:orion_atlas_concepts", "orion-hub"),
                       ("chroma:orion_day_letter_parts", "orion-hub")):
        assert f"part:{svc}" in organ[f"part:{store}"].members, store
    alone = sorted(o.members[0] for o in atlas.organs if len(o.members) == 1)
    assert alone == [f"part:falkor:{g}" for g in
                     ("orion_bus_mirror", "orion_episodic", "orion_express", "orion_kg", "orion_turns")]
    # the same body without code evidence: the eight lone stores of before
    for st in body.stores:
        st.links = [ln for ln in st.links if ln[2] != "declared_code"]
    before = build_atlas(catalog, observed, now=now, body=body)
    assert before.stats.singleton_organs == 8


def test_a_linked_store_left_alone_fails_the_eval():
    from orion.body_atlas.sources import BodyInputs, StoreIn
    catalog = [{"name": "orion:a:b", "producer_services": ["orion-a"], "consumer_services": ["orion-b"]}]
    atlas = build_atlas(catalog, None, now=NOW, body=BodyInputs(
        stores=[StoreIn(name="s", store="chroma", rows=1, links=[("orion-a", "writes", "declared_code", "m.py")])]))
    atlas.organs = [o.model_copy(update={"members": [m for m in o.members if m != "part:chroma:s"]})
                    for o in atlas.organs] + [atlas.organs[0].model_copy(update={"members": ["part:chroma:s"]})]
    assert any("stores with a writes, reads or uses flow sit alone" in f for f in _EVAL.evaluate(atlas))


def test_store_directions_on_this_checkout_are_true():
    """Ground truth for writes / reads: hand-checked facts from the live repo (run_store_direction_eval.py)."""
    spec = importlib.util.spec_from_file_location(
        "run_store_direction_eval", Path(__file__).parents[1] / "evals" / "run_store_direction_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    failures, summary = ev.evaluate(Path(__file__).resolve().parents[3])
    assert failures == [], summary


_WSPEC = importlib.util.spec_from_file_location(
    "run_wire_eval", Path(__file__).parents[1] / "evals" / "run_wire_eval.py")
_WIRE = importlib.util.module_from_spec(_WSPEC)
_WSPEC.loader.exec_module(_WIRE)


def test_the_wire_rules_pass_on_both_frozen_bodies_and_fail_on_a_wrong_expected_set(monkeypatch):
    for path in (_WIRE.HELD_OUT, _WIRE.TODAY):
        assert _WIRE.rule_failures(path, verbose=False) == []
    wrong = {**_WIRE.EXPECTED, _WIRE.TODAY.name: {**_WIRE.EXPECTED[_WIRE.TODAY.name],
                                                  "R-broadcast-key": {"part:redis:orion:situation:latest"}}}
    monkeypatch.setattr(_WIRE, "EXPECTED", wrong)
    assert any("R-broadcast-key: precision 0.00 recall 0.00" in f for f in _WIRE.rule_failures(_WIRE.TODAY, verbose=False))
