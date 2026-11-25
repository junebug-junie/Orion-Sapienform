"""PE retire Patch 4 (design rev 4 item 5): node:prediction.street_rhythm is a WORLD node in SHADOW.

Shadow = written, recorded and visible on debug surfaces, but it does not compete in the attention contests
(so it cannot win, author a goal or reach Prime) and cannot seed curiosity until its 7-day calm check passes.
Going live is one line: orion/schemas/attention_candidate.py STREET_RHYTHM_SHADOW = False. These tests pin
both states, so the flip cannot silently do nothing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orion.schemas import attention_candidate as ac

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
NID = "node:prediction.street_rhythm"
REPO = Path(__file__).resolve().parents[1]


def _node(value: float, *, age_s: float = 30.0):
    return SimpleNamespace(
        node_id=NID, label=NID, node_kind="concept", metadata={"prediction_error": value},
        signals=SimpleNamespace(confidence=1.0),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=age_s)),
    )


def _magnitude(pct: float, *, value: float = 0.9, age: float = 30.0, band: str = "unusual"):
    from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

    return PredictionErrorMagnitudeV1(value=value, age_sec=age, percentile_now=pct, n_readings_7d=4000,
                                      band=band, trend="flat")


def test_ships_in_shadow():
    assert ac.STREET_RHYTHM_SHADOW is True
    assert ac.SHADOW_NODE_IDS == frozenset({NID})
    assert ac.STREET_RHYTHM_NODE_ID == NID


def test_it_is_world_not_body():
    from orion.attention.world_first import external_max_age_sec, source_kind_for_node

    assert ac.is_world_source_id(NID)
    assert source_kind_for_node(NID) == "external"
    assert external_max_age_sec(NID) == 600.0


def test_body_only_gates_never_read_it():
    from orion.attention.self_signal_outcomes import _still_a_body_signal

    assert not _still_a_body_signal(NID)
    import importlib.util
    import sys

    path = REPO / "services" / "orion-equilibrium-service" / "app" / "self_signal_episode_gate.py"
    sys.path.insert(0, str(path.parents[1]))
    try:
        spec = importlib.util.spec_from_file_location("ssg_for_test", path)
        gate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gate
        spec.loader.exec_module(gate)
    finally:
        sys.path.remove(str(path.parents[1]))
    obs = gate.observe_signals({NID: (_magnitude(0.999), NOW, 0.999)}, now=NOW)
    assert all(o.signal_id != NID for o in obs)


def test_shadow_fences_it_from_the_contests_and_curiosity():
    assert ac.world_first_fenced(NID)
    assert NID in ac.NO_CURIOSITY_CONSUMER_NODE_IDS


def test_shadow_node_is_no_contest_candidate_in_the_broadcast():
    from orion.substrate import attention_broadcast as ab

    frame = ab.build_substrate_attention_frame(nodes=[_node(1.0)], now=NOW, world_first=True,
                                               magnitude_by_node_id={NID: _magnitude(0.999, value=1.0)})
    assert all(c["source_id"] != NID for c in frame.debug["world_first"]["candidates"])


def test_shadow_node_never_seeds_curiosity():
    from orion.substrate.endogenous_curiosity import EndogenousCuriosityConfig, endogenous_curiosity_candidates

    out = endogenous_curiosity_candidates(nodes=[_node(1.0)], config=EndogenousCuriosityConfig(enabled=True, budget=8),
                                          now=NOW)
    assert all(NID not in c.focal_node_refs for c in out)


def test_after_the_flip_it_competes_and_seeds(monkeypatch):
    """What STREET_RHYTHM_SHADOW = False does: the same two sets lose the id, nothing else changes."""
    from orion.substrate import endogenous_curiosity as ec

    monkeypatch.setattr(ac, "SHADOW_NODE_IDS", frozenset())
    monkeypatch.setattr(ec, "NO_CURIOSITY_CONSUMER_NODE_IDS", frozenset({ac.STREET_NODE_ID}))
    assert not ac.world_first_fenced(NID)
    out = ec.endogenous_curiosity_candidates(nodes=[_node(1.0)],
                                             config=ec.EndogenousCuriosityConfig(enabled=True, budget=8), now=NOW)
    assert any(NID in c.focal_node_refs for c in out)


def test_curiosity_bar_is_set_for_go_live():
    from orion.substrate import curiosity_seed_bar as csb

    assert csb.CURIOSITY_SEED_BAR_BY_NODE_ID[NID] == pytest.approx(1.8 / 3.0)
    assert NID not in csb.CURIOSITY_SEED_CEILING_BY_NODE_ID


def test_glossary_entry_is_an_honest_event_written_forecast_miss():
    from orion.attention.world_first import node_prediction_error_event_written, node_prediction_error_semantics
    from orion.field.channel_glossary import resolve_channel_entry

    e = resolve_channel_entry("prediction_error", node=NID)
    assert e is not None and e.node == NID
    assert "how surprised the walkway rhythm forecast was by who did or didn't appear" in e.meaning.lower()
    sem = dict(e.semantics)
    assert sem["value_kind"] == "surprise_excess" and sem["sparsity"] == "event_gated"
    assert node_prediction_error_semantics(NID) == ("surprise_excess", "higher_is_worse")
    assert node_prediction_error_event_written(NID)


def test_registered_in_the_prediction_organ():
    doc = yaml.safe_load((REPO / "config" / "integration" / "organs.yaml").read_text())
    organs = doc.get("organs", doc)
    assert "orion:prediction:street_rhythm_surprise" in organs["prediction"]["channels"]
