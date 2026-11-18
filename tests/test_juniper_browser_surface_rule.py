"""One rule for "a turn Juniper sent from a Hub browser surface" in SQL and in Python (#3602 review).

``JUNIPER_INGRESS_PREDICATE`` (SQL over cockpit_turn_sighting: the promise resolver, the report, the curiosity readout)
and ``is_browser_surface_turn`` (Python: the Hub promise view, Prime's hand-back) must agree on the edge cases: an empty
``{}`` surface_context and the operator smoke's ``operator_review`` surface are not Juniper.
"""

from __future__ import annotations

from orion.regulation.juniper_turns import JUNIPER_INGRESS_PREDICATE, SMOKE_SURFACE, is_browser_surface_turn
from orion.schemas.situation import SurfaceContextV1


def test_python_rule_edges():
    assert is_browser_surface_turn({"surface_context": {"surface": "hub_desktop", "input_modality": "typed"}})
    assert is_browser_surface_turn({"surface_context": {"hub_chat_lane": "grounded_small"}})  # thought-process.js
    assert not is_browser_surface_turn({"surface_context": {}})
    assert not is_browser_surface_turn({"surface_context": {"surface": SMOKE_SURFACE}})
    assert not is_browser_surface_turn({"mode": "orion"})


def test_sql_rule_excludes_the_same_edges():
    assert "raw->'surface_context' <> '{}'::jsonb" in JUNIPER_INGRESS_PREDICATE
    assert f"<> '{SMOKE_SURFACE}'" in JUNIPER_INGRESS_PREDICATE


def test_the_smoke_surface_is_a_valid_situation_surface():
    # The Hub builds a SurfaceContextV1 from the payload's surface; a value outside its Literal would fail the smoke.
    assert SurfaceContextV1(surface=SMOKE_SURFACE, input_modality="typed").surface == SMOKE_SURFACE


def test_the_skill_runner_smoke_marks_itself_operator_review():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "scripts" / "trace_hub_skill_runner_e2e.py").read_text()
    assert f'"surface": "{SMOKE_SURFACE}"' in text and '"surface": "hub_desktop"' not in text


def test_hub_copies_only_a_non_empty_surface_context_into_the_ingress_hop():
    import pytest

    # Hub's import tree needs Hub's requirements; the promise-ledger CI job may not install them all.
    _cockpit_ingress_payload = pytest.importorskip("orion.hub.turn_orchestrator")._cockpit_ingress_payload

    def raw(surface_context):
        return _cockpit_ingress_payload(payload={"surface_context": surface_context}, user_message="hi", session_id="s")

    assert "surface_context" not in raw({})
    assert "surface_context" not in raw(None)
    assert raw({"surface": "hub_desktop"})["surface_context"] == {"surface": "hub_desktop"}
