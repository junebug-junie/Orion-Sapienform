"""The hub must answer "what did Orion change, and is a surface stuck?" without SQL.

On 2026-09-02 Orion changed its own routing threshold and then held the routing
surface lock for thirteen hours while 77 proposals were refused behind it.
Neither fact was visible from any hub surface, and the value it had changed
*from* was recorded nowhere at all -- it had to be inferred from a pytest
fixture that had been leaking writes onto the live row.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")
from orion.core.schemas.substrate_mutation import MutationAdoptionV1
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_queue import SubstrateMutationStore
from scripts import api_routes

HUB_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_surface(monkeypatch, tmp_path):
    store = mutation_control_surface.RuntimeControlSurfaceStore(
        sql_db_path=str(tmp_path / "control.sqlite3")
    )
    monkeypatch.setattr(mutation_control_surface, "_CONTROL_SURFACE_STORE", store)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", SubstrateMutationStore())
    return store


def _adopt(*, applied_at: datetime, window_sec: int = 900) -> MutationAdoptionV1:
    adoption = MutationAdoptionV1(
        proposal_id="p-1",
        decision_id="d-1",
        target_surface="routing",
        applied_patch={"chat_reflective_lane_threshold": 0.58},
        rollback_payload={"chat_reflective_lane_threshold": 0.5},
        applied_at=applied_at,
        rollback_window_sec=window_sec,
    )
    assert api_routes.SUBSTRATE_MUTATION_STORE.record_adoption(adoption) == []
    return adoption


def test_panel_reports_the_value_a_change_replaced(isolated_surface) -> None:
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="operator")
    mutation_control_surface.set_chat_reflective_lane_threshold(
        value=0.58, actor="mutation_apply", proposal_id="p-1"
    )

    data = api_routes._self_modification_panel_payload()

    assert data["last_change"]["previous_value"] == 0.5
    assert data["last_change"]["new_value"] == 0.58
    assert data["last_change"]["actor"] == "mutation_apply"
    assert data["current"]["value"] == 0.58


def test_no_recorded_changes_reads_as_calm_not_as_a_fault(isolated_surface) -> None:
    """An empty history is the normal state until the first write lands."""
    data = api_routes._self_modification_panel_payload()

    assert data["history"] == []
    assert data["history_available"] is True
    assert "last_change" not in data
    assert "history_error" not in data


def test_an_unreadable_history_does_not_read_as_an_empty_one(
    isolated_surface, tmp_path
) -> None:
    """The distinction the panel exists for, driven by actually breaking the table.

    ``history()`` swallows its own backend errors and returns ``[]``, so a
    dropped table arrives looking exactly like a calm one. The previous version
    of this test only ever built a healthy empty store, so it asserted a
    postcondition that could not fail and the real case shipped broken.
    """
    import sqlite3

    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.5, actor="operator")
    with sqlite3.connect(tmp_path / "control.sqlite3") as conn:
        conn.execute("DROP TABLE substrate_runtime_control_surface_history")
        conn.commit()

    data = api_routes._self_modification_panel_payload()

    assert data["history"] == []          # same as the calm case ...
    assert data["history_available"] is False  # ... but distinguishable
    assert "no such table" in str(data["history_error"])


def test_the_rendered_line_for_unreadable_history_differs_from_the_calm_one() -> None:
    """Couple the JS branch order to the payload, not just the Python.

    A JS-only regression -- reading the wrong key, or checking the branches in
    the wrong order -- renders "nothing has changed" over a broken table and no
    Python test notices. This pins the two things that must stay true of the
    frontend: which key path it reads, and that the error branch is checked
    before the empty branch.
    """
    app_js = (HUB_ROOT / "static" / "js" / "app.js").read_text()

    assert "const selfMod = routing.self_modification || {};" in app_js
    error_branch = app_js.index("if (selfMod.history_error)")
    empty_branch = app_js.index("selfMod.history_available")
    assert error_branch < empty_branch, "error branch must be checked first"
    assert "HISTORY UNREADABLE" in app_js
    assert "LOCKS UNREADABLE" in app_js


def test_the_payload_exposes_the_key_path_the_frontend_reads() -> None:
    """The other half of the contract: the nesting the JS walks must exist."""
    payload = api_routes._autonomy_readiness_payload()

    assert "self_modification" in payload.get("routing", {})
    assert "self_modification" not in payload  # not at the top level


def test_panel_flags_a_surface_held_past_its_window(isolated_surface) -> None:
    """The live shape: a 900s window, held for 13 hours, settlement not running."""
    _adopt(applied_at=datetime.now(timezone.utc) - timedelta(hours=13), window_sec=900)

    holds = api_routes._self_modification_panel_payload()["surface_holds"]

    assert len(holds) == 1
    assert holds[0]["target_surface"] == "routing"
    assert holds[0]["window_elapsed"] is True
    assert holds[0]["held_for_sec"] > 13 * 3600 - 60
    assert holds[0]["rollback_window_sec"] == 900


def test_a_fresh_hold_is_not_flagged_as_overdue(isolated_surface) -> None:
    _adopt(applied_at=datetime.now(timezone.utc), window_sec=900)

    holds = api_routes._self_modification_panel_payload()["surface_holds"]

    assert holds[0]["window_elapsed"] is False


def test_no_hold_reports_an_empty_list_not_an_error(isolated_surface) -> None:
    data = api_routes._self_modification_panel_payload()

    assert data["surface_holds"] == []
    assert "surface_holds_error" not in data


def test_a_settled_adoption_stops_being_reported_as_held(isolated_surface) -> None:
    adoption = _adopt(applied_at=datetime.now(timezone.utc) - timedelta(hours=13))
    assert api_routes.SUBSTRATE_MUTATION_STORE.record_settlement(adoption.adoption_id) is True

    assert api_routes._self_modification_panel_payload()["surface_holds"] == []
