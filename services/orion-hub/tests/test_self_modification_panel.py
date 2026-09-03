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


def test_no_recorded_changes_is_distinguishable_from_unreadable_history(
    isolated_surface,
) -> None:
    """Absence must not read as a value, and must not read as a failure either."""
    data = api_routes._self_modification_panel_payload()

    assert data["history"] == []
    assert data["history_available"] is True
    assert "last_change" not in data
    assert "history_error" not in data


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
