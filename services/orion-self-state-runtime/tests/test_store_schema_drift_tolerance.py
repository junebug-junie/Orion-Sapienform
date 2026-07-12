"""Regression tests for a real live incident (2026-07-12): after Phase 0
removed `policy_pressure` from SelfStateV1.dimensions' valid dimension_id
values, the service crash-looped on every single tick in production --
`load_latest_self_state()` naively `model_validate()`d the last row saved
*before* the schema change, which still had `dimension_id: "policy_pressure"`
baked into its stored JSON, and let the ValidationError propagate uncaught
out of `_tick()`'s "load previous state" step. The service never recovered
on its own because every subsequent load of the same stale row hit the same
error.

These loaders must degrade to None (or skip the row, for the list loader)
on a schema-incompatible legacy row, the same way they already degrade to
None for other "previous is unusable" reasons (policy_id mismatch, staleness)
in worker.py's _tick().
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.store import SelfStateRuntimeStore

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _store_with_row(row) -> SelfStateRuntimeStore:
    store = SelfStateRuntimeStore.__new__(SelfStateRuntimeStore)
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.first.return_value = row
    store._engine = engine
    return store


def _store_with_rows(rows) -> SelfStateRuntimeStore:
    store = SelfStateRuntimeStore.__new__(SelfStateRuntimeStore)
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.fetchall.return_value = rows
    store._engine = engine
    return store


def _legacy_self_state_payload_with_policy_pressure() -> dict:
    # Mirrors the exact real production row: a valid SelfStateV1 shape
    # except for one dimension_id no longer accepted by the current schema.
    return {
        "schema_version": "self.state.v1",
        "self_state_id": "self.state:legacy",
        "generated_at": NOW.isoformat(),
        "source_field_tick_id": "tick",
        "source_field_generated_at": NOW.isoformat(),
        "source_attention_frame_id": "frame",
        "source_attention_generated_at": NOW.isoformat(),
        "self_state_policy_id": "self_state_policy.v1",
        "overall_condition": "steady",
        "overall_intensity": 0.4,
        "overall_confidence": 0.8,
        "dimensions": {
            "policy_pressure": {
                "dimension_id": "policy_pressure",
                "score": 0.0,
                "confidence": 0.5,
                "dominant_evidence": [],
                "reasons": ["policy_pressure from field+attention channel synthesis"],
            }
        },
    }


def _valid_self_state_payload() -> dict:
    payload = _legacy_self_state_payload_with_policy_pressure()
    payload["dimensions"] = {
        "coherence": {
            "dimension_id": "coherence",
            "score": 0.8,
            "confidence": 0.7,
            "dominant_evidence": [],
            "reasons": ["no contributing channel evidence this tick"],
        }
    }
    return payload


def test_load_latest_self_state_degrades_to_none_on_legacy_incompatible_row():
    store = _store_with_row(
        {"self_state_json": _legacy_self_state_payload_with_policy_pressure()}
    )
    assert store.load_latest_self_state() is None


def test_load_latest_self_state_still_parses_a_valid_row():
    store = _store_with_row({"self_state_json": _valid_self_state_payload()})
    state = store.load_latest_self_state()
    assert state is not None
    assert "coherence" in state.dimensions


def test_load_self_state_for_attention_frame_degrades_to_none_on_legacy_row():
    store = _store_with_row(
        {"self_state_json": _legacy_self_state_payload_with_policy_pressure()}
    )
    assert store.load_self_state_for_attention_frame("frame") is None


def test_load_recent_self_states_skips_legacy_row_keeps_valid_ones():
    store = _store_with_rows(
        [
            {"self_state_json": _valid_self_state_payload()},
            {"self_state_json": _legacy_self_state_payload_with_policy_pressure()},
        ]
    )
    states = store.load_recent_self_states(limit=10)
    # The incompatible row is skipped, not raised; the valid one still comes through.
    assert len(states) == 1
    assert "coherence" in states[0].dimensions
