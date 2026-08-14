"""Guard the fill-only contract of the chat_history_log backfill.

The backfill repairs rows that lost their turn-level write to the concurrent
same-primary-key race in orion-sql-writer. It reads a preserved copy of the turn
out of bus_fallback_log, so the one thing that must never happen is a preserved
copy overwriting text that is already stored -- that would turn a recovery tool
into a second source of loss.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backfill_chat_history_from_bus_fallback.py"
)
spec = importlib.util.spec_from_file_location("backfill_chat_history", MODULE_PATH)
backfill = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backfill)


def _row(**overrides):
    row = {
        "id": "corr-1",
        "correlation_id": "corr-1",
        "cur_prompt": "",
        "cur_response": "",
        "cur_source": None,
        "cur_session_id": None,
        "cur_spark_meta": None,
        "created_at": None,
        "fallback_id": 1,
        "fallback_payload": {
            "prompt": "restored prompt",
            "response": "restored response",
            "source": "hub_orion",
            "session_id": "orion_journal",
            "spark_meta": {"mode": "orion"},
        },
    }
    row.update(overrides)
    return row


def test_fills_every_empty_field_from_the_preserved_payload():
    (planned,) = backfill.plan_updates([_row()])

    assert planned["updates"]["prompt"] == "restored prompt"
    assert planned["updates"]["response"] == "restored response"
    assert planned["updates"]["source"] == "hub_orion"
    assert planned["updates"]["session_id"] == "orion_journal"


@pytest.mark.parametrize(
    "column, stored",
    [
        ("cur_prompt", "the real thing Juniper typed"),
        ("cur_response", "the real thing Orion said"),
    ],
)
def test_never_overwrites_text_that_is_already_stored(column, stored):
    """The half that survived the race is the authoritative copy -- keep it."""
    (planned,) = backfill.plan_updates([_row(**{column: stored})])

    overwritten = column.removeprefix("cur_")
    assert overwritten not in planned["updates"]


def test_whitespace_only_stored_value_counts_as_empty():
    (planned,) = backfill.plan_updates([_row(cur_prompt="   \n  ")])

    assert planned["updates"]["prompt"] == "restored prompt"


def test_row_needing_nothing_is_dropped_from_the_plan():
    """A fully populated row produces no UPDATE at all, not an empty one."""
    complete = _row(
        cur_prompt="p",
        cur_response="r",
        cur_source="hub_orion",
        cur_session_id="orion_journal",
        cur_spark_meta={"mode": "orion"},
    )

    assert backfill.plan_updates([complete]) == []


def test_payload_missing_a_half_does_not_invent_one():
    """A fallback copy that itself lacks the response must not write an empty
    string over the column -- that would erase the NULL/'' distinction and make
    the row look repaired when nothing was recovered."""
    row = _row(fallback_payload={"prompt": "restored prompt", "response": "  "})

    (planned,) = backfill.plan_updates([row])

    assert planned["updates"]["prompt"] == "restored prompt"
    assert "response" not in planned["updates"]


def test_json_encoded_payload_is_parsed():
    """bus_fallback_log.payload is JSONB, but a text column would hand back str."""
    import json

    row = _row(fallback_payload=json.dumps(_row()["fallback_payload"]))

    (planned,) = backfill.plan_updates([row])

    assert planned["updates"]["response"] == "restored response"


def test_undecodable_payload_is_skipped_rather_than_raising():
    assert backfill.plan_updates([_row(fallback_payload="{not json")]) == []
