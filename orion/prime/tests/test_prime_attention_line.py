"""Prime's attention schema line (choice-as-EFE Step 2a, S2.1): what is salient, why, confidence; stale reads absent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.prime.attention_line import HEADER, MAX_AGE_SEC, attention_schema_line
from orion.prime.state import PrimeStateV1
from orion.prime.thought import FrameInputs, build_frame

NOW = datetime(2026, 11, 25, 7, 20, tzinfo=timezone.utc)

# The live row of 2026-11-25T07:14:28Z (substrate_attention_self_model), trimmed to the fields the line reads.
LIVE = {
    "attention_reason": "bottom_up_salience",
    "confidence": 0.6,
    "confidence_basis": "broadcast.coalition_stability_score (fresh, bottom-up)",
    "prediction_error_confidence": None,
    "prediction_error_confidence_basis": "unknown: no Active-Inference prediction-error domain is live "
                                         "(execution, the last, retired 2026-11-24, #3576)",
    "broadcast_selected_open_loop_id": "open-loop-f24f69eaa370",
    "broadcast_selected_description": "Orion is reading (2 accepted links in the last 15 min)",
    "field_salient_target_ids": ["node:prediction.room", "world:reading"],
    "predicted_shift": None,
}


def row(model=None, *, age_sec=330.0):
    return {"generated_at": NOW - timedelta(seconds=age_sec), "self_model_json": {**LIVE, **(model or {})}}


def test_a_fresh_bottom_up_row_names_what_why_and_the_confidence_basis():
    line = attention_schema_line(row(), now=NOW)
    assert line == (
        'read 5 min ago: most salient: "Orion is reading (2 accepted links in the last 15 min)"; '
        "bottom-up: salience alone chose it; no goal changed the winner; "
        "confidence 0.60 (how long the winner has held: 0.3 new or nothing won, 0.6 held 1-3 ticks, 0.9 longer)")


def test_top_down_is_named_as_top_down():
    line = attention_schema_line(row({"attention_reason": "top_down_override", "confidence": 0.9}), now=NOW)
    assert "top-down: a goal's bias changed which loop won" in line and "confidence 0.90" in line


def test_null_confidence_says_unknown_never_a_number():
    line = attention_schema_line(row({"confidence": None, "confidence_basis": ""}), now=NOW)
    assert line.endswith("confidence unknown (the schema recorded none)")
    assert "0." not in line.split("confidence")[-1]


def test_out_of_range_or_non_numeric_confidence_is_unknown():
    for bad in (1.5, -0.1, float("nan"), "0.6", True):
        assert "confidence unknown" in attention_schema_line(row({"confidence": bad}), now=NOW)


def test_the_dead_unconditional_confidence_is_never_shown():
    # Null on every live row since #3576; the line does not read it, so a stray value cannot leak in either.
    line = attention_schema_line(row({"confidence": None, "prediction_error_confidence": 0.81}), now=NOW)
    assert "0.81" not in line and line.endswith("confidence unknown (the schema recorded none)")


def test_a_stale_row_reads_absent():
    line = attention_schema_line(row(age_sec=MAX_AGE_SEC + 1), now=NOW)
    assert line.startswith("(absent: the latest schema reading is 10 min old")
    assert "salient" not in line and "confidence" not in line
    # Exactly at the limit is still fresh.
    assert attention_schema_line(row(age_sec=MAX_AGE_SEC), now=NOW).startswith("read 10 min ago")


def test_no_row_a_failed_read_and_a_malformed_row_read_absent_with_why():
    assert attention_schema_line(None, now=NOW) == "(absent: no schema reading is recorded)"
    assert attention_schema_line(None, now=NOW, read_error="OperationalError: timeout").startswith(
        "(absent: the schema could not be read: OperationalError")
    assert "unreadable" in attention_schema_line({"generated_at": NOW, "self_model_json": "x"}, now=NOW)
    assert "no time" in attention_schema_line({"generated_at": "nope", "self_model_json": {}}, now=NOW)


def test_no_winner_and_field_only_rows():
    # The live no-winner shape: still tagged bottom_up_salience, confidence the empty-coalition constant 0.3.
    base = {"broadcast_selected_open_loop_id": None, "broadcast_selected_description": None, "confidence": 0.3}
    line = attention_schema_line(row({**base, "field_salient_target_ids": []}), now=NOW)
    assert line == ("read 5 min ago: nothing won attention this tick; no loop won, so nothing was chosen; "
                    "confidence n/a (nothing won)")
    line = attention_schema_line(row(base), now=NOW)
    assert line.startswith("read 5 min ago: most salient in the field: node:prediction.room, world:reading")
    assert "salience alone chose it" not in line and "0.30" not in line
    field_only = row({"attention_reason": "field_salience_only", "broadcast_selected_open_loop_id": None,
                      "broadcast_selected_description": None, "confidence": None})
    line = attention_schema_line(field_only, now=NOW)
    assert "most salient in the field: node:prediction.room, world:reading" in line
    assert "only field salience was read" in line and "confidence unknown" in line
    assert "no attention data" in attention_schema_line(row({"attention_reason": "no_data"}), now=NOW)


def test_a_row_dated_in_the_future_reads_absent():
    assert attention_schema_line(row(age_sec=-120), now=NOW) == (
        "(absent: the latest schema reading is dated in the future)")
    assert attention_schema_line(row(age_sec=-30), now=NOW).startswith("read 0 min ago")


def test_an_iso_string_time_is_read():
    r = {"generated_at": (NOW - timedelta(seconds=60)).isoformat().replace("+00:00", "Z"), "self_model_json": LIVE}
    assert attention_schema_line(r, now=NOW).startswith("read 1 min ago")


def test_the_frame_shows_the_section_only_when_the_line_is_set():
    off = build_frame(PrimeStateV1(), FrameInputs(now=NOW))
    assert HEADER not in off
    line = attention_schema_line(row(), now=NOW)
    on = build_frame(PrimeStateV1(), FrameInputs(now=NOW, attention_schema=line))
    assert f"{HEADER}\n{line}" in on
    stale = build_frame(PrimeStateV1(), FrameInputs(
        now=NOW, attention_schema=attention_schema_line(row(age_sec=900), now=NOW)))
    assert f"{HEADER}\n(absent: the latest schema reading is 15 min old" in stale
