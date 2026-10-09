"""Curiosity runs as introspect items: labels, lengths, failures kept, outcome counts."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.curiosity_introspect import (
    INDEX_TEXT_CHARS,
    RUN_KIND,
    answer_section,
    index_text,
    run_item,
    self_question_item,
)
from orion.schemas.introspect import (
    CURIOSITY_FULL_JSON_BUDGET,
    DEFAULT_TEXT_CAP,
    SHORT_FIELD_CAP,
    IntrospectResultV1,
)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
MS = int(NOW.timestamp() * 1000)
MCP_TOOL_RESULT_MAX_CHARS = 12000


def _story(body="", **run):
    base = {
        "run_id": "abc123", "line": "investigate", "status": "completed", "error": "",
        "started_at": MS, "finished_at": MS + 60_000, "hops": 2, "findings": 1, "revisions": 1,
        "prior_touched": {"prior_id": "p1", "claim": "c" * 500, "from": 0.4, "to": 0.7,
                          "from_status": "open", "to_status": "revised"},
        "outcome_kind": "finished",
        "reach_out": {"wanted": True, "decision": "blocked:quiet_hours"},
    }
    base.update(run)
    return {"run": base, "journal_body": body}


def test_answer_section_takes_the_answer_heading_until_the_next_heading():
    body = "All verified. Now the write-up.\n\n## Answer\n\nI am more than I was.\n\n## What I still cannot see\n\nX"
    assert answer_section(body) == "I am more than I was."
    assert answer_section("Prose with no headings.\n\nMore.") == "Prose with no headings.\n\nMore."
    assert answer_section("## Answer\n\nlast section only") == "last section only"
    assert answer_section(None) == ""
    # A heading that only mentions an answer is not the Answer section.
    assert answer_section("# Answering the question\nbody") == "# Answering the question\nbody"


def test_index_text_is_the_answer_section_clipped():
    assert len(index_text("x" * (INDEX_TEXT_CHARS + 50))) == INDEX_TEXT_CHARS


def test_a_run_with_a_write_up_is_unsettled_and_lists_its_answer():
    body = "preamble\n## Answer\n" + "a" * 2000
    item = run_item(_story(body), {"turn_ok": True, "n_tested": 3, "n_moved": 1, "n_formed": 0,
                                   "unknown_reason": None, "realized_nats": 9.9})
    assert item.kind == RUN_KIND and item.id == "abc123"
    assert item.epistemic_status == "unsettled"
    assert item.text == "a" * DEFAULT_TEXT_CAP and item.truncated
    assert item.occurred_at == NOW
    extra = item.extra
    assert (extra["line"], extra["status"], extra["hops"], extra["findings"], extra["revisions"]) == (
        "investigate", "completed", 2, 1, 1)
    assert extra["reach_out"] == "blocked:quiet_hours" and extra["outcome_kind"] == "finished"
    assert len(extra["prior_touched"]["claim"]) == SHORT_FIELD_CAP
    assert (extra["prior_touched"]["from"], extra["prior_touched"]["to"]) == (0.4, 0.7)
    assert (extra["turn_ok"], extra["n_tested"], extra["n_moved"], extra["n_formed"]) == (True, 3, 1, 0)
    assert "realized_nats" not in extra, "only the named outcome counts are carried"
    assert extra["has_write_up"] is True


def test_one_run_by_id_returns_the_write_up_up_to_the_json_budget():
    body = 'He said "no"\n\\ é ' * 3000
    item = run_item(_story(body), full=True)
    assert item.truncated
    assert len(json.dumps(item.text, ensure_ascii=False)) <= CURIOSITY_FULL_JSON_BUDGET
    result = IntrospectResultV1(ok=True, operation="curiosity", as_of=NOW, total_available=1, items=[item])
    assert len(json.dumps(result.model_dump(mode="json"))) < MCP_TOOL_RESULT_MAX_CHARS


def test_a_failed_run_without_a_write_up_is_a_short_record():
    item = run_item(_story("", status="failed", error="fcc_context_ceiling_exceeded", line="self_inquiry",
                           outcome_kind="died", prior_touched=None, reach_out={}))
    assert item.epistemic_status == "record" and not item.truncated
    assert item.text == "Self question run failed with no write-up; error: fcc_context_ceiling_exceeded."
    assert item.extra["error"] == "fcc_context_ceiling_exceeded"
    assert item.extra["prior_touched"] is None and item.extra["reach_out"] is None
    assert item.extra["has_write_up"] is False
    assert "turn_ok" not in item.extra, "no outcome row -> no outcome fields, not zeros"
    quiet = run_item(_story("", status="completed", outcome_kind="wrote_nothing"))
    assert quiet.text == "World question run completed with no write-up; outcome: wrote_nothing."


def test_occurred_at_falls_back_to_finished_and_a_clockless_run_is_skipped():
    item = run_item(_story("x", started_at=None))
    assert item.occurred_at.timestamp() * 1000 == MS + 60_000
    assert run_item(_story("x", started_at=None, finished_at=None)) is None


def test_self_question_item_is_a_record():
    item = self_question_item({
        "question_id": "lived.orion.camera_layer", "text": "When the porch camera goes live?",
        "family": "lived", "pinned": False, "ask_count": 1,
        "last_asked_at": NOW, "created_at": datetime(2026, 10, 6, 19, 25),
    })
    assert (item.kind, item.epistemic_status, item.id) == ("self_question", "record", "lived.orion.camera_layer")
    assert item.occurred_at.tzinfo is not None
    assert item.extra == {"family": "lived", "ask_count": 1, "last_asked_at": NOW.isoformat(), "pinned": False}
