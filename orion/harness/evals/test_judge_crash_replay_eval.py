"""Pure parts of judge_crash_replay_eval (no DB, no model)."""
from __future__ import annotations

import json
import re

from orion.harness.evals.judge_crash_replay_eval import (
    FIXTURE,
    build_request,
    classify,
    new_response_format,
    summarize,
)
from orion.harness.finalize import FINALIZE_REFLECT_USER_TURN, finalize_reflect_json_schema
from orion.harness.tests.fixtures import make_appraisal, make_thought

JUDGEMENT = {"imperative": "x", "tone": "y", "strain_refs": [], "alignment_notes": ["n"],
             "alignment_verdict": "uncertain", "strain_unresolved": False,
             "world_contact_opportunity": False, "recommended_tool": None, "recommended_tool_reason": None}


def test_classify_follows_the_harness_parse_path():
    assert classify(json.dumps(JUDGEMENT)) == {"ok": True, "verdict": "uncertain", "notes": ["n"]}
    assert classify('{"what_i_learned": "a draft"}')["failure"] == "validation"
    assert classify("We need answer user's request. Need output strict JSON")["failure"].startswith("unparseable")
    assert classify("")["failure"] == "unparseable:empty"
    assert classify(None, error="http 502")["failure"] == "engine:http 502"


def test_new_request_matches_what_the_gateway_sends_per_backend():
    turn = {"user_message": "fetch it", "draft_text": '{"what_i_learned": "x"}', "thought": make_thought(),
            "appraisal": make_appraisal(), "notes": None, "receipts": []}
    template = "draft={{ draft_text }}"
    old = build_request("old", template, turn, "llamacpp")
    new = build_request("new", template, turn, "llamacpp")
    assert [m["role"] for m in old["messages"]] == ["system"]
    assert "response_format" not in old and "chat_template_kwargs" not in old
    assert [m["role"] for m in new["messages"]] == ["system", "user"]
    assert new["messages"][-1]["content"] == FINALIZE_REFLECT_USER_TURN
    assert new["chat_template_kwargs"] == {"enable_thinking": False}
    assert new["response_format"] == {"type": "json_object", "schema": finalize_reflect_json_schema()}
    # Strata takes only OpenAI's named json_schema (llm-gateway response_format_for_backend).
    strata = new_response_format("strata")
    assert strata["type"] == "json_schema" and strata["json_schema"]["schema"] == finalize_reflect_json_schema()


def test_summary_counts_failures_and_control_agreement():
    rows = [
        {"bucket": "failed", "lane": "chat", "stored_verdict": None,
         "old": {"ok": False, "failure": "validation"}, "new": {"ok": True, "verdict": "aligned"}},
        {"bucket": "control", "lane": "agent", "stored_verdict": "aligned",
         "old": {"ok": True, "verdict": "aligned"}, "new": {"ok": True, "verdict": "misaligned"}},
    ]
    s = summarize(rows, ("old", "new"))
    assert s["failed"]["old"]["failures"] == 1 and s["failed"]["new"]["failures"] == 0
    assert s["control"]["old"]["agrees_with_stored_verdict"] == 1
    assert s["control"]["new"]["agrees_with_stored_verdict"] == 0


def test_fixture_holds_ids_and_labels_only():
    fixture = json.loads(FIXTURE.read_text())
    uuid = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    allowed = {"correlation_id", "failed_at", "judged_at", "lane", "lane_known", "failure_class", "family"}
    for case in fixture["failed"] + fixture["control"]:
        assert uuid.match(case["correlation_id"])
        assert set(case) <= allowed
