from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from orion.harness.reading_receipts import (
    ReadingReceiptTracker,
    enforce_reading_receipt_grounding,
)
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1
from orion.schemas.reading import ReadingToolBindingV1
from orion.world_pulse_read.tools import deterministic_reading_request_id
from orion.world_pulse_read.urls import normalize_source_url

URL = "https://example.org/papers/agent-systems"
WHY = "Compare this source with our runtime evidence"
BINDING = ReadingToolBindingV1(
    invocation_context="unified_chat",
    parent_run_id="turn-17",
    parent_trace_id="trace-17",
)


def _tool_use(tool_use_id: str, *, name: str = "mcp__orion-reading__recommend_reading", url: str = URL):
    arguments = {"url": url, "why_now": WHY} if "recommend_reading" in name else {"url": url}
    return {
        "type": "assistant",
        "raw": {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": name,
                        "input": arguments,
                    }
                ]
            },
        },
    }


def _tool_result(tool_use_id: str, body: object, *, is_error: bool = False):
    text = body if isinstance(body, str) else json.dumps(body)
    return {
        "type": "user",
        "raw": {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "is_error": is_error,
                        "content": [{"type": "text", "text": text}],
                    }
                ]
            },
        },
    }


def _request_id():
    return deterministic_reading_request_id(
        BINDING, url=normalize_source_url(URL), why_now=WHY
    )


def _accepted_payload(*, request_id: object | None = None, status: str = "queued"):
    return {
        "ok": True,
        "result": {
            "request_id": str(request_id or _request_id()),
            "status": status,
            "seed_id": f"reading:{request_id or _request_id()}",
        },
        "error": None,
    }


def test_success_requires_explicit_ok_and_matching_durable_request_id():
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", _accepted_payload()))

    [outcome] = tracker.outcomes()
    assert outcome.acceptance == "accepted"
    assert outcome.request_id == _request_id()
    assert outcome.status == "queued"
    assert outcome.attempt_count == 1

    final = enforce_reading_receipt_grounding("Done.", [outcome])
    assert str(_request_id()) in final
    assert "current status: `queued`" in final
    assert enforce_reading_receipt_grounding(final, [outcome]) == final


@pytest.mark.parametrize("content", [None, 1, {"type": "tool_use"}, "not-blocks"])
def test_malformed_step_content_is_ignored(content):
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe({"raw": {"message": {"content": content}}})
    assert tracker.outcomes() == []


@pytest.mark.parametrize(
    "body",
    [
        {"status": "queued", "request_id": "unwrapped-is-not-authority"},
        {"ok": True, "result": {"status": "queued"}},
        {"ok": True, "result": {"request_id": "not-a-uuid", "status": "queued", "seed_id": "s"}},
        {"ok": True, "result": {"request_id": "00000000-0000-0000-0000-000000000000", "status": "queued", "seed_id": "s"}},
        {"ok": True, "result": _accepted_payload()["result"], "error": "contradiction"},
        "not json",
    ],
)
def test_malformed_or_mismatched_receipt_is_acceptance_unknown(body):
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", body))

    [outcome] = tracker.outcomes()
    assert outcome.acceptance == "unknown"
    assert outcome.failure_kind == "malformed_receipt"
    assert outcome.request_id is None


def test_rpc_timeout_is_acceptance_unknown():
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("tool-timeout"))
    tracker.observe(
        _tool_result("tool-timeout", "RPC timed out while waiting for Hub", is_error=True)
    )

    [outcome] = tracker.outcomes()
    assert outcome.acceptance == "unknown"
    assert outcome.failure_kind == "rpc_timeout"


def test_two_failed_retries_preserve_one_request_identity_and_remain_failure():
    tracker = ReadingReceiptTracker(BINDING)
    for tool_use_id in ("tool-1", "tool-2"):
        tracker.observe(_tool_use(tool_use_id))
        tracker.observe(
            _tool_result(tool_use_id, "queue unavailable", is_error=True)
        )

    [outcome] = tracker.outcomes()
    assert outcome.acceptance == "unknown"
    assert outcome.attempt_count == 2
    assert outcome.tool_use_ids == ["tool-1", "tool-2"]
    assert outcome.request_id is None
    final = enforce_reading_receipt_grounding("I saved it for later.", [outcome])
    assert "after 2 attempts" in final
    assert "acceptance is unknown" in final


def test_failed_then_successful_idempotent_retry_is_confirmed_by_later_receipt():
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", "timeout", is_error=True))
    tracker.observe(_tool_use("tool-2"))
    tracker.observe(_tool_result("tool-2", _accepted_payload(status="started")))

    [outcome] = tracker.outcomes()
    assert outcome.acceptance == "accepted"
    assert outcome.attempt_count == 2
    assert outcome.request_id == _request_id()
    assert outcome.status == "started"


def test_more_than_twenty_retries_cannot_abort_receipt_grounding():
    tracker = ReadingReceiptTracker(BINDING)
    for index in range(21):
        tool_use_id = f"tool-{index}"
        tracker.observe(_tool_use(tool_use_id))
        tracker.observe(_tool_result(tool_use_id, "timeout", is_error=True))

    [outcome] = tracker.outcomes()
    assert outcome.acceptance == "unknown"
    assert outcome.attempt_count == 21
    assert len(outcome.tool_use_ids) == 21


@pytest.mark.parametrize(
    "false_response",
    [
        "I queued it and will process it later.",
        "The source is saved; my future self will digest it.",
        "I recorded a persistent intention to revisit this source.",
        "Submission succeeded, and it is waiting in the pipeline.",
    ],
)
def test_unknown_acceptance_replaces_semantically_equivalent_persistence_claims(false_response):
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", "service error", is_error=True))

    final = enforce_reading_receipt_grounding(false_response, tracker.outcomes())
    assert false_response not in final
    assert "acceptance is unknown" in final
    assert "I did not read that source during this turn." in final


def test_failed_recommendation_cannot_preserve_an_invented_source_summary():
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", "service error", is_error=True))
    fabricated = "The unread paper proves FABRICATED_BENCHMARK_CLAIM across five agents."

    final = enforce_reading_receipt_grounding(fabricated, tracker.outcomes())
    assert "FABRICATED_BENCHMARK_CLAIM" not in final
    assert "did not read" in final


def test_successful_fetch_of_same_source_avoids_false_not_read_statement():
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("fetch-1", name="WebFetch"))
    tracker.observe(_tool_result("fetch-1", "source contents"))
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", "service error", is_error=True))

    final = enforce_reading_receipt_grounding("I queued it.", tracker.outcomes())
    assert "acceptance is unknown" in final
    assert "did not read" not in final


def test_successful_context_mode_fetch_of_same_source_counts_as_read():
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(
        _tool_use(
            "fetch-1",
            name="mcp__plugin_context-mode_context-mode__ctx_fetch_and_index",
        )
    )
    tracker.observe(_tool_result("fetch-1", "source fetched and indexed"))
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", "service error", is_error=True))

    final = enforce_reading_receipt_grounding("I queued it.", tracker.outcomes())
    assert "acceptance is unknown" in final
    assert "did not read" not in final


@pytest.mark.parametrize(
    "tool_name,body",
    [
        ("WebFetch", ""),
        ("mcp__firecrawl__scrape", {"success": False, "error": "blocked"}),
        ("mcp__firecrawl__scrape", {"success": True, "data": {}}),
        (
            "mcp__plugin_context-mode_context-mode__ctx_fetch_and_index",
            "Cached: **paper** — 12 sections. Use ctx_search to read it.",
        ),
    ],
)
def test_empty_or_failed_fetch_does_not_count_as_read(tool_name, body):
    tracker = ReadingReceiptTracker(BINDING)
    tracker.observe(_tool_use("fetch-1", name=tool_name))
    tracker.observe(_tool_result("fetch-1", body))
    tracker.observe(_tool_use("tool-1"))
    tracker.observe(_tool_result("tool-1", "service error", is_error=True))

    final = enforce_reading_receipt_grounding("I queued it.", tracker.outcomes())
    assert "I did not read that source during this turn." in final


@pytest.mark.asyncio
async def test_harness_runner_replaces_false_success_transcript_after_two_failures():
    async def failed_reading_turn(**_: Any) -> AsyncIterator[dict[str, Any]]:
        for tool_use_id in ("tool-1", "tool-2"):
            yield {"type": "step", "step": _tool_use(tool_use_id)}
            yield {
                "type": "step",
                "step": _tool_result(
                    tool_use_id, "reading service unavailable", is_error=True
                ),
            }
        yield {
            "type": "final",
            "llm_response": (
                "I saved the source for later. It is an imaginary benchmark "
                "with FABRICATED_RESULTS."
            ),
            "metadata": {"exit_code": 0},
        }

    request = HarnessRunRequestV1(
        correlation_id="reading-turn",
        thought_event=make_thought(),
        user_message=f"Read {URL} asynchronously",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        reading_binding=BINDING,
    )
    result = await HarnessRunner(
        AsyncMock(), fcc_runner=failed_reading_turn
    ).run(request)

    assert "FABRICATED_RESULTS" not in result.draft_text
    assert "acceptance is unknown" in result.draft_text
    assert "did not read" in result.draft_text
    assert len(result.reading_receipts) == 1
    assert result.reading_receipts[0].attempt_count == 2
