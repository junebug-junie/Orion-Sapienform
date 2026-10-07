"""tool_progress heartbeat frames must not become grammar step atoms.

2026-09-29: a 183-step harness run (137 of them tool_progress) flushed 743 grammar events in
~20ms at run end and overflowed one sql-writer lane. Progress frames stay in the live stream,
receipts and step_count; they just are not recorded as started/completed atom pairs.
"""
from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from orion.harness.fcc_motor import is_progress_frame
from orion.harness.grammar_emit import HarnessGrammarCollector, build_harness_grammar_events
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


def _frame(rtype: str, **extra: Any) -> dict[str, Any]:
    raw = {"type": rtype, **extra}
    return {"type": "step", "step": {"type": rtype, "raw": raw}}


def test_is_progress_frame_matches_only_tool_progress() -> None:
    assert is_progress_frame({"type": "tool_progress", "raw": {"type": "tool_progress"}})
    assert is_progress_frame({"type": "tool_progress"})
    assert not is_progress_frame({"type": "assistant", "raw": {"type": "assistant"}})
    assert not is_progress_frame({"type": "user", "raw": {"type": "user"}})
    assert not is_progress_frame({})
    assert not is_progress_frame("nope")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_runner_records_no_step_atoms_for_progress_frames_but_keeps_step_count() -> None:
    async def _fcc(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield _frame("assistant", message={"content": "one"})
        for _ in range(5):
            yield _frame("tool_progress")
        yield _frame("assistant", message={"content": "twotwo"})
        yield {"type": "final", "llm_response": "draft", "metadata": {"exit_code": 0}}

    grammar_events: list[Any] = []

    async def _publish(channel: str, envelope: Any) -> None:
        if channel == "orion:grammar:event":
            grammar_events.append(envelope.payload)

    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=_publish)
    request = HarnessRunRequestV1(
        correlation_id="c-progress",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(bus, step_channel="orion:harness:run:step", fcc_runner=_fcc).run(request)

    atoms = [
        e["atom"] for e in grammar_events
        if e.get("atom") and e["trace_id"].endswith(":harness_motor")
    ]
    started = [a for a in atoms if a["semantic_role"] == "exec_step_started"]
    completed = [a for a in atoms if a["semantic_role"] == "exec_step_completed"]
    assert len(started) == 2 and len(completed) == 2  # the 2 assistant frames, not the 5 pings
    assert not any("tool_progress" in a["summary"] for a in started)
    assembled = [a for a in atoms if a["semantic_role"] == "exec_result_assembled"]
    # the runner's own bookkeeping is unchanged: every frame still gets a live receipt
    assert len(result.grammar_receipts) == 7
    # avg_step_chars numerator covers only the recorded frames: 3 + 6 chars, not the pings'
    assert "step_char_sum=9" in assembled[0]["summary"]
    assert "step_char_max=6" in assembled[0]["summary"]
    assert result.draft_text == "draft"


def test_collector_still_chains_successors_across_a_gap_in_order() -> None:
    from datetime import datetime, timezone

    c = HarnessGrammarCollector(node_name="athena", correlation_id="c1", observed_at=datetime.now(timezone.utc))
    c.record_step_started(order=1, summary="a")
    c.record_step_completed(order=1)
    c.record_step_started(order=7, summary="b")  # orders 2-6 were progress frames
    c.record_step_completed(order=7)
    events = build_harness_grammar_events(c)
    successors = [e for e in events if e.edge and e.edge.relation_type == "temporal_successor"]
    assert len(successors) == 1  # completed(1) -> started(7): the chain is unbroken


@pytest.mark.asyncio
async def test_error_after_trailing_progress_frames_links_failure_to_last_real_step() -> None:
    async def _fcc(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield _frame("assistant", message={"content": "real"})
        for _ in range(4):
            yield _frame("tool_progress")
        yield {"type": "error", "error": "boom", "error_code": "fcc_timeout"}

    grammar_events: list[Any] = []

    async def _publish(channel: str, envelope: Any) -> None:
        if channel == "orion:grammar:event":
            grammar_events.append(envelope.payload)

    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=_publish)
    request = HarnessRunRequestV1(
        correlation_id="c-progress-err",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    await HarnessRunner(bus, step_channel="orion:harness:run:step", fcc_runner=_fcc).run(request)

    motor = [e for e in grammar_events if e["trace_id"].endswith(":harness_motor")]
    failed = [e["atom"] for e in motor if e.get("atom") and e["atom"]["semantic_role"] == "exec_step_failed"]
    assert len(failed) == 1 and "order=1," in failed[0]["summary"]  # the real step, not frame 5
    derived = [e["edge"] for e in motor if e.get("edge") and e["edge"]["relation_type"] == "derived_from"]
    assert any(ed["to_atom_id"] == failed[0]["atom_id"] for ed in derived)  # linked to its started atom
