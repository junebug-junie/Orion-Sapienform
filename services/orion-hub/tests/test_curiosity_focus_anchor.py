"""A run that took a curiosity focus carries it, typed, all the way to the governor's task anchor.

Live 2026-10-17: run 480ee593ead6 took focus a21f96c9, compacted, lost the task and wrote no
focus_disposition. The focus now rides the durable brief -> CuriosityTurnRequestV1 -> _generate
payload -> HarnessRunRequestV1.curiosity_focus, where the governor puts it in the system prompt.
"""

from __future__ import annotations

import asyncio
from orion.schemas.curiosity_focus import CuriosityFocusAnchorV1
from orion.schemas.durable_run import CuriosityTurnRequestV1, DurableRunRequestV1

from test_curiosity_focus import _loop as _focus_loop, _queue
from test_curiosity_investigation import _CortexBus
from test_curiosity_recall_retrieval_query import _real_generate_turn
from scripts.curiosity_investigation import INVESTIGATION_TAG

RUN = "480ee593ead6"
FOCUS = CuriosityFocusAnchorV1(
    focus_id="a21f96c9-f1aa-5c19-9562-91ed82a64b80", run_id=RUN,
    question="Do walkway regulars keep a weekly rhythm?", origin="self", graph="orion_worldview",
)


def test_in_process_focus_run_hands_the_typed_focus_to_the_turn(monkeypatch):
    loop, table, pool = _focus_loop(monkeypatch)
    focus = _queue(pool)
    asyncio.run(loop.tick())
    [row] = table.rows
    sent = loop.seen_generate_kwargs["curiosity_focus"]
    assert isinstance(sent, CuriosityFocusAnchorV1)
    assert sent.question == focus.question and sent.focus_id == focus.focus_id
    assert sent.run_id == row["offered_run_id"] and sent.origin == "self"
    assert sent.graph == loop.graph_own


def test_a_run_without_a_focus_hands_none(monkeypatch):
    loop, _table, _pool = _focus_loop(monkeypatch)
    asyncio.run(loop.tick())
    assert "curiosity_focus" not in loop.seen_generate_kwargs


def test_focus_switched_off_hands_none(monkeypatch):
    loop, _table, pool = _focus_loop(monkeypatch, enabled=False)
    _queue(pool)
    asyncio.run(loop.tick())
    assert "curiosity_focus" not in loop.seen_generate_kwargs


def test_durable_kickoff_puts_the_focus_on_the_brief(monkeypatch):
    loop, table, pool = _focus_loop(monkeypatch)
    bus = _CortexBus()
    loop._bus = bus
    loop.kickoff_via_cortex = True
    focus = _queue(pool)
    assert asyncio.run(loop.tick()) == "dispatched"
    [(_channel, envelope, _reply)] = bus.rpc_calls
    request = DurableRunRequestV1.model_validate(envelope.payload["context"]["metadata"]["durable_run"])
    assert request.brief.focus is not None
    assert request.brief.focus.question == focus.question
    assert request.brief.focus.run_id == request.run_id == table.rows[0]["offered_run_id"]


def test_durable_kickoff_without_a_focus_sends_no_focus_key(monkeypatch):
    loop, _table, _pool = _focus_loop(monkeypatch)
    bus = _CortexBus()
    loop._bus = bus
    loop.kickoff_via_cortex = True
    assert asyncio.run(loop.tick()) == "dispatched"
    [(_channel, envelope, _reply)] = bus.rpc_calls
    assert "focus" not in envelope.payload["context"]["metadata"]["durable_run"]["brief"]


def test_runner_turn_request_focus_reaches_the_unified_turn_payload(monkeypatch):
    loop, _table, _pool = _focus_loop(monkeypatch)
    captured = _real_generate_turn(loop, CuriosityTurnRequestV1(
        run_id=RUN, correlation_id="corr-f", prompt="p", timeout_sec=60.0,
        source_tag=INVESTIGATION_TAG, focus=FOCUS,
    ))
    assert CuriosityFocusAnchorV1.model_validate(captured["payload"]["curiosity_focus"]) == FOCUS


def test_runner_turn_request_without_focus_leaves_the_payload_unchanged(monkeypatch):
    loop, _table, _pool = _focus_loop(monkeypatch)
    captured = _real_generate_turn(loop, CuriosityTurnRequestV1(
        run_id=RUN, correlation_id="corr-n", prompt="p", timeout_sec=60.0, source_tag=INVESTIGATION_TAG,
    ))
    assert "curiosity_focus" not in captured["payload"]


def test_a_row_that_does_not_fit_the_contract_is_logged_not_fatal(monkeypatch, caplog):
    from orion.curiosity.focus import FocusRow

    loop, _table, _pool = _focus_loop(monkeypatch)
    bad = FocusRow(focus_id="not-a-uuid!", question="q", why="w", origin="self", status="taken")
    assert loop._focus_anchor(bad, RUN) is None
    assert any("curiosity_focus_anchor_invalid" in r.message for r in caplog.records)
    loop.graph_own = "weird graph; name"
    good = FocusRow(focus_id=FOCUS.focus_id, question="q " * 400, why="w", origin="juniper", status="taken")
    anchor = loop._focus_anchor(good, RUN)
    assert anchor is not None and anchor.graph == "" and len(anchor.question) <= 500


# The turn-orchestrator half (payload -> HarnessRunRequestV1.curiosity_focus) lives in
# test_turn_orchestrator_utterance_origin.py: importing that module purges `scripts.*`.
