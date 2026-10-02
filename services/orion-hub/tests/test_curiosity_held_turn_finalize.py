"""Held curiosity turns finalize while the run still holds its GPU (run a153451fe423).

durable-runs stopped attempt 1 and released the run's hold at the turn's 900 s limit; the
harness motor had been given the same 900 s, so finalize ran after the hold was gone and
failed with ``gpu_pool_unavailable:hold_not_granted:released``, discarding a 7,506-char draft.
Hub now keeps a finalize reserve out of the motor's budget, tells the harness when it stops
waiting, and an urgent turn whose finalize still fails ends with Orion's draft, flagged.
"""
from __future__ import annotations

import asyncio
import time

from orion.hub import turn_orchestrator
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.harness_finalize import HarnessRunV1
from scripts import curiosity_investigation as ci
from test_curiosity_investigation import _CortexBus, _loop

REPAIR_FAILED = "orion_response_repair exec failed: gpu_pool_unavailable:hold_not_granted:released"
DRAFT = "Verdict so far: the cabinet reading is real. " * 167  # ~7.5k chars, as in the incident


def _ref() -> GpuLeaseRefV1:
    return GpuLeaseRefV1(lease_id="hold-a153451fe423", generation=1, role="agent", holder="durable-runs:a153451fe423")


def _generate_with(monkeypatch, frames, *, reserve=330.0, **gen_kw):
    captured = {}

    async def turn(**kwargs):
        captured.update(kwargs)
        return frames
    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", turn)
    loop = _loop(_CortexBus(), held_turn_finalize_reserve_sec=reserve)
    del loop._generate
    before = time.monotonic()
    text, debug = asyncio.run(loop._generate("prompt", "corr", fcc_model_label="llamacpp/agent", **gen_kw))
    return text, debug, captured.get("payload") or {}, before


def test_motor_budget_leaves_the_finalize_reserve_inside_the_turn(monkeypatch):
    _, _, payload, before = _generate_with(
        monkeypatch, [{"type": "final", "llm_response": "grounded", "harness_step_count": 14}],
        timeout_sec=900, gpu_lease=_ref())
    # The bug: inference_timeout_sec == 900 == the attempt limit durable-runs releases the hold at.
    assert payload["inference_timeout_sec"] == 570.0
    deadline = payload[ci.REPLY_DEADLINE_PAYLOAD_KEY]
    assert before + 899 <= deadline <= time.monotonic() + 900


def test_unheld_turn_gets_no_budget_or_deadline(monkeypatch):
    _, _, payload, _ = _generate_with(
        monkeypatch, [{"type": "final", "llm_response": "grounded", "harness_step_count": 14}], timeout_sec=900)
    assert "inference_timeout_sec" not in payload and ci.REPLY_DEADLINE_PAYLOAD_KEY not in payload


def test_reserve_larger_than_the_turn_never_starves_the_motor():
    assert ci.held_turn_fcc_budget_sec(900, 330) == 570
    assert ci.held_turn_fcc_budget_sec(300, 330) == 300
    assert ci.held_turn_fcc_budget_sec(900, 0) == 900


def test_urgent_turn_whose_finalize_failed_ends_with_the_salvaged_draft(monkeypatch):
    frame = {"type": "turn_error", "phase": "finalize", "error": REPAIR_FAILED, "partial_draft": DRAFT,
             "partial": 107}
    text, debug, _, _ = _generate_with(monkeypatch, [frame], timeout_sec=900, gpu_lease=_ref(), urgent=True)
    assert text == DRAFT.strip()
    assert debug["draft_salvaged"] is True and debug["salvaged_from_error"] == REPAIR_FAILED
    assert debug["harness_step_count"] == 107


def test_ordinary_turn_never_salvages_a_draft(monkeypatch):
    frame = {"type": "turn_error", "phase": "finalize", "error": REPAIR_FAILED, "partial_draft": DRAFT}
    text, debug, _, _ = _generate_with(monkeypatch, [frame], timeout_sec=900, gpu_lease=_ref())
    assert text == "" and debug["error"] == "no_final_frame" and "draft_salvaged" not in debug


def test_urgent_turn_error_without_a_draft_still_fails(monkeypatch):
    frame = {"type": "turn_error", "error": "urgent_stance_unavailable"}
    text, debug, _, _ = _generate_with(monkeypatch, [frame], timeout_sec=900, gpu_lease=_ref(), urgent=True)
    assert text == "" and debug["error"] == "urgent_stance_unavailable"


# --- turn_orchestrator: what the harness is told -------------------------------


def test_reply_budget_is_measured_when_the_harness_request_is_built():
    # 60 s of stance/recall already gone from a 900 s turn: the harness has ~830 s (minus transit slack).
    deadline = time.monotonic() + 840.0
    inference, reply = turn_orchestrator._held_turn_budgets(
        {"inference_timeout_sec": 570.0, ci.REPLY_DEADLINE_PAYLOAD_KEY: deadline})
    assert inference == 570.0
    assert 820.0 <= reply <= 830.0


def test_motor_budget_is_clamped_to_the_reply_budget():
    deadline = time.monotonic() + 200.0
    inference, reply = turn_orchestrator._held_turn_budgets(
        {"inference_timeout_sec": 570.0, ci.REPLY_DEADLINE_PAYLOAD_KEY: deadline})
    assert inference == reply and reply <= 190.0


def test_no_deadline_leaves_the_request_unchanged():
    assert turn_orchestrator._held_turn_budgets({"inference_timeout_sec": 42}) == (42, None)
    assert turn_orchestrator._held_turn_budgets({}) == (None, None)


def _incident_run() -> HarnessRunV1:
    return HarnessRunV1(correlation_id="6ddea5f1", final_text=None, draft_text=DRAFT, finalize_ran=False,
                        step_count=107, exit_code=-9, compliance_verdict="failed", grounding_status=REPAIR_FAILED)


def test_urgent_error_frame_carries_the_whole_draft():
    frame = turn_orchestrator._harness_error_frame(
        _incident_run(), correlation_id="c", partial_max_len=turn_orchestrator._URGENT_PARTIAL_DRAFT_MAX_LEN)
    assert frame["partial_draft"] == DRAFT
    # Chat frames keep their old cap.
    chat = turn_orchestrator._harness_error_frame(_incident_run(), correlation_id="c")
    assert len(chat["partial_draft"]) == turn_orchestrator._PARTIAL_DRAFT_MAX_LEN


def test_context_overflow_draft_is_never_salvaged():
    assert ci.salvage_urgent_draft({"partial_draft": DRAFT, "context_overflow": True}) == ""
    assert ci.salvage_urgent_draft({"partial_draft": DRAFT}) == DRAFT.strip()


def test_held_turn_limit_counts_from_receipt(monkeypatch):
    # durable-runs' timer started before Hub got the request: time spent fencing the hold
    # comes out of the limit handed to _generate.
    from unittest.mock import AsyncMock

    from orion.schemas.durable_run import CuriosityTurnRequestV1

    async def slow_fence(*_a, **_kw):
        await asyncio.sleep(0.3)
    monkeypatch.setattr(ci, "validate_hold_ref", slow_fence)
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._generate = AsyncMock(return_value=("grounded finding", {}))
    req = CuriosityTurnRequestV1(run_id="a153451fe423", correlation_id="c", prompt="p", timeout_sec=900,
                                 gpu_lease=_ref())
    asyncio.run(loop._turn_result_for(req, hold_lock=False))
    assert 899.0 < loop._generate.await_args.kwargs["timeout_sec"] <= 899.75
