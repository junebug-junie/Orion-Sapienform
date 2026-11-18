"""The governor rewrites a final reply that names a correction_id no tool returned."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.harness.runner import HarnessMotorResult, build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1

REAL = "8f0c1c3e-5b7a-4f5e-9a77-1f3d2a9b6c10"
LIVE_REPLY = "I recorded the correction for ¶9.\n- correction_id: corr_9_218_reproduce\n"


class _Chain:
    substrate_appraisal = make_appraisal()
    reflection = make_reflection()
    verdict_molecule_id = "verdict-1"
    finalize_changed = True
    response_repair_ran = True
    response_repair_reason = "x"
    quick_lane_skipped_5b = False
    outcome_molecule = MagicMock(surprise_resolved=True)

    def __init__(self, final_text):
        self.final_text = final_text


async def _run(final_text, tool_result_texts):
    from app import bus_listener

    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="00000000-0000-4000-8000-000000000009", thought_event=thought,
        user_message="record it", permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
    )
    molecule = build_draft_molecule(
        correlation_id=request.correlation_id, thought=thought, draft_text="draft",
        grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    motor = HarnessMotorResult(draft_text="draft", grammar_receipts=[], step_count=3, exit_code=0,
                               draft_molecule=molecule, correction_results=tuple(tool_result_texts))
    with patch.object(bus_listener, "HarnessRunner", return_value=AsyncMock(run=AsyncMock(return_value=motor))), \
            patch.object(bus_listener, "run_harness_finalize_chain", AsyncMock(return_value=_Chain(final_text))):
        return await bus_listener.handle_harness_run_request(AsyncMock(), request, reply_to="r")


@pytest.mark.asyncio
async def test_an_invented_id_is_replaced_and_the_reply_says_nothing_was_recorded():
    run = await _run(LIVE_REPLY, [])
    assert "Nothing was recorded." in run.final_text
    assert "correction_id: [unverified]" in run.final_text


@pytest.mark.asyncio
async def test_an_id_a_tool_returned_is_left_alone():
    reply = f"Recorded. correction_id: {REAL}"
    run = await _run(reply, [f'{{"correction_id": "{REAL}"}}'])
    assert run.final_text == reply


@pytest.mark.asyncio
async def test_the_degraded_path_without_substrate_appraisal_is_gated_too():
    """Substrate appraisal down: the raw motor draft becomes the reply. It must not skip the gate."""
    from app import bus_listener
    SubstrateAppraisalUnavailableError = bus_listener.SubstrateAppraisalUnavailableError

    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="00000000-0000-4000-8000-00000000000a", thought_event=thought,
        user_message="record it", permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
    )
    molecule = build_draft_molecule(
        correlation_id=request.correlation_id, thought=thought, draft_text=LIVE_REPLY,
        grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    motor = HarnessMotorResult(draft_text=LIVE_REPLY, grammar_receipts=[], step_count=3, exit_code=0,
                               draft_molecule=molecule, correction_results=())
    with patch.object(bus_listener, "HarnessRunner", return_value=AsyncMock(run=AsyncMock(return_value=motor))), \
            patch.object(bus_listener, "run_harness_finalize_chain",
                         AsyncMock(side_effect=SubstrateAppraisalUnavailableError("down"))):
        run = await bus_listener.handle_harness_run_request(AsyncMock(), request, reply_to="r")
    assert run.final_text and "Nothing was recorded." in run.final_text
    assert "correction_id: [unverified]" in run.final_text
    assert "corr_9_218_reproduce" not in (run.draft_text or "").split("[Unverified")[0]


def test_the_gate_helper_passes_none_through_and_uses_only_correction_results():
    from app.bus_listener import _gate_claims

    assert _gate_claims(None, object(), "c") is None
    motor = HarnessMotorResult(draft_text="", grammar_receipts=[], correction_results=(f'{{"correction_id": "{REAL}"}}',))
    assert _gate_claims(f"correction_id: {REAL}", motor, "c") == f"correction_id: {REAL}"
    assert "Nothing was recorded." in _gate_claims("correction_id: nope_nope_9", motor, "c")


@pytest.mark.asyncio
async def test_the_draft_preview_shown_to_the_hub_is_gated_before_it_is_published():
    from app import bus_listener

    request = HarnessRunRequestV1(
        correlation_id="00000000-0000-4000-8000-00000000000b", thought_event=make_thought(),
        user_message="record it", permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
        draft_preview=True,
    )
    motor = HarnessMotorResult(draft_text=LIVE_REPLY, grammar_receipts=[], correction_results=())
    publish = AsyncMock()
    with patch.object(bus_listener, "draft_preview_hold_reason", return_value=None), \
            patch.object(bus_listener, "publish_harness_run_draft_preview", publish):
        shown, held = await bus_listener._maybe_publish_draft_preview(
            AsyncMock(), request, motor, repair_overlay=make_repair_overlay(), corr="c", received_monotonic=0.0
        )
    assert held is None
    published_text = publish.await_args.kwargs["text"]
    assert "Nothing was recorded." in published_text and "correction_id: [unverified]" in published_text
    assert shown == published_text
