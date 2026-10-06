"""Draft-first display (spec L8): governor publishes the grounded draft before
the finalize judge, holds sensitive turns, and records what was shown."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.finalize import HarnessFinalizeChainResult
from orion.harness.runner import HarnessMotorResult, build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunDraftPreviewV1, HarnessRunRequestV1

DRAFT = "You -- all containers are up."
FINAL = "Juniper -- I checked two containers."


def _motor(thought) -> HarnessMotorResult:
    molecule = build_draft_molecule(
        correlation_id="c-1",
        thought=thought,
        draft_text=DRAFT,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    return HarnessMotorResult(
        draft_text=DRAFT, grammar_receipts=[], step_count=1, exit_code=0, draft_molecule=molecule
    )


def _request(thought, *, draft_preview: bool = True, corr: str = "c-1") -> HarnessRunRequestV1:
    return HarnessRunRequestV1(
        correlation_id=corr,
        thought_event=thought,
        user_message="are the containers up?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        draft_preview=draft_preview,
    )


def _draft_publishes(bus, channel: str) -> list:
    return [c for c in bus.publish.await_args_list if c.args[0] == channel]


async def _run(request, *, bus, chain_seen: dict):
    from app import bus_listener

    thought = request.thought_event

    async def _fake_chain(**kwargs):
        # Record whether the draft was already on the bus when the judge started.
        chain_seen["draft_published_before_judge"] = bool(
            _draft_publishes(bus, bus_listener.settings.channel_harness_run_draft_preview)
        )
        from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule

        reflection = make_reflection()
        verdict = await emit_verdict_molecule(correlation_id="c-1", reflection=reflection, publish_fn=AsyncMock())
        outcome = await emit_turn_outcome_molecule(
            correlation_id="c-1",
            thought=thought,
            substrate_appraisal=make_appraisal(),
            reflection=reflection,
            verdict_molecule=verdict,
            draft_text=DRAFT,
            final_text=FINAL,
            finalize_changed=True,
            publish_fn=AsyncMock(),
        )
        return HarnessFinalizeChainResult(
            final_text=FINAL,
            substrate_appraisal=make_appraisal(),
            reflection=reflection,
            verdict_molecule=verdict,
            outcome_molecule=outcome,
            finalize_changed=True,
            quick_lane_skipped_5b=False,
            verdict_molecule_id="verdict-1",
            response_repair_ran=True,
            response_repair_reason="strain_unresolved",
        )

    with patch.object(
        bus_listener, "HarnessRunner", return_value=AsyncMock(run=AsyncMock(return_value=_motor(thought)))
    ), patch.object(bus_listener, "run_harness_finalize_chain", _fake_chain), patch.object(
        bus_listener, "emit_post_turn_closure", AsyncMock(return_value=AsyncMock())
    ):
        return await bus_listener.handle_harness_run_request(bus, request, reply_to="orion:harness:run:result:c-1")


@pytest.mark.asyncio
async def test_draft_published_before_judge_and_recorded_with_final() -> None:
    from app import bus_listener

    bus = AsyncMock()
    seen: dict = {}
    run = await _run(_request(make_thought()), bus=bus, chain_seen=seen)

    assert seen["draft_published_before_judge"] is True
    published = _draft_publishes(bus, bus_listener.settings.channel_harness_run_draft_preview)
    assert len(published) == 1
    envelope = published[0].args[1]
    assert envelope.kind == "harness.run.draft_preview.v1"
    payload = HarnessRunDraftPreviewV1.model_validate(envelope.payload)
    assert payload.correlation_id == "c-1"
    assert payload.text == DRAFT
    # harness_turn_trace.run_artifact keeps both: what was shown, what it became.
    assert run.draft_preview_text == DRAFT
    assert run.draft_preview_held_reason is None
    assert run.final_text == FINAL
    assert run.response_repair_reason == "strain_unresolved"


@pytest.mark.asyncio
async def test_not_requested_publishes_nothing() -> None:
    from app import bus_listener

    bus = AsyncMock()
    run = await _run(_request(make_thought(), draft_preview=False), bus=bus, chain_seen={})
    assert _draft_publishes(bus, bus_listener.settings.channel_harness_run_draft_preview) == []
    assert run.draft_preview_text is None
    assert run.draft_preview_held_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"boundary_register": True}, "sensitive:boundary_register"),
        ({"trust_rupture_score": 0.99}, "sensitive:trust_rupture_score"),
        ({"repair_pressure_level": 0.99}, "sensitive:repair_pressure_level"),
    ],
)
async def test_sensitive_turns_stay_judge_first(overrides, reason) -> None:
    from app import bus_listener

    bus = AsyncMock()
    run = await _run(_request(make_thought(**overrides)), bus=bus, chain_seen={})
    assert _draft_publishes(bus, bus_listener.settings.channel_harness_run_draft_preview) == []
    assert run.draft_preview_text is None
    assert run.draft_preview_held_reason == reason
    assert run.final_text == FINAL


@pytest.mark.asyncio
async def test_publish_failure_costs_only_the_early_display() -> None:
    from app import bus_listener

    channel = bus_listener.settings.channel_harness_run_draft_preview

    async def _publish(ch, env):
        if ch == channel:
            raise RuntimeError("redis hiccup")

    bus = AsyncMock()
    bus.publish.side_effect = _publish
    run = await _run(_request(make_thought()), bus=bus, chain_seen={})
    assert run.final_text == FINAL
    assert run.draft_preview_text is None
    assert run.draft_preview_held_reason == "publish_failed"


@pytest.mark.asyncio
async def test_substrate_degraded_path_still_records_the_shown_draft() -> None:
    from app import bus_listener

    thought = make_thought()

    class _Timeout:
        async def finalize_appraisal(self, molecule, *, correlation_id=None):
            raise TimeoutError("rpc timeout")

    bus = AsyncMock()
    with patch.object(
        bus_listener, "HarnessRunner", return_value=AsyncMock(run=AsyncMock(return_value=_motor(thought)))
    ):
        run = await bus_listener.handle_harness_run_request(
            bus, _request(thought), reply_to="orion:harness:run:result:c-1", substrate_client=_Timeout()
        )
    assert run.finalize_degraded_reason is not None
    assert run.draft_preview_text == DRAFT
