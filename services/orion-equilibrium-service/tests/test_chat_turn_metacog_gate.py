from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from app.chat_turn_metacog_gate import (
    ChatTurnCorrelator,
    _capped_grounding_capsule_for_upstream,
    build_chat_turn_metacog_trigger,
    evaluate_chat_turn_gate_conditions,
    is_chat_turn_evidence_terminal,
)


def _thought_event(
    *,
    correlation_id="corr-1",
    disposition="proceed",
    boundary_register=False,
) -> dict:
    return {
        "correlation_id": correlation_id,
        "disposition": disposition,
        "disposition_reasons": [] if disposition == "proceed" else ["some_reason"],
        "boundary_register": boundary_register,
        "grounding_capsule": {"identity_summary": ["x"]},
        "autonomy_slice": None,
    }


def _run_artifact(
    *,
    correlation_id="corr-1",
    compliance_verdict="completed",
    exit_code=0,
    finalize_degraded_reason=None,
    reflection=None,
    substrate_appraisal=None,
) -> dict:
    return {
        "correlation_id": correlation_id,
        "compliance_verdict": compliance_verdict,
        "grounding_status": "grounded",
        "exit_code": exit_code,
        "finalize_degraded_reason": finalize_degraded_reason,
        "reflection": reflection,
        "substrate_appraisal": substrate_appraisal,
        "finalize_ran": reflection is not None,
    }


# --- evaluate_chat_turn_gate_conditions --------------------------------------


def test_unremarkable_turn_fires_nothing():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == []


def test_timeout_fires_with_reason():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=None,
        run_artifact=None,
        timed_out=True,
        timeout_reason="exec_turn_timeout",
        surprise_threshold=0.7,
    )
    assert fired == ["timeout=exec_turn_timeout"]


def test_stance_react_timeout_fires_with_reason():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=None,
        run_artifact=None,
        timed_out=True,
        timeout_reason="stance_react_timeout",
        surprise_threshold=0.7,
    )
    assert fired == ["timeout=stance_react_timeout"]


def test_timeout_without_reason_falls_back():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=None,
        run_artifact=None,
        timed_out=True,
        surprise_threshold=0.7,
    )
    assert fired == ["timeout=unknown"]


def test_timeout_preserves_already_accumulated_boundary_register():
    """A thought_event already accumulated (e.g. exec_turn_timeout arriving
    after a normal ThoughtEventV1) must not lose its own conditions just
    because run_artifact will never arrive."""
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(boundary_register=True),
        run_artifact=None,
        timed_out=True,
        timeout_reason="exec_turn_timeout",
        surprise_threshold=0.7,
    )
    assert fired == ["timeout=exec_turn_timeout", "boundary_register=true"]


def test_timeout_short_circuits_run_artifact_fields_even_if_present():
    """run_artifact should never actually be non-None alongside timed_out=True
    in real usage (they're mutually exclusive terminal paths), but the
    function must not evaluate run_artifact-only conditions in that case."""
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(exit_code=1),
        timed_out=True,
        timeout_reason="exec_turn_timeout",
        surprise_threshold=0.7,
    )
    assert fired == ["timeout=exec_turn_timeout"]


def test_deferred_disposition_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(disposition="defer"),
        run_artifact=None,
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["disposition=defer"]


def test_refused_disposition_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(disposition="refuse"),
        run_artifact=None,
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["disposition=refuse"]


def test_boundary_register_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(boundary_register=True),
        run_artifact=_run_artifact(),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert "boundary_register=true" in fired


def test_misaligned_reflection_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(reflection={"alignment_verdict": "misaligned", "strain_unresolved": False}),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["alignment_verdict=misaligned"]


def test_strain_unresolved_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(reflection={"alignment_verdict": "aligned", "strain_unresolved": True}),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["strain_unresolved=true"]


def test_surprise_at_or_above_threshold_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(substrate_appraisal={"surprise_level": 0.7}),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["surprise_level=0.700"]


def test_surprise_below_threshold_does_not_fire():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(substrate_appraisal={"surprise_level": 0.69}),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == []


def test_compliance_short_of_complete_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(compliance_verdict="partial"),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["compliance_verdict=partial"]


def test_non_zero_exit_code_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(exit_code=1),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["exit_code=1"]


def test_none_exit_code_does_not_fire():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(exit_code=None),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == []


def test_degraded_finalize_fires():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(),
        run_artifact=_run_artifact(finalize_degraded_reason="substrate_rpc_timeout"),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["finalize_degraded_reason=substrate_rpc_timeout"]


def test_run_artifact_none_and_proceed_disposition_fires_nothing_yet():
    """Mid-flight: thought_event arrived, run_artifact hasn't -- not terminal,
    should not evaluate any run_artifact-only condition as fired."""
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(disposition="proceed"),
        run_artifact=None,
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == []


def test_multiple_conditions_all_reported():
    fired = evaluate_chat_turn_gate_conditions(
        thought_event=_thought_event(boundary_register=True),
        run_artifact=_run_artifact(exit_code=1, compliance_verdict="failed"),
        timed_out=False,
        surprise_threshold=0.7,
    )
    assert fired == ["boundary_register=true", "compliance_verdict=failed", "exit_code=1"]


# --- is_chat_turn_evidence_terminal ------------------------------------------


def test_terminal_on_timeout():
    assert is_chat_turn_evidence_terminal(thought_event=None, run_artifact=None, timed_out=True)


def test_terminal_on_run_artifact():
    assert is_chat_turn_evidence_terminal(thought_event=None, run_artifact=_run_artifact(), timed_out=False)


def test_terminal_on_defer_disposition_without_run_artifact():
    assert is_chat_turn_evidence_terminal(
        thought_event=_thought_event(disposition="defer"), run_artifact=None, timed_out=False
    )


def test_not_terminal_on_proceed_disposition_without_run_artifact():
    assert not is_chat_turn_evidence_terminal(
        thought_event=_thought_event(disposition="proceed"), run_artifact=None, timed_out=False
    )


def test_not_terminal_with_no_evidence_at_all():
    assert not is_chat_turn_evidence_terminal(thought_event=None, run_artifact=None, timed_out=False)


# --- build_chat_turn_metacog_trigger ------------------------------------------


def test_build_trigger_returns_none_when_nothing_fired():
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=_thought_event(),
        run_artifact=_run_artifact(),
        timed_out=False,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is None


def test_build_trigger_on_defer_matches_acceptance_check():
    """Acceptance check from docs/superpowers/design/2026-07-18-collapse-mirror-
    metacog-redesign.md: a defer-disposition turn produces exactly one
    trigger_kind="chat_turn" with fired_conditions containing "disposition=defer"."""
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=_thought_event(disposition="defer"),
        run_artifact=None,
        timed_out=False,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "chat_turn"
    assert "disposition=defer" in trigger.upstream["fired_conditions"]
    assert trigger.signal_refs == ["corr-1"]


def test_build_trigger_upstream_carries_full_evidence():
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=_thought_event(),
        run_artifact=_run_artifact(
            reflection={"alignment_verdict": "misaligned", "alignment_notes": ["note"], "strain_unresolved": True},
            substrate_appraisal={"surprise_level": 0.9},
        ),
        timed_out=False,
        zen_state="not_zen",
        pressure=0.5,
        recall_enabled=True,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert trigger.upstream["alignment_verdict"] == "misaligned"
    assert trigger.upstream["alignment_notes"] == ["note"]
    assert trigger.upstream["strain_unresolved"] is True
    assert trigger.upstream["surprise_level"] == 0.9
    assert trigger.upstream["timed_out"] is False
    assert trigger.recall_enabled is True


# --- grounding_capsule digest-field truncation (code review, PR follow-up) --


def test_capped_grounding_capsule_truncates_long_digest_fields():
    long_digest = "x" * 1000
    capsule = {
        "identity_summary": ["a", "b"],
        "continuity_digest": long_digest,
        "belief_digest": long_digest,
        "memory_digest": long_digest,
    }
    capped = _capped_grounding_capsule_for_upstream(capsule)
    assert len(capped["continuity_digest"]) == 300 + len("...(truncated)")
    assert capped["continuity_digest"].endswith("...(truncated)")
    assert len(capped["belief_digest"]) == 300 + len("...(truncated)")
    assert len(capped["memory_digest"]) == 300 + len("...(truncated)")
    assert capped["identity_summary"] == ["a", "b"]


def test_capped_grounding_capsule_leaves_short_digests_untouched():
    capsule = {"continuity_digest": "short", "belief_digest": None, "memory_digest": "also short"}
    capped = _capped_grounding_capsule_for_upstream(capsule)
    assert capped["continuity_digest"] == "short"
    assert capped["belief_digest"] is None
    assert capped["memory_digest"] == "also short"


def test_capped_grounding_capsule_handles_none_and_non_dict():
    assert _capped_grounding_capsule_for_upstream(None) is None
    assert _capped_grounding_capsule_for_upstream("not a dict") == "not a dict"


def test_build_trigger_caps_grounding_capsule_digest_fields():
    long_digest = "y" * 1000
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event={
            **_thought_event(disposition="defer"),
            "grounding_capsule": {"memory_digest": long_digest},
        },
        run_artifact=None,
        timed_out=False,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert len(trigger.upstream["grounding_capsule"]["memory_digest"]) == 300 + len("...(truncated)")


def test_build_trigger_on_timeout_has_null_run_artifact_fields():
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=None,
        run_artifact=None,
        timed_out=True,
        timeout_reason="exec_turn_timeout",
        zen_state="not_zen",
        pressure=0.5,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert trigger.upstream["fired_conditions"] == ["timeout=exec_turn_timeout"]
    assert trigger.upstream["timeout_reason"] == "exec_turn_timeout"
    assert trigger.upstream["compliance_verdict"] is None
    assert trigger.upstream["disposition"] is None


def test_build_trigger_on_stance_react_timeout():
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=None,
        run_artifact=None,
        timed_out=True,
        timeout_reason="stance_react_timeout",
        zen_state="not_zen",
        pressure=0.5,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert trigger.upstream["fired_conditions"] == ["timeout=stance_react_timeout"]
    assert trigger.upstream["timeout_reason"] == "stance_react_timeout"


def test_build_trigger_on_timeout_preserves_boundary_register_in_upstream():
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=_thought_event(boundary_register=True),
        run_artifact=None,
        timed_out=True,
        timeout_reason="exec_turn_timeout",
        zen_state="not_zen",
        pressure=0.5,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert "boundary_register=true" in trigger.upstream["fired_conditions"]
    assert trigger.upstream["boundary_register"] is True


def test_build_trigger_empty_correlation_id_yields_empty_signal_refs():
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="",
        thought_event=_thought_event(disposition="refuse"),
        run_artifact=None,
        timed_out=False,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert trigger.signal_refs == []


# --- ChatTurnCorrelator (fake redis) ------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_correlator_accumulates_across_two_calls_then_terminal():
    redis = _FakeRedis()
    correlator = ChatTurnCorrelator(redis, ttl_seconds=600)

    thought, run, timed_out, timeout_reason = await correlator.accumulate(
        correlation_id="corr-1", thought_event=_thought_event()
    )
    assert thought is not None
    assert run is None
    assert timeout_reason is None
    assert not is_chat_turn_evidence_terminal(thought_event=thought, run_artifact=run, timed_out=timed_out)
    # Mid-flight state persisted (not cleared) since not yet terminal.
    assert redis.store

    thought2, run2, timed_out2, timeout_reason2 = await correlator.accumulate(
        correlation_id="corr-1", run_artifact=_run_artifact(exit_code=1)
    )
    assert thought2 is not None  # carried over from the first call
    assert run2 is not None
    assert timeout_reason2 is None
    assert is_chat_turn_evidence_terminal(thought_event=thought2, run_artifact=run2, timed_out=timed_out2)
    # Terminal evidence is consumed -- no leaked key.
    assert not redis.store


@pytest.mark.asyncio
async def test_correlator_clears_immediately_on_defer_without_waiting_for_run():
    redis = _FakeRedis()
    correlator = ChatTurnCorrelator(redis, ttl_seconds=600)

    thought, run, timed_out, _timeout_reason = await correlator.accumulate(
        correlation_id="corr-1", thought_event=_thought_event(disposition="defer")
    )
    assert is_chat_turn_evidence_terminal(thought_event=thought, run_artifact=run, timed_out=timed_out)
    assert not redis.store


@pytest.mark.asyncio
async def test_correlator_clears_on_timeout_even_with_no_prior_thought_event():
    redis = _FakeRedis()
    correlator = ChatTurnCorrelator(redis, ttl_seconds=600)

    thought, run, timed_out, timeout_reason = await correlator.accumulate(
        correlation_id="corr-1", timed_out=True, timeout_reason="exec_turn_timeout"
    )
    assert timed_out is True
    assert timeout_reason == "exec_turn_timeout"
    assert is_chat_turn_evidence_terminal(thought_event=thought, run_artifact=run, timed_out=timed_out)
    assert not redis.store


@pytest.mark.asyncio
async def test_correlator_clears_on_stance_react_timeout():
    """The earlier-in-the-pipeline gap (Finding 1): ThoughtClient.react() itself
    never returns to Hub, so Hub gives up before ever calling the harness
    governor -- run_artifact will never arrive. Must fire on this too, not
    just the later exec_turn_timeout case."""
    redis = _FakeRedis()
    correlator = ChatTurnCorrelator(redis, ttl_seconds=600)

    thought, run, timed_out, timeout_reason = await correlator.accumulate(
        correlation_id="corr-1", timed_out=True, timeout_reason="stance_react_timeout"
    )
    assert timeout_reason == "stance_react_timeout"
    assert is_chat_turn_evidence_terminal(thought_event=thought, run_artifact=run, timed_out=timed_out)
    assert not redis.store


@pytest.mark.asyncio
async def test_correlator_preserves_boundary_register_across_exec_turn_timeout():
    """Regression for Finding 4: a thought_event accumulated before a later
    exec_turn_timeout must not lose its boundary_register/disposition signal."""
    redis = _FakeRedis()
    correlator = ChatTurnCorrelator(redis, ttl_seconds=600)

    await correlator.accumulate(
        correlation_id="corr-1", thought_event=_thought_event(boundary_register=True)
    )
    thought, run, timed_out, timeout_reason = await correlator.accumulate(
        correlation_id="corr-1", timed_out=True, timeout_reason="exec_turn_timeout"
    )
    assert thought is not None
    assert thought["boundary_register"] is True
    trigger = build_chat_turn_metacog_trigger(
        correlation_id="corr-1",
        thought_event=thought,
        run_artifact=run,
        timed_out=timed_out,
        timeout_reason=timeout_reason,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        surprise_threshold=0.7,
    )
    assert trigger is not None
    assert "boundary_register=true" in trigger.upstream["fired_conditions"]
    assert "timeout=exec_turn_timeout" in trigger.upstream["fired_conditions"]


@pytest.mark.asyncio
async def test_correlator_noop_on_empty_correlation_id():
    redis = _FakeRedis()
    correlator = ChatTurnCorrelator(redis, ttl_seconds=600)

    thought, run, timed_out, timeout_reason = await correlator.accumulate(
        correlation_id="", thought_event=_thought_event()
    )
    assert thought is not None
    assert not redis.store
