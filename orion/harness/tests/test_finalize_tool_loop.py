from __future__ import annotations

import json

import pytest

from orion.harness.finalize import (
    FINALIZE_LOOP_TOOL_ALLOWLIST,
    maybe_run_finalize_tool_retry,
    run_harness_finalize_chain,
)
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.schemas.harness_finalize import GrammarReceiptV1


def _retry_kwargs(**overrides):
    thought = make_thought()
    # surprise_level=0.5 forces the deterministic quick lane blocked (matches
    # test_harness_finalize_chain.py's convention) -- a "misaligned" verdict is
    # only real if the quick lane (which can only ever produce "aligned")
    # didn't fire, so the retry's own internal re-reflection call must go
    # through the real LLM path too, not silently short-circuit past it.
    appraisal = make_appraisal(surprise_level=0.5)
    base = dict(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=make_repair_overlay(),
        user_message="what do you see?",
        grammar_receipts=[],
    )
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_flag_off_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", raising=False)
    reflection = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")

    async def cortex_client(_req):  # pragma: no cover -- must never be called
        raise AssertionError("cortex_client must not be called when the flag is off")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=reflection, cortex_client=cortex_client)
    )

    assert result is reflection
    assert receipts == []
    assert retried is False
    assert tool is None


@pytest.mark.asyncio
async def test_aligned_verdict_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    reflection = make_reflection(alignment_verdict="aligned", recommended_tool="look_at_camera")

    async def cortex_client(_req):  # pragma: no cover
        raise AssertionError("must not fire on an aligned verdict")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=reflection, cortex_client=cortex_client)
    )

    assert result is reflection
    assert retried is False
    assert tool is None


@pytest.mark.asyncio
async def test_no_recommendation_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    reflection = make_reflection(alignment_verdict="misaligned", recommended_tool=None)

    async def cortex_client(_req):  # pragma: no cover
        raise AssertionError("must not fire with no recommendation")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=reflection, cortex_client=cortex_client)
    )

    assert retried is False
    assert tool is None


@pytest.mark.asyncio
async def test_unrecognized_tool_dropped_not_dispatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool name outside the allowlist (hallucinated or stale) must be
    logged and ignored, never dispatched -- this is the load-bearing safety
    check, not a cosmetic one. It must also be SCRUBBED from the returned
    reflection, not just skipped at the dispatch site -- otherwise the
    unvetted name keeps flowing into the verdict molecule (published +
    persisted) and 5c's voice-finalize prompt looking like a legitimate
    recommendation."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    reflection = make_reflection(
        alignment_verdict="misaligned",
        recommended_tool="delete_all_the_things",
        recommended_tool_reason="a hallucinated justification",
    )
    assert "delete_all_the_things" not in FINALIZE_LOOP_TOOL_ALLOWLIST

    async def cortex_client(_req):  # pragma: no cover
        raise AssertionError("an unrecognized tool must never be dispatched")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=reflection, cortex_client=cortex_client)
    )

    assert retried is False
    assert tool is None
    assert result.recommended_tool is None
    assert result.recommended_tool_reason is None
    # Everything else about the reflection is preserved -- only the unvetted
    # recommendation fields are scrubbed.
    assert result.alignment_verdict == "misaligned"


@pytest.mark.asyncio
async def test_tool_already_called_this_turn_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound enforcement: if the recommended tool already has a receipt this
    turn (motor stage or an earlier retry), it is never called again -- this
    is what actually prevents a runaway loop, independent of the retry count."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    reflection = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")
    existing = [GrammarReceiptV1(step_index=0, tool_name="look_at_camera", summary="already ran")]

    async def cortex_client(_req):  # pragma: no cover
        raise AssertionError("must not call a tool already called this turn")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=reflection, grammar_receipts=existing, cortex_client=cortex_client)
    )

    assert retried is False
    assert tool is None
    assert receipts == existing


@pytest.mark.asyncio
async def test_successful_retry_calls_tool_and_re_reflects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    original = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")
    re_reflected = make_reflection(alignment_verdict="aligned")
    calls: list[str] = []

    async def cortex_client(req):
        verb = req.plan.verb_name
        calls.append(verb)
        if verb == "look_at_camera":
            return {"final_text": "chair, table, door visible"}
        if verb == "harness_finalize_reflect":
            return {"final_text": json.dumps(re_reflected.model_dump(mode="json"))}
        raise AssertionError(f"unexpected verb {verb}")

    published: list[dict] = []

    async def grammar_publish_fn(event, **kwargs):
        published.append({"event": event, **kwargs})

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(
            reflection=original,
            cortex_client=cortex_client,
            grammar_publish_fn=grammar_publish_fn,
        )
    )

    assert calls[0] == "look_at_camera"
    assert calls[1] == "harness_finalize_reflect"
    assert retried is True
    assert tool == "look_at_camera"
    assert result.alignment_verdict == "aligned"
    assert len(receipts) == 1
    assert receipts[0].tool_name == "look_at_camera"
    assert len(published) == 1  # the synthetic grammar step actually published


@pytest.mark.asyncio
async def test_dispatch_failure_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    reflection = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")

    async def cortex_client(_req):
        raise RuntimeError("vision-window unreachable")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=reflection, cortex_client=cortex_client)
    )

    assert result is reflection
    assert receipts == []
    assert retried is False
    assert tool is None


@pytest.mark.asyncio
async def test_grammar_publish_failure_still_completes_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grammar-publish failure must degrade to an unpublished (grammar_event_id=
    None) receipt, not abandon the retry -- the tool's evidence is still real
    and still worth re-reflecting on even if it can't be durably recorded."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    original = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")
    re_reflected = make_reflection(alignment_verdict="aligned")

    async def cortex_client(req):
        if req.plan.verb_name == "look_at_camera":
            return {"final_text": "chair visible"}
        return {"final_text": json.dumps(re_reflected.model_dump(mode="json"))}

    async def broken_publish_fn(*_a, **_k):
        raise RuntimeError("bus down")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(
            reflection=original,
            cortex_client=cortex_client,
            grammar_publish_fn=broken_publish_fn,
        )
    )

    assert retried is True
    assert tool == "look_at_camera"
    assert len(receipts) == 1
    assert receipts[0].grammar_event_id is None
    assert result.alignment_verdict == "aligned"


@pytest.mark.asyncio
async def test_re_reflect_failure_degrades_honestly_not_silently_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_finalize_reflection has its own internal fail-open (it never raises
    -- an LLM/parse failure degrades to reflection_source=
    "degraded_llm_failure_fallback" rather than propagating), so
    maybe_run_finalize_tool_retry's own except-Exception wrapper around it is
    defense-in-depth, not the path this scenario actually takes. The real
    contract: the tool call's real evidence is still recorded (receipts grows
    by one), and the caller gets an honest "reflection itself failed" verdict
    -- not a silent revert to the stale pre-retry reflection, which would hide
    that a second LLM call was even attempted."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    original = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")

    async def cortex_client(req):
        if req.plan.verb_name == "look_at_camera":
            return {"final_text": "chair visible"}
        raise RuntimeError("gateway down for the re-reflect call")

    result, receipts, retried, tool, _trace_id = await maybe_run_finalize_tool_retry(
        **_retry_kwargs(reflection=original, cortex_client=cortex_client)
    )

    assert retried is True
    assert tool == "look_at_camera"
    assert len(receipts) == 1
    assert result.alignment_verdict == "misaligned"
    assert result.reflection_source == "degraded_llm_failure_fallback"
    assert any("reflect_llm_failed" in note for note in result.alignment_notes)


@pytest.mark.asyncio
async def test_harness_finalize_chain_records_retry_on_outcome_molecule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a misaligned-with-recommendation reflection flowing through
    the real chain lands finalize_loop_retried/finalize_loop_tool on the
    published outcome molecule, and the second reflection's verdict (not the
    first) is what verdict/outcome/voice-finalize actually see."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    thought = make_thought()
    coalition = build_coalition_snapshot(thought)
    receipts: list[GrammarReceiptV1] = []
    draft_text = "internal draft"
    molecule = build_draft_molecule(
        correlation_id="c-1",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=receipts,
        coalition_snapshot=coalition,
        repair_overlay=make_repair_overlay(),
    )
    appraisal = make_appraisal(surprise_level=0.5)
    first_reflection = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")
    second_reflection = make_reflection(alignment_verdict="aligned")
    cortex_calls: list[str] = []
    outcome_holder: list[object] = []

    async def substrate_client(_mol):
        return appraisal

    async def cortex_client(req):
        verb = req.plan.verb_name
        cortex_calls.append(verb)
        if verb == "look_at_camera":
            return {"final_text": "chair, table, door visible"}
        if verb == "harness_finalize_reflect":
            first_call = cortex_calls.count("harness_finalize_reflect") == 1
            reflection = first_reflection if first_call else second_reflection
            return {
                "final_text": reflection.model_dump(mode="json"),
                "trace_id": "trace-first" if first_call else "trace-retry",
            }
        if verb == "orion_voice_finalize":
            return {"final_text": "final for juniper"}
        raise AssertionError(f"unexpected verb {verb}")

    async def outcome_publish_fn(molecule, **_):
        outcome_holder.append(molecule)

    verdict_holder: list[object] = []

    async def verdict_publish_fn(molecule, **_):
        verdict_holder.append(molecule)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda result: result["final_text"],
        )
        mp.setattr(
            "orion.harness.finalize.extract_voice_finalize_text",
            lambda _result: "final for juniper",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-1",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=receipts,
            repair_overlay=make_repair_overlay(),
            user_message="what do you see?",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
            outcome_publish_fn=outcome_publish_fn,
            verdict_publish_fn=verdict_publish_fn,
        )

    assert cortex_calls == ["harness_finalize_reflect", "look_at_camera", "harness_finalize_reflect", "orion_voice_finalize"]
    assert chain.reflection.alignment_verdict == "aligned"  # the SECOND reflection won
    assert len(outcome_holder) == 1
    outcome = outcome_holder[0]
    assert outcome.finalize_loop_retried is True
    assert outcome.finalize_loop_tool == "look_at_camera"
    assert outcome.alignment_verdict == "aligned"
    # The verdict molecule's cortex_trace_id must point at the RETRY's LLM
    # call (trace-retry), not the stale pre-retry one (trace-first) -- a
    # durably-persisted trace id pointing at the wrong transcript is exactly
    # the kind of bug that only shows up when someone tries to debug via it.
    assert len(verdict_holder) == 1
    assert verdict_holder[0].cortex_trace_id == "trace-retry"


@pytest.mark.asyncio
async def test_max_retries_constant_is_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAX_FINALIZE_LOOP_RETRIES must actually bound the loop in
    run_harness_finalize_chain, not just exist as documentation -- set it to 0
    and confirm the tool is never called even though the reflection recommends
    it and the flag is on."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    monkeypatch.setattr("orion.harness.finalize.MAX_FINALIZE_LOOP_RETRIES", 0)
    thought = make_thought()
    coalition = build_coalition_snapshot(thought)
    draft_text = "internal draft"
    molecule = build_draft_molecule(
        correlation_id="c-1",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=coalition,
        repair_overlay=make_repair_overlay(),
    )
    appraisal = make_appraisal(surprise_level=0.5)
    reflection = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")
    outcome_holder: list[object] = []

    async def substrate_client(_mol):
        return appraisal

    async def cortex_client(req):
        if req.plan.verb_name == "look_at_camera":  # pragma: no cover
            raise AssertionError("must not fire when MAX_FINALIZE_LOOP_RETRIES is 0")
        if req.plan.verb_name == "harness_finalize_reflect":
            return {"final_text": reflection.model_dump(mode="json")}
        return {"final_text": "final for juniper"}

    async def outcome_publish_fn(molecule, **_):
        outcome_holder.append(molecule)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda result: result["final_text"],
        )
        mp.setattr(
            "orion.harness.finalize.extract_voice_finalize_text",
            lambda _result: "final for juniper",
        )
        await run_harness_finalize_chain(
            correlation_id="c-1",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(),
            user_message="what do you see?",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
            outcome_publish_fn=outcome_publish_fn,
        )

    assert outcome_holder[0].finalize_loop_retried is False
    assert outcome_holder[0].finalize_loop_tool is None


@pytest.mark.asyncio
async def test_harness_finalize_chain_flag_off_no_retry_field_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", raising=False)
    thought = make_thought()
    coalition = build_coalition_snapshot(thought)
    draft_text = "internal draft"
    molecule = build_draft_molecule(
        correlation_id="c-1",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=coalition,
        repair_overlay=make_repair_overlay(),
    )
    appraisal = make_appraisal(surprise_level=0.5)
    reflection = make_reflection(alignment_verdict="misaligned", recommended_tool="look_at_camera")
    outcome_holder: list[object] = []

    async def substrate_client(_mol):
        return appraisal

    async def cortex_client(req):
        if req.plan.verb_name == "look_at_camera":  # pragma: no cover
            raise AssertionError("must not fire when the flag is off")
        return {"final_text": reflection.model_dump(mode="json") if req.plan.verb_name == "harness_finalize_reflect" else "final for juniper"}

    async def outcome_publish_fn(molecule, **_):
        outcome_holder.append(molecule)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda result: result["final_text"],
        )
        mp.setattr(
            "orion.harness.finalize.extract_voice_finalize_text",
            lambda _result: "final for juniper",
        )
        await run_harness_finalize_chain(
            correlation_id="c-1",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(),
            user_message="what do you see?",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
            outcome_publish_fn=outcome_publish_fn,
        )

    assert outcome_holder[0].finalize_loop_retried is False
    assert outcome_holder[0].finalize_loop_tool is None
