"""orion_response_repair must never ship a reasoning model's chain-of-thought as Orion's reply.

Live 2026-09-24..30: on the thinking-on agent lane the repair verb spent all 8000 tokens
reasoning and returned content=""; extract_cortex_payload_text() fell back to the 26-36k-char
reasoning_content and the harness shipped it as final_text ("We need answer user's request:
repair Orion's draft reply to Juniper..."), 17 turns in 7 days. The payload below mirrors the
stored cognition_traces step for corr 0ba83fec-e66d-5685-9182-d9f867fc81c2.
"""
from __future__ import annotations

from typing import Any

import pytest

from orion.cognition.cortex_payload_extract import (
    cortex_payload_truncated,
    extract_cortex_answer_text,
    extract_cortex_payload_text,
)
from orion.harness.finalize import (
    RESPONSE_REPAIR_MAX_TOKENS,
    HarnessFinalizeFailedError,
    build_response_repair_context,
    build_response_repair_plan_request,
    extract_response_repair_text,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)

REASONING = (
    "We need answer user's request: repair Orion's draft reply to Juniper after integrative "
    "reflection rejected it. Need output only the repaired reply. "
) * 240  # ~30k chars, like the stored traces


def _gateway_step(*, content: str, reasoning: str, finish_reason: str) -> dict[str, Any]:
    return {
        "status": "success",
        "verb_name": "orion_response_repair",
        "step_name": "llm_orion_response_repair",
        "order": 0,
        "result": {
            "LLMGatewayService": {
                "content": content,
                "reasoning_content": reasoning,
                "inline_think_content": None,
                "thinking_source": "provider_reasoning",
                "reasoning_trace": {"role": "reasoning", "stage": "post_answer", "content": reasoning},
                "usage": {"completion_tokens": 8000},
                "raw": {
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "reasoning_content": reasoning,
                            },
                        }
                    ]
                },
            }
        },
        "error": None,
    }


def _stored_payload(*, content: str = "", finish_reason: str = "length") -> dict[str, Any]:
    return {
        "status": "success",
        "final_text": None,
        "steps": [_gateway_step(content=content, reasoning=REASONING, finish_reason=finish_reason)],
    }


def test_stored_reasoning_only_payload_is_refused() -> None:
    payload = _stored_payload()
    # The shared extractor still falls back (JSON-verb callers rely on it) ...
    assert extract_cortex_payload_text(payload).startswith("We need answer")
    # ... the answer-only one does not, and the repair extractor refuses.
    assert extract_cortex_answer_text(payload) == ""
    with pytest.raises(ValueError, match="reasoning only, empty answer"):
        extract_response_repair_text(payload)


def test_truncated_repair_reply_is_refused() -> None:
    """Live a84fc74a shipped this 52-char fragment cut at finish_reason=length."""
    payload = _stored_payload(content="I checked rather than guessed — the mesh is up (90+…")
    assert cortex_payload_truncated(payload) is True
    with pytest.raises(ValueError, match="truncated at max_tokens"):
        extract_response_repair_text(payload)


def test_answer_wins_over_reasoning_when_both_present() -> None:
    payload = _stored_payload(content="The ring is a set of relationships, yes.", finish_reason="stop")
    assert cortex_payload_truncated(payload) is False
    assert extract_response_repair_text(payload) == "The ring is a set of relationships, yes."


def test_repair_request_turns_thinking_off_with_its_own_budget() -> None:
    ctx = build_response_repair_context(
        correlation_id="c-1",
        draft_text="draft",
        reflection=make_reflection(alignment_verdict="misaligned"),
        user_message="hi",
    )
    assert ctx["chat_template_kwargs"] == {"enable_thinking": False}
    assert ctx["max_tokens"] == RESPONSE_REPAIR_MAX_TOKENS == 3072
    req = build_response_repair_plan_request(
        correlation_id="c-1",
        draft_text="draft",
        reflection=make_reflection(alignment_verdict="misaligned"),
        user_message="hi",
    )
    assert req.context["chat_template_kwargs"] == {"enable_thinking": False}
    assert req.context["max_tokens"] == 3072


@pytest.mark.asyncio
async def test_finalize_chain_never_ships_reasoning_as_final_text() -> None:
    """End to end through the chain: the reasoning-only repair result takes the existing
    failure path (HarnessFinalizeFailedError), never becomes final_text."""
    thought = make_thought()
    draft_text = "Here is what I found in my own records."
    molecule = build_draft_molecule(
        correlation_id="c-leak",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="misaligned")
    seen_verbs: list[str] = []

    async def substrate_client(_mol: object):
        return make_appraisal()

    async def cortex_client(req: Any):
        seen_verbs.append(req.plan.verb_name)
        return _stored_payload()

    with pytest.MonkeyPatch.context() as mp:
        # Force the full 5b path (the quick lane would otherwise verdict this aligned).
        mp.setattr("orion.harness.finalize.maybe_quick_lane_verdict", lambda **_: None)
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        with pytest.raises(HarnessFinalizeFailedError, match="reasoning only"):
            await run_harness_finalize_chain(
                correlation_id="c-leak",
                draft_text=draft_text,
                draft_molecule=molecule,
                thought=thought,
                grammar_receipts=[],
                repair_overlay=make_repair_overlay(),
                user_message="what did you do today?",
                voice_contract=None,
                cortex_client=cortex_client,
                substrate_client=substrate_client,
            )
    assert "orion_response_repair" in seen_verbs


def test_inline_think_in_content_is_never_the_reply() -> None:
    """Review finding: reasoning delivered as inline <think> text inside content (llama.cpp
    reasoning_format=none / thinking left on) must not slip through the answer-only path."""
    think = "<think>We need answer user request: repair draft...</think>"
    step = _gateway_step(content=think, reasoning="", finish_reason="stop")
    step["result"]["LLMGatewayService"]["text"] = ""
    step["result"]["LLMGatewayService"]["inline_think_content"] = "We need answer..."
    payload = {"status": "success", "final_text": None, "steps": [step]}
    assert extract_cortex_answer_text(payload) == ""
    with pytest.raises(ValueError, match="reasoning only, empty answer"):
        extract_response_repair_text(payload)

    step["result"]["LLMGatewayService"]["raw"]["choices"][0]["message"]["content"] = think + "\n\nThe real reply."
    step["result"]["LLMGatewayService"]["content"] = think + "\n\nThe real reply."
    assert extract_response_repair_text(payload) == "The real reply."


def test_top_level_truncation_flag_is_refused() -> None:
    """Live shape: router-stripped top-level final_text fragment, truncation flagged only in
    runtime diagnostics (no finish_reason in steps)."""
    payload = {
        "status": "success",
        "final_text": "I checked rather than guessed — the mesh is up (90+…",
        "steps": [],
        "metadata": {"runtime_response_diagnostics": {"truncation_detected": True}},
    }
    assert cortex_payload_truncated(payload) is True
    with pytest.raises(ValueError, match="truncated at max_tokens"):
        extract_response_repair_text(payload)
