"""The repair pass gets a reply budget sized from the draft (#3607).

Live 2026-11-16..17: the day letter's first attempt failed with
``harness_step_failed:letter:no_final_frame`` on 3 of the last 6 days. 5b marked
the letter aligned but ``strain_unresolved`` (3 of 4 failures) or misaligned (1),
so the repair pass rewrote the whole letter under a fixed 3072-token cap. Letters
are 18-24k chars, 5.4-6.9k tokens (Qwen tokenizer, 3.07-3.55 chars/token). Every
repair hit finish_reason=length at exactly 3072 completion tokens with thinking
off (gateway reasoning_len=0), the extractor refused the cut reply, and the turn
failed (harness_turn_trace 9061adc5, 390c58ab, 8d3ac56d, 139e7e46).
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from orion.harness.finalize import (
    RESPONSE_REPAIR_MAX_TOKENS,
    RESPONSE_REPAIR_MAX_TOKENS_CEILING,
    HarnessFinalizeFailedError,
    build_response_repair_context,
    response_repair_max_tokens,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought

# Lowest chars/token measured on the six live letters (139e7e46: 21,186 chars, 6,905 tokens).
# Hex ids and markdown tokenize densely, so a letter needs MORE tokens than plain prose.
LIVE_MIN_CHARS_PER_TOKEN = 3.07

_PARAGRAPH = (
    "At 07:14 the walkway camera saw a vehicle [arc:46e924094d7a7c32] and I held the "
    "prediction error at 0.25 until the dream at 17:36 [experience:gpu_wait:0df49e68] "
    "settled it. Juniper asked about the promise ledger [juniper_chat:67af2b9d-42c]; "
    "I said I would replay it tomorrow.\n\n"
)
# Same size as the longest live letter first attempt (9061adc5, 24,118 chars).
LETTER = ("## Today\n\n" + _PARAGRAPH * 200)[:24_118]
SHORT_REPLY = "You're right -- I should have checked the cabinet sensor before saying it ran hot."


def _tokens_needed(text: str) -> int:
    return math.ceil(len(text) / LIVE_MIN_CHARS_PER_TOKEN)


def _repair_payload(*, content: str, finish_reason: str) -> dict[str, Any]:
    return {
        "status": "success",
        "final_text": None,
        "trace_id": "t-repair",
        "steps": [{
            "status": "success",
            "verb_name": "orion_response_repair",
            "step_name": "llm_orion_response_repair",
            "order": 0,
            "result": {"LLMGatewayService": {
                "content": content,
                "raw": {"choices": [{"finish_reason": finish_reason, "index": 0,
                                     "message": {"role": "assistant", "content": content}}]},
            }},
            "error": None,
        }],
    }


def _model_rewrite(draft: str, max_tokens: int, *, grow: float = 1.0) -> dict[str, Any]:
    """A model that rewrites the whole reply (``grow`` x the draft) and stops at max_tokens."""
    if grow == 1.0:
        rewrite = draft + " (corrected)"
    else:
        rewrite = (draft * math.ceil(grow))[: int(len(draft) * grow)]
    needed = _tokens_needed(rewrite)
    if needed > max_tokens:
        cut = int(max_tokens * LIVE_MIN_CHARS_PER_TOKEN)
        return _repair_payload(content=rewrite[:cut], finish_reason="length")
    return _repair_payload(content=rewrite, finish_reason="stop")


async def _chain(draft: str, reflection, *, grow: float = 1.0):
    thought = make_thought()
    molecule = build_draft_molecule(correlation_id="c-letter", thought=thought, draft_text=draft,
                                    grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought),
                                    repair_overlay=make_repair_overlay())
    seen: dict[str, Any] = {}

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(req: Any):
        if req.plan.verb_name == "orion_response_repair":
            seen["max_tokens"] = req.context["max_tokens"]
            return _model_rewrite(draft, req.context["max_tokens"], grow=grow)
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t-reflect"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("orion.harness.finalize.maybe_quick_lane_verdict", lambda **_: None)
        mp.setattr("orion.harness.finalize.extract_finalize_reflection_payload",
                   lambda _result: reflection.model_dump(mode="json"))
        chain = await run_harness_finalize_chain(
            correlation_id="c-letter", draft_text=draft, draft_molecule=molecule, thought=thought,
            grammar_receipts=[], repair_overlay=make_repair_overlay(),
            user_message="Write tonight's end-of-day letter.",
            voice_contract=None, cortex_client=cortex_client, substrate_client=substrate_client)
    return chain, seen


# --- the budget -------------------------------------------------------------------------


def test_short_replies_keep_todays_budget() -> None:
    assert RESPONSE_REPAIR_MAX_TOKENS == 3072
    for draft in ("", SHORT_REPLY, "x" * 6_000):
        assert response_repair_max_tokens(draft) == 3072, len(draft)


def test_a_letter_gets_room_for_the_whole_rewrite() -> None:
    for size in (18_179, 21_186, 24_118):   # live letter drafts
        draft = LETTER[:size]
        budget = response_repair_max_tokens(draft)
        assert budget > 3072
        # Headroom over the measured worst case, so a slightly longer rewrite still fits.
        assert budget >= 1.2 * _tokens_needed(draft), (size, budget)
        assert budget <= RESPONSE_REPAIR_MAX_TOKENS_CEILING


def test_budget_is_capped_for_a_huge_draft() -> None:
    assert response_repair_max_tokens("x" * 500_000) == RESPONSE_REPAIR_MAX_TOKENS_CEILING


def test_ceiling_plus_a_letter_repair_prompt_fits_the_smallest_repair_lane() -> None:
    # Live letter repair prompts: 38,465-42,125 tokens (llm_call_usage, 11-16..17). The
    # repair runs on the turn's hold or agent/chat; the smallest of those windows is 131072.
    assert 42_125 + RESPONSE_REPAIR_MAX_TOKENS_CEILING < 131_072


def test_repair_request_carries_the_sized_budget() -> None:
    ctx = build_response_repair_context(correlation_id="c-1", draft_text=LETTER,
                                        reflection=make_reflection(strain_unresolved=True),
                                        user_message="letter")
    assert ctx["max_tokens"] == response_repair_max_tokens(LETTER) > 3072
    assert ctx["chat_template_kwargs"] == {"enable_thinking": False}
    short = build_response_repair_context(correlation_id="c-1", draft_text=SHORT_REPLY,
                                          reflection=make_reflection(alignment_verdict="misaligned"),
                                          user_message="hi")
    assert short["max_tokens"] == 3072


# --- end to end through the chain -------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reflection", [
    make_reflection(alignment_verdict="aligned", strain_unresolved=True),   # 390c58ab, 8d3ac56d, 139e7e46
    make_reflection(alignment_verdict="misaligned", strain_unresolved=True),  # 9061adc5
])
async def test_a_letter_longer_than_3072_tokens_does_not_fail(reflection) -> None:
    assert _tokens_needed(LETTER) > 3072
    chain, seen = await _chain(LETTER, reflection)
    assert seen["max_tokens"] > 3072
    assert chain.final_text.startswith(LETTER)
    assert chain.response_repair_ran is True
    assert "kept_draft" not in (chain.response_repair_reason or "")


@pytest.mark.asyncio
async def test_a_short_reply_repairs_as_before() -> None:
    chain, seen = await _chain(SHORT_REPLY, make_reflection(alignment_verdict="misaligned"))
    assert seen["max_tokens"] == 3072
    assert chain.final_text == SHORT_REPLY + " (corrected)"
    assert chain.response_repair_reason == "misaligned"


# --- a repair that is still cut off -----------------------------------------------------


@pytest.mark.asyncio
async def test_truncated_repair_of_an_aligned_draft_keeps_the_draft_and_says_so() -> None:
    """5b passed the draft's content (aligned, no unsupported claim) and asked for repair only
    for open strain: when the rewrite still does not fit, Juniper gets the draft, and the trace
    says the repair was cut off."""
    chain, _ = await _chain(LETTER, make_reflection(alignment_verdict="aligned", strain_unresolved=True), grow=3.0)
    assert chain.final_text == LETTER
    assert chain.finalize_changed is False
    assert chain.response_repair_ran is True
    assert chain.response_repair_reason == "strain_unresolved_kept_draft:truncated"
    assert chain.outcome_molecule.response_repair_reason == "strain_unresolved_kept_draft:truncated"
    assert chain.outcome_molecule.finalize_failed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reflection", [
    make_reflection(alignment_verdict="misaligned"),
    make_reflection(alignment_verdict="uncertain"),
    make_reflection(alignment_verdict="aligned", strain_unresolved=True,
                    alignment_notes=["unsupported claim: the draft says X; notes say Y"]),
])
async def test_truncated_repair_of_a_rejected_draft_still_fails(reflection) -> None:
    """The judge found a problem in the draft itself: shipping it would be a silent fallback
    to the thing the judge rejected. The turn fails, as before, and the retry path runs."""
    with pytest.raises(HarnessFinalizeFailedError, match="truncated at max_tokens"):
        await _chain(LETTER, reflection, grow=3.0)
