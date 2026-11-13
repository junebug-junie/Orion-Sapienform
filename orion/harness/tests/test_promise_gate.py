"""Intention ledger D0: the judge names promises, plain code decides which are real.

Pins acceptance checks 2 and 5 of docs/superpowers/specs/2026-11-11-intention-ledger-design.md
and the offer rule (Juniper, 2026-11-12: an offer counts only after Juniper accepts it).
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from jinja2 import Environment

from orion.harness.finalize import (
    FINALIZE_REFLECT_MODEL_FIELDS,
    build_finalize_reflect_context,
    finalize_reflect_json_schema,
    finalize_reflect_user_turn,
    run_finalize_reflection,
)
from orion.harness.prefix import compile_harness_prefix
from orion.harness.promise_gate import (
    PROMISE_FIELDS,
    counted_promise_quotes,
    gate_promises,
    quote_in_text,
)
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought
from orion.schemas.harness_finalize import (
    REFLECTION_SOURCE_JUDGE_FAILED,
    FinalizeReflectionV1,
    OpenPromiseRefV1,
    PromiseV1,
)

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "orion/cognition/prompts/harness_finalize_reflect.j2"

REPLY = (
    "The read is still queued.\n\nI’ll hold the deep read until the pipeline digests it, "
    "then pull it back. If you want, I'll re-read it the next time I'm cut."
)
FIRM = "I'll hold the deep read until the pipeline digests it, then pull it back."
OFFER = "If you want, I'll re-read it the next time I'm cut."
USER = "How is the paper read going?"

JUDGEMENT = {
    "imperative": "Report the read status.",
    "tone": "steady",
    "strain_refs": [],
    "alignment_notes": ["draft reports the queue state"],
    "alignment_verdict": "aligned",
    "strain_unresolved": False,
    "world_contact_opportunity": False,
    "recommended_tool": None,
    "recommended_tool_reason": None,
    "promises_made": [],
    "offers_accepted": [],
    "promises_kept": [],
}


def _reflection(**promise_fields: object) -> FinalizeReflectionV1:
    base = {k: v for k, v in JUDGEMENT.items() if k not in PROMISE_FIELDS}
    return FinalizeReflectionV1(
        correlation_id="c", thought_event_id="t", substrate_appraisal_id="a", draft_hash="h",
        **base, **promise_fields,
    )


def _render(ctx: dict) -> str:
    return Environment(autoescape=False).from_string(TEMPLATE.read_text(encoding="utf-8")).render(**ctx)


def _ctx(**kw: object) -> dict:
    return build_finalize_reflect_context(
        correlation_id="c-promise", draft_text=REPLY, thought=make_thought(),
        substrate_appraisal=make_appraisal(), repair_overlay=make_repair_overlay(), user_message=USER, **kw)


# --- acceptance check 2: a "promise" quote that is not in the reply never survives ---------

def test_quote_not_in_reply_is_dropped_and_real_quote_survives() -> None:
    paraphrase = "I will bring the deep read back once the pipeline has digested it."
    reflection = _reflection(promises_made=[PromiseV1(quote=paraphrase), PromiseV1(quote=FIRM)])
    result = gate_promises(reflection, reply_text=REPLY, enabled=True)
    assert [p.quote for p in result.reflection.promises_made] == [FIRM]
    assert result.dropped_not_in_text == [paraphrase]


def test_quote_match_tolerates_whitespace_and_curly_quotes_only() -> None:
    # REPLY has a curly apostrophe and a paragraph break; the judge wrote straight ones.
    assert quote_in_text("The read is still queued. I'll hold the deep read", REPLY)
    assert not quote_in_text("the read is still queued.", REPLY)  # case is kept
    assert not quote_in_text("I'll", REPLY)  # a fragment is not a promise
    assert not quote_in_text("", REPLY)
    # Dashes and the ellipsis character are rewritten by models when they copy.
    assert quote_in_text("lands -- then I'll pull it back...", "It lands \u2014 then I'll pull it back\u2026")
    assert quote_in_text("I'll write the perception_reader.py patch", "I'll write the `perception_reader.py` patch")


def test_short_quote_is_counted_as_too_short_not_as_missing() -> None:
    result = gate_promises(_reflection(promises_made=[PromiseV1(quote="I'll do it.")]),
                           reply_text="Sure. I'll do it.", enabled=True)
    assert result.reflection.promises_made == []
    assert result.dropped_too_short == 1 and result.dropped_not_in_text == []


def test_over_long_duplicate_and_over_cap_quotes_are_dropped() -> None:
    long_reply = "x" * 300
    reflection = _reflection(promises_made=[PromiseV1(quote=long_reply), PromiseV1(quote=FIRM),
                                            PromiseV1(quote=" " + FIRM)])
    result = gate_promises(reflection, reply_text=long_reply + " " + REPLY, enabled=True)
    assert [p.quote for p in result.reflection.promises_made] == [FIRM]
    assert result.dropped_too_long == 1


@pytest.mark.asyncio
async def test_judge_paraphrase_never_reaches_the_reflection() -> None:
    async def _judge(_req: object) -> dict[str, object]:
        return {"final_text": json.dumps({**JUDGEMENT, "promises_made": [
            {"quote": "I promise to finish the paper tomorrow.", "offer": False},
            {"quote": FIRM, "offer": False}]})}

    reflection, quick, _ = await run_finalize_reflection(
        correlation_id="c-gate", draft_text=REPLY, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=USER, cortex_client=_judge)
    assert quick is False
    assert reflection.alignment_verdict == "aligned"
    assert [p.quote for p in reflection.promises_made] == [FIRM]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"promises_made": ["not an object", {"offer": True}], "promises_kept": [7]},
    {"promises_made": 7, "offers_accepted": True, "promises_kept": "act_1"},
])
async def test_malformed_promise_list_does_not_sink_the_verdict(bad: dict) -> None:
    async def _judge(_req: object) -> dict[str, object]:
        return {"final_text": json.dumps({**JUDGEMENT, **bad})}

    reflection, _, _ = await run_finalize_reflection(
        correlation_id="c-bad", draft_text=REPLY, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=USER, cortex_client=_judge)
    assert reflection.reflection_source != REFLECTION_SOURCE_JUDGE_FAILED
    assert reflection.alignment_verdict == "aligned"
    assert reflection.promises_made == [] and reflection.promises_kept == []
    assert reflection.offers_accepted == []


def test_string_false_offer_is_not_an_offer() -> None:
    from orion.harness.promise_gate import coerce_promise_fields

    made = coerce_promise_fields({"promises_made": [{"quote": FIRM, "offer": "false"}]})["promises_made"]
    assert made[0].offer is False


def test_kept_entry_that_is_not_an_open_promise_is_dropped() -> None:
    reflection = _reflection(promises_kept=["act_open", "act_invented"])
    result = gate_promises(reflection, reply_text=REPLY, enabled=True,
                           open_promises=[OpenPromiseRefV1(id="act_open", quote=FIRM)])
    assert result.reflection.promises_kept == ["act_open"]
    assert result.dropped_unknown_kept == ["act_invented"]


def test_regate_against_final_text_drops_a_promise_repair_removed() -> None:
    """The chain gates again on the text Juniper saw; accepted offers and kept ids survive."""
    reflection = _reflection(promises_made=[PromiseV1(quote=FIRM)], offers_accepted=[OFFER],
                             promises_kept=["act_open"])
    repaired = "The read is still queued."
    result = gate_promises(reflection, reply_text=repaired, enabled=True,
                           pending_offers=reflection.offers_accepted, open_promises=reflection.promises_kept)
    assert result.reflection.promises_made == []
    assert result.reflection.offers_accepted == [OFFER]
    assert result.reflection.promises_kept == ["act_open"]


# --- the offer rule -----------------------------------------------------------------------

def test_offer_is_recorded_but_counts_only_after_acceptance() -> None:
    reflection = _reflection(promises_made=[PromiseV1(quote=FIRM), PromiseV1(quote=OFFER, offer=True)])
    gated = gate_promises(reflection, reply_text=REPLY, enabled=True).reflection
    assert [p.offer for p in gated.promises_made] == [False, True]
    assert counted_promise_quotes(gated) == [FIRM]
    assert counted_promise_quotes(gated, accepted_later=[OFFER]) == [FIRM, OFFER]


def test_offer_acceptance_must_name_an_offer_the_judge_was_shown() -> None:
    reflection = _reflection(offers_accepted=[OFFER.replace("’", "'"), "I'll write the summary tonight."])
    result = gate_promises(reflection, reply_text="Yes please.", enabled=True, pending_offers=[OFFER])
    assert result.reflection.offers_accepted == [OFFER]
    assert result.dropped_unknown_offer == ["I'll write the summary tonight."]


def test_acceptance_is_asked_from_the_next_user_message_not_a_word_list() -> None:
    rendered = _render(_ctx(pending_offers=[OFFER]))
    assert f'- "{OFFER}"' in rendered
    assert "user_message accepts" in rendered
    assert "offers_accepted: [] (no offers are open)" not in rendered
    # Without pending offers the judge is told none are open.
    assert "offers_accepted: [] (no offers are open)" in _render(_ctx())


# --- the judge contract with the flag on and off -----------------------------------------

def test_flag_on_schema_asks_for_promise_fields_and_validates_them() -> None:
    schema = finalize_reflect_json_schema(True)
    assert list(schema["properties"])[-3:] == list(PROMISE_FIELDS)
    jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": FIRM, "offer": False}]}, schema)
    # No caps in the schema: Strata 502s an answer that breaks it, which would sink the
    # verdict. Four promises or a long quote must pass the schema; the gate caps them.
    jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": FIRM, "offer": False}] * 4}, schema)
    jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": "y" * 241, "offer": False}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": FIRM}]}, schema)
    rendered = _render(_ctx())
    required = rendered.split("\nREQUIRED JSON FIELDS (", 1)[1].split("\nFAIL-CLOSED", 1)[0]
    listed = [line[2:].split(":", 1)[0] for line in required.splitlines() if line.startswith("- ")]
    assert listed == list(FINALIZE_REFLECT_MODEL_FIELDS)
    assert "PROMISES (independent of alignment_verdict" in rendered


@pytest.mark.parametrize("flag", ["true", "false"])
def test_opening_instruction_is_intact_with_flag_on_and_off(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_PROMISE_LEDGER_ENABLED", flag)
    rendered = _render(_ctx())
    assert "Output only the judgement JSON described under REQUIRED JSON FIELDS.\n" in rendered
    # Promise rules come after every alignment rule, never before INPUTS.
    if flag == "true":
        assert rendered.index("INPUTS") < rendered.index("TOOL RECOMMENDATION") < rendered.index("PROMISES (")


@pytest.mark.asyncio
async def test_chain_regates_promises_against_the_final_text() -> None:
    """Response repair rewrites the draft and drops the promise: the stored reflection
    must not keep a promise Juniper never saw."""
    from orion.harness.finalize import run_harness_finalize_chain
    from orion.harness.runner import build_coalition_snapshot, build_draft_molecule

    thought = make_thought()
    molecule = build_draft_molecule(
        correlation_id="c-regate", thought=thought, draft_text=REPLY, grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=make_repair_overlay())
    judged = {**JUDGEMENT, "alignment_verdict": "misaligned",
              "promises_made": [{"quote": FIRM, "offer": False}]}

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(_req: object):
        return {"final_text": json.dumps(judged), "trace_id": "t"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("orion.harness.finalize.extract_response_repair_text",
                   lambda _result: "The read is still queued.")
        chain = await run_harness_finalize_chain(
            correlation_id="c-regate", draft_text=REPLY, draft_molecule=molecule, thought=thought,
            grammar_receipts=[], repair_overlay=make_repair_overlay(), user_message=USER,
            voice_contract=None, cortex_client=cortex_client, substrate_client=substrate_client)

    assert chain.response_repair_ran is True
    assert chain.final_text == "The read is still queued."
    assert chain.verdict_molecule.reflection.promises_made  # draft-gated copy, published first
    assert chain.reflection.promises_made == []  # what the run artifact stores


def test_flag_off_judge_is_not_asked_and_gate_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_PROMISE_LEDGER_ENABLED", "false")
    ctx = _ctx(pending_offers=[OFFER], open_promises=[OpenPromiseRefV1(id="a1", quote=FIRM)])
    assert not set(PROMISE_FIELDS) & set(ctx["structured_output_schema"]["properties"])
    assert ctx["messages"][0]["content"] == finalize_reflect_user_turn(False)
    assert "promises_made" not in ctx["messages"][0]["content"]
    rendered = _render(ctx)
    # The draft itself holds both quotes; the promise section and its lists are absent.
    assert "PROMISES (" not in rendered and f'- "{OFFER}"' not in rendered and "[a1]" not in rendered
    gated = gate_promises(_reflection(promises_made=[PromiseV1(quote=FIRM)]), reply_text=REPLY).reflection
    assert gated.promises_made == []


# --- acceptance check 5: open-promise text never reaches the motor prompt -----------------

def test_open_promise_text_is_in_the_judge_prompt_but_never_the_motor_prompt() -> None:
    quote = "I'll bring what surprised me next time we're in a technical thread"
    thought = make_thought()
    judge_prompt = _render(_ctx(open_promises=[OpenPromiseRefV1(id="act_1a2b", quote=quote)]))
    assert f'[act_1a2b] "{quote}"' in judge_prompt
    motor_prompt = compile_harness_prefix(thought, repair_overlay=make_repair_overlay(), user_message=USER)
    assert quote not in motor_prompt
    assert "promise" not in motor_prompt.lower()
    # Structural: the motor-prefix module reads no promise state at all.
    assert "promise" not in (REPO / "orion/harness/prefix.py").read_text(encoding="utf-8").lower()
