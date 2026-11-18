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
    PromiseKeptV1,
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
# A completion test, as the judge writes it (rule 0b keeps only items with one).
ACT = "bring the deep read back"
DONE_WHEN = "a later reply gives Juniper the digested deep read"

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
    "promise_conditions_met": [],
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
    reflection = _reflection(promises_made=[PromiseV1(force="commits", quote=paraphrase, doer="orion", done_when=DONE_WHEN), PromiseV1(force="commits", quote=FIRM, doer="orion", done_when=DONE_WHEN)])
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
    result = gate_promises(_reflection(promises_made=[PromiseV1(force="commits", quote="I'll do it.", doer="orion", done_when=DONE_WHEN)]),
                           reply_text="Sure. I'll do it.", enabled=True)
    assert result.reflection.promises_made == []
    assert result.dropped_too_short == 1 and result.dropped_not_in_text == []


def test_over_long_duplicate_and_over_cap_quotes_are_dropped() -> None:
    long_reply = "x" * 300
    reflection = _reflection(promises_made=[PromiseV1(force="commits", quote=long_reply, doer="orion", done_when=DONE_WHEN), PromiseV1(force="commits", quote=FIRM, doer="orion", done_when=DONE_WHEN),
                                            PromiseV1(force="commits", quote=" " + FIRM, doer="orion", done_when=DONE_WHEN)])
    result = gate_promises(reflection, reply_text=long_reply + " " + REPLY, enabled=True)
    assert [p.quote for p in result.reflection.promises_made] == [FIRM]
    assert result.dropped_too_long == 1


@pytest.mark.asyncio
async def test_judge_paraphrase_never_reaches_the_reflection() -> None:
    async def _judge(_req: object) -> dict[str, object]:
        return {"final_text": json.dumps({**JUDGEMENT, "promises_made": [
            {"quote": "I promise to finish the paper tomorrow.", "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN},
            {"quote": FIRM, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN}]})}

    reflection, quick, _ = await run_finalize_reflection(
        correlation_id="c-gate", draft_text=REPLY, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=USER, cortex_client=_judge)
    assert quick is False
    assert reflection.alignment_verdict == "aligned"
    assert [p.quote for p in reflection.promises_made] == [FIRM]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"promises_made": ["not an object", {"offer": True, "doer": "orion", "act": ACT, "done_when": DONE_WHEN}], "promises_kept": [7]},
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

    made = coerce_promise_fields({"promises_made": [{"quote": FIRM, "offer": "false", "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN}]})["promises_made"]
    assert made[0].offer is False


DELIVERY = "The read is still queued."  # a span of REPLY, standing in for "the words that deliver it"


def _kept(*ids: str, quote: str = DELIVERY) -> dict:
    return {"promises_kept": list(ids), "promises_kept_evidence": [PromiseKeptV1(id=i, quote=quote) for i in ids]}


def test_kept_entry_that_is_not_an_open_promise_is_dropped() -> None:
    reflection = _reflection(**_kept("act_open", "act_invented"))
    result = gate_promises(reflection, reply_text=REPLY, enabled=True,
                           open_promises=[OpenPromiseRefV1(id="act_open", quote=FIRM)])
    assert result.reflection.promises_kept == ["act_open"]
    assert [e.id for e in result.reflection.promises_kept_evidence] == ["act_open"]
    assert result.dropped_unknown_kept == ["act_invented"]


def test_regate_against_final_text_drops_a_promise_repair_removed() -> None:
    """The chain gates again on the text Juniper saw; accepted offers survive, and a kept id survives
    only while its delivering quote is still in the final text."""
    reflection = _reflection(promises_made=[PromiseV1(force="commits", quote=FIRM, doer="orion", done_when=DONE_WHEN)], offers_accepted=[OFFER],
                             **_kept("act_open"))
    repaired = "The read is still queued."
    result = gate_promises(reflection, reply_text=repaired, enabled=True,
                           pending_offers=reflection.offers_accepted, open_promises=reflection.promises_kept)
    assert result.reflection.promises_made == []
    assert result.reflection.offers_accepted == [OFFER]
    assert result.reflection.promises_kept == ["act_open"]
    gone = gate_promises(reflection, reply_text="Nothing new yet.", enabled=True,
                         pending_offers=reflection.offers_accepted, open_promises=reflection.promises_kept)
    assert gone.reflection.promises_kept == [] and gone.dropped_kept_not_in_text == ["act_open"]


# --- amendment 5: a "kept" mark must quote the words that deliver it ----------------------

def test_kept_mark_without_a_quote_or_with_an_invented_quote_is_dropped() -> None:
    open_refs = [OpenPromiseRefV1(id="act_a", quote=FIRM), OpenPromiseRefV1(id="act_b", quote=OFFER),
                 OpenPromiseRefV1(id="act_c", quote=FIRM)]
    reflection = _reflection(
        promises_kept=["act_a", "act_b", "act_c"],
        promises_kept_evidence=[PromiseKeptV1(id="act_a", quote=DELIVERY),
                                PromiseKeptV1(id="act_b", quote="Here is the full deep read you asked for.")])
    result = gate_promises(reflection, reply_text=REPLY, enabled=True, open_promises=open_refs)
    assert result.reflection.promises_kept == ["act_a"]
    assert result.dropped_kept_not_in_text == ["act_b", "act_c"]  # invented quote; no quote at all


def test_kept_mark_that_only_repeats_the_promise_is_dropped() -> None:
    """Re-promising is not keeping: a delivering quote inside the promise's own words is refused."""
    reflection = _reflection(promises_kept=["act_a"],
                             promises_kept_evidence=[PromiseKeptV1(id="act_a", quote="I'll hold the deep read")])
    result = gate_promises(reflection, reply_text=REPLY, enabled=True,
                           open_promises=[OpenPromiseRefV1(id="act_a", quote=FIRM)])
    assert result.reflection.promises_kept == [] and result.dropped_kept_repeats_promise == ["act_a"]


def test_judge_kept_objects_parse_and_a_bare_id_fails_closed() -> None:
    from orion.harness.promise_gate import coerce_promise_fields

    got = coerce_promise_fields({"promises_kept": [{"id": "act_a", "quote": DELIVERY}, "act_bare", {"quote": "x"}, 7]})
    assert got["promises_kept"] == ["act_a", "act_bare"]
    assert [(e.id, e.quote) for e in got["promises_kept_evidence"]] == [("act_a", DELIVERY)]
    reflection = _reflection(**got)
    result = gate_promises(reflection, reply_text=REPLY, enabled=True,
                           open_promises=[OpenPromiseRefV1(id="act_a", quote=FIRM),
                                          OpenPromiseRefV1(id="act_bare", quote=FIRM)])
    assert result.reflection.promises_kept == ["act_a"] and result.dropped_kept_not_in_text == ["act_bare"]


def test_schema_asks_for_an_id_and_a_quote_per_kept_promise() -> None:
    kept = finalize_reflect_json_schema(True)["properties"]["promises_kept"]
    assert kept["items"]["required"] == ["id", "quote"]
    jsonschema.validate([{"id": "act_a", "quote": DELIVERY}], kept)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(["act_a"], kept)
    text = _render(_ctx(open_promises=[OpenPromiseRefV1(id="act_a", quote=FIRM)]))
    assert '"quote": the exact words of' in text and "[act_a]" in text


def test_judge_view_holds_more_open_promises_than_the_made_cap() -> None:
    """Amendment 8: the judge sees up to PROMISE_JUDGE_VIEW_MAX open promises, so a 4th open
    promise is not structurally un-closeable."""
    from orion.schemas.harness_finalize import PROMISE_ITEMS_MAX, PROMISE_JUDGE_VIEW_MAX

    assert PROMISE_JUDGE_VIEW_MAX > PROMISE_ITEMS_MAX
    refs = [OpenPromiseRefV1(id=f"act_{i}", quote=FIRM) for i in range(PROMISE_JUDGE_VIEW_MAX)]
    text = _render(_ctx(open_promises=refs))
    assert all(f"[act_{i}]" in text for i in range(PROMISE_JUDGE_VIEW_MAX))


# --- doer: only Orion's own future act is a promise (2026-11-14 exclusions) ----------------
# The D0 replay listed forecasts about a system and asks for Juniper to act as promises
# (5 of 8 false items). The judge now says who does the act; plain code keeps only Orion.

DOER_REPLY = (
    "The backfill is third in line, so it should finish overnight. Send me the log when it "
    "finishes. I'll compare the two runs tomorrow and tell you which one drifted."
)
FORECAST = "so it should finish overnight"
ASK = "Send me the log when it finishes."
REAL = "I'll compare the two runs tomorrow and tell you which one drifted."


@pytest.mark.asyncio
async def test_forecast_and_ask_are_dropped_and_the_real_promise_is_kept() -> None:
    async def _judge(_req: object) -> dict[str, object]:
        return {"final_text": json.dumps({**JUDGEMENT, "promises_made": [
            {"quote": FORECAST, "offer": False, "force": "commits", "doer": "a_system", "act": ACT, "done_when": DONE_WHEN},
            {"quote": ASK, "offer": True, "force": "commits", "doer": "juniper", "act": ACT, "done_when": DONE_WHEN},
            {"quote": REAL, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN}]})}

    reflection, _, _ = await run_finalize_reflection(
        correlation_id="c-doer", draft_text=DOER_REPLY, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=USER, cortex_client=_judge)
    assert [(p.quote, p.doer) for p in reflection.promises_made] == [(REAL, "orion")]
    assert counted_promise_quotes(reflection) == [REAL]


def test_gate_reports_what_it_dropped_for_the_doer() -> None:
    reflection = _reflection(promises_made=[
        PromiseV1(force="commits", quote=FORECAST, doer="a_system", done_when=DONE_WHEN), PromiseV1(force="commits", quote=ASK, offer=True, doer="juniper", done_when=DONE_WHEN),
        PromiseV1(force="commits", quote=REAL, doer="someone_else", done_when=DONE_WHEN), PromiseV1(force="commits", quote=REAL, doer="orion", done_when=DONE_WHEN)])
    result = gate_promises(reflection, reply_text=DOER_REPLY, enabled=True)
    assert [p.quote for p in result.reflection.promises_made] == [REAL]
    assert result.dropped_not_orion == [("a_system", FORECAST), ("juniper", ASK), ("someone_else", REAL)]
    assert result.dropped_total == 3


@pytest.mark.parametrize("doer", [None, "Orion", "me", 3])
def test_missing_or_unknown_doer_fails_closed(doer: object) -> None:
    from orion.harness.promise_gate import coerce_promise_fields

    item: dict[str, object] = {"quote": REAL, "offer": False}
    if doer is not None:
        item["doer"] = doer
    made = coerce_promise_fields({"promises_made": [item]})["promises_made"]
    assert made[0].doer is None
    result = gate_promises(_reflection(promises_made=made), reply_text=DOER_REPLY, enabled=True)
    assert result.reflection.promises_made == [] and result.dropped_not_orion == [(None, REAL)]


def test_non_orion_items_do_not_use_up_the_cap() -> None:
    """Dropped before the cap: three forecasts listed first must not crowd out the promise."""
    reflection = _reflection(promises_made=[PromiseV1(force="commits", quote=FORECAST, doer="a_system", done_when=DONE_WHEN)] * 3
                             + [PromiseV1(force="commits", quote=REAL, doer="orion", done_when=DONE_WHEN)])
    result = gate_promises(reflection, reply_text=DOER_REPLY, enabled=True)
    assert [p.quote for p in result.reflection.promises_made] == [REAL]
    assert result.dropped_over_cap == 0


def test_schema_requires_a_doer_from_the_closed_set_and_the_prompt_defines_each() -> None:
    schema = finalize_reflect_json_schema(True)
    item = schema["properties"]["promises_made"]["items"]
    assert "doer" in item["required"]
    assert set(item["properties"]["doer"]["enum"]) == {"orion", "juniper", "someone_else", "a_system"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": REAL, "offer": False}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": REAL, "offer": False, "force": "commits", "doer": "me",
                                                             "act": ACT, "done_when": DONE_WHEN}]},
                            schema)
    rendered = _render(_ctx())
    section = rendered.split("PROMISES (", 1)[1].split("REQUIRED JSON FIELDS", 1)[0]
    for doer in item["properties"]["doer"]["enum"]:
        assert f'- "{doer}":' in section
    assert 'Plain code keeps only items with doer "orion".' in section


# --- rule 0b: a promise has a completion that could fail ---------------------------------

PRESENCE_REPLY = (
    "Good luck with the move. I'm around all evening if anything comes up. "
    "I'll compare the two runs tomorrow and tell you which one drifted. "
    "Send me the log when it finishes, so it should finish overnight."
)
PRESENCE = "I'm around all evening if anything comes up."


@pytest.mark.asyncio
async def test_presence_line_is_dropped_real_promise_kept_forecast_and_ask_still_dropped() -> None:
    """Orion is the subject of the presence line, so the doer rule keeps it; the judge finds
    no completion that could fail (done_when null) and rule 0b drops it."""
    async def _judge(_req: object) -> dict[str, object]:
        return {"final_text": json.dumps({**JUDGEMENT, "promises_made": [
            {"quote": PRESENCE, "offer": False, "force": "commits", "doer": "orion", "act": "stay available", "done_when": None},
            {"quote": REAL, "offer": False, "force": "commits", "doer": "orion", "act": "compare the two runs",
             "done_when": "a reply names which of the two runs drifted"},
            {"quote": ASK, "offer": False, "force": "commits", "doer": "juniper", "act": "send the log", "done_when": "the log arrives"},
            {"quote": FORECAST, "offer": False, "force": "commits", "doer": "a_system", "act": "finish", "done_when": None}]})}

    reflection, _, _ = await run_finalize_reflection(
        correlation_id="c-presence", draft_text=PRESENCE_REPLY, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=USER, cortex_client=_judge)
    assert [p.quote for p in reflection.promises_made] == [REAL]
    kept = reflection.promises_made[0]
    assert (kept.act, kept.done_when) == ("compare the two runs", "a reply names which of the two runs drifted")
    assert counted_promise_quotes(reflection) == [REAL]


def test_act_and_done_when_are_cut_to_the_quote_cap() -> None:
    from orion.harness.promise_gate import coerce_promise_fields
    from orion.schemas.harness_finalize import PROMISE_QUOTE_MAX_CHARS

    made = coerce_promise_fields({"promises_made": [{"quote": REAL, "offer": False, "force": "commits",
                                                     "doer": "orion", "act": "a" * 900, "done_when": "d" * 900}]})
    item = made["promises_made"][0]
    assert len(item.act) == len(item.done_when) == PROMISE_QUOTE_MAX_CHARS


@pytest.mark.parametrize("done_when", [None, "", "   ", "n/a", "none", 3])
def test_missing_or_placeholder_completion_fails_closed(done_when: object) -> None:
    from orion.harness.promise_gate import coerce_promise_fields

    item: dict[str, object] = {"quote": REAL, "offer": False, "force": "commits", "doer": "orion", "act": "compare"}
    if done_when is not None:
        item["done_when"] = done_when
    made = coerce_promise_fields({"promises_made": [item]})["promises_made"]
    result = gate_promises(_reflection(promises_made=made), reply_text=DOER_REPLY, enabled=True)
    assert result.reflection.promises_made == [] and result.dropped_no_completion == [REAL]
    assert result.dropped_total == 1


def test_presence_lines_do_not_use_up_the_cap() -> None:
    reflection = _reflection(promises_made=[PromiseV1(force="commits", quote=PRESENCE, doer="orion")] * 3
                             + [PromiseV1(force="commits", quote=REAL, doer="orion", done_when=DONE_WHEN)])
    result = gate_promises(reflection, reply_text=PRESENCE_REPLY, enabled=True)
    assert [p.quote for p in result.reflection.promises_made] == [REAL]
    assert result.dropped_over_cap == 0 and len(result.dropped_no_completion) == 3


def test_schema_requires_act_and_done_when_and_the_prompt_defines_them() -> None:
    schema = finalize_reflect_json_schema(True)
    item = schema["properties"]["promises_made"]["items"]
    assert {"act", "done_when"} <= set(item["required"])
    base = {"quote": REAL, "offer": False, "force": "commits", "doer": "orion", "act": "compare",
            "settles_in": "chat", "condition": None}
    jsonschema.validate({**JUDGEMENT, "promises_made": [{**base, "done_when": None}]}, schema)
    jsonschema.validate({**JUDGEMENT, "promises_made": [{**base, "done_when": DONE_WHEN, "settles_in": "chat", "condition": None}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**JUDGEMENT, "promises_made": [base]}, schema)
    section = _render(_ctx()).split("PROMISES (", 1)[1].split("REQUIRED JSON FIELDS", 1)[0]
    assert "- act:" in section and "- done_when:" in section
    assert "Plain code\n  keeps only items with a done_when." in section


# --- rule 0a: the line must commit -----------------------------------------------------------

SPEECH_REPLY = (
    "I filed the trace in the queue. Best to try the small model first. I'd enjoy trying "
    "the camera sometime. I'll compare the two runs tomorrow and tell you which one drifted."
)


@pytest.mark.asyncio
async def test_report_suggestion_and_wish_are_dropped_and_the_commitment_kept() -> None:
    """All four lines have Orion as the doer; only the one that binds Orion survives."""
    def _item(quote: str, force: str) -> dict[str, object]:
        return {"quote": quote, "offer": False, "force": force, "doer": "orion", "act": "x",
                "done_when": "a later reply shows it done"}

    async def _judge(_req: object) -> dict[str, object]:
        return {"final_text": json.dumps({**JUDGEMENT, "promises_made": [
            _item("I filed the trace in the queue.", "reports"),
            _item("Best to try the small model first.", "suggests"),
            _item("I'd enjoy trying the camera sometime.", "wishes"),
            _item(REAL, "commits")]})}

    reflection, _, _ = await run_finalize_reflection(
        correlation_id="c-force", draft_text=SPEECH_REPLY, thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5), repair_overlay=make_repair_overlay(),
        user_message=USER, cortex_client=_judge)
    assert [(p.quote, p.force) for p in reflection.promises_made] == [(REAL, "commits")]


@pytest.mark.parametrize("force", [None, "promise", "Commits", 1])
def test_missing_or_unknown_force_fails_closed(force: object) -> None:
    from orion.harness.promise_gate import coerce_promise_fields

    item: dict[str, object] = {"quote": REAL, "offer": False, "doer": "orion", "done_when": DONE_WHEN}
    if force is not None:
        item["force"] = force
    made = coerce_promise_fields({"promises_made": [item]})["promises_made"]
    assert made[0].force is None
    result = gate_promises(_reflection(promises_made=made), reply_text=DOER_REPLY, enabled=True)
    assert result.reflection.promises_made == [] and result.dropped_not_commit == [(None, REAL)]


def test_schema_requires_a_force_from_the_closed_set_and_the_prompt_defines_each() -> None:
    schema = finalize_reflect_json_schema(True)
    item = schema["properties"]["promises_made"]["items"]
    assert "force" in item["required"]
    assert set(item["properties"]["force"]["enum"]) == {"commits", "reports", "predicts", "suggests", "wishes"}
    good = {"quote": REAL, "offer": False, "doer": "orion", "act": ACT, "done_when": DONE_WHEN,
            "settles_in": "chat", "condition": None}
    jsonschema.validate({**JUDGEMENT, "promises_made": [{**good, "force": "commits"}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**JUDGEMENT, "promises_made": [{**good, "force": "promise"}]}, schema)
    section = _render(_ctx()).split("PROMISES (", 1)[1].split("REQUIRED JSON FIELDS", 1)[0]
    for force in item["properties"]["force"]["enum"]:
        assert f'- "{force}":' in section
    assert 'Plain code keeps only items with force "commits".' in section


# --- the offer rule -----------------------------------------------------------------------

def test_offer_is_recorded_but_counts_only_after_acceptance() -> None:
    reflection = _reflection(promises_made=[PromiseV1(force="commits", quote=FIRM, doer="orion", done_when=DONE_WHEN), PromiseV1(force="commits", quote=OFFER, offer=True, doer="orion", done_when=DONE_WHEN)])
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
    assert list(schema["properties"])[-len(PROMISE_FIELDS):] == list(PROMISE_FIELDS)
    jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": FIRM, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN, "settles_in": "chat", "condition": None}]}, schema)
    # No caps in the schema: Strata 502s an answer that breaks it, which would sink the
    # verdict. Four promises or a long quote must pass the schema; the gate caps them.
    jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": FIRM, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN, "settles_in": "chat", "condition": None}] * 4}, schema)
    jsonschema.validate({**JUDGEMENT, "promises_made": [{"quote": "y" * 241, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN, "settles_in": "chat", "condition": None}]}, schema)
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
              "promises_made": [{"quote": FIRM, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN}]}

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
    gated = gate_promises(_reflection(promises_made=[PromiseV1(force="commits", quote=FIRM, doer="orion", done_when=DONE_WHEN)]), reply_text=REPLY).reflection
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


@pytest.mark.parametrize("judge_copy, reply", [
    # Replay turn 868ed31f (2026-11-16): markdown emphasis in the reply, none in the judge's copy.
    ("that's ledger entry #3 for free: first public identity record — under my name.",
     "If so, that's ledger entry #3 for free: *first public identity record — under my name.* One question."),
    # Apostrophe forms a model or renderer can produce: modifier letter, fullwidth, curly.
    ("Iʼll compare the two runs tomorrow.", "Sure. I'll compare the two runs tomorrow."),
    ("I＇ll compare the two runs tomorrow.", "Sure. I’ll compare the two runs tomorrow."),
    # A zero-width space and a non-breaking space inside the copy.
    ("I'll compare​ the two runs tomorrow.", "Sure. I'll compare the two runs tomorrow."),
    # A zero-width space standing in for a real one.
    ("I'll compare\u200bthe two runs tomorrow.", "Sure. I'll compare the two runs tomorrow."),
])
def test_quote_match_folds_emphasis_apostrophes_and_invisible_spacing(judge_copy, reply):
    """Orion's PR #3532 second-round question 5: a promise must not be refused for a copy artifact."""
    from orion.harness.promise_gate import quote_in_text

    assert quote_in_text(judge_copy, reply)
    assert not quote_in_text("I'll compare the three runs tomorrow.", reply)


# --- 2026-11-18: long quotes (#3530), settle classes and conditions (#3595) ----------------------------

# Faithful to replay turn d05f08e3 (11-16 replay, dev2.part5): the judge quoted the whole 2-clause
# sentence (backticks, an em dash, ~250 chars), and the 240 cap dropped the hand-found hold-deep-read.
LONG_PROMISE = (
    "I'll hold the deep read until the pipeline digests it — then pull it back through `reading_results` "
    "and we can see what it actually says about feasibility on hardware like circe's V100s before we get "
    "excited about applying it to my own cortex."
)
LONG_REPLY = "Queued it.\n\n" + LONG_PROMISE.replace("hardware like", "hardware  like") + " Sleep well."


def test_a_long_literal_quote_is_kept_as_its_leading_part_and_counted_trimmed() -> None:
    from orion.harness.promise_gate import leading_quote, normalize_for_quote_match
    from orion.schemas.harness_finalize import PROMISE_QUOTE_MAX_CHARS

    assert len(LONG_PROMISE) > PROMISE_QUOTE_MAX_CHARS
    item = PromiseV1(force="commits", quote=LONG_PROMISE, doer="orion", done_when=DONE_WHEN)
    result = gate_promises(_reflection(promises_made=[item]), reply_text=LONG_REPLY, enabled=True)
    [kept] = result.reflection.promises_made
    assert result.trimmed_long == 1 and result.dropped_too_long == 0
    assert len(kept.quote) <= PROMISE_QUOTE_MAX_CHARS
    assert LONG_PROMISE.startswith(kept.quote) and kept.quote == leading_quote(LONG_PROMISE)
    assert not kept.quote.endswith((" ", ",", "-", "—"))           # cut at a word boundary, tail tidied
    assert normalize_for_quote_match(kept.quote) in normalize_for_quote_match(LONG_REPLY)
    # The second pass (against the text Juniper saw) keeps it unchanged: it is within the cap now.
    again = gate_promises(result.reflection, reply_text=LONG_REPLY, enabled=True)
    assert [p.quote for p in again.reflection.promises_made] == [kept.quote] and again.trimmed_long == 0


def test_a_long_quote_that_is_not_literal_is_still_dropped_too_long() -> None:
    invented = LONG_PROMISE.replace("hold the deep read", "keep the deep read")
    item = PromiseV1(force="commits", quote=invented, doer="orion", done_when=DONE_WHEN)
    result = gate_promises(_reflection(promises_made=[item]), reply_text=LONG_REPLY, enabled=True)
    assert result.reflection.promises_made == []
    assert result.dropped_too_long == 1 and result.trimmed_long == 0


def test_leading_quote_edges() -> None:
    from orion.harness.promise_gate import leading_quote

    assert leading_quote("short and fine") == "short and fine"
    assert leading_quote("x" * 300) is None                      # no word boundary under the cap
    assert leading_quote("a " + "x" * 300) is None                # the only cut leaves a fragment
    words = ("word " * 60).strip()                                # 299 chars
    cut = leading_quote(words, cap=240)
    assert cut is not None and len(cut) <= 240 and words.startswith(cut) and cut.endswith("word")


def test_settles_in_and_condition_parse_leniently() -> None:
    from orion.harness.promise_gate import coerce_promise_fields

    base = {"quote": FIRM, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN}
    parsed = coerce_promise_fields({"promises_made": [
        {**base, "settles_in": "run", "condition": "  the three proposals appear "},
        {**base, "settles_in": "someday", "condition": 7},
        base,
    ], "promise_conditions_met": ["act-1", 3, " ", "act-2"]})
    made = parsed["promises_made"]
    assert [(p.settles_in, p.condition) for p in made] == [
        ("run", "the three proposals appear"), (None, None), (None, None)]
    assert parsed["promise_conditions_met"] == ["act-1", "act-2"]


def test_only_shown_ids_survive_as_conditions_met_and_the_final_pass_keeps_them() -> None:
    shown = [OpenPromiseRefV1(id="act-cond", quote="I'll read the proposals.", condition="the proposals appear"),
             OpenPromiseRefV1(id="act-plain", quote="I'll send the list.")]
    reflection = _reflection(promise_conditions_met=["act-cond", "act-cond", "act-unknown"])
    first = gate_promises(reflection, reply_text=REPLY, open_promises=shown, enabled=True)
    assert first.reflection.promise_conditions_met == ["act-cond"]
    assert first.dropped_unknown_condition == ["act-unknown"]
    # The chain's second pass gives only ids (kept + conditions met), as run_harness_finalize_chain does.
    r = first.reflection
    second = gate_promises(r, reply_text="different final text", enabled=True,
                           open_promises=[*r.promises_kept, *r.promise_conditions_met])
    assert second.reflection.promise_conditions_met == ["act-cond"]


def test_flag_off_clears_conditions_met() -> None:
    gated = gate_promises(_reflection(promise_conditions_met=["act-cond"]), reply_text=REPLY, enabled=False)
    assert gated.reflection.promise_conditions_met == []


def test_schema_asks_for_settles_in_condition_and_conditions_met_and_the_prompt_defines_them() -> None:
    schema = finalize_reflect_json_schema(True)
    item = schema["properties"]["promises_made"]["items"]
    assert {"settles_in", "condition"} <= set(item["required"])
    assert set(item["properties"]["settles_in"]["enum"]) == {"chat", "run", "absence"}
    good = {"quote": FIRM, "offer": False, "force": "commits", "doer": "orion", "act": ACT, "done_when": DONE_WHEN,
            "settles_in": "run", "condition": "the run lands"}
    jsonschema.validate({**JUDGEMENT, "promises_made": [good]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**JUDGEMENT, "promises_made": [{**good, "settles_in": "later"}]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({k: v for k, v in JUDGEMENT.items() if k != "promise_conditions_met"}, schema)
    section = _render(_ctx()).split("PROMISES (", 1)[1].split("REQUIRED JSON FIELDS", 1)[0]
    assert "- settles_in:" in section and "- condition:" in section
    for cls in ("chat", "run", "absence"):
        assert f'- "{cls}":' in section
    assert "- promise_conditions_met: [] (no promises are open)." in section


def test_the_judge_sees_each_open_promises_condition() -> None:
    open_refs = [OpenPromiseRefV1(id="act-cond", quote="I'll read the proposals.", condition="the proposals appear"),
                 OpenPromiseRefV1(id="act-plain", quote="I'll send the list.")]
    section = _render(_ctx(open_promises=open_refs)).split("PROMISES (", 1)[1].split("REQUIRED JSON FIELDS", 1)[0]
    assert '[act-cond] "I\'ll read the proposals." (waits on: the proposals appear)\n' in section
    assert '[act-plain] "I\'ll send the list."\n' in section
    assert "- promise_conditions_met: for each open promise above marked \"waits on\"" in section
