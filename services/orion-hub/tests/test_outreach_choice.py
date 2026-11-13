"""The outreach choice point (scripts/outreach_choice.py, personality spec rev 4 move 1).

The background tick shows Orion its grounded topics plus "say nothing" in the SAME compose call,
Orion picks one or declines, and every readable pick publishes one PersonalityChoiceV1. A subject
Juniper already got (or Orion already declined) in the last 7 days is not shown again unless it
changed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from orion.schemas.personality import PERSONALITY_CHOICE_CHANNEL, PERSONALITY_CHOICE_KIND, PersonalityChoiceV1
from scripts.endogenous_outreach import (
    EndogenousOutreach,
    OutreachContext,
    build_outreach_prompt,
    is_pass_response,
)
from scripts.outreach_choice import (
    INVALID_PICK_REASON,
    NO_PICK_REASON,
    ORION_DECLINED_REASON,
    OutreachTopic,
    build_choice,
    choice_record,
    filter_unsaid,
    fold_said_subjects,
    parse_pick,
    rank_topics,
    version_of,
)
from scripts.outreach_door_policy import daydream_content_id
from scripts.tension_outreach_trigger import TensionTriggerReason

NOW = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)


def _topic(subject: str, *, kind: str = "reading", minutes_ago: float = 10.0, version: str = "",
           **kw: Any) -> OutreachTopic:
    return OutreachTopic(subject=subject, kind=kind, version=version,
                         observed_at=NOW - timedelta(minutes=minutes_ago),
                         option_text=f"option {subject}", prompt_text=f"prompt {subject}", **kw)


# --- pure: parse_pick ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,kind,index,body", [
    ("Pick: 2\nThe second thing.\nWhy Juniper: x", "chosen", 1, "The second thing.\nWhy Juniper: x"),
    ("**Pick:** 1\nFirst.", "chosen", 0, "First."),
    ("Pick: [3]\nThird.", "chosen", 2, "Third."),
    ("pick 2.\nSecond.", "chosen", 1, "Second."),
    ("Pick: option 1\nFirst.", "chosen", 0, "First."),
    ("Pick: 0", "declined", None, ""),
    ("Pick: none", "declined", None, ""),
    ("PASS", "declined", None, "PASS"),
    ("Pick: 2\nThinking.\nPASS", "declined", None, "Thinking.\nPASS"),
    ("Pick: 7\nSeventh.", "invalid", None, "Seventh."),
    ("A message with no pick line.", "missing", None, "A message with no pick line."),
    # Review 2026-11-13: realistic variants.
    ("Pick: say nothing", "declined", None, ""),
    ("Pick: zero", "declined", None, ""),
    ("Pick \u2014 2\nSecond.", "chosen", 1, "Second."),
    ("My pick: 2\nSecond.", "chosen", 1, "Second."),
    ("Pick 2\nSecond.", "chosen", 1, "Second."),
    ("Thinking about it.\nPick: 3\nThird.", "chosen", 2, "Thinking about it.\nThird."),
    ("Pick 3 things matter here, and the read is one.", "missing", None,
     "Pick 3 things matter here, and the read is one."),
])
def test_parse_pick_three_options(raw, kind, index, body) -> None:
    pick = parse_pick(raw, 3, is_pass=is_pass_response)
    assert (pick.kind, pick.index, pick.body) == (kind, index, body)


def test_parse_pick_one_option_without_pick_line_is_an_implicit_pick() -> None:
    """One topic shown: speaking at all is the pick (speak vs say nothing is still a choice)."""
    pick = parse_pick("The daydream.\nWhy Juniper: x", 1, is_pass=is_pass_response)
    assert (pick.kind, pick.index, pick.shape) == ("chosen", 0, "implicit")


@pytest.mark.parametrize("raw", ["Pick: two\nThe daydream.", "My pick: maybe\nThe daydream.",
                                 "Pick: say nothing", "Pick \u2014 0"])
def test_one_topic_a_broken_or_declining_pick_line_is_never_an_implicit_pick(raw) -> None:
    """Review 2026-11-13: with one topic, an unparsed Pick line used to fall through to an implicit
    pick, so "Pick: say nothing" published a chosen option and "My pick: ..." reached Juniper."""
    pick = parse_pick(raw, 1, is_pass=is_pass_response)
    assert pick.kind in ("declined", "missing")


def test_parse_pick_words_in_a_sentence_are_not_a_pick_line() -> None:
    pick = parse_pick("I would pick the second one if I had to.", 2, is_pass=is_pass_response)
    assert pick.kind == "missing"


# --- pure: subject ledger, ranking ------------------------------------------------------------


def test_filter_unsaid_drops_said_subjects_and_keeps_a_new_version() -> None:
    v1, v2 = version_of(["belief:a"]), version_of(["belief:a", "belief:b"])
    said = {"reading:s1": {v1}, "daydream:abc": {""}}
    topics = [_topic("reading:s1", version=v1), _topic("reading:s1b", version=v1),
              _topic("daydream:abc", kind="daydream"), _topic("reading:s1", version=v2)]
    kept, dropped = filter_unsaid(topics, said)
    assert [(t.subject, t.version) for t in kept] == [("reading:s1b", v1), ("reading:s1", v2)]
    assert dropped == ["reading:s1", "daydream:abc"]


def test_version_is_order_free() -> None:
    assert version_of(["b", "a"]) == version_of(["a", "b", "a"]) != version_of(["a"])
    assert version_of([]) == ""


def test_rank_topics_freshest_first_one_per_subject_capped() -> None:
    topics = [_topic("a", minutes_ago=50), _topic("b", minutes_ago=5), _topic("a", minutes_ago=1),
              OutreachTopic(subject="c", kind="reading", option_text="c", prompt_text="c"),
              _topic("d", minutes_ago=20), _topic("e", minutes_ago=30)]
    assert [t.subject for t in rank_topics(topics, 4)] == ["b", "d", "e", "a"]
    assert len(rank_topics(topics, 99)) == 5  # hard ceiling: at most 5 shown
    assert len(rank_topics(topics, 0)) == 1


def test_fold_said_subjects_reads_choice_rows_and_legacy_daydream_rows() -> None:
    rows = [
        {"reason": "sent", "choice": {"chosen_subject": "reading:s1", "chosen_version": "v1",
                                      "shown": [{"subject": "reading:s1"}, {"subject": "daydream:d1"}]}},
        {"reason": ORION_DECLINED_REASON, "choice": {"chosen_subject": None,
                                                     "shown": [{"subject": "daydream:d2", "version": ""},
                                                               {"subject": "reading:s3", "version": "v3"}]}},
        {"reason": "no_why_juniper", "choice": {"chosen_subject": "daydream:d4", "chosen_version": ""}},
        {"reason": "sent", "choice": None, "daydream_id": "daydream:legacy"},
        {"reason": "juniper_spoke_after_generation", "choice": {"chosen_subject": "reading:x"}},
        # A reply we could not read as a pick: its shown subjects are decided too (no recompose loop).
        {"reason": NO_PICK_REASON, "choice": {"shown": [{"subject": "reading:np", "version": "v"}]}},
        "garbage",
        {"reason": "sent", "choice": {"shown": ["not a dict"]}},
    ]
    said = fold_said_subjects(rows)
    assert said == {"reading:s1": {"v1"}, "daydream:d2": {""}, "reading:s3": {"v3"},
                    "daydream:d4": {""}, "daydream:legacy": {""}, "reading:np": {"v"}}
    # The daydream shown beside a sent pick was not chosen: it stays free.
    assert "daydream:d1" not in said


# --- pure: the PersonalityChoiceV1 ---------------------------------------------------------------


def test_build_choice_chosen_keeps_ranks_and_the_others_as_alternatives() -> None:
    topics = [_topic("daydream:d", kind="daydream"), _topic("reading:s1"), _topic("reading:s2")]
    pick = parse_pick("Pick: 2\nmsg", 3, is_pass=is_pass_response)
    choice = build_choice(decision_key="want-1", topics=topics, pick=pick, occurred_at=NOW,
                          situation={"trigger": "tension", "nested": {"no": 1}}, act_id="want-1",
                          correlation_id="gen-1")
    assert isinstance(choice, PersonalityChoiceV1)
    assert choice.choice_id == "outreach:want-1" and choice.lane == "outreach"
    assert choice.chosen.key == "reading:s1" and choice.chosen_market_rank == 2
    assert [(a.key, a.market_rank) for a in choice.alternatives] == [("daydream:d", 1), ("reading:s2", 3)]
    assert choice.declined_all is False and choice.picks_in_decision == 1
    assert choice.situation["trigger"] == "tension" and isinstance(choice.situation["nested"], str)
    assert choice.act_id == "want-1" and choice.correlation_id == "gen-1"


def test_build_choice_declined_all_lists_every_shown_option() -> None:
    topics = [_topic("daydream:d", kind="daydream"), _topic("reading:s1")]
    choice = build_choice(decision_key="w", topics=topics, pick=parse_pick("Pick: 0", 2, is_pass=is_pass_response),
                          occurred_at=NOW, situation={}, act_id="w", correlation_id="w")
    assert choice.declined_all and choice.chosen is None
    assert [a.key for a in choice.alternatives] == ["daydream:d", "reading:s1"]


def test_build_choice_one_topic_picked_turns_down_say_nothing() -> None:
    topics = [_topic("daydream:d", kind="daydream")]
    choice = build_choice(decision_key="w", topics=topics, pick=parse_pick("msg", 1, is_pass=is_pass_response),
                          occurred_at=NOW, situation={}, act_id=None, correlation_id=None)
    assert choice.chosen.key == "daydream:d"
    assert [a.key for a in choice.alternatives] == ["say_nothing"]


def test_build_choice_none_when_no_readable_pick() -> None:
    topics = [_topic("a"), _topic("b")]
    for raw in ("no pick line", "Pick: 9\nx"):
        assert build_choice(decision_key="w", topics=topics, pick=parse_pick(raw, 2, is_pass=is_pass_response),
                            occurred_at=NOW, situation={}, act_id="w", correlation_id="w") is None


def test_choice_record_names_shown_subjects_and_the_pick() -> None:
    topics = [_topic("daydream:d", kind="daydream"), _topic("reading:s1", version="v")]
    rec = choice_record(topics, parse_pick("Pick: 2\nx", 2, is_pass=is_pass_response), "outreach:w")
    assert rec["chosen_subject"] == "reading:s1" and rec["chosen_version"] == "v" and rec["chosen_rank"] == 2
    assert [s["subject"] for s in rec["shown"]] == ["daydream:d", "reading:s1"]


# --- prompt ---------------------------------------------------------------------------------------


def test_choice_prompt_numbers_topics_offers_say_nothing_and_drops_the_cue() -> None:
    ctx = OutreachContext(curiosity_summaries=[], recent_turns=[], presence=None,
                          daydream=(600.0, "A brass orrery turning slowly over a dark desk."),
                          personality_cue='- "Wrote the answer into the graph first, then looked."')
    topics = [_topic("daydream:d", kind="daydream"), _topic("reading:s1")]
    prompt = build_outreach_prompt(ctx, topics=topics)
    assert "1. prompt daydream:d" in prompt and "2. prompt reading:s1" in prompt
    assert "0. Say nothing to her this time." in prompt
    assert 'Start your reply with one line: "Pick: <number>"' in prompt
    assert "Why Juniper:" in prompt
    # The cue rode in as free material with no subject: that is the 3x repeat. Not in choice mode.
    assert "graph first" not in prompt
    # The daydream is an option, not a separate block.
    assert "brass orrery" not in prompt
    # Without topics the one-topic prompt is unchanged (cue and daydream render as before).
    old = build_outreach_prompt(ctx)
    assert "graph first" in old and "brass orrery" in old and "Pick:" not in old


# --- runtime --------------------------------------------------------------------------------------

DAYDREAM = (600.0, "A brass orrery turning slowly over a dark desk, one planet missing.")


def _fires() -> TensionTriggerReason:
    return TensionTriggerReason(target_id="node:test", run_length=99, peak_deviation_pressure=1.0)


class _Bus:
    def __init__(self) -> None:
        self.published: List[tuple] = []

    async def publish(self, channel: str, env: Any) -> None:
        self.published.append((channel, env))


def _choice_outreach(**overrides: Any) -> EndogenousOutreach:
    base = dict(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=40,
                quiet_start_hour=-1, quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0,
                notify_channel="orion:notify:in_app", fallback_session_id="orion_outreach",
                trigger_evaluator=_fires, choice_enabled=True, choice_max_options=4)
    base.update(overrides)
    o = EndogenousOutreach(**base)
    o._bus = _Bus()
    return o


class _Ledger:
    """The decision log, in memory: every recorded row, folded back the way Postgres would."""

    def __init__(self, monkeypatch) -> None:
        import scripts.endogenous_outreach_decisions as decisions

        self.rows: List[Dict[str, Any]] = []
        monkeypatch.setattr(decisions, "record_decision",
                            lambda result, *, tension_reason=None, forced=False: self.rows.append(dict(result)))
        monkeypatch.setattr(decisions, "fetch_said_outreach_subjects",
                            lambda lookback_days=7: fold_said_subjects(
                                {"reason": r.get("reason"), "choice": r.get("choice"),
                                 "daydream_id": (r.get("grounding") or {}).get("daydream_id")}
                                for r in self.rows if not r.get("source")))

        def used(lookback_days=7):
            return set(), set(), {(r.get("grounding") or {}).get("daydream_id") for r in self.rows
                                  if r.get("reason") in ("sent", "orion_passed", ORION_DECLINED_REASON,
                                                         "no_why_juniper")
                                  and (r.get("grounding") or {}).get("daydream_id")}

        monkeypatch.setattr(decisions, "fetch_recently_used_outreach_content_ids", used)


def _stub_tick(monkeypatch, *, daydream=DAYDREAM, replies: List[str], prompts: List[str],
               delivered: List[str]) -> None:
    async def gather(self, session_id, daydream=None, _dd=daydream):
        return OutreachContext(curiosity_summaries=[], recent_turns=[], presence=None, daydream=_dd,
                               daydream_id=daydream_content_id(_dd[1]) if _dd else "",
                               tension_reason=self._last_tension_reason)

    async def generate(self, prompt, session_id, correlation_id, **_kw):
        prompts.append(prompt)
        return replies.pop(0), {"stub": True}

    async def deliver(self, *, text, **_kw):
        delivered.append(text)

    monkeypatch.setattr(EndogenousOutreach, "_gather_context", gather)
    monkeypatch.setattr(EndogenousOutreach, "_generate", generate)
    monkeypatch.setattr(EndogenousOutreach, "_deliver", deliver)


def _reading_source(subject_seed: str = "seed-1", finding_ids=("belief:p1@r1",), sent: List[Any] = None):
    def make():
        ids = tuple(finding_ids)
        return OutreachTopic(subject=f"reading:{subject_seed}", kind="reading", version=version_of(ids),
                             observed_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                             option_text='A read, "Graphs first": belief moved',
                             prompt_text='A read that changed you: "Graphs first".',
                             appendix="What the piece says -- Graphs first: a summary.",
                             finding_ids=ids, seed_id=subject_seed,
                             on_sent=(lambda: sent.append(ids)) if sent is not None else None)

    async def source():
        return [make()]

    return source


def _choices(o: EndogenousOutreach) -> List[PersonalityChoiceV1]:
    out = []
    for channel, env in o._bus.published:
        if channel == PERSONALITY_CHOICE_CHANNEL:
            assert env.kind == PERSONALITY_CHOICE_KIND
            out.append(PersonalityChoiceV1.model_validate(env.payload))
    return out


def test_tick_shows_topics_and_orion_picks_the_second(monkeypatch) -> None:
    ledger = _Ledger(monkeypatch)
    prompts: List[str] = []
    delivered: List[str] = []
    sent_ids: List[Any] = []
    o = _choice_outreach()
    o.topic_sources.append(_reading_source(sent=sent_ids))
    _stub_tick(monkeypatch, replies=["Pick: 2\nThe read moved a belief of mine.\nWhy Juniper: she asked."],
               prompts=prompts, delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["outreach"] is True and result["reason"] == "sent"
    # Freshest first: the read (5 min) outranks the daydream (10 min).
    assert "1. A read that changed you" in prompts[0] and "2. A daydream" in prompts[0]
    assert "0. Say nothing" in prompts[0]
    assert result["choice"]["chosen_subject"] == daydream_content_id(DAYDREAM[1])
    assert result["choice"]["chosen_rank"] == 2 and result["choice"]["published"] is True
    assert "finding_ids" not in result  # a daydream was picked, not the read
    assert delivered == ["The read moved a belief of mine."]
    (choice,) = _choices(o)
    assert choice.chosen.key == daydream_content_id(DAYDREAM[1]) and choice.chosen_market_rank == 2
    assert [a.key for a in choice.alternatives] == ["reading:seed-1"]
    assert choice.situation["trigger"] == "tension" and choice.situation["options_shown"] == 2
    assert sent_ids == [] and ledger.rows[-1]["reason"] == "sent"
    assert o._claimed_subjects == set()  # the tick's claims are released when it ends


def test_picking_a_read_delivers_its_appendix_and_marks_it_told(monkeypatch) -> None:
    _Ledger(monkeypatch)
    prompts: List[str] = []
    delivered: List[str] = []
    sent_ids: List[Any] = []
    o = _choice_outreach()
    o.topic_sources.append(_reading_source(sent=sent_ids))
    _stub_tick(monkeypatch, replies=["Pick: 1\nThat read changed how I check my writes.\nWhy Juniper: x."],
               prompts=prompts, delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["reason"] == "sent" and result["finding_ids"] == ["belief:p1@r1"]
    assert result["seed_id"] == "seed-1"
    assert delivered == ["That read changed how I check my writes.\n\n"
                         "What the piece says -- Graphs first: a summary."]
    assert sent_ids == [("belief:p1@r1",)]


@pytest.mark.parametrize("reply,shape", [("Pick: 0", "zero"), ("PASS", "pass"),
                                         ("Pick: 1\nHm.\nPASS\nWhy Juniper: x", "pass")])
def test_say_nothing_is_declined_all_and_nothing_is_sent(monkeypatch, reply, shape) -> None:
    ledger = _Ledger(monkeypatch)
    delivered: List[str] = []
    o = _choice_outreach()
    o.topic_sources.append(_reading_source())
    _stub_tick(monkeypatch, replies=[reply], prompts=[], delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["outreach"] is False and result["reason"] == ORION_DECLINED_REASON
    assert result["pass_shape"] == shape and result["choice"]["declined_all"] is True
    assert delivered == []
    (choice,) = _choices(o)
    assert choice.declined_all and choice.chosen is None and len(choice.alternatives) == 2
    # Declined subjects are decided: the next tick does not show them again.
    said = fold_said_subjects([{"reason": ledger.rows[-1]["reason"], "choice": ledger.rows[-1]["choice"]}])
    assert set(said) == {"reading:seed-1", daydream_content_id(DAYDREAM[1])}


def test_message_without_pick_line_among_two_topics_is_not_sent(monkeypatch) -> None:
    _Ledger(monkeypatch)
    delivered: List[str] = []
    o = _choice_outreach()
    o.topic_sources.append(_reading_source())
    _stub_tick(monkeypatch, replies=["Something.\nWhy Juniper: x"], prompts=[], delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["reason"] == NO_PICK_REASON and delivered == [] and _choices(o) == []


def test_out_of_range_pick_is_not_sent(monkeypatch) -> None:
    _Ledger(monkeypatch)
    delivered: List[str] = []
    o = _choice_outreach()
    _stub_tick(monkeypatch, replies=["Pick: 4\nSomething.\nWhy Juniper: x"], prompts=[], delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["reason"] == INVALID_PICK_REASON and delivered == []


def test_one_topic_still_a_choice_between_it_and_saying_nothing(monkeypatch) -> None:
    _Ledger(monkeypatch)
    prompts: List[str] = []
    delivered: List[str] = []
    o = _choice_outreach()
    _stub_tick(monkeypatch, replies=["The missing planet.\nWhy Juniper: x"], prompts=prompts, delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["reason"] == "sent" and "1. A daydream" in prompts[0] and "0. Say nothing" in prompts[0]
    (choice,) = _choices(o)
    assert [a.key for a in choice.alternatives] == ["say_nothing"]


def test_a_topic_another_composer_holds_is_not_shown(monkeypatch) -> None:
    _Ledger(monkeypatch)
    prompts: List[str] = []
    o = _choice_outreach()
    o.topic_sources.append(_reading_source())
    assert o.claim_subjects(["reading:seed-1"]) == {"reading:seed-1"}  # the readings door composing it
    _stub_tick(monkeypatch, replies=["Pick: 0"], prompts=prompts, delivered=[])

    asyncio.run(o.maybe_outreach())

    assert "Graphs first" not in prompts[0]
    assert o._claimed_subjects == {"reading:seed-1"}  # the door's claim is untouched


def test_a_failing_topic_source_does_not_end_the_tick(monkeypatch) -> None:
    _Ledger(monkeypatch)
    o = _choice_outreach()

    async def broken():
        raise RuntimeError("db down")

    o.topic_sources.append(broken)
    _stub_tick(monkeypatch, replies=["Pick: 1\nThe planet.\nWhy Juniper: x"], prompts=[], delivered=[])

    assert asyncio.run(o.maybe_outreach())["reason"] == "sent"


def test_choice_mode_does_not_read_the_personality_cue(monkeypatch) -> None:
    seen: List[bool] = []
    module_globals = EndogenousOutreach._gather_context.__globals__  # the module this test's class lives in

    async def fake_cue(url, *, enabled):
        seen.append(enabled)
        from scripts.personality_cue_reader import _skip

        return _skip(datetime.now(timezone.utc), "disabled")

    monkeypatch.setitem(module_globals, "read_personality_cue", fake_cue)
    for fn in ("_fetch_recent_turns", "_fetch_embodied_presence", "_fetch_current_daydream"):
        monkeypatch.setitem(module_globals, fn, lambda *a: None)

    async def no_situation(bus):
        return None, "no_key"

    monkeypatch.setitem(module_globals, "_read_situation", no_situation)
    for choice_enabled in (True, False):
        o = _choice_outreach(choice_enabled=choice_enabled, personality_cue_enabled=True)
        asyncio.run(o._gather_context("s"))
    assert seen == [False, True]


def test_regression_the_same_subject_is_not_sent_three_times_in_two_days(monkeypatch) -> None:
    """The subject ledger half of the 3x-repeat fix (the cue half is
    ``test_choice_mode_does_not_read_the_personality_cue``: the live repeat came in through the cue).

    Live 2026-11-11 15:15, 11-12 04:37 and 15:05 UTC: three ticks, three different daydreams
    (daydream:5b6b..., daydream:6c10..., daydream:c915...), and all three messages were about the
    same thing ("wrote the answer into the graph first, then looked"). The daydream-hash ledger
    saw three new daydreams and let all three through, because the real subject rode in as free
    prompt material with no id. At the choice point every subject has an id, and a subject Juniper
    got in the last 7 days is not shown again; a new daydream each tick does not reopen it."""
    ledger = _Ledger(monkeypatch)
    prompts: List[str] = []
    delivered: List[str] = []
    o = _choice_outreach()
    # The recurring subject, with a stable id every tick (here a read, seed "graph-first").
    o.topic_sources.append(_reading_source("graph-first"))
    daydreams = [(600.0, f"Daydream number {i}: a desk lamp over two laptops and a mug.") for i in range(3)]
    for i, dd in enumerate(daydreams):
        _stub_tick(monkeypatch, daydream=dd, prompts=prompts, delivered=delivered,
                   replies=[f"Pick: 1\nMessage {i}.\nWhy Juniper: x"])
        asyncio.run(o.maybe_outreach())

    rows = [r for r in ledger.rows if r.get("choice")]
    shown = [[s["subject"] for s in r["choice"]["shown"]] for r in rows]
    # Tick 1 shows the recurring subject and Orion picks it; ticks 2 and 3 never see it again.
    assert shown[0][0] == "reading:graph-first"
    assert all("reading:graph-first" not in s for s in shown[1:])
    assert [r["choice"]["chosen_subject"] for r in rows] == [
        "reading:graph-first", daydream_content_id(daydreams[1][1]), daydream_content_id(daydreams[2][1])]
    assert sum('"Graphs first"' in p for p in prompts) == 1
    assert len(delivered) == 3  # Orion still speaks; just not about the same subject again


def test_the_same_read_with_a_new_finding_is_shown_again(monkeypatch) -> None:
    """Changed materially: a belief the read moved days later is a new finding id, a new version."""
    ledger = _Ledger(monkeypatch)
    o = _choice_outreach()
    ids = ["belief:p1@r1"]

    async def source():
        return [OutreachTopic(subject="reading:seed-1", kind="reading", version=version_of(ids),
                              observed_at=datetime.now(timezone.utc), option_text="read", prompt_text="A read.",
                              finding_ids=tuple(ids), seed_id="seed-1")]

    o.topic_sources.append(source)
    for dd_i in range(3):
        _stub_tick(monkeypatch, daydream=(600.0, f"Daydream {dd_i} of a quiet harbor at night, no boats."),
                   prompts=[], delivered=[], replies=["Pick: 1\nThe read.\nWhy Juniper: x"])
        asyncio.run(o.maybe_outreach())
        if dd_i == 1:
            ids.append("belief:p2@r2")
    picks = [r["choice"]["chosen_subject"] for r in ledger.rows if r.get("choice")]
    assert picks[0] == "reading:seed-1" and picks[1] != "reading:seed-1" and picks[2] == "reading:seed-1"


def test_choice_off_keeps_the_one_topic_tick(monkeypatch) -> None:
    _Ledger(monkeypatch)
    prompts: List[str] = []
    o = _choice_outreach(choice_enabled=False)
    o.topic_sources.append(_reading_source())
    _stub_tick(monkeypatch, replies=["PASS"], prompts=prompts, delivered=[])

    result = asyncio.run(o.maybe_outreach())

    assert result["reason"] == "orion_passed" and "Pick:" not in prompts[0] and _choices(o) == []


def test_declining_a_read_at_the_tick_tells_the_readings_door(monkeypatch) -> None:
    """Review 2026-11-13 (MUST): a tick decline left the read pending, and the readings door then sent
    it on its own. The declined row carries the shown reads' finding ids, and on_declined mirrors it."""
    ledger = _Ledger(monkeypatch)
    declined: List[str] = []
    o = _choice_outreach()

    async def source():
        return [OutreachTopic(subject="reading:seed-1", kind="reading", version="v",
                              observed_at=datetime.now(timezone.utc), option_text="read", prompt_text="A read.",
                              finding_ids=("belief:p1@r1",), seed_id="seed-1",
                              on_declined=lambda: declined.append("seed-1"))]

    o.topic_sources.append(source)
    _stub_tick(monkeypatch, replies=["Pick: 0"], prompts=[], delivered=[])
    result = asyncio.run(o.maybe_outreach())
    assert result["reason"] == ORION_DECLINED_REASON and result["finding_ids"] == ["belief:p1@r1"]
    assert declined == ["seed-1"] and ledger.rows[-1]["finding_ids"] == ["belief:p1@r1"]


def test_a_tick_compose_failure_on_a_read_does_not_carry_its_finding_ids(monkeypatch) -> None:
    """Review 2026-11-13: tick failures (empty message, ungrounded name) must not count toward the
    readings door's max_attempts for that read."""
    _Ledger(monkeypatch)
    o = _choice_outreach()
    o.topic_sources.append(_reading_source())
    _stub_tick(monkeypatch, replies=["Pick: 1"], prompts=[], delivered=[])
    result = asyncio.run(o.maybe_outreach())
    assert result["reason"] == "empty_generation" and "finding_ids" not in result


def test_a_choice_that_fails_to_build_never_raises_out_of_the_tick(monkeypatch) -> None:
    import scripts.outreach_choice as oc

    _Ledger(monkeypatch)
    o = _choice_outreach()

    def boom(**_kw):
        raise ValueError("bad choice")

    monkeypatch.setattr(oc, "build_choice", boom)
    _stub_tick(monkeypatch, replies=["Pick: 1\nThe planet.\nWhy Juniper: x"], prompts=[], delivered=[])
    result = asyncio.run(o.maybe_outreach())
    assert result["reason"] == "sent" and result["choice"]["published"] is False and _choices(o) == []


def test_choice_prompt_leaves_out_material_with_no_subject_id() -> None:
    """Review 2026-11-13: priors / curiosity summaries have no subject in the choice ledger, so they
    stay out of a choice prompt (the same no-id path the cue repeat came through)."""
    ctx = OutreachContext(curiosity_summaries=["repair pressure rising"], recent_turns=[], presence=None,
                          open_prior_previews=["an open claim"])
    prompt = build_outreach_prompt(ctx, topics=[_topic("reading:s1")])
    assert "repair pressure rising" not in prompt and "an open claim" not in prompt


# --- personality rev 4: the bid carries the topics' text; a care-only fund is care_promoted -------


def test_outreach_object_text_joins_shown_topics_and_caps() -> None:
    from scripts.outreach_choice import OBJECT_TEXT_MAX, outreach_object_text

    topics = [_topic("daydream:d", kind="daydream"), _topic("reading:s1")]
    text = outreach_object_text(topics)
    assert text == "; ".join(t.option_text for t in topics)
    assert outreach_object_text([]) is None
    long = [OutreachTopic(subject=f"s{i}", kind="reading", option_text="w " * 300, prompt_text="p") for i in range(3)]
    assert len(outreach_object_text(long)) <= OBJECT_TEXT_MAX


def test_build_choice_carries_care_promoted() -> None:
    topics = [_topic("daydream:d", kind="daydream"), _topic("reading:s1")]
    pick = parse_pick("Pick: 1\nmsg", 2, is_pass=is_pass_response)
    kw = dict(decision_key="w", topics=topics, pick=pick, occurred_at=NOW, situation={}, act_id="w",
              correlation_id="w")
    assert build_choice(**kw).care_promoted is False
    assert build_choice(**kw, care_promoted=True).care_promoted is True


def test_tick_bid_sends_topics_text_and_a_care_only_fund_marks_the_choice(monkeypatch) -> None:
    """Live gap 2026-11-13: outreach wants carried object_ref "juniper" and no text, so care was always 0.
    The binding bid now carries the shown topics' text; when the want was funded only through care, the
    published choice is ``care_promoted`` (the reducer and source (c) weigh it half)."""
    import scripts.outreach_bind as ob
    from orion.schemas.physiology_want import WantDecisionV1

    seen: List[Any] = []

    async def fake_abid(want, timeout_sec, object_text=None):
        seen.append(object_text)
        now = datetime.now(timezone.utc)
        return WantDecisionV1(
            decision_id=f"{want.want_id}@x", want_id=want.want_id, family=want.family, producer=want.producer,
            verb=want.verb, tier=want.tier, decided_at=now, outcome="funded", reason="net_positive", net=0.05,
            charge=0.1, want=want, why="hub.outreach: care lifted it",
            warnings=["personality:arm=care", "personality:care=0.6000:tendency=d1:cos=0.9000",
                      "personality:net_without_care=-0.010000"])

    async def fake_release(want_id):
        return 1

    monkeypatch.setattr(ob, "_abid", fake_abid)
    monkeypatch.setattr(ob, "_arelease", fake_release)
    _Ledger(monkeypatch)
    prompts: List[str] = []
    delivered: List[str] = []
    o = _choice_outreach()
    o.economy_mode = "on"
    o.topic_sources.append(_reading_source())
    _stub_tick(monkeypatch, replies=["Pick: 1\nThe read moved a belief of mine.\nWhy Juniper: she asked."],
               prompts=prompts, delivered=delivered)

    result = asyncio.run(o.maybe_outreach())

    assert result["outreach"] is True
    assert len(seen) == 1 and seen[0] and 'A read, "Graphs first": belief moved' in seen[0]
    assert "A daydream" in seen[0]
    (choice,) = _choices(o)
    assert choice.care_promoted is True
