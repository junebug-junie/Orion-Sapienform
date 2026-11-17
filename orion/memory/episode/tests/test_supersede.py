"""Supersede linking logic (issue #3587), made-up fixtures only, no database, no model."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from orion.memory.episode import supersede as s

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def row(mid, statement, *, days=0, purpose="about_juniper", state="auto", refs=()):
    return s.MemoryRow(memory_id=mid, statement=statement, purpose=purpose, created_at=T0 + timedelta(days=days),
                       confirmation_state=state, referents=frozenset(refs))


def answers(*texts):
    """An ask() that replays canned answers in order and records every prompt."""
    seen: list[str] = []
    queue = list(texts)

    async def ask(prompt):
        seen.append(prompt)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return ask, seen


YES = json.dumps({"corrects": "yes", "sure": "yes", "reason": "B says the pet is a dog, A said a cat."})
YES_UNSURE = json.dumps({"corrects": "yes", "sure": "no", "reason": "maybe"})
NO = json.dumps({"corrects": "no", "sure": "yes", "reason": "both can be true"})
BOTH_FALSE = json.dumps({"both_true": "no", "reason": "B says A was wrong."})
BOTH_TRUE = json.dumps({"both_true": "yes", "reason": "different times"})

OLD = row("old", "Juniper told me her pet Biscuit is a cat.", refs=("pet:biscuit",))
NEW = row("new", "Juniper told me Biscuit is a dog, not a cat.", days=3, refs=("pet:biscuit",))


def corpus(*rows):
    return s.Corpus.build(rows)


# --- candidates ------------------------------------------------------------------------------

def test_candidates_need_an_older_row_that_shares_something_specific():
    unrelated = row("x", "The office was loud all evening.", days=1)
    later = row("later", "Juniper said Biscuit is a cat.", days=5, refs=("pet:biscuit",))
    c = corpus(OLD, unrelated, NEW, later)
    got = s.candidates_for(NEW, c, dense={"old": 0.7, "x": 0.7, "later": 0.9})
    assert [x.old.memory_id for x in got] == ["old"]  # 'x' shares nothing rare; 'later' is newer
    assert got[0].shared_rare_referents == ("pet:biscuit",)


def test_dense_floor_and_dense_alone():
    near = row("near", "A very close paraphrase with no shared rare word.", days=1)
    c = corpus(OLD, near, NEW)
    assert s.candidates_for(NEW, c, dense={"old": 0.59}) == []           # under the floor
    assert [x.old.memory_id for x in s.candidates_for(NEW, c, dense={"near": s.DENSE_ALONE})] == ["near"]
    assert s.candidates_for(NEW, c, dense={"near": s.DENSE_ALONE - 0.01}) == []


def test_confirmed_and_follow_up_rows_are_never_candidates():
    confirmed = row("conf", "Juniper told me Biscuit is a cat.", state="confirmed", refs=("pet:biscuit",))
    todo = row("todo", "I should ask Juniper whether Biscuit is a cat.", purpose="follow_up", refs=("pet:biscuit",))
    c = corpus(confirmed, todo, NEW)
    assert s.candidates_for(NEW, c, dense={"conf": 0.9, "todo": 0.9}) == []
    # A follow-up never corrects anything either.
    assert s.candidates_for(row("t2", "Ask about Biscuit the cat.", days=9, purpose="follow_up"), corpus(OLD)) == []


def test_common_referents_do_not_count_as_specific():
    rows = [row(f"r{i}", f"Juniper did thing {i}.", refs=("person:juniper",)) for i in range(s.RARE_REFERENT_MAX_ROWS + 1)]
    new = row("n", "Juniper did something else entirely.", days=40, refs=("person:juniper",))
    assert s.candidates_for(new, corpus(*rows, new), dense={r.memory_id: 0.7 for r in rows}) == []


def test_without_dense_two_specific_shares_are_needed():
    c = corpus(OLD, NEW)
    got = s.candidates_for(NEW, c, dense=None)
    assert [x.old.memory_id for x in got] == ["old"]  # rare referent + rare word "biscuit"
    lone = row("lone", "Juniper went to Millbrook.", days=4)
    assert s.candidates_for(row("n2", "Millbrook is far.", days=5), corpus(lone), dense=None) == []


def test_at_most_k_candidates_best_first():
    olds = [row(f"o{i}", f"Biscuit fact {i}.", refs=("pet:biscuit",)) for i in range(6)]
    got = s.candidates_for(NEW, corpus(*olds, NEW), dense={f"o{i}": 0.6 + i / 100 for i in range(6)})
    assert [x.old.memory_id for x in got] == ["o5", "o4", "o3"]


# --- the check -------------------------------------------------------------------------------

def test_parse_check_refuses_anything_malformed():
    assert s.parse_check(YES).first_yes
    assert s.parse_check("```json\n" + YES + "\n```").first_yes
    for bad in ("", "yes", "[1]", '{"corrects": "maybe", "sure": "yes", "reason": "x"}',
                '{"corrects": "yes", "sure": "yes", "reason": ""}'):
        v = s.parse_check(bad)
        assert not v.first_yes and not v.accepted, bad


def test_two_agreeing_answers_link():
    ask, seen = answers(YES, BOTH_FALSE)
    v = asyncio.run(s.run_check(OLD, NEW, ask))
    assert v.accepted and v.both_true is False and len(seen) == 2
    assert OLD.statement in seen[0] and NEW.statement in seen[0] and "both be true" in seen[1]


def test_second_question_can_veto():
    ask, _ = answers(YES, BOTH_TRUE)
    v = asyncio.run(s.run_check(OLD, NEW, ask))
    assert v.first_yes and not v.accepted


def test_no_unsure_or_failure_never_links_and_skips_the_second_question():
    for first in (NO, YES_UNSURE, "garbage", RuntimeError("gateway down")):
        ask, seen = answers(first)
        v = asyncio.run(s.run_check(OLD, NEW, ask))
        assert not v.accepted and len(seen) == 1
    ask, _ = answers(YES, RuntimeError("timeout"))
    v = asyncio.run(s.run_check(OLD, NEW, ask))
    assert not v.accepted and v.both_true is None


def test_budget_caps_model_calls():
    olds = [row(f"o{i}", f"Biscuit fact {i}.", refs=("pet:biscuit",)) for i in range(3)]
    ask, seen = answers(YES, BOTH_FALSE, NO, NO)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(*olds), ask, max_checks=3,
                                           dense=_dense({f"o{i}": 0.7 for i in range(3)})))
    assert len(seen) == 3 and plan["budget_left"] == 0
    assert len(plan["links"]) == 1


def _dense(scores):
    async def run(_statement):
        return dict(scores)
    return run


def test_plan_falls_back_when_the_index_is_down():
    async def down(_statement):
        return None

    ask, _ = answers(YES, BOTH_FALSE)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=down))
    assert plan["dense"] == "unavailable" and [l["old_id"] for l in plan["links"]] == ["old"]


def test_dedupe_one_link_per_old_row_and_no_chains():
    a, b, c = (s.Link("A", "B", "r"), s.Link("A", "C", "r"), s.Link("B", "D", "r"))
    assert s.dedupe_links([a, b, c]) == [a]
    assert s.dedupe_links([s.Link("X", "X", "r")]) == []


def test_link_round_trips_through_the_checkpoint():
    link = s.Link("A", "B", "reason", dense=0.71, shared=("pet:biscuit",))
    assert s.Link.from_dict(json.loads(json.dumps(link.as_dict()))) == link


def test_live_rows_sql_reuses_the_serving_gate():
    from orion.memory.lived_memory import EPISODE_SQL

    assert EPISODE_SQL in s.LIVE_ROWS_SQL
