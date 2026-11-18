"""Supersede linking logic (issue #3587), made-up fixtures only, no database, no model."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from orion.memory.episode import supersede as s

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def row(mid, statement, *, days=0, purpose="about_juniper", state="auto", refs=()):
    return s.MemoryRow(memory_id=mid, statement=statement, purpose=purpose, created_at=T0 + timedelta(days=days),
                       confirmation_state=state, referents=frozenset(refs))


def answers(*texts):
    """An ask() that replays canned answers in order and records every prompt."""
    seen: list[str] = []
    temps: list[float] = []
    queue = list(texts)

    async def ask(prompt, *, temperature=0.0):
        seen.append(prompt)
        temps.append(temperature)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    ask.temps = temps
    return ask, seen


def vote(relation, conf=0.9, same="yes", reason="r", kind="lasting_state", changed="yes", referent="yes"):
    return json.dumps({"a_kind": kind, "b_says_a_wrong_or_changed": changed, "same_referent": referent,
                       "same_attribute": same, "relation": relation, "confidence": conf, "reason": reason})


COR = vote("correction", reason="B says the pet is a dog, A said a cat.")
UPD = vote("update")
COMPAT = vote("compatible", same="no", changed="no")
LOWC = vote("correction", conf=0.5)
COR3 = (COR, COR, COR)

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


def test_dense_floor_and_no_dense_alone_leg():
    near = row("near", "A very close paraphrase with no shared rare word.", days=1)
    c = corpus(OLD, near, NEW)
    assert s.candidates_for(NEW, c, dense={"old": 0.59}) == []           # under the floor
    # Issue #3659: a cosine alone, however high, never makes a pair.
    assert s.candidates_for(NEW, c, dense={"near": 0.99}) == []
    assert not hasattr(s, "DENSE_ALONE")


def test_candidates_carry_their_legs_without_text():
    got = s.candidates_for(NEW, corpus(OLD, NEW), dense={"old": 0.7123})
    legs = got[0].legs()
    assert legs == {"cosine": 0.7123, "rare_tokens": len(got[0].shared_rare_tokens),
                    "rare_token_rows": [2] * len(got[0].shared_rare_tokens), "rare_referents": 1,
                    "rare_referent_rows": [2], "rank": 1, "k": s.K_PER_ROW}
    assert "Biscuit" not in json.dumps(legs)


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


# --- the classifier -------------------------------------------------------------------------

def test_parse_vote_refuses_anything_malformed():
    assert s.parse_vote(COR).acting
    assert s.parse_vote("```json\n" + UPD + "\n```").acting
    for bad in ("", "yes", "[1]", vote("maybe"), vote("correction", same="perhaps"), vote("correction", conf=1.5),
                json.dumps({"relation": "update", "same_attribute": "yes", "confidence": "high"})):
        assert not s.parse_vote(bad).acting, bad
        assert s.parse_vote(bad).error, bad
    assert not s.parse_vote(vote("correction", same="no")).acting   # not the same attribute
    assert not s.parse_vote(vote("refine")).acting
    weird = s.parse_vote(vote("correction", kind="weird"))
    assert weird.error is None and not weird.acting          # a bad checklist field abstains, not an error
    assert s.parse_vote(vote("correction", kind="lasting state")).acting   # spacing normalized
    assert s.parse_vote(json.dumps({"relation": "correction", "same_attribute": "yes", "confidence": True,
                                    "a_kind": "event", "b_says_a_wrong_or_changed": "yes"})).error
    assert s.parse_vote(json.dumps({"relation": "correction", "same_attribute": "yes", "confidence": "nan",
                                    "a_kind": "event", "b_says_a_wrong_or_changed": "yes"})).error


def test_the_checklist_must_agree_with_an_acting_relation():
    assert not s.parse_vote(vote("update", kind="plan")).acting          # a plan is never updated
    assert not s.parse_vote(vote("update", kind="event")).acting
    assert not s.parse_vote(vote("correction", kind="past_period")).acting
    assert not s.parse_vote(vote("correction", kind="view")).acting
    assert not s.parse_vote(vote("correction", changed="no")).acting      # B's words must say so
    assert s.parse_vote(vote("correction", kind="measurement")).acting
    ask, _ = answers(vote("update", kind="plan"))
    v = asyncio.run(s.classify_pair(OLD, NEW, ask))
    assert not v.acts and v.abstain == "checklist:update"


def test_three_unanimous_confident_votes_act_above_temperature_zero():
    ask, seen = answers(*COR3)
    v = asyncio.run(s.classify_pair(OLD, NEW, ask))
    assert v.acts and v.relation == "correction" and len(seen) == 3 and v.abstain == ""
    assert all(t > 0 for t in ask.temps) and OLD.statement in seen[0] and NEW.statement in seen[0]


def test_any_non_acting_vote_abstains_and_stops_asking():
    for votes in ((COMPAT,), (COR, vote("refine")), (COR, COR, vote("unrelated", same="no", changed="no"))):
        ask, seen = answers(*votes)
        v = asyncio.run(s.classify_pair(OLD, NEW, ask))
        assert not v.acts and not v.juniper_acts and len(seen) == len(votes)
        assert v.abstain.startswith("vote:")


def test_split_or_low_confidence_abstains():
    ask, _ = answers(COR, UPD, COR)
    v = asyncio.run(s.classify_pair(OLD, NEW, ask))
    assert not v.acts and v.abstain == "split_update_correction"
    ask, _ = answers(COR, LOWC, COR)
    v = asyncio.run(s.classify_pair(OLD, NEW, ask))
    assert not v.acts and v.abstain == "low_confidence"


def test_a_failed_or_malformed_vote_is_an_error_not_a_link():
    for bad in (RuntimeError("gateway down"), "garbage"):
        ask, _ = answers(COR, bad)
        v = asyncio.run(s.classify_pair(OLD, NEW, ask))
        assert not v.acts and v.error and s.check_failed(v)


def test_juniper_words_outrank_low_confidence_but_not_a_dissent():
    mine = replace(NEW, voice="juniper_said", juniper_quoted=True)
    ask, _ = answers(COR, LOWC, UPD)
    v = asyncio.run(s.classify_pair(OLD, mine, ask))
    assert not v.acts and v.juniper_acts and v.relation == "correction"
    ask, _ = answers(COR, COMPAT)
    v = asyncio.run(s.classify_pair(OLD, mine, ask))
    assert not v.juniper_acts                       # one "compatible" vote: no override
    ask, _ = answers(COR, LOWC, COR)
    assert not asyncio.run(s.classify_pair(OLD, NEW, ask)).juniper_acts   # not her words


def test_juniper_path_needs_one_clear_target():
    mine = replace(NEW, voice="juniper_said", juniper_quoted=True)
    o2 = row("o2", "Juniper told me Biscuit is a tabby cat.", refs=("pet:biscuit",))
    cands = s.candidates_for(mine, corpus(OLD, o2, mine), dense={"old": 0.75, "o2": 0.7})
    ask, _ = answers(COR, LOWC, COR, COMPAT)       # old: her path; o2: compatible
    links, recs = asyncio.run(s.check_pairs(mine, cands, ask))
    assert [(l.old_id, l.path) for l in links] == [("old", "juniper")] and recs[0].linked
    ask, _ = answers(COR, LOWC, COR, COR, LOWC, COR)   # both qualify: no clear target
    links, _ = asyncio.run(s.check_pairs(mine, cands, ask))
    assert links == []


def test_link_carries_relation_path_and_votes():
    ask, _ = answers(UPD, UPD, UPD)
    cands = s.candidates_for(NEW, corpus(OLD, NEW), dense={"old": 0.75})
    links, recs = asyncio.run(s.check_pairs(NEW, cands, ask))
    assert links[0].relation == "update" and links[0].path == "model" and links[0].votes == ("update",) * 3
    audit = recs[0].audit()
    assert audit["linked"] and audit["legs"]["cosine"] == 0.75 and len(audit["votes"]) == 3
    assert OLD.statement not in json.dumps(audit) and NEW.statement not in json.dumps(audit)


def test_invalid_at_update_is_the_new_time_correction_is_the_old_time():
    t_old, t_new, now = T0, T0 + timedelta(days=30), T0 + timedelta(days=40)
    old, new = {"occurred_at": t_old, "created_at": t_old}, {"occurred_at": t_new, "created_at": now}
    assert s.invalid_at_for("update", old, new, now) == t_new
    assert s.invalid_at_for("correction", old, new, now) == t_old
    assert s.invalid_at_for("update", old, {"occurred_at": None, "created_at": now}, now) == now


def test_budget_caps_model_calls():
    olds = [row(f"o{i}", f"Biscuit fact {i}.", refs=("pet:biscuit",)) for i in range(3)]
    ask, seen = answers(*COR3, COMPAT, COMPAT)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(*olds), ask, max_checks=4,
                                           dense=_dense({f"o{i}": 0.7 for i in range(3)}), apply_gate=OPEN))
    assert len(seen) == 4 and plan["budget_left"] == 0
    assert len(plan["links"]) == 1 and plan["apply"] == "on" and s.links_to_apply(plan) == plan["links"]
    assert len(plan["decisions"]) == 2 and plan["decisions"][0]["linked"]


def _dense(scores):
    async def run(_statement):
        return dict(scores)
    return run


OPEN = (True, "eval_passed")


# --- never act degraded (issue #3659) --------------------------------------------------------

def test_index_down_links_nothing_asks_nothing_and_says_why():
    async def down(_statement):
        return None

    ask, seen = answers(*COR3)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=down, apply_gate=OPEN))
    assert plan["dense"] == "unavailable" and plan["links"] == [] and seen == []
    assert plan["apply"] == "refused" and plan["refused_because"] == [s.REFUSED_DENSE_UNAVAILABLE]
    assert s.links_to_apply(plan) == []


def test_index_raising_links_nothing():
    async def boom(_statement):
        raise ConnectionError("chroma down")

    ask, seen = answers(*COR3)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=boom, apply_gate=OPEN))
    assert plan["links"] == [] and seen == []
    assert plan["refused_because"] == [f"{s.REFUSED_DENSE_UNAVAILABLE}:ConnectionError"]


def test_no_index_configured_never_falls_back_to_word_overlap():
    ask, seen = answers(*COR3)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=None, apply_gate=OPEN))
    assert plan["dense"] == "off" and plan["links"] == [] and seen == []
    assert plan["refused_because"] == [s.REFUSED_DENSE_OFF]


def test_a_malformed_answer_anywhere_holds_every_link():
    other_old = row("o2", "Biscuit sleeps on the green sofa.", refs=("pet:biscuit",))
    ask, _ = answers(*COR3, "not json")
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD, other_old), ask,
                                           dense=_dense({"old": 0.75, "o2": 0.7}), apply_gate=OPEN))
    assert plan["links"] == [] and [l["old_id"] for l in plan["held_links"]] == ["old"]
    assert plan["refused_because"] == [s.REFUSED_CHECK_ERRORS] and s.links_to_apply(plan) == []


def test_a_failed_vote_on_a_later_pair_holds_every_link():
    other_old = row("o2", "Biscuit sleeps on the green sofa.", refs=("pet:biscuit",))
    ask, _ = answers(*COR3, COR, RuntimeError("hold lost"))
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD, other_old), ask,
                                           dense=_dense({"old": 0.75, "o2": 0.7}), apply_gate=OPEN))
    assert plan["check_errors"] == 1 and s.links_to_apply(plan) == []
    assert [l["old_id"] for l in plan["held_links"]] == ["old"]


def test_a_closed_eval_gate_proposes_but_holds():
    ask, seen = answers(*COR3)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=_dense({"old": 0.75}),
                                           apply_gate=(False, "eval_missing")))
    assert len(seen) == 3 and plan["links"] == [] and [l["old_id"] for l in plan["held_links"]] == ["old"]
    assert plan["refused_because"] == [f"{s.REFUSED_EVAL_GATE}:eval_missing"]


def test_the_gate_defaults_closed():
    ask, _ = answers(*COR3)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=_dense({"old": 0.75})))
    assert plan["links"] == [] and plan["apply"] == "refused"


def test_a_checkpoint_from_before_the_rule_applies_nothing():
    legacy = {"links": [{"old_id": "a", "new_id": "b", "reason": "r"}], "asked": 1, "dense": "unavailable"}
    assert s.links_to_apply(legacy) == [] and s.links_to_apply(None) == []
    assert s.links_to_apply({"links": legacy["links"], "apply": "on", "refused_because": ["x"]}) == []


class _GateConn:
    def __init__(self, row=None, exc=None):
        self.row, self.exc = row, exc

    async def execute(self, sql, params):
        if self.exc:
            raise self.exc
        assert "memory_supersede_eval_run" in sql and params == (s.CHECK_VERSION, s.VOTES, s.EVAL_ROUTE)
        row = self.row

        class Cur:
            description = None

            async def fetchone(self):
                return row
        return Cur()


def test_eval_gate_open_only_on_a_passing_run_for_this_check_version():
    run = asyncio.run
    fresh = datetime.now(timezone.utc)
    assert run(s.eval_gate(_GateConn({"passed": True, "finished_at": fresh}))) == (True, "eval_passed")
    stale = fresh - timedelta(days=s.EVAL_MAX_AGE_DAYS + 1)
    assert run(s.eval_gate(_GateConn({"passed": True, "finished_at": stale}))) == (False, "eval_stale")
    assert run(s.eval_gate(_GateConn({"passed": False}))) == (False, "eval_failed")
    assert run(s.eval_gate(_GateConn(None))) == (False, "eval_missing")
    assert run(s.eval_gate(_GateConn(exc=RuntimeError("relation does not exist")))) == (
        False, "eval_unreadable:RuntimeError")


def test_dedupe_one_link_per_old_row_and_no_chains():
    a, b, c = (s.Link("A", "B", "r"), s.Link("A", "C", "r"), s.Link("B", "D", "r"))
    assert s.dedupe_links([a, b, c]) == [a]
    assert s.dedupe_links([s.Link("X", "X", "r")]) == []


def test_link_round_trips_through_the_checkpoint():
    link = s.Link("A", "B", "reason", dense=0.71, shared=("pet:biscuit",), relation="update", path="juniper",
                  votes=("update", "update", "correction"))
    assert s.Link.from_dict(json.loads(json.dumps(link.as_dict()))) == link
    legacy = s.Link.from_dict({"old_id": "A", "new_id": "B", "reason": "r"})   # a pre-v3 checkpoint
    assert legacy.relation == "correction" and legacy.path == "model"


def test_live_rows_sql_reuses_the_serving_gate():
    from orion.memory.lived_memory import EPISODE_SQL

    assert EPISODE_SQL in s.LIVE_ROWS_SQL


def test_open_asks_rejected_rows_and_unsettled_corrections_are_out():
    pending = row("p", "Juniper told me Biscuit is a cat.", state="pending_confirmation", refs=("pet:biscuit",))
    unconf = row("u", "Juniper told me Biscuit is a cat again.", state="unconfirmed", refs=("pet:biscuit",))
    assert s.candidates_for(NEW, corpus(pending, unconf, NEW), dense={"p": 0.9, "u": 0.9}) == []
    unsettled = row("n", "Juniper told me Biscuit is a dog.", days=3, state="pending_confirmation",
                    refs=("pet:biscuit",))
    assert s.candidates_for(unsettled, corpus(OLD, unsettled), dense={"old": 0.9}) == []


def test_same_episode_siblings_and_undone_pairs_are_never_candidates():
    sib_old = s.MemoryRow("so", OLD.statement, "about_juniper", T0, episode_id="ep-1", referents=OLD.referents)
    sib_new = s.MemoryRow("sn", NEW.statement, "about_juniper", T0 + timedelta(days=1), episode_id="ep-1",
                          referents=NEW.referents)
    assert s.candidates_for(sib_new, corpus(sib_old, sib_new), dense={"so": 0.9}) == []
    assert s.candidates_for(NEW, corpus(OLD, NEW), dense={"old": 0.9}, excluded=frozenset({("old", "new")})) == []


def test_a_failed_call_stops_the_whole_plan():
    olds = [row(f"o{i}", f"Biscuit fact {i}.", refs=("pet:biscuit",)) for i in range(3)]
    other = row("n2", "Juniper said Biscuit is a dog now.", days=4, refs=("pet:biscuit",))
    ask, seen = answers(RuntimeError("gpu_hold_lost"))
    plan = asyncio.run(s.plan_for_new_rows([NEW, other], corpus(*olds), ask, max_checks=10,
                                           dense=_dense({f"o{i}": 0.7 for i in range(3)}), apply_gate=OPEN))
    assert len(seen) == 1 and plan["links"] == [] and plan["budget_left"] == 0
    assert s.REFUSED_CHECK_ERRORS in plan["refused_because"]


def test_a_budget_cut_disables_the_juniper_path():
    mine = replace(NEW, voice="juniper_said", juniper_quoted=True)
    o2 = row("o2", "Juniper told me Biscuit is a tabby cat.", refs=("pet:biscuit",))
    cands = s.candidates_for(mine, corpus(OLD, o2, mine), dense={"old": 0.75, "o2": 0.7})
    ask, _ = answers(COR, LOWC, COR, COR)          # o2 is cut after one acting vote
    links, recs = asyncio.run(s.check_pairs(mine, cands, ask, budget=[4]))
    assert links == [] and recs[1].verdict.abstain == "budget_spent"


def test_links_to_apply_keeps_only_current_version_links_with_a_relation():
    good = {"old_id": "a", "new_id": "b", "check_version": s.CHECK_VERSION, "relation": "update"}
    plan = {"apply": "on", "refused_because": [], "links": [
        good, {**good, "check_version": "supersede_check.v2"}, {**good, "relation": None}]}
    assert s.links_to_apply(plan) == [good]


def test_an_update_invalid_at_is_the_new_time():
    old = {"occurred_at": T0 + timedelta(days=5)}
    new = {"occurred_at": T0, "created_at": T0 + timedelta(days=9)}
    assert s.invalid_at_for("update", old, new, T0) == T0    # apply_link refuses this one (PG test)


def test_juniper_words_still_need_the_checklist():
    mine = replace(NEW, voice="juniper_said", juniper_quoted=True)
    ask, _ = answers(*(vote("update", kind="plan"),) * 3)
    v = asyncio.run(s.classify_pair(OLD, mine, ask))
    assert not v.acts and not v.juniper_acts


def test_a_third_vote_failing_the_checklist_abstains_after_three_calls():
    ask, seen = answers(COR, COR, vote("correction", changed="no"))
    v = asyncio.run(s.classify_pair(OLD, NEW, ask))
    assert not v.acts and v.abstain == "checklist:correction" and len(seen) == 3


def test_missing_or_mixed_checklist_fields():
    no_changed = json.dumps({"relation": "correction", "same_attribute": "yes", "confidence": 0.9, "a_kind": "event"})
    no_kind = json.dumps({"relation": "update", "same_attribute": "yes", "confidence": 0.9,
                          "b_says_a_wrong_or_changed": "yes"})
    for text in (no_changed, no_kind):
        v = s.parse_vote(text)
        assert v.error is None and not v.acting
    compat_bad = json.dumps({"relation": "compatible", "same_attribute": "no", "confidence": 0.9, "a_kind": "??",
                             "b_says_a_wrong_or_changed": "unsure"})
    ask, _ = answers(compat_bad)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=_dense({"old": 0.75}), apply_gate=OPEN))
    assert plan["check_errors"] == 0 and plan["refused_because"] == []    # abstained, plan not held
    assert not s.parse_vote(vote("update", kind="measurement")).acting
    assert not s.parse_vote(vote("update", changed="no")).acting


def test_a_correction_needs_the_same_specific_referent():
    """v3.2: a shared project word is not the same item (two pairs Juniper rejected, 2026-11-18)."""
    assert not s.parse_vote(vote("correction", kind="measurement", referent="no")).acting
    assert not s.parse_vote(vote("update", referent="no")).acting
    missing = json.loads(vote("correction"))
    del missing["same_referent"]
    v = s.parse_vote(json.dumps(missing))
    assert v.error is None and not v.acting                     # missing = abstain, not an error
    ask, seen = answers(vote("correction", referent="no"))
    c = asyncio.run(s.classify_pair(OLD, NEW, ask))
    assert not c.acts and c.abstain == "checklist:correction" and len(seen) == 1
    assert '"same_referent"' in s.build_classify_prompt(OLD, NEW)


def test_a_named_pr_needs_the_same_number_in_b():
    """v3.3: Juniper's rejected pair, made up: a named PR's deploy vs a general 'nothing deployed'."""
    a = row("a", "Juniper said the login fix is deployed and linked PR #41.")
    b = row("b", "Juniper said nothing is deployed yet; the ticket closes when a fix lands.", days=14)
    same = row("c", "Juniper said PR #41 is not deployed after all.", days=1)
    assert s.numbered_items(a.statement) == {"41"} and not s.same_numbered_item(a, b) and s.same_numbered_item(a, same)
    assert s.same_numbered_item(row("x", "Biscuit is a cat."), b)          # A names no number: no constraint
    ask, seen = answers()
    v = asyncio.run(s.classify_pair(a, b, ask, budget=[3]))
    assert not v.acts and not v.juniper_acts and v.abstain == s.NUMBERED_ITEM_NOT_IN_B and seen == []
    ask, _ = answers(*(vote("correction", kind="event"),) * 3)
    assert asyncio.run(s.classify_pair(a, same, ask)).acts


import pytest  # noqa: E402


@pytest.mark.parametrize("text,ids", [
    ("merged PR #2968, then left", {"2968"}), ("see #41.", {"41"}), ("PR-41 and PRs 42", {"41", "42"}),
    ("issues 12 and ticket #0099", {"12", "99"}), ("orion#41", {"41"}),
    ("about 40 W in 2026-11-18", set()), ("#1 and PR 7", set()), ("# 2026 review", set()), ("abc#41st", set()),
])
def test_numbered_items_regex(text, ids):
    assert s.numbered_items(text) == ids


def test_numbered_rule_edge_cases():
    a2 = row("a2", "PR #14 and PR #15 are deployed.")
    assert s.same_numbered_item(a2, row("b2", "PR #15 is not deployed.", days=1))       # one of two named
    a = row("a", "Juniper said the fix is PR #14.")
    assert s.same_numbered_item(a, row("b", "It was PR #41, not #14.", days=1))          # renumbering that names the old one
    assert not s.same_numbered_item(a, row("c", "It was PR #41.", days=1))               # blocked: an accepted cost


def test_juniper_words_correct_a_numbered_item_without_restating_it():
    """v3.5: her verified words outrank the number rule, but still need every vote acting with the
    same referent; the strict (model-only) path never bypasses it."""
    a = row("a", "Juniper said the login fix is deployed and linked PR #41.")
    hers = row("b", "Juniper said: no, that login fix is not deployed, I reverted it.", days=1)
    hers = replace(hers, voice="juniper_said", juniper_quoted=True)
    ask, seen = answers(*(vote("correction", kind="event"),) * 3)
    v = asyncio.run(s.classify_pair(a, hers, ask))
    assert not v.acts and v.juniper_acts and len(seen) == 3 and v.abstain == s.NUMBERED_JUNIPER_ONLY
    ask, _ = answers(vote("correction", kind="event"), vote("update"), vote("correction", kind="event"))
    v = asyncio.run(s.classify_pair(a, hers, ask))
    assert v.juniper_acts and v.relation == "correction"      # split votes, majority correction, her path
    two = row("a2", "Juniper said the login fix is in PR #41 and deployed.", days=0)
    cands = s.candidates_for(hers, corpus(a, two, hers), dense={"a": 0.8, "a2": 0.79})
    ask, _ = answers(*(vote("correction", kind="event"),) * 6)
    links, _ = asyncio.run(s.check_pairs(hers, cands, ask))
    assert links == []                                          # two qualify: no clear target
    ask, _ = answers(vote("correction", kind="event", referent="no"))
    v = asyncio.run(s.classify_pair(a, hers, ask))
    assert not v.juniper_acts and v.abstain == "checklist:correction"
    cands = s.candidates_for(hers, corpus(a, hers), dense={"a": 0.8})
    ask, _ = answers(*(vote("correction", kind="event"),) * 3)
    links, _ = asyncio.run(s.check_pairs(hers, cands, ask))
    assert [(l.old_id, l.path) for l in links] == [("a", "juniper")]
    plan = asyncio.run(s.plan_for_new_rows([hers], corpus(a, hers), answers(*(vote("correction", kind="event"),) * 3)[0],
                                           dense=_dense({"a": 0.8}), apply_gate=OPEN))
    assert len(plan["links"]) == 1 and plan["abstained"] == {}     # a linked pair is not counted as abstained


def test_the_plan_counts_why_pairs_abstained():
    a = row("old", "Juniper said the login fix is deployed (PR #41).", refs=("pet:biscuit",))
    ask, seen = answers()
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(a), ask, dense=_dense({"old": 0.75}), apply_gate=OPEN))
    assert plan["abstained"] == {s.NUMBERED_ITEM_NOT_IN_B: 1} and seen == []
    assert plan["decisions"][0]["abstain"] == s.NUMBERED_ITEM_NOT_IN_B


def test_index_dense_retries_once_then_refuses(monkeypatch):
    import orion.introspect.semantic_index as si

    calls = []

    async def flaky_embed(client, cfg, text, *, doc_prefix):
        calls.append(1)
        if len(calls) == 1:
            raise si.SearchUnavailableError("chroma unavailable: ReadTimeout")
        return [0.1], "m"

    async def near(client, cfg, vector, n, *, where=None):
        return [("episode_memory:abc", 0.9)]

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(si, "embed", flaky_embed)
    monkeypatch.setattr(si, "nearest", near)
    monkeypatch.setattr(s.asyncio, "sleep", no_sleep)
    cfg = si.SearchConfig(chroma_url="http://c", embed_url="http://e", collection="x", min_similarity=0.0)
    got = asyncio.run(s.index_dense(cfg)("statement"))
    assert len(calls) == 2 and got is not None

    async def always_down(client, cfg, text, *, doc_prefix):
        calls.append(1)
        raise si.SearchUnavailableError("down")

    calls.clear()
    monkeypatch.setattr(si, "embed", always_down)
    dense = s.index_dense(cfg)
    assert asyncio.run(dense("statement")) is None and len(calls) == 1     # "down" is not transient
    async def timeout_down(client, cfg, text, *, doc_prefix):
        calls.append(1)
        raise si.SearchUnavailableError("chroma unavailable: ReadTimeout")
    calls.clear()
    monkeypatch.setattr(si, "embed", timeout_down)
    dense = s.index_dense(cfg)
    assert asyncio.run(dense("one")) is None and len(calls) == 2           # one retry
    assert asyncio.run(dense("two")) is None and len(calls) == 3           # the plan's retry is spent


def test_plan_records_dense_time_and_retries():
    async def dense(_statement):
        return {"old": 0.75}
    dense.retries = lambda: 1
    ask, _ = answers(*COR3)
    plan = asyncio.run(s.plan_for_new_rows([NEW], corpus(OLD), ask, dense=dense, apply_gate=OPEN))
    assert plan["dense_calls"] == 1 and plan["dense_ms"] >= 0 and plan["dense_retries"] == 1


def test_record_run_derives_abstained_from_decisions_on_every_path():
    class Conn:
        def __init__(self):
            self.params = None

        async def execute(self, sql, params):
            self.params = params

    conn = Conn()
    plan = {"apply": "on", "check_version": s.CHECK_VERSION,
            "decisions": [{"old_id": "a", "new_id": "b", "abstain": "vote:compatible", "linked": False},
                          {"old_id": "c", "new_id": "b", "abstain": "", "linked": True}]}
    asyncio.run(s.record_run(conn, run_id="pass_apply:x", source="pass_apply", plan=plan))
    counts = conn.params[9].obj
    assert counts["abstained"] == {"vote:compatible": 1}
