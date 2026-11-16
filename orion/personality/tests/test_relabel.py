"""#3444 relabel planner: rebuild a choice node's label from its stored choices with the current rules. Pure."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.personality.graph import DEFAULT_PARAMS, choice_label, pair_label
from orion.personality.relabel import Member, approved, choice_sides, plan_relabel, replay_label

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)


def ev(did, eid, sim, chosen, declined, *, lane="curiosity", meta=None, at=0, created=0, declined_all=False,
       asyncpg=False):
    alts = [{"key": f"k{i}", "text": t, "market_rank": i + 1} for i, t in enumerate(declined)]
    ch = None if declined_all else {"key": "kc", "text": chosen, "market_rank": None}
    row = {"disposition_id": did, "evidence_id": f"choice:{eid}", "occurred_at": T0 + timedelta(minutes=at),
           "created_at": T0 + timedelta(hours=created), "similarity": sim, "meta": meta or {}, "excerpt": "stale",
           "choice_id": eid, "lane": lane, "declined_all": declined_all, "chosen": ch, "alternatives": alts}
    if asyncpg:  # asyncpg hands jsonb back as text
        row = {**row, "meta": json.dumps(row["meta"]), "chosen": json.dumps(ch), "alternatives": json.dumps(alts)}
    return row


def node(did, label, *, n=1, opposed=0, sim=0.5):
    return {"disposition_id": did, "status": "candidate", "label": label, "label_similarity": sim,
            "evidence_count": n, "opposed_count": opposed}


def test_old_more_tail_label_is_rebuilt_from_the_founding_choice():
    declined = [f"option {i}" for i in range(15)]
    [r] = plan_relabel([node("d1", "peer briefs refused budget over curiosity dies as… (+14 more), 1 of 1")],
                       [ev("d1", "c1", 1.0, "peer briefs refused budget", declined)])
    assert r.changed and r.new_label == "peer briefs refused budget, 1 of 1"
    assert r.new_similarity == DEFAULT_PARAMS.match_threshold and r.anomalies == ()


def test_replay_follows_reinforce_closer_match_takes_the_label_and_echo_never_does():
    rows = [ev("d", "a", 1.0, "first subject", ["x"], created=0),
            ev("d", "b", 0.7, "closer subject", ["y"], created=1),
            ev("d", "c", 0.9, "own name echo", ["z"], created=2, meta={"cites_own_name": True}),
            ev("d", "e", 0.6, "weaker subject", ["w"], created=3),
            ev("d", "f", 0.95, "said no", ["v"], created=4, meta={"juniper_no": True})]
    [r] = plan_relabel([node("d", "old, 5 of 6", n=5, opposed=1, sim=0.7)], rows)
    assert r.new_label == choice_label(pair_label("closer subject", ["y"]), 5, 6)
    assert r.new_similarity == 0.7 and r.base_evidence_id == "choice:b"


def test_category_only_base_gives_way_and_never_displaces_a_subject():
    a = Member("e1", 1.0, "a promise I made")
    b = Member("e2", 0.55, "walkway logs over budget")  # weaker, but names a subject: takes a category base
    c = Member("e3", 0.99, "a thread from my letter")  # closer, but category-only: never displaces a subject
    label, sim, base = replay_label([a, b, c], evidence_count=3, opposed_count=0)
    assert label == "walkway logs over budget, 3 of 3" and base == "e2" and sim == 0.55


def test_fold_order_is_commit_time_first_and_asyncpg_text_jsonb_reads_the_same():
    psy = [ev("d", "a", 1.0, "alpha", ["x"]), ev("d", "b", 0.8, "beta", ["y"], created=1)]
    apg = [ev("d", "a", 1.0, "alpha", ["x"], asyncpg=True), ev("d", "b", 0.8, "beta", ["y"], created=1, asyncpg=True)]
    n = [node("d", "z, 2 of 2", n=2)]
    assert plan_relabel(n, psy) == plan_relabel(n, apg)
    assert plan_relabel(n, psy)[0].new_label == "beta over y, 2 of 2"


def test_old_prime_category_text_is_remade_subject_first():
    rows = [ev("d", "p", 1.0, "a promise I made in a curiosity run (1 promise, harness_turn_trace:ab#promises_made[0])",
               ["a thread from my letter for 2026-11-13: **The walkway render break is unresolved.** more"],
               lane="prime")]
    [r] = plan_relabel([node("d", "a promise I made over a thread from my letter, 1 of 1")], rows)
    assert r.new_label == "Curiosity run promise over The walkway render break is unresolved, 1 of 1"


def test_declined_all_uses_the_lane_decline_phrase_and_missing_choice_keeps_stored_excerpt():
    assert choice_sides("prime", True, None, [("b", 2), ("a", 1), ("u", None)]) == ("no move", ["a", "b", "u"])
    row = ev("d", "a", 1.0, "", ["x"])
    row.update(choice_id=None, excerpt="kept words")
    [r] = plan_relabel([node("d", "old, 1 of 1")], [row])
    assert r.new_label == "kept words, 1 of 1" and "choice_rows_missing:1" in r.anomalies
    [none] = plan_relabel([node("q", "lonely, 1 of 1")], [])
    assert not none.changed and none.anomalies == ("no_evidence",)


def test_approved_writes_only_reviewed_exact_pairs():
    plan = plan_relabel([node("d1", "old one, 1 of 1"), node("d2", "old two, 1 of 1")],
                        [ev("d1", "a", 1.0, "new one", ["x", "y", "z"]), ev("d2", "b", 1.0, "new two", ["x", "y", "z"])])
    assert [r.disposition_id for r in approved(plan, None)[0]] == ["d1", "d2"]
    ok, refused = approved(plan, {"d1": ["old one, 1 of 1", "new one, 1 of 1"], "d2": ["old two, 1 of 1", "other"],
                                  "gone": ["a", "b"]})
    assert [r.disposition_id for r in ok] == ["d1"] and refused == ["d2", "gone"]
