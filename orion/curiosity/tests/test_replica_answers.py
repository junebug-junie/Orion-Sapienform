"""The curiosity replica's answers to Prime, pure parts (orion/curiosity/replica/answers.py; design section 9)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from orion.curiosity.replica.answers import (
    AnswerRecordV1,
    TargetV1,
    answer_act,
    answer_blocker,
    answer_note,
    answer_produced,
    draw_answer_arm,
    focus_target,
    prime_thought_act_id,
    question_act_id,
    refusal_alarm,
    refusal_rate,
    refusal_rate_line,
    replies_view,
    signal_target,
    stance_from_note,
)
from orion.curiosity.replica.frame import ReplyFrameInputs, build_reply_frame, parse_reply, reply_system_prompt
from orion.grammar.acts import stable_act_id
from orion.prime.acts import nomination_act, thought_act

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
SIG = TargetV1(kind="signal", target_id="sig1", prime_act_id="act_r1", about="slow curiosity: five runs")


def rec(prime_act_id="act_r1", *, stance="refuse", arm="treated", delivered=True, at=NOW, kind="signal"):
    return AnswerRecordV1(answer_act_id=f"ans_{prime_act_id}", prime_act_id=prime_act_id, kind=kind, target_id="t",
                          stance=stance, reason="a reason", arm=arm, delivered=delivered, at=at, about="about")


def test_the_holdback_draw_is_deterministic_near_its_fraction_and_never_holds_a_support():
    ids = [f"act_{i}" for i in range(4000)]
    for stance in ("refuse", "challenge"):
        arms = [draw_answer_arm(i, stance, refusal_fraction=0.2, challenge_fraction=0.2) for i in ids]
        assert arms == [draw_answer_arm(i, stance, refusal_fraction=0.2, challenge_fraction=0.2) for i in ids]
        assert 0.18 < arms.count("control") / len(ids) < 0.22
    assert {draw_answer_arm(i, "support", refusal_fraction=1.0, challenge_fraction=1.0) for i in ids} == {"treated"}
    assert draw_answer_arm("a", "refuse", refusal_fraction=1.0, challenge_fraction=0.0) == "control"
    assert draw_answer_arm("a", "challenge", refusal_fraction=1.0, challenge_fraction=0.0) == "treated"


def test_targets_name_primes_own_acts_and_only_primes_focus_is_one():
    # The ids are Prime's own (orion/prime/acts.py), computed the same way.
    t = focus_target(focus_id="f1", question="Q?", chat_correlation_id="prime:th1")
    assert t.prime_act_id == nomination_act(focus_id="f1", arm="treated", thought_act_id="x", concern_id=None).act_id
    assert t.parent_act_id == thought_act("th1", llm_timeout_sec=1).act_id == prime_thought_act_id("th1")
    assert question_act_id("f1") == stable_act_id("prime", "ask", "f1")
    for corr in (None, "", "chat-123", "prime:"):
        assert focus_target(focus_id="f1", question="Q?", chat_correlation_id=corr) is None
    s = signal_target(SimpleNamespace(signal_id="s9", act_id="act_9", direction="slow", why="busy morning",
                                      expires_at=NOW))
    assert (s.kind, s.prime_act_id, s.about) == ("signal", "act_9", "slow curiosity: busy morning")


def test_one_act_per_prime_act_whatever_the_stance():
    a, b = answer_act(SIG, arm="treated"), answer_act(SIG, arm="control")
    assert a.act_id == b.act_id and a.parent_act_id == "act_r1" and a.actor == "curiosity.replica.answer" and a.verb == "ask"
    assert (a.test_before, b.test_before) == ("holdback:treated", "holdback:control")


def test_the_blocker_rules():
    caps = dict(per_day=3, refractory_sec=1800)
    assert answer_blocker([], SIG, NOW, **caps) is None
    b = answer_blocker([rec()], SIG, NOW + timedelta(hours=5), **caps)
    assert (b.code, b.why, b.final) == ("answered", "I already answered it", True)
    other = SIG.model_copy(update={"prime_act_id": "act_r2"})
    b = answer_blocker([rec(at=NOW - timedelta(minutes=10))], other, NOW, **caps)
    assert b.code == "refractory" and not b.final and "min between answers" in b.why
    three = [rec(f"act_{i}", at=NOW - timedelta(hours=h)) for i, h in enumerate((1, 5, 20))]
    b = answer_blocker(three, other, NOW, **caps)
    assert b.code == "daily_cap" and "at most 3" in b.why and not b.final
    assert answer_blocker(three, other, NOW + timedelta(hours=4, minutes=1), **caps) is None   # oldest is 24 h+ old
    loop = other.model_copy(update={"parent_act_id": "ans_act_r1"})
    assert answer_blocker([rec(at=NOW - timedelta(hours=3))], loop, NOW, **caps).code == "answer_parent"
    mine = TargetV1(kind="signal", target_id="x", prime_act_id="act_x", actor="curiosity.replica")
    assert answer_blocker([], mine, NOW, **caps).code == "own"
    theirs = mine.model_copy(update={"actor": "curiosity.replica.answer"})
    assert answer_blocker([], theirs, NOW, **caps).code == "own"


def test_two_answers_of_one_thought_skip_the_refractory_but_count_against_the_cap():
    """Review finding: one plan answering both the focus and the signal had its second stance dropped."""
    first = rec("act_focus", kind="focus", at=NOW).model_copy(update={"thought_id": "th1"})
    other = SIG.model_copy(update={"prime_act_id": "act_r2"})
    assert answer_blocker([first], other, NOW, per_day=3, refractory_sec=1800, thought_id="th1") is None
    assert answer_blocker([first], other, NOW, per_day=3, refractory_sec=1800, thought_id="th2").code == "refractory"
    assert answer_blocker([first], other, NOW, per_day=1, refractory_sec=1800, thought_id="th1").code == "daily_cap"


def test_prime_hears_only_delivered_challenges_and_refusals_of_the_last_48_h():
    answers = [rec("a1"), rec("a2", stance="challenge"), rec("a3", arm="control"), rec("a4", stance="support"),
               rec("a5", delivered=False), rec("a6", at=NOW - timedelta(hours=49))]
    v = replies_view(answers, NOW)
    assert [r.prime_act_id for r in v.replies] == ["a1", "a2"] and v.as_of == NOW


def test_the_act_note_starts_with_the_stance_and_produced_follows_the_arm():
    for r, produced, tail in ((rec(), True, "delivered"), (rec(delivered=False), False, "not delivered"),
                              (rec(arm="control", delivered=False), True, "recorded, not applied"),
                              (rec(stance="challenge", arm="control", delivered=False), True, "not shown to Prime"),
                              (rec(stance="support"), True, "recorded")):
        note = answer_note(r)
        assert stance_from_note(note) == r.stance and tail in note and answer_produced(r) is produced
    assert stance_from_note("plan for run x | ...") is None and stance_from_note(None) is None


def test_the_refusal_rate_covers_the_last_20_answers_and_flags_over_half():
    answers = [rec(f"x{i}", stance="refuse" if i % 3 else "support", at=NOW - timedelta(hours=30 - i)) for i in range(30)]
    assert refusal_rate(answers) == (sum(1 for i in range(10, 30) if i % 3), 20)
    assert "above 50%" in refusal_rate_line(answers)
    assert refusal_rate_line([]) == "your refusal rate: no answers to Prime yet"
    assert "above" not in refusal_rate_line([rec(stance="challenge"), rec("b", stance="refuse")])


def test_the_14_day_alarm_needs_14_known_days_all_over_half():
    days = [(f"d{i}", 0.6) for i in range(14)]
    assert refusal_alarm(days)
    assert not refusal_alarm(days[:13])
    assert not refusal_alarm([*days[:13], ("d13", 0.5)])
    assert not refusal_alarm([*days[:13], ("d13", None)])     # unknown is never an alarm


def test_the_reply_frame_and_prompt():
    inp = ReplyFrameInputs(now=NOW, target=SIG.model_copy(update={"direction": "slow", "expires_at": NOW + timedelta(hours=3)}),
                           runs_error="hub down", concerns_error="Prime has published no concerns view",
                           digest_error="pg down", answers=[rec(at=NOW - timedelta(hours=2))])
    frame = build_reply_frame(inp)
    assert "WHAT PRIME DID: sent a slow signal to curiosity (signal_id sig1, until 15:00 UTC)" in frame
    assert "(unavailable: hub down)" in frame and "(unavailable: pg down)" in frame
    assert "1 of your last 1 answers to Prime were refusals" in frame
    off = build_reply_frame(ReplyFrameInputs(now=NOW, target=SIG, answers_on=False))
    assert "answering Prime is switched off" in off
    text = reply_system_prompt(per_day=3, refractory_min=30)
    assert "never refused" in text and "at most 3 times a day" in text
    ans, err = parse_reply({"stance": "refuse", "reason": "r", "extra": 1})
    assert err is None and ans.stance == "refuse"
    assert parse_reply({"stance": "veto", "reason": "r"})[1].startswith("invalid_answer")
    assert parse_reply({"stance": "refuse", "reason": ""})[1].startswith("invalid_answer")


def test_answer_acts_stay_out_of_the_replica_familys_history_in_the_body_economy():
    """The answer actor is never a want family: no families.yaml entry, and Appetite, the learned weights and
    Choice's Tier 1 lane use all skip its acts, so delivered answers (graded 1) cannot lift curiosity.replica."""
    from pathlib import Path

    from orion.physiology.appetite import ActRecord, build_family_appetite
    from orion.physiology.choice import build_context
    from orion.physiology.families import load_families
    from orion.physiology.weights import weights_now
    from orion.schemas.curiosity_replica import ANSWER_ACTOR, REPLICA_FAMILY

    fams, err = load_families(Path(__file__).resolve().parents[3] / "config" / "physiology" / "families.yaml")
    assert err is None and ANSWER_ACTOR not in fams.families and REPLICA_FAMILY in fams.families
    plans = [ActRecord(act_id=f"p{i}", family=REPLICA_FAMILY, opened_at=NOW - timedelta(hours=10 - i),
                       closed_at=NOW - timedelta(hours=10 - i, minutes=-1), outcome_value=0.0, outcome_source="producer",
                       signal_id="artifact_produced") for i in range(4)]
    answers = [ActRecord(act_id=f"a{i}", family=ANSWER_ACTOR, opened_at=NOW - timedelta(hours=1, minutes=i),
                         closed_at=NOW - timedelta(hours=1, minutes=i), outcome_value=1.0, outcome_source="producer",
                         signal_id="artifact_produced") for i in range(6)]
    cfg = fams.families[REPLICA_FAMILY]
    assert build_family_appetite(cfg, plans + answers, NOW) == build_family_appetite(cfg, plans, NOW)
    assert weights_now(fams, plans + answers, [], NOW)[REPLICA_FAMILY] == weights_now(fams, plans, [], NOW)[REPLICA_FAMILY]
    ctx = build_context(fams, plans + answers, [], None, NOW)
    assert ANSWER_ACTOR not in ctx.appetite and ANSWER_ACTOR not in ctx.weights
