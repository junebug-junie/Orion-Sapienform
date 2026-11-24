"""The concern contest's rules (design 2026-11-17 patch 2, acceptance check 3). Pure, no I/O."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.attention import concern_contest as cc

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
DAY = timedelta(hours=24)


def _e(cid, quiet_h, *, sal=None, ttl=DAY, never=None, flight=None, done=False, unseen=False):
    return cc.ContestEntry(concern_id=cid, last_evidence_at=NOW - timedelta(hours=quiet_h), ttl=ttl, salience=sal,
                           never_displaced=never, in_flight=flight, done=done, unseen_quiet=unseen)


def test_score_is_salience_times_the_fade_full_until_half_the_life_then_linear_to_zero():
    assert cc.score(_e("a", 0, sal=0.8), NOW) == 0.8
    assert cc.score(_e("a", 12, sal=0.8), NOW) == 0.8
    assert abs(cc.score(_e("a", 18, sal=0.8), NOW) - 0.4) < 1e-9
    assert cc.score(_e("a", 24, sal=0.8), NOW) == 0.0
    assert cc.score(_e("a", 3), NOW) is None  # unscored, never 0


def test_a_concern_quiet_for_less_than_half_its_life_is_never_displaced_whatever_the_score():
    held = [_e("h", 11.9, sal=0.0)]
    assert not cc.displaceable(held[0], NOW)
    assert cc.displacements(held, [_e("w", 0, sal=1.0)], NOW) == []
    assert cc.displaceable(_e("h", 12), NOW)


def test_an_open_question_or_a_take_on_protects_it_and_a_promise_is_never_displaced():
    assert not cc.displaceable(_e("h", 20, flight="a question I asked about it is open"), NOW)
    assert not cc.displaceable(_e("h", 20, never="I took it on"), NOW)
    assert not cc.displaceable(_e("h", 60, ttl=timedelta(hours=72), never="a promise"), NOW)
    assert cc.displacements([_e("h", 20, never="a promise")], [_e("w", 0, sal=1.0)], NOW) == []


def test_a_quiet_concern_its_holder_has_not_thought_about_since_is_protected():
    # Replay 11-17..11-24 without this: concerns were displaced at exactly 12.0 h, before Prime could act.
    assert not cc.displaceable(_e("h", 20, unseen=True), NOW)
    assert cc.displacements([_e("h", 20, unseen=True)], [_e("w", 0, sal=1.0)], NOW) == []
    # Promise intake ignores it.
    assert cc.intake_victim([_e("h", 20, unseen=True)], NOW).concern_id == "h"


def test_at_most_max_n_displacements_per_tick():
    held = [_e(f"h{i}", 20) for i in range(3)]
    waiting = [_e(f"w{i}", 0) for i in range(3)]
    assert len(cc.displacements(held, waiting, NOW)) == cc.MAX_DISPLACEMENTS_PER_TICK == 2
    assert len(cc.displacements(held, waiting, NOW, max_n=1)) == 1


def test_unscored_ranks_below_scored():
    # A scored candidate beats an unscored incumbent; an unscored candidate never beats a scored one.
    assert cc.beats(_e("w", 0, sal=0.05), _e("h", 13), NOW)
    assert cc.beats(_e("w", 0), _e("h", 13, sal=0.01), NOW) is None
    assert cc.strength(_e("u", 0), NOW) < cc.strength(_e("s", 20, sal=0.0), NOW)


def test_a_tie_keeps_the_incumbent():
    h = _e("h", 13, sal=0.5)
    assert cc.beats(_e("w", 13, sal=0.5), h, NOW) is None
    assert cc.beats(_e("w", 13, sal=0.59), h, NOW) is None  # under the 0.1 margin
    assert cc.beats(_e("w", 13, sal=0.61), h, NOW)
    # Both unscored with the same evidence time: the incumbent stays.
    assert cc.beats(_e("w", 13), _e("h", 13), NOW) is None
    assert cc.beats(_e("w", 1), _e("h", 13), NOW) == "both unscored, newer evidence"


def test_every_displacement_carries_both_scores_and_the_reason():
    [d] = cc.displacements([_e("h", 20, sal=0.2)], [_e("w", 0, sal=0.9)], NOW)
    assert (d.held_id, d.by_id) == ("h", "w")
    assert abs(d.held_score - 0.2 * (1 - 8 / 12)) < 1e-9 and d.by_score == 0.9
    assert "quiet 20.0 h, past half its life (12.0 h)" in d.reason and "score 0.90 >=" in d.reason


def test_the_weakest_displaceable_goes_first():
    held = [_e("old", 20), _e("older", 22), _e("young", 2)]
    [d] = cc.displacements(held, [_e("w", 0)], NOW)
    assert d.held_id == "older"


def test_a_settled_concern_still_held_can_always_yield():
    done = _e("h", 1, done=True)
    assert cc.displaceable(done, NOW) and cc.beats(_e("w", 5), done, NOW) == "its act already closed"


def test_promise_intake_skips_promises_and_take_ons_and_prefers_no_act_in_flight():
    held = [_e("p", 30, ttl=timedelta(hours=72), never="a promise"), _e("asked", 30, flight="q open"), _e("fresh", 1)]
    assert cc.intake_victim(held, NOW).concern_id == "fresh"  # no quiet rule at intake; in flight ranks last
    assert cc.intake_victim(held[:2], NOW).concern_id == "asked"
    assert cc.intake_victim([held[0], _e("t", 30, never="I took it on")], NOW) is None


def test_standings_say_why():
    lines = {s.concern_id: s.why for s in cc.standings(
        [_e("q", 13), _e("y", 2), _e("p", 2, never="a promise"), _e("a", 20, flight="a question I asked about it is open"),
         _e("u", 13, unseen=True)], NOW)}
    assert lines == {"q": "can be displaced: quiet past half its life (12.0 h)", "y": "protected until quiet 12.0 h",
                     "p": "never displaced: a promise", "a": "protected: a question I asked about it is open",
                     "u": "quiet past half its life; it can be displaced after this thought unless I act on it"}
