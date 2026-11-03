"""Turn self-bets (unified-turn spec A3): settle rules, void rules, and the per-caller grade."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import app.turn_self_bets as tsb
from app.turn_self_bets import FACULTIES, SizeHistory, TurnSelfBets, caller_of, grade_faculty, live_summary, settle

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
PRED = {"asks_question_p": 0.1, "fetches_outside_p": 0.9, "steps_lo": 3, "steps_hi": 5,
        "reply_chars_lo": 100, "reply_chars_hi": 400, "coverage": 0.8, "schema_version": "turn.self_prediction.v1"}


def _bet(cid="c1", session="orion_curiosity:abc", age=timedelta(minutes=10), pred=PRED):
    return {"correlation_id": cid, "session_id": session, "created_at": NOW - age, "self_prediction": pred}


REPLY = "Here is what I found. " * 10   # 220 characters, inside the predicted 100-400


def _art(text=REPLY, steps=4, fetches=("https://x",), code=0):
    return {"final_text": text, "step_count": steps, "source_fetches": list(fetches), "exit_code": code}


def _by(rows):
    return {r["faculty"]: r for r in rows}


def test_a_turn_that_went_as_predicted_upholds_every_faculty():
    rows = _by(settle(_bet(), _art(), NOW))
    assert set(rows) == set(FACULTIES)
    # A v1 bet states no relative faculty; the size it would compare is still recorded.
    assert rows["turn.self.more_steps_than_usual"]["void_reason"] == "ungradeable_no_stated"
    assert rows["turn.self.more_steps_than_usual"]["observed"] == 4.0
    assert rows["turn.self.steps"]["status"] == rows["turn.self.reply_length"]["status"] == "upheld"
    # A 10% bet on a question that was not asked is settled, not broken.
    assert rows["turn.self.asks_question"]["status"] == "not_happened"
    assert rows["turn.self.asks_question"]["hit"] is False and rows["turn.self.asks_question"]["stated_p"] == 0.1
    assert rows["turn.self.fetches_outside"]["status"] == "happened"
    assert rows["turn.self.fetches_outside"]["hit"] is True
    assert rows["turn.self.steps"]["observed"] == 4.0
    assert rows["turn.self.reply_length"]["observed"] == float(len(REPLY))
    assert rows["turn.self.steps"]["bet_id"] == "c1:turn.self.steps"


def test_ranges_break_outside_and_probabilities_record_the_event():
    art = _art(text="x" * 50 + " Want me to dig further?", steps=9, fetches=())
    rows = _by(settle(_bet(), art, NOW))
    assert rows["turn.self.steps"]["status"] == "broken"
    assert rows["turn.self.reply_length"]["status"] == "broken"
    # A probability bet is settled, not judged: the event happened or not; Brier grades the p.
    assert rows["turn.self.asks_question"]["hit"] is True
    assert rows["turn.self.fetches_outside"]["hit"] is False


def test_question_mark_only_counts_near_the_end():
    early = "Why? " + "x" * 400
    assert _by(settle(_bet(), _art(text=early), NOW))["turn.self.asks_question"]["hit"] is False


@pytest.mark.parametrize("art", [_art(code=-9), _art(code=1), _art(text=""), _art(text="   "),
                                 {"exit_code": 0, "step_count": 2}])
def test_failed_or_cut_short_turns_are_void_not_misses(art):
    rows = settle(_bet(), art, NOW)
    assert {r["status"] for r in rows} == {"void"}
    assert {r["void_reason"] for r in rows} == {"failed_to_execute"}
    assert all(r["hit"] is None for r in rows)


def test_no_record_waits_then_expires():
    assert settle(_bet(age=timedelta(hours=1)), None, NOW) is None
    rows = settle(_bet(age=timedelta(hours=7)), None, NOW)
    assert {r["void_reason"] for r in rows} == {"expired_unsettled"}


def test_a_missing_stated_value_is_never_filled_in():
    pred = dict(PRED, asks_question_p=None, steps_hi=None)
    rows = _by(settle(_bet(pred=pred), _art(), NOW))
    assert rows["turn.self.asks_question"]["void_reason"] == "ungradeable_no_stated"
    assert rows["turn.self.steps"]["void_reason"] == "ungradeable_no_stated"
    assert rows["turn.self.fetches_outside"]["status"] == "happened"


def test_caller_is_the_session_prefix():
    assert caller_of("orion_world_pulse_read:2026-10-29") == "orion_world_pulse_read"
    assert caller_of("orion_world_pulse_read_stage_7") == "orion_world_pulse_read_stage"
    assert caller_of("self-abc") == "self"
    assert caller_of(None) == "unknown" and caller_of("") == "unknown" and caller_of("___") == "unknown"


@pytest.mark.parametrize("sid", ["f9e8d7c6-1a2b-4c3d-8e9f-0a1b2c3d4e5f", "ab047634-0000-4000-8000-000000000000",
                                 "9c91d646-0000-4000-8000-000000000000"])
def test_a_chat_session_uuid_is_one_caller_not_split_by_its_hex_letters(sid):
    # Review finding: 'f9e8...' became caller 'f', 'ab04...' caller 'ab', '9c91...' caller 'none'.
    assert caller_of(sid) == "chat"


def test_a_cut_short_notice_is_void_not_a_graded_reply():
    # Review finding: exit_code null + the harness's stop notice was graded as a 230-step reply.
    from orion.harness.cut_short import CUT_SHORT_MARKER

    art = _art(text=CUT_SHORT_MARKER + " This turn was stopped after 230 steps.", steps=230, code=None)
    assert {r["void_reason"] for r in settle(_bet(), art, NOW)} == {"failed_to_execute"}


@pytest.mark.parametrize("bet_pred,art", [("not an object", _art()), (PRED, ["not", "an", "object"])])
def test_a_malformed_row_settles_void_instead_of_stalling_every_pass(bet_pred, art):
    rows = settle(_bet(pred=bet_pred), art, NOW)
    assert {r["void_reason"] for r in rows} == {"ungradeable_bad_record"}


def _p_rows(caller, ps_hits):
    return [{"faculty": "turn.self.fetches_outside", "caller": caller, "status": "upheld" if h else "broken",
             "void_reason": None, "stated_p": p, "hit": h} for p, h in ps_hits]


def test_knowing_only_the_caller_earns_no_skill_against_the_caller_base_rate():
    # curiosity fetches 90% of the time, chat 10%; Orion states exactly the caller's rate, every turn.
    rows = (_p_rows("orion_curiosity", [(0.9, True)] * 27 + [(0.9, False)] * 3)
            + _p_rows("chat", [(0.1, True)] * 3 + [(0.1, False)] * 27))
    g = grade_faculty("turn.self.fetches_outside", rows)
    assert g["skill"] == pytest.approx(0.64)       # against one global base rate it looks skilled
    assert g["skill_vs_caller"] == pytest.approx(0.0, abs=1e-9)   # against each caller's own rate: nothing


def test_knowing_which_turns_inside_a_caller_fetch_earns_skill():
    hits = [True, False] * 30
    rows = _p_rows("orion_curiosity", [(0.9 if h else 0.1, h) for h in hits])
    g = grade_faculty("turn.self.fetches_outside", rows)
    assert g["skill_vs_caller"] == pytest.approx(1 - 0.01 / 0.25)


def _r_rows(lo, hi, observed, n=60):
    return [{"faculty": "turn.self.steps", "caller": "self", "status": "upheld" if lo <= x <= hi else "broken",
             "void_reason": None, "stated_lo": lo, "stated_hi": hi, "observed": x, "hit": lo <= x <= hi,
             "coverage": 0.8} for x in observed[:n]]


def test_a_huge_always_right_range_reads_too_wide_and_scores_its_width():
    rows = _r_rows(1, 100, [4] * 60)
    g = grade_faculty("turn.self.steps", rows)
    assert g["realized_coverage"] == 1.0 and g["reads"] == "too_wide"
    assert g["interval_score"] == pytest.approx(99.0)


def test_misses_are_penalised_by_distance():
    rows = _r_rows(3, 5, [3, 4, 5, 9] * 15)
    g = grade_faculty("turn.self.steps", rows)
    assert g["realized_coverage"] == pytest.approx(0.75) and g["reads"] == "calibrated"
    # width 2, plus 2/0.2 * (9 - 5) = 40 on each miss (1 in 4)
    assert g["interval_score"] == pytest.approx(2 + 40 / 4, rel=0.05)


def test_below_the_floor_there_is_a_count_but_no_grade():
    g = grade_faculty("turn.self.steps", _r_rows(3, 5, [4] * 5, n=5))
    assert g == {"n": 5, "enough": False}
    # Same floor for probabilities: grading.grade() reports numbers from 1 row, this does not below 50.
    assert grade_faculty("turn.self.fetches_outside", _p_rows("self", [(0.5, True)] * 49)) == {"n": 49, "enough": False}


def test_void_rows_are_counted_but_never_graded():
    rows = [dict(r, status="happened") for r in _p_rows("self", [(0.5, True)] * 55)]
    rows += [dict(r, status="void", void_reason="failed_to_execute", hit=None) for r in _p_rows("self", [(0.5, False)] * 9)]
    out = live_summary(rows)["faculties"]["turn.self.fetches_outside"]
    assert out["counts"] == {"happened": 55, "failed_to_execute": 9}
    assert out["overall"]["n"] == 55


class FakeStore:
    def __init__(self, pending):
        self.pending = pending
        self.rows: dict[str, dict] = {}

    def pending_turn_self_bets(self, since, limit):
        done = {r["correlation_id"] for r in self.rows.values()}
        return [b for b in self.pending if b["correlation_id"] not in done and b["created_at"] >= since][:limit]

    def insert_turn_self_bets(self, rows):
        new = [r for r in rows if r["bet_id"] not in self.rows]
        self.rows.update({r["bet_id"]: r for r in new})
        return len(new)

    def turn_size_history(self, since):
        return [r for r in self.rows.values() if r["faculty"] in tsb.SIZE_FACULTIES
                and r["observed"] is not None and r["placed_at"] >= since]

    def turn_self_bets(self, since, limit=20000):
        return [r for r in self.rows.values() if r["placed_at"] >= since][:limit]


def test_settle_pass_writes_finished_turns_once_and_leaves_running_ones():
    pending = [dict(_bet("done"), run_artifact=_art()), dict(_bet("running"), run_artifact=None),
               dict(_bet("lost", age=timedelta(hours=8)), run_artifact=None)]
    store = FakeStore(pending)
    svc = TurnSelfBets(store, clock=lambda: NOW)
    out = svc.settle_pending()
    assert out == {"at": NOW, "read": 3, "settled_turns": 2, "waiting": 1, "rows_written": 12}
    assert svc.settle_pending()["rows_written"] == 0     # idempotent: settled turns are not read again
    live = svc.live(force=True)
    assert live["faculties"]["turn.self.steps"]["counts"] == {"upheld": 1, "expired_unsettled": 1}


def test_live_grades_a_rolling_window_so_old_prompt_bets_age_out():
    pending = [dict(_bet("new"), run_artifact=_art()),
               dict(_bet("old", age=tsb.LIVE_WINDOW + timedelta(days=1)), run_artifact=_art())]
    svc = TurnSelfBets(FakeStore(pending), clock=lambda: NOW)
    svc.settle_pending()
    assert svc.live(force=True)["faculties"]["turn.self.steps"]["counts"] == {"upheld": 1}
    everything = svc.live(since=NOW - timedelta(days=30))
    assert everything["faculties"]["turn.self.steps"]["counts"] == {"upheld": 2}


def test_bets_older_than_the_lookback_are_not_read(monkeypatch):
    store = FakeStore([dict(_bet("old", age=tsb.LOOKBACK + timedelta(days=1)), run_artifact=_art())])
    assert TurnSelfBets(store, clock=lambda: NOW).settle_pending()["read"] == 0


PRED_V2 = {"schema_version": "turn.self_prediction.v2", "asks_question_p": 0.1, "fetches_outside_p": 0.9,
           "more_steps_than_usual_p": 0.7, "longer_reply_than_usual_p": 0.2}


def test_a_v2_bet_settles_bigger_than_usual_against_the_reference():
    rows = _by(settle(_bet(pred=PRED_V2), _art(steps=9), NOW,
                      {"turn.self.steps": 8.0, "turn.self.reply_length": float(len(REPLY))}))
    more, longer = rows["turn.self.more_steps_than_usual"], rows["turn.self.longer_reply_than_usual"]
    assert (more["status"], more["hit"], more["stated_p"], more["reference"], more["observed"]) == \
        ("happened", True, 0.7, 8.0, 9.0)
    # Exactly the usual size is not bigger than usual.
    assert (longer["status"], longer["hit"]) == ("not_happened", False)
    # v2 states no range, but the turn's sizes stay on the range rows as reference history.
    assert rows["turn.self.steps"]["void_reason"] == "ungradeable_no_stated"
    assert rows["turn.self.steps"]["observed"] == 9.0


def test_no_reference_is_void_never_guessed():
    rows = _by(settle(_bet(pred=PRED_V2), _art(), NOW))
    assert rows["turn.self.more_steps_than_usual"]["void_reason"] == "ungradeable_no_reference"
    assert rows["turn.self.more_steps_than_usual"]["hit"] is None


def _hist(n, size, caller="orion_curiosity", start=NOW - timedelta(days=1)):
    return [{"caller": caller, "faculty": "turn.self.steps", "placed_at": start + timedelta(minutes=i),
             "observed": size} for i in range(n)]


def test_reference_uses_only_earlier_turns_inside_the_window():
    h = SizeHistory(_hist(20, 10.0) + _hist(20, 1000.0, start=NOW - tsb.REFERENCE_WINDOW - timedelta(days=1))
                    + _hist(20, 1000.0, start=NOW))   # too old, and not yet placed at bet time
    assert h.reference("orion_curiosity", "turn.self.steps", NOW) == 10.0
    assert h.reference("chat", "turn.self.steps", NOW) is None       # per caller
    assert SizeHistory(_hist(19, 10.0)).reference("orion_curiosity", "turn.self.steps", NOW) is None


def test_a_settle_pass_builds_the_reference_from_earlier_turns_never_the_turn_itself():
    # 25 earlier v1 turns of 10 steps, then a v2 turn of 11 steps: its reference is 10, not a median
    # including its own 11, and it is bigger than usual.
    pending = [dict(_bet(f"old{i}", age=timedelta(hours=5, minutes=-i)), run_artifact=_art(steps=10))
               for i in range(25)]
    pending.append(dict(_bet("new", pred=PRED_V2, age=timedelta(minutes=5)), run_artifact=_art(steps=11)))
    store = FakeStore(pending)
    TurnSelfBets(store, clock=lambda: NOW).settle_pending()
    row = store.rows["new:turn.self.more_steps_than_usual"]
    assert (row["reference"], row["hit"], row["status"]) == (10.0, True, "happened")
    assert store.rows["old0:turn.self.more_steps_than_usual"]["void_reason"] == "ungradeable_no_stated"


def test_relative_faculties_are_graded_as_probabilities():
    rows = [{"faculty": "turn.self.more_steps_than_usual", "caller": "self", "status": "happened" if h else "not_happened",
             "void_reason": None, "stated_p": 0.8 if h else 0.2, "hit": h} for h in [True, False] * 30]
    g = grade_faculty("turn.self.more_steps_than_usual", rows)
    assert g["brier"] == pytest.approx(0.04) and g["skill_vs_caller"] == pytest.approx(1 - 0.04 / 0.25)


def test_relative_grade_reports_the_within_caller_sorting_score():
    rows = [{"faculty": "turn.self.longer_reply_than_usual", "caller": c, "status": "happened" if h else "not_happened",
             "void_reason": None, "stated_p": p, "hit": h}
            for c in ("self", "chat") for p, h in [(0.7, True), (0.3, False), (0.6, True), (0.4, True)] * 15]
    g = grade_faculty("turn.self.longer_reply_than_usual", rows)
    assert g["sort_auc"] == pytest.approx(1.0) and g["sort_pairs"] == 2 * 45 * 15 and g["sort_p"] < 1e-6
