"""The promise ledger report (pure parts): close kinds, un-closeable promises, the guard, the blinded sample."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.grammar.promise_acts import (VOID_ABSENCE_NOTE, VOID_NOTE, VOID_NOT_CHAT_NOTE, VOID_NOT_DUE_NOTE,
                                       promise_ref)
from orion.prime.handback import turn_concern_id

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("report_promise_ledger", REPO / "scripts/report_promise_ledger.py")
rpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpt)

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
H = timedelta(hours=72)


def _act(i, *, opened, closed=None, source=None, note=None, corr=None):
    return {"act_id": f"act_{i}", "test_before": promise_ref(corr or f"c{i}", 0), "opened_at": opened,
            "horizon_at": opened + H, "closed_at": closed, "outcome_source": source, "outcome_note": note}


def test_close_summary_counts_each_kind_and_leaves_void_and_unclosable_out_of_the_keep_rate():
    acts = [
        _act(1, opened=T0, closed=T0 + timedelta(hours=5), source="producer", note="kept_in:cx evidence=reply"),
        _act(2, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon"),
        _act(3, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE),
        _act(4, opened=T0),
    ]
    out = rpt.close_summary(acts, unclosable=set())
    assert (out["kept"], out["expired"], out["void"], out["open"]) == (1, 1, 1, 1)
    assert out["kept_by_evidence"] == {"reply": 1}
    assert out["keep_rate"]["n"] == 2 and out["keep_rate"]["mean"] == 0.5
    moved = rpt.close_summary(acts, unclosable={"act_2"})
    assert moved["expired"] == 0 and moved["unclosable"] == 1 and moved["keep_rate"]["n"] == 1


def test_an_expired_promise_never_in_the_judges_view_is_unclosable():
    # view of 1: at every Juniper turn the older act_0 fills the view, so act_1 is never shown.
    acts = [_act(0, opened=T0), _act(1, opened=T0 + timedelta(hours=1), closed=T0 + H + timedelta(hours=1),
                                     source="expired", note="no keep within horizon")]
    starts = [T0 + timedelta(hours=10), T0 + timedelta(hours=30)]
    assert rpt.unclosable_ids(acts, starts, view_max=1) == {"act_1"}
    assert rpt.unclosable_ids(acts, starts, view_max=2) == set()


def test_expired_promise_with_a_carry_reply_is_named():
    acts = [_act(1, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon", corr="c-carry")]
    carry = {turn_concern_id("c-carry"): [T0 + timedelta(hours=20)]}
    out = rpt.close_summary(acts, unclosable=set(), carry_concerns=carry)
    assert out["expired_with_carry_reply"] == 1 and out["expired_with_carry_reply_acts"] == ["act_1"]


def test_guard_baseline_is_frozen_for_shadow():
    # Orion's PR #3561 question 3: the baseline is set before shadow data and never re-based on it.
    # Changing this pin is a Juniper decision recorded in #3531, not a quiet edit.
    assert rpt.REPLAY_BASELINE == (23, 289) and rpt.GUARD_RATIO_RED == 2.0


def test_guard_prints_an_interval_beside_the_ratio():
    g = rpt.guard_summary([_turn(str(i), [FIRM] if i < 2 else []) for i in range(20)])
    lo, hi = g["rate_poisson90"]
    assert g["rate"] == 0.1 and lo < 0.1 < hi
    assert rpt.guard_summary([])["rate_poisson90"] is None


def test_guard_leaves_out_turns_before_the_finder_went_live():
    early = {**_turn("old", [{"quote": "old row", "offer": False, "doer": "orion"}]),
             "created_at": rpt.FINDER_LIVE_AT - timedelta(hours=1)}
    late = {**_turn("new", [FIRM]), "created_at": rpt.FINDER_LIVE_AT + timedelta(hours=1)}
    g = rpt.guard_summary([early, late])
    assert (g["judged_turns"], g["turns_before_finder"], g["firm_promises"]) == (1, 1, 1)
    assert g["skipped_pre_force"] == 1 and g["turns_not_judged"] == 0


def test_guard_interval_handles_more_promises_than_turns():
    # A turn can hold several promises: the rate is a count rate and can exceed 1 (Wilson crashed here).
    g = rpt.guard_summary([_turn("a", [FIRM, {**FIRM, "quote": "I'll also check the other run tonight."}])])
    lo, hi = g["rate_poisson90"]
    assert g["rate"] == 2.0 and lo < 2.0 < hi


def test_text_report_shows_raw_counts_beside_the_keep_rate(monkeypatch, capsys):
    acts = [
        _act(1, opened=T0, closed=T0 + timedelta(hours=5), source="producer", note="kept_in:cx evidence=reply"),
        _act(2, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon"),
        _act(3, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE),
    ]
    acts += [_act(10 + i, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon",
                  corr=f"c-carry{i}") for i in range(3)]
    carry = {turn_concern_id(f"c-carry{i}"): [T0 + timedelta(hours=20)] for i in range(3)}
    monkeypatch.setattr(rpt, "gather", lambda dsn, since: {"acts": acts, "starts": [T0 + timedelta(hours=1)],
                                                           "turns": [], "carry": carry})
    assert rpt.main(["--dsn", "x"]) == 0
    out = capsys.readouterr().out
    assert "keep rate (Beta(1,1)) over kept 1 + expired 4 (void 1 and un-closeable 0 left out)" in out
    assert "expired with a carry reply: 3 (act_10, act_11, act_12)" in out


def test_carry_act_ids_are_capped_in_the_text_report():
    assert rpt._ids([f"act_{i}" for i in range(12)]) == " (" + ", ".join(f"act_{i}" for i in range(10)) + " +2 more)"
    assert rpt._ids([]) == ""


def _turn(corr, made, source="substrate_informed_pass", text="Some reply."):
    return {"correlation_id": corr, "reflection_source": source, "made": made, "final_text": text}


FIRM = {"quote": "I'll compare the two runs tomorrow.", "offer": False, "doer": "orion", "force": "commits",
        "act": "compare", "done_when": "a later reply names which run drifted"}


def test_guard_counts_judged_turns_only_and_skips_pre_force_items():
    turns = [_turn("a", [FIRM]), _turn("b", [{**FIRM, "offer": True}]),
             _turn("c", [{"quote": "old row", "offer": False, "doer": "orion"}]),
             _turn("d", [FIRM], source="deterministic_quick_gate"), _turn("e", [])]
    g = rpt.guard_summary(turns, baseline=(1, 10), min_judged=1)
    assert g["judged_turns"] == 4 and g["turns_not_judged"] == 1
    assert (g["firm_promises"], g["offers"], g["skipped_pre_force"]) == (1, 1, 1)
    assert g["rate"] == 0.25 and g["ratio"] == 2.5 and g["red"] is True


def test_blinded_sample_hides_the_judges_reading_and_scores_back():
    text = "Lots of context before. I'll compare the two runs tomorrow. And after."
    sheet = rpt.blinded_sample([_turn("a", [FIRM], text=text)], 5)
    (item,) = sheet
    assert set(item) == {"key", "quote", "juniper_said", "reply_context", "label", "why"}
    assert "compare the two runs" in item["reply_context"]
    blob = json.dumps(sheet)
    for leaked in ("force", "doer", "done_when", "commits", "offer"):
        assert leaked not in blob
    item["label"] = True
    scored = rpt.score_labels(sheet + [{"label": False}, {"label": None}])
    assert scored["labelled"] == 2 and scored["true"] == 1 and scored["precision"] == 0.5
    lo, hi = scored["wilson95"]
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_wilson_matches_the_pr_3532_numbers():
    lo, hi = rpt.wilson(36, 43)
    assert (round(lo, 2), round(hi, 2)) == (0.70, 0.92)


# --- Orion's second review round (PR #3561 / #3598 / #3532 threads, 2026-11-17) ---------------------


def test_guard_cannot_go_red_before_it_is_armed():
    # 0c1e question 3: a high ratio on a handful of turns is noise, not a red guard.
    turns = [_turn(str(i), [FIRM] if i < 3 else []) for i in range(10)]
    g = rpt.guard_summary(turns)
    assert g["ratio"] > rpt.GUARD_RATIO_RED and g["armed"] is False and g["red"] is False
    assert rpt.GUARD_MIN_JUDGED_TURNS == 60


def test_guard_red_boundary_at_the_arming_point():
    # At 60 turns and the frozen 23/289, red (ratio > 2) needs 10 firm promises: 9 is not red.
    n = rpt.GUARD_MIN_JUDGED_TURNS
    for firm, red in ((9, False), (10, True)):
        g = rpt.guard_summary([_turn(str(i), [FIRM] if i < firm else []) for i in range(n)])
        assert g["armed"] is True and g["red"] is red, (firm, g["ratio"])


def test_guard_leaves_out_turns_not_sent_from_a_hub_browser_surface():
    # 0c1e question 2 / #3602: a coding session's message posted over the Hub API is not Juniper's turn.
    session = {**_turn("agent", [FIRM, {**FIRM, "quote": "I'll read the three proposals closely."}]), "from_browser": False}
    g = rpt.guard_summary([_turn("j", []), session], min_judged=1)
    assert (g["judged_turns"], g["turns_not_from_browser"], g["firm_promises"], g["turns_not_judged"]) == (1, 1, 0, 0)


def _sturn(corr, made, *, session="s1", at=T0, accepted=None, prev=None, prev_is_juniper=True):
    return {**_turn(corr, made), "session_id": session, "created_at": at, "accepted": accepted or [],
            "prev_correlation_id": prev, "prev_is_juniper": prev_is_juniper if prev else None}


def test_duplicate_candidates_pair_reworded_repeats_in_one_session_only():
    a = {**FIRM, "quote": "I'll compare the two runs tomorrow morning and tell you which drifted."}
    b = {**FIRM, "quote": "Tomorrow I'll compare the two runs and tell you which drifted.", "act": "other"}
    turns = [_sturn("c1", [a], at=T0), _sturn("c2", [b], at=T0 + timedelta(hours=5)),
             _sturn("c3", [b], session="s2", at=T0 + timedelta(hours=6)),
             _sturn("c4", [{**FIRM, "quote": "Different promise entirely, about the camera.",
                            "act": "x", "done_when": "y happened later on"}], at=T0 + timedelta(hours=7)),
             _sturn("c5", [a], at=T0 + timedelta(hours=80))]
    pairs = rpt.duplicate_candidates(turns)
    assert [(p["a"], p["b"]) for p in pairs] == [("c1#0", "c2#0")]  # c5 is 75 h after c2: never open together
    assert "same_done_when" in pairs[0]["why"] and any(w.startswith("shared_span_") for w in pairs[0]["why"])


def test_offers_summary_counts_accepted_and_unmatched_against_the_previous_turn():
    offer = {**FIRM, "quote": "If you want, I'll queue the read tonight.", "offer": True}
    turns = [_sturn("c1", [offer], at=T0),
             _sturn("c2", [], at=T0 + timedelta(hours=1), prev="c1",
                    accepted=["If you want, I\u2019ll queue the read tonight."]),
             _sturn("c3", [], at=T0 + timedelta(hours=2), prev="c2", accepted=["Something never offered here."]),
             _sturn("c4", [], session="s9", at=T0, prev="before-window", accepted=["No earlier turn in the window."]),
             # Hub's rule: the previous row was Orion's outreach, so the judge was shown no offers.
             _sturn("c5", [], at=T0 + timedelta(hours=3), prev="outreach-row", prev_is_juniper=False,
                    accepted=["If you want, I'll queue the read tonight."])]
    out = rpt.offers_summary(turns)
    assert (out["accepted"], out["accepted_unmatched"], out["accepted_no_prior"],
            out["accepted_prev_not_juniper"]) == (4, 1, 1, 1)
    assert out["accepted_unmatched_turns"] == ["c3"]


def test_blinded_sample_shows_the_message_juniper_sent():
    sheet = rpt.blinded_sample([{**_turn("a", [FIRM], text="I'll compare the two runs tomorrow."),
                                 "juniper_said": "can you   check the runs?"}], 1)
    assert sheet[0]["juniper_said"] == "can you check the runs?"


def test_text_report_names_the_keep_rate_ceiling_and_the_new_lines(monkeypatch, capsys):
    monkeypatch.setattr(rpt, "gather", lambda dsn, since: {"acts": [], "starts": [], "turns": [], "carry": {}})
    assert rpt.main(["--dsn", "x"]) == 0
    out = capsys.readouterr().out
    assert "ceiling (chat-showable share) 0.39 (9/23" in out
    assert "not armed: red needs 60 judged turns" in out
    assert "offers accepted: 0; accepted_unmatched 0" in out
    assert "duplicate candidates" in out


def test_duplicate_candidates_skip_items_with_no_time():
    a = {**FIRM, "quote": "I'll compare the two runs tomorrow morning and tell you which drifted."}
    turns = [{**_sturn("c1", [a]), "created_at": None}, _sturn("c2", [a], at=T0)]
    assert rpt.duplicate_candidates(turns) == []


def test_report_leaves_session_turns_out_of_offers_and_duplicates(monkeypatch):
    a = {**FIRM, "quote": "I'll compare the two runs tomorrow morning and tell you which drifted."}
    at = rpt.FINDER_LIVE_AT + timedelta(hours=1)
    turns = [_sturn("c1", [a], at=at), {**_sturn("c2", [a], at=at + timedelta(hours=1)), "from_browser": False}]
    out = rpt.report({"acts": [], "starts": [], "turns": turns, "carry": {}})
    assert out["duplicate_candidates"] == []
    turns[1]["from_browser"] = True
    assert len(rpt.report({"acts": [], "starts": [], "turns": turns, "carry": {}})["duplicate_candidates"]) == 1


def test_judged_sql_qualifies_both_predicates_by_alias():
    sql = rpt.JUDGED_SQL
    assert "k.stage = 'ingress' AND k.seq = 0 AND k.raw->'surface_context' IS NOT NULL" in sql
    assert "btrim(c.prompt)" in sql and "c.client_meta->>'unsolicited'" in sql
    assert "btrim(p.prompt)" in sql and "p.client_meta->>'unsolicited'" in sql
    import pytest
    with pytest.raises(ValueError):
        rpt.qualify("stage = 'x'", "k", ("seq",))
    assert rpt.qualify("raw = 'raw seq'", "k", ("raw",)) == "k.raw = 'raw seq'"


def test_baseline_and_ceiling_print_on_the_clean_basis_too(monkeypatch, capsys):
    # Orion's PR #3561 third-round question 2: the 2 coding-session turns out, beside the frozen numbers.
    assert rpt.REPLAY_BASELINE_CLEAN == (21, 287) and rpt.KEEP_RATE_CEILING_CLEAN == (9, 21)
    g = rpt.guard_summary([_turn("j", [FIRM])], min_judged=1)
    assert (g["baseline_rate"], g["baseline_clean_rate"]) == (0.0796, 0.0732)
    assert g["ratio_clean"] == round(1 / (21 / 287), 2) and g["red"] is True  # red still on the frozen basis
    monkeypatch.setattr(rpt, "gather", lambda dsn, since: {"acts": [], "starts": [], "turns": [], "carry": {}})
    assert rpt.main(["--dsn", "x"]) == 0
    out = capsys.readouterr().out
    assert "clean 0.43 9/21" in out and "[clean baseline 0.0732 (21/287 replay" in out


def test_duplicate_window_pairs_counts_every_pair_the_rule_compares():
    # Orion's PR #3598 third-round question 5: the number of same-session pairs within 72 h, before the
    # similarity test. The live example (780e83f2 / 295b0ef5, 2 minutes apart) is one such pair, and the
    # shipped rule does not flag it: its longest shared run is 11 characters and act/done_when differ.
    float_up = {**FIRM, "quote": "I'll float it back up when you've had some sleep — or whenever you poke me first.",
                "act": "float the ticket back up",
                "done_when": "Orion brings up the GitHub ticket in a subsequent turn after Juniper has rested"}
    abstain = {**FIRM, "quote": "I won't float anything back up while you're asleep",
               "act": "not resurface open loops while user sleeps",
               "done_when": "no new messages regarding open loops are sent during the user's sleep period"}
    turns = [_sturn("c1", [float_up], at=T0), _sturn("c2", [abstain], at=T0 + timedelta(minutes=2)),
             _sturn("c3", [FIRM], session="s2", at=T0), _sturn("c4", [FIRM], at=T0 + timedelta(hours=80))]
    assert rpt.duplicate_window_pairs(turns) == 1
    assert rpt.duplicate_candidates(turns) == []
    out = rpt.report({"acts": [], "starts": [], "turns": [{**t, "created_at": rpt.FINDER_LIVE_AT + (t["created_at"] - T0)}
                                                          for t in turns], "carry": {}})
    assert out["duplicate_window_pairs"] == 1


# --- Orion's PR #3641 round (thread 0a4514841253): lanes, browser window, Prime by exact key ----------------

import pytest  # noqa: E402

LIVE = rpt.FINDER_LIVE_AT + timedelta(hours=1)


@pytest.fixture(autouse=True)
def _no_live_redis(monkeypatch):
    # main() reads Prime from ORION_BUS_URL when the redis package is installed: keep these tests offline.
    monkeypatch.delenv("ORION_BUS_URL", raising=False)


def _stored(corr, made, *, browser=False, session="", workflow="chat", at=LIVE, mode=False):
    return {"correlation_id": corr, "created_at": at, "workflow": workflow, "made": made,
            "session_id": session, "from_browser": browser, "has_mode": mode}


OFFER = {**FIRM, "quote": "Want me to pull the other run too?", "offer": True}


def test_lane_split_names_the_browser_lane_and_folds_round_ups():
    stored = [_stored("j", [FIRM], browser=True, session="9c91"),
              _stored("e1", [OFFER, FIRM], session="memory-scoreboard-eval"),
              _stored("e2", [OFFER], session="memory-scoreboard-eval"),
              _stored("r1", [FIRM], session="orion_session-roundup-20261117T1800"),
              _stored("r2", [FIRM], session="orion_session-roundup-20261117T1900"),
              _stored("w", [FIRM], session="", workflow="curiosity_investigation"),
              _stored("none", [])]
    out = rpt.lane_split(stored)
    assert (out["turns"], out["items"], out["browser_turns"]) == (6, 7, 1)
    assert out["by_lane"]["memory-scoreboard-eval"] == {"turns": 2, "items": 3, "firm": 1}
    assert out["by_lane"]["orion_session-roundup"]["turns"] == 2
    assert out["by_lane"]["juniper_browser"]["firm"] == 1
    assert "curiosity_investigation" in out["by_lane"]


def test_browser_window_reports_the_last_browser_turn_and_the_gap_after_it():
    now = LIVE + timedelta(hours=10)
    turns = [{**_turn("b1-aaaaaaaa", []), "created_at": LIVE, "from_browser": True},
             {**_turn("b2-bbbbbbbb", [], source="deterministic_quick_gate"),
              "created_at": LIVE + timedelta(hours=4), "from_browser": True},
             {**_turn("api", []), "created_at": LIVE + timedelta(hours=8), "from_browser": False},
             {**_turn("old", []), "created_at": rpt.FINDER_LIVE_AT - timedelta(hours=1), "from_browser": True}]
    bw = rpt.browser_window(turns, now)
    assert (bw["browser_turns"], bw["judged"], bw["last_corr"], bw["hours_since_last"]) == (2, 1, "b2-bbbbb", 6.0)
    assert rpt.browser_window([], now)["last_at"] is None


def test_guard_prints_expected_beside_observed():
    g = rpt.guard_summary([{**_turn(str(i), []), "created_at": LIVE} for i in range(6)])
    assert g["expected_firm"] == round(6 * 23 / 289, 2) and g["firm_promises"] == 0
    # Live 2026-11-18: 0 of 6 judged browser turns; at the frozen rate that happens about 62% of the time.
    assert g["p_at_most_observed"] == round(rpt.poisson_cdf(0, 6 * 23 / 289), 3) == 0.62
    assert g["p_at_least_observed"] == 1.0
    hi = rpt.guard_summary([{**_turn(str(i), [FIRM] if i < 3 else []), "created_at": LIVE} for i in range(6)])
    assert hi["p_at_least_observed"] == round(1 - rpt.poisson_cdf(2, 6 * 23 / 289), 3) < 0.02
    assert rpt.guard_summary([])["p_at_most_observed"] is None
    assert rpt.poisson_cdf(3, 0.0) == 1.0 and abs(rpt.poisson_cdf(1, 2.0) - 3 * 2.718281828 ** -2) < 1e-6


def test_void_share_is_void_over_every_closed_act():
    acts = [_act(1, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon"),
            _act(2, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE),
            _act(3, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE),
            _act(4, opened=T0)]
    assert rpt.close_summary(acts, unclosable=set())["void_share"] == round(2 / 3, 3)
    assert rpt.close_summary([_act(5, opened=T0)], unclosable=set())["void_share"] is None


def test_prime_by_ref_joins_on_the_exact_turn_concern_id():
    # Prime's classify_who: a non-orion_ session sent from a Hub browser surface (#3602).
    chat = {"session": "9c91", "mode": True, "browser": True}
    stored = [_stored("open-c", [FIRM], **chat), _stored("closed-c", [FIRM], session="orion_curiosity"),
              _stored("refused-c", [FIRM], **chat), _stored("none-c", [FIRM], **chat),
              _stored("unread-c", [FIRM], session="self-sense-eval"),  # no mode: never handed back
              _stored("offer-only", [OFFER], **chat),
              _stored("no-quote", [{**FIRM, "quote": "  "}], **chat)]  # Prime counts no empty quote
    state = {"concerns": [{"concern_id": turn_concern_id("open-c")}],
             "concern_record": [{"concern_id": turn_concern_id("closed-c"), "end": "settled"}]}
    turns = {"recent": [{"correlation_id": "refused-c", "thread_refused": "already holding 3 concerns"}]}
    # A carry on the open turn's key counts; one on a different turn's key in the same session does not.
    carry = {turn_concern_id("open-c"): [LIVE], turn_concern_id("elsewhere"): [LIVE, LIVE]}
    out = rpt.prime_by_ref(stored, carry, state, turns)
    assert out["turns"] == 5  # offer-only and empty-quote turns are not promises Prime counts
    assert out["status"] == {"concern_open": 1, "concern_settled": 1, "refused": 1, "no_concern_or_aged_out": 1,
                             "not_handed_back": 1}
    assert {r["corr"]: r["prime_who"] for r in out["rows"]}["closed-c"] == "curiosity"
    assert out["carry_replies"] == 1
    assert {r["corr"]: r["refused_why"] for r in out["rows"]}["refused-"] == "already holding 3 concerns"
    assert rpt.prime_by_ref(stored, carry, None, None)["status"] == {"not_read": 4, "not_handed_back": 1}


def test_read_prime_from_files_and_never_raises(tmp_path):
    state, turns = tmp_path / "s.json", tmp_path / "t.json"
    state.write_text(json.dumps({"concerns": []}))
    turns.write_text(json.dumps({"recent": []}))
    got_state, got_turns, how = rpt.read_prime(None, str(state), str(turns))
    assert got_state == {"concerns": []} and got_turns == {"recent": []} and how.startswith("files")
    assert rpt.read_prime(None, str(tmp_path / "missing.json"))[2].startswith("not read")
    assert rpt.read_prime(None)[2].startswith("not read")


def test_text_report_prints_the_window_lanes_expected_and_prime_lines(monkeypatch, capsys):
    stored = [_stored("e1-xxxxxxxx", [FIRM], session="memory-scoreboard-eval", mode=True)]
    turns = [{**_turn("b1-aaaaaaaa", []), "created_at": LIVE, "from_browser": True, "session_id": "s"}]
    monkeypatch.setattr(rpt, "gather", lambda dsn, since: {"acts": [], "starts": [], "turns": turns, "carry": {},
                                                           "stored": stored})
    assert rpt.main(["--dsn", "x"]) == 0
    out = capsys.readouterr().out
    assert "expected vs observed: 0.08 firm promises expected over 1 judged browser turns" in out
    assert "browser window: 1 Juniper browser turns since 2026-11-16 18:05Z (1 judged); last " in out
    assert "(b1-aaaaa)" in out and "no promise_view/promise_ledger line is expected in that gap" in out
    assert ("stored promise items since 2026-11-16 18:05Z: 1 on 1 turns, 0 of them Juniper browser turns"
            in out) and "memory-scoreboard-eval 1t/1i/1f" in out
    assert "Prime by exact key (turn_concern_id) on 1 item-bearing turns" in out and "not read" in out
    assert "void share of closed acts: -" in out
    # An eval session with ``mode`` and no browser surface is not Juniper's chat to Prime (#3602, fixed).
    assert "who=None not_handed_back" in out and "who=chat_juniper" not in out


def test_prime_by_ref_an_api_chat_post_in_her_own_session_is_not_handed_back():
    """#3602: a coding session's note posted to /api/chat lands in Juniper's own session with ``mode`` and no
    browser surface; Prime does not hand it back, so the report does not expect a concern for it."""
    stored = [_stored("api-note", [FIRM], session="9c91", mode=True, browser=False),
              _stored("typed", [FIRM], session="9c91", mode=True, browser=True)]
    out = rpt.prime_by_ref(stored, {}, None, None)
    assert {r["corr"]: r["prime_who"] for r in out["rows"]} == {"api-note": None, "typed": "chat_juniper"}


def test_prime_turns_file_needs_the_state_file():
    with pytest.raises(SystemExit):
        rpt.main(["--dsn", "x", "--prime-turns", "t.json"])


def test_close_summary_splits_void_by_settle_class_and_keeps_them_out_of_the_keep_rate():
    acts = [_act(1, opened=T0, closed=T0 + timedelta(hours=5), source="producer", note="kept_in:cx evidence=reply"),
            _act(2, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon"),
            _act(3, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE),
            _act(4, opened=T0, closed=T0 + H, source="expired", note=VOID_NOT_DUE_NOTE),
            _act(5, opened=T0, closed=T0 + H, source="expired", note=VOID_NOT_CHAT_NOTE),
            _act(6, opened=T0, closed=T0 + H, source="expired", note=VOID_ABSENCE_NOTE)]
    # A carry for act 5's turn lands after its horizon: a later-delivery note, the act stays void.
    out = rpt.close_summary(acts, unclosable=set(), carry_concerns={turn_concern_id("c5"): [T0 + H + timedelta(hours=2)]})
    assert out["void"] == 4 and out["kept"] == 1 and out["expired"] == 1
    assert out["void_by_kind"] == {"no_juniper_turn": 1, "not_due": 1, "not_chat": 1, "absence_unchecked": 1}
    assert out["void_with_later_carry_acts"] == ["act_5"]
    assert out["keep_rate"]["n"] == 2  # graded class only
    assert out["settle_class_void_share"] == round(3 / 6, 3)


def test_prime_vs_acts_counts_each_disagreement():
    acts = [_act(1, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE, corr="both-closed"),
            _act(2, opened=T0, corr="act-only"),
            _act(3, opened=T0, corr="both-open")]
    state = {"concerns": [{"concern_id": turn_concern_id("both-closed"), "event_ref": "turn:both-closed"},
                          {"concern_id": turn_concern_id("both-open"), "event_ref": "turn:both-open"},
                          {"concern_id": turn_concern_id("concern-only"), "event_ref": "turn:concern-only"},
                          {"concern_id": "letter", "event_ref": "orion_day_letter:2026-11-16#thread0"}],
             "concern_record": []}
    out = rpt.prime_vs_acts(acts, state)
    assert out["concern_no_act_ids"] == [turn_concern_id("concern-only")]
    assert out["act_no_concern_ids"] == ["act_2"]
    assert out["concern_open_act_closed_ids"] == [turn_concern_id("both-closed")]
    assert rpt.prime_vs_acts(acts, None) == {"read": False}
    # A concern Prime already closed is in concern_record: its act is not "with no concern".
    state["concern_record"] = [{"concern_id": turn_concern_id("act-only")}]
    assert rpt.prime_vs_acts(acts, state)["act_no_concern"] == 0
