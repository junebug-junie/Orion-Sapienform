"""The promise ledger report (pure parts): close kinds, un-closeable promises, the guard, the blinded sample."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.grammar.promise_acts import VOID_NOTE, promise_ref
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
    lo, hi = g["rate_wilson90"]
    assert g["rate"] == 0.1 and lo < 0.1 < hi
    assert rpt.guard_summary([])["rate_wilson90"] is None


def test_text_report_shows_raw_counts_beside_the_keep_rate(monkeypatch, capsys):
    acts = [
        _act(1, opened=T0, closed=T0 + timedelta(hours=5), source="producer", note="kept_in:cx evidence=reply"),
        _act(2, opened=T0, closed=T0 + H, source="expired", note="no keep within horizon"),
        _act(3, opened=T0, closed=T0 + H, source="expired", note=VOID_NOTE),
    ]
    monkeypatch.setattr(rpt, "gather", lambda dsn, since: {"acts": acts, "starts": [T0 + timedelta(hours=1)], "turns": [], "carry": {}})
    assert rpt.main(["--dsn", "x"]) == 0
    out = capsys.readouterr().out
    assert "keep rate (Beta(1,1)) over kept 1 + expired 1 (void 1 and un-closeable 0 left out)" in out


def _turn(corr, made, source="substrate_informed_pass", text="Some reply."):
    return {"correlation_id": corr, "reflection_source": source, "made": made, "final_text": text}


FIRM = {"quote": "I'll compare the two runs tomorrow.", "offer": False, "doer": "orion", "force": "commits",
        "act": "compare", "done_when": "a later reply names which run drifted"}


def test_guard_counts_judged_turns_only_and_skips_pre_force_items():
    turns = [_turn("a", [FIRM]), _turn("b", [{**FIRM, "offer": True}]),
             _turn("c", [{"quote": "old row", "offer": False, "doer": "orion"}]),
             _turn("d", [FIRM], source="deterministic_quick_gate"), _turn("e", [])]
    g = rpt.guard_summary(turns, baseline=(1, 10))
    assert g["judged_turns"] == 4 and g["turns_not_judged"] == 1
    assert (g["firm_promises"], g["offers"], g["skipped_pre_force"]) == (1, 1, 1)
    assert g["rate"] == 0.25 and g["ratio"] == 2.5 and g["red"] is True


def test_blinded_sample_hides_the_judges_reading_and_scores_back():
    text = "Lots of context before. I'll compare the two runs tomorrow. And after."
    sheet = rpt.blinded_sample([_turn("a", [FIRM], text=text)], 5)
    (item,) = sheet
    assert set(item) == {"key", "quote", "reply_context", "label", "why"}
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
