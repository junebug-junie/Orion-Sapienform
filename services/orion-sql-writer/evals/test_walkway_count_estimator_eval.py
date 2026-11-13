"""Offline checks of the walkway count-estimator eval (no database)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("walkway_count_eval", HERE / "run_walkway_count_estimator_eval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["walkway_count_eval"] = ev  # dataclasses look their module up
_spec.loader.exec_module(ev)

H = datetime(2026, 10, 20, 16, 0, tzinfo=timezone.utc)


def test_synthetic_gates_pass_for_the_gated_estimators_and_catch_clusters() -> None:
    gates = ev.synthetic_gates()
    for gate, k, got, want, ok in gates:
        if k in ev.GATED:
            assert ok, (gate, k, got, want)
    assert any(k == "clusters" and not ok for _, k, _, _, ok in gates)  # the over-count is visible


def test_checked_in_labels_file_parses_and_holds_no_invented_rows() -> None:
    labels, errs = ev.load_labels(ev.DEFAULT_LABELS)
    assert errs == []
    for lb in labels:
        assert lb.note, "every hand count says how it was checked"


def test_labels_loader_rejects_bad_rows(tmp_path) -> None:
    p = tmp_path / "l.csv"
    p.write_text(
        "# comment\nhour_start_utc,stream,true_people,note\n"
        "2026-10-20T16:00:00Z,walkway,3,counted from strips\n"
        "2026-10-20T16:30:00Z,walkway,3,not on the hour\n"
        "2026-10-20T17:00:00,walkway,3,no timezone\n"
        "2026-10-20T18:00:00Z,walkway,-1,negative\n"
        "2026-10-20T16:00:00Z,walkway,4,duplicate\n",
        encoding="utf-8")
    labels, errs = ev.load_labels(p)
    assert [(lb.hour, lb.true_people) for lb in labels] == [(H, 3)]
    assert len(errs) == 4
    bad = tmp_path / "b.csv"
    bad.write_text("hour,count\n", encoding="utf-8")
    assert ev.load_labels(bad)[0] == [] and ev.load_labels(bad)[1]
    assert ev.load_labels(tmp_path / "missing.csv")[1]


def _hour(est, low=1, high=9, camera="ok"):
    return {"low": low, "high": high, "camera": camera, "stories": [], "windows": [], "est": est}


def test_score_ranks_by_mean_absolute_error_and_reports_the_sign_test() -> None:
    hours, labels = {}, []
    for i in range(10):
        t = H + timedelta(hours=i)
        est = {k: 9 for k in ev.ESTIMATORS}
        est["stories"] = 3
        est["run_peaks"] = 4
        hours[t] = _hour(est)
        labels.append(ev.Label(t, "walkway", 3, "hand"))
    res = ev.score(labels, hours)
    assert res["scored_hours"] == 10 and res["best"] == "stories" and res["runner_up"] == "run_peaks"
    assert res["wins"] == 10 and res["sign_test_p"] < 0.01
    assert res["inside_range"] == 10


def test_sign_test_is_not_significant_on_a_coin_flip() -> None:
    assert ev.sign_test_p(5, 5) == 1.0
    assert ev.sign_test_p(0, 0) == 1.0
    assert round(ev.sign_test_p(9, 1), 4) == 0.0215


def test_hours_to_check_prefers_disagreement_and_skips_down_labelled_and_pre_story_hours() -> None:
    est_lo = {k: 2 for k in ev.ESTIMATORS}
    est_hi = dict(est_lo, clusters=14)
    hours = {
        H: _hour(est_hi, high=14),
        H + timedelta(hours=1): _hour(est_lo, high=2),
        H + timedelta(hours=2): _hour(est_hi, high=14, camera="down"),
        H + timedelta(hours=3): _hour(est_hi, high=14),            # labelled already
        H - timedelta(days=3): _hour(est_hi, high=14),             # before stories ran
        H + timedelta(hours=4): _hour(est_lo, high=1),             # too few people to be worth it
    }
    got = ev.hours_to_check(hours, since_stories=H - timedelta(hours=1), labelled={H + timedelta(hours=3)})
    assert [t for t, _ in got] == [H, H + timedelta(hours=1)]


def test_offline_run_passes(capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["eval", "--offline"])
    assert ev.main() == 0
    out = capsys.readouterr().out
    assert "VERDICT: OFFLINE" in out
