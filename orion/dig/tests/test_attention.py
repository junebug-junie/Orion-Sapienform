"""`dig attention` (issue #3600): relative attention rates. No DB: live rows are a pinned fixture.

The fixture is what ``frames_sql`` returns for the issue's baseline window (24 h to 2026-11-17T08:00Z,
whole hours), built on 2026-11-18 by running every real frame of that window and its kept prior (read-only,
before the 72 h prune) through ``orion.attention.frame_rollup.frame_increments`` -- the same function
orion-attention-runtime now uses to write the rollup -- and summing per side.
"""
from __future__ import annotations

import csv
import io
import json
import re
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.dig import __main__ as dig
from orion.dig import attention as A

FIXTURE = json.loads((Path(__file__).parent / "attention_2026-11-17_baseline.json").read_text())


def _csv(rows):
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


class Runner:
    """Stands in for ReadOnlyPsql: answers each of the three queries from the fixture."""

    def __init__(self, fx=FIXTURE, fail: str | None = None):
        self.fx, self.fail, self.seen = fx, fail, []

    def run(self, sql: str, flags: list[str]) -> tuple[str, int]:
        self.seen.append(sql)
        if self.fail:
            return self.fail, 1
        if "min(hour)" in sql:
            return _csv(self.fx["coverage"]), 0
        if "FROM temporal_self_event" in sql:
            return _csv(self.fx["events"]), 0
        return _csv(self.fx["frames"]), 0


def _baseline_text(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    return dig.cmd_attention(FIXTURE["window"], FIXTURE["end"], dig.Out("t"), runner=Runner())


def _read():
    w = A.parse_window(FIXTURE["window"], FIXTURE["end"])
    run = Runner()
    from orion.body_atlas.sources import psql_rows

    return A.read(w, lambda sql: psql_rows(run, sql))


# --- the rule: no raw cross-source count without its rate ------------------------------------------

bare_counts = A.bare_counts


def test_the_checker_catches_raw_cross_source_counts():
    # the 11-17 mistake Juniper called out: winner counts across sources, no rate
    assert bare_counts("body won 3,915 frames, chat only 834") == ["3,915", "834"]
    assert bare_counts("  node:prediction.body: won 1,805 frames") == ["1,805"]
    assert bare_counts("  world:chat: won 100% (834 of 834)") == []
    assert bare_counts("  whole UTC hours, every frame, about 2.2 s a frame") == []
    assert bare_counts("  1 in 4 frames sampled") == ["1", "4"]  # nothing is sampled any more: no exception for it
    # a count glued to a ref or written as a bare ratio is still a raw count (review 2026-11-17)
    assert bare_counts("world:chat=834 frames") == ["834"]
    assert bare_counts("chat won 834/40846") == ["834", "40846"]
    assert bare_counts("full rates: /tmp/x/dig/t/attention-24h-1.json") == []


def test_every_printed_line_is_a_rate_or_a_ref(tmp_path, monkeypatch):
    text = _baseline_text(tmp_path, monkeypatch)
    for line in text.splitlines():
        assert bare_counts(line) == [], f"raw count with no rate: {line!r}"


def test_every_line_with_any_number_in_the_whole_printout_carries_a_rate_or_ref():
    # render() uncut (the 4,000-char cut only drops lines), plus the never-eligible and no-frames paths
    r = _read()
    lines = A.render(r) + A.render(A.AttentionRead("24h", "a", "a", "b", "c", A.Side(), A.Side()))
    for line in lines:
        assert bare_counts(line) == [], f"raw count with no rate: {line!r}"


# --- the live baseline (issue #3600) ---------------------------------------------------------------

def test_won_when_eligible_matches_the_issue_baseline(tmp_path, monkeypatch):
    text = _baseline_text(tmp_path, monkeypatch)
    expect = {
        "world:chat": "won 100% (834 of 834)",
        "world:reading": "won 100% (426 of 426)",
        "node:substrate.street": "won 100% (96 of 96)",
        "node:substrate.perception_walkway": "won 96% (",
        "node:substrate.perception": "won 91% (",
        "node:prediction.body": "won 65% (",
        "node:prediction.room": "won 69% (",  # issue: 68% (its query's window edge was 1 frame different)
        "node:substrate.execution": "won 68% (",
        "node:substrate.strain": "won 48% (",
    }
    for target, want in expect.items():
        line = next(ln for ln in text.splitlines() if ln.startswith(f"  {target} ["))
        assert want in line, line


def test_rest_and_perception_eligibility_match_the_issue_baseline(tmp_path, monkeypatch):
    text = _baseline_text(tmp_path, monkeypatch)
    # issue (24 h to 08:23Z, from raw frames): rest 70% (28,641 of 40,835), perception eligible 10.7%.
    # The rollup reads whole hours, so the window ends 08:00Z: 23 minutes earlier, 5 more frames.
    assert "rest (no eligible target): 70% (28,724 of 40,840) of frames" in text
    assert "eligible 10.6% (4,321 of 40,840) vs design 10.0%" in text


def test_event_sources_are_measured_against_their_own_events():
    r = _read()
    chat = A.eligibility(r.cur, "world:chat", "event")
    # 2 Juniper turns x 900 s (CHAT_RATE_WINDOW), windows merged: 1,755 s lit / ~2.1 s a frame = 830 expected
    # frames, 834 eligible. The issue's "~74% of 8 turns" counted Orion's 6 outreach rows too; they do not
    # light world:chat (world_first._CHAT_TURN_WHERE counts Juniper's turns only), so they are not its activity.
    assert (chat.num, chat.den) == (834, 830) and chat.reference == "2 chat_turn x 900 s, overlaps merged"
    street = A.eligibility(r.cur, "node:substrate.street", "event")
    assert (street.num, street.den) == (96, 792) and street.reference == "7 street_event x 300 s, overlaps merged"
    reading = A.eligibility(r.cur, "world:reading", "event")
    # 39 readings, merged windows 25,459 s; only one 15-min link window lit it (loose source, see EVENT_SOURCES)
    assert reading.reference == "39 reading_turn x 900 s, overlaps merged" and (reading.num, reading.den) == (426, 12034)


def test_lit_seconds_merges_overlapping_windows_and_counts_window_edges():
    t0 = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)
    lo, hi = t0, t0 + timedelta(hours=1)
    # two events 5 min apart, 15-min windows: lit 5 + 15 = 20 min, not 30
    two = [t0 + timedelta(minutes=10), t0 + timedelta(minutes=15)]
    assert A.lit_seconds(two, 900, lo, hi) == 20 * 60
    # an event 5 min before the window still lights its first 10 min; one 5 min before the end only 5 min
    assert A.lit_seconds([t0 - timedelta(minutes=5)], 900, lo, hi) == 10 * 60
    assert A.lit_seconds([hi - timedelta(minutes=5)], 900, lo, hi) == 5 * 60


def test_contested_frames_and_wins_inside_them_are_reported():
    r = _read()
    assert r.cur.contested == 3047
    assert (r.cur.present_contested["node:substrate.perception"], r.cur.won_contested["node:substrate.perception"]) \
        == (1118, 721)


def test_not_attended_causes_are_counted_per_target(tmp_path, monkeypatch):
    r = _read()
    assert r.cur.causes["world:chat"] == {"quiet for itself": 40006}
    walk = r.cur.causes["node:substrate.perception_walkway"]
    assert walk["below its band floor"] == 37798 and walk["absent"] == 2592


# --- causes ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reason,cause", [
    ("world-first: not attended -- body quiet for itself; prediction.body prediction error; value 0.06", "quiet for itself"),
    ("world-first: not attended -- world usual for itself: not busy; chat is quiet; value 0.0", "usual for itself"),
    ("world-first: not attended -- absent: camera frames stale (vision_frame_staleness 1.0); per", "absent"),
    ("world-first: not attended -- insufficient_history; prediction.chat_turn prediction error", "insufficient history"),
    ("world-first: not attended -- event 412s old: past the 300s orienting window (value carried forward since); x",
     "past the orienting window"),
    ("world-first: not attended -- body high for itself, not sustained (hot 100s of last 600s); bio", "not sustained"),
    ("world-first: not attended -- high for itself (mid-rank), below this source's unusual floor; walk",
     "below its band floor"),
    ("world-first: not attended -- stale: reading 2000s old; x", "stale reading"),
    ("world-first: not attended -- placeholder: not a measurement; x", "not a measurement"),
    # a quiet verdict whose later text says insufficient_history is still quiet: only the verdict counts
    ("world-first: not attended -- body quiet for itself; x; band quiet, trend insufficient_history", "quiet for itself"),
    ("world-first: not attended -- something new", "other"),
    ("no separator at all, quiet for itself", "other"),  # same as the SQL: no ' -- ' -> '' -> other
])
def test_classify(reason, cause):
    assert A.classify(reason) == cause


def test_the_rollup_is_classified_at_write_time_by_the_shared_rule():
    # dig reads causes as written; the one rule lives in orion.attention.frame_rollup
    from orion.attention import frame_rollup

    assert A.CAUSES is frame_rollup.CAUSES and A.classify is frame_rollup.classify
    assert "frame_json" not in A.frames_sql(A.parse_window("24h", "2026-11-17T08:23Z"))


# --- window, coverage, sampling --------------------------------------------------------------------

NOW = datetime(2026, 11, 17, 8, 23, tzinfo=timezone.utc)
HOUR = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)


def test_parse_window():
    w = A.parse_window(None, None, now=NOW)
    assert (w.label, w.end - w.start, w.start - w.base_start) == ("24h", timedelta(hours=24), timedelta(days=7))
    assert w.end == HOUR  # whole hours: the current hour is still filling
    assert A.parse_window("7d", "2026-11-17T08:23Z").end == HOUR
    for bad in ("31d", "0h", "week", "24"):
        with pytest.raises(ValueError):
            A.parse_window(bad, None, now=NOW)
    with pytest.raises(ValueError):
        A.parse_end("yesterday")


def test_coverage_clamps_both_sides_to_the_first_rollup_hour():
    w = A.parse_window("24h", None, now=NOW)
    c = w.covered(HOUR - timedelta(hours=60))  # rollup started 60 h ago: 36 h of baseline, not 7 days
    assert c.start == w.start and c.base_seconds == 36 * 3600
    c = A.parse_window("7d", None, now=NOW).covered(HOUR - timedelta(hours=72))
    assert c.seconds == 72 * 3600 and c.base_seconds == 0


def test_a_clamped_window_and_a_missing_baseline_are_said(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    fx = dict(FIXTURE, coverage=[{"first": "2026-11-16 20:00:00+00"}],
              frames=[r for r in FIXTURE["frames"] if r["cur"] == "t"],
              events=[r for r in FIXTURE["events"] if r["cur"] == "t"])
    text = dig.cmd_attention("24h", FIXTURE["end"], dig.Out("t"), runner=Runner(fx))
    assert "counted only from 2026-11-16T20:00Z: the hourly rollup starts there" in text
    assert "drift: n/a, no counted hours before the window yet" in text


def test_no_frames_reads_as_unread_not_calm(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    fx = dict(FIXTURE, coverage=[{"first": ""}], frames=[], events=[])
    run = Runner(fx)
    text = dig.cmd_attention("24h", FIXTURE["end"], dig.Out("t"), runner=run)
    assert "no attention frames in this window: unread, not calm" in text
    assert len(run.seen) == 1  # no count read when the rollup has nothing


# --- kinds come from the semantic layer, events from the experience contract ------------------------

def test_target_kind_uses_world_first_semantics(monkeypatch):
    import orion.attention.world_first as wf

    monkeypatch.setattr(wf, "node_prediction_error_continuous", lambda n: n == "node:x")
    monkeypatch.setattr(wf, "node_prediction_error_event_written", lambda n: n == "node:y")
    assert A.target_kind("node:x") == "continuous"
    assert A.target_kind("node:y") == "event-written"
    assert A.target_kind("node:z") == "percentile"
    assert A.target_kind("world:chat") == "event"


def test_event_kinds_are_real_temporal_source_kinds():
    from orion.schemas.temporal_self import TemporalSourceKind

    kinds = set(typing.get_args(TemporalSourceKind))
    for target, src in A.EVENT_SOURCES.items():
        assert src.kind in kinds, target


def test_event_windows_come_from_world_first():
    from orion.attention import world_first as wf

    assert A.EVENT_SOURCES["world:chat"].window_sec == wf.CHAT_RATE_WINDOW.total_seconds()
    assert A.EVENT_SOURCES["world:reading"].window_sec == wf.READING_RATE_WINDOW.total_seconds()
    assert A.EVENT_SOURCES["node:substrate.street"].window_sec == wf.EVENT_ORIENTING_WINDOW_SEC


# --- read-only, bounded, honest failure -------------------------------------------------------------

def test_every_query_is_read_only_single_pass_and_time_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    run = Runner()
    dig.cmd_attention("24h", FIXTURE["end"], dig.Out("t"), runner=run)
    assert len(run.seen) == 3
    # never the frame documents: the rollup, its coverage, and events (2026-11-18: frame_json reads timed out)
    assert not any("substrate_attention_frames" in s or "frame_json" in s for s in run.seen)
    assert sum("FROM substrate_attention_hour_rollup" in s for s in run.seen) == 2
    for sql in run.seen:
        assert f"SET LOCAL statement_timeout = '{A.STATEMENT_TIMEOUT}'" in sql
        assert "\\" not in sql
        for word in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "GRANT ", "TRUNCATE "):
            assert word not in sql.upper()


def test_a_missing_grant_is_said_not_shown_as_calm(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    run = Runner(fail="ERROR:  permission denied for table substrate_attention_hour_rollup\n")
    with pytest.raises(dig.DigError, match="needs GRANT SELECT ON substrate_attention_hour_rollup TO orion_readonly"):
        dig.cmd_attention("24h", None, dig.Out("t"), runner=run)


def test_main_routes_attention_and_writes_the_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    monkeypatch.setattr(dig, "cmd_attention", lambda w, e, out: f"ok {w} {e}")
    assert dig.main(["attention"]) == 0 and capsys.readouterr().out.strip() == "ok None None"
    assert dig.main(["attention", "7d", "2026-11-17T08:23Z"]) == 0
    assert capsys.readouterr().out.strip() == "ok 7d 2026-11-17T08:23Z"
    assert "attention [<window>]" in dig.HELP


def test_result_file_holds_both_sides(tmp_path, monkeypatch):
    text = _baseline_text(tmp_path, monkeypatch)
    path = Path(text.rsplit(": ", 1)[1].strip())
    data = json.loads(path.read_text())
    chat = data["rates"]["world:chat"]
    assert chat["kind"] == "event" and chat["cur"]["won_when_eligible"] == 1.0
    assert data["base"]["frames"] > 0 and data["cur"]["frames"] == 40840


def test_a_target_judged_only_before_the_window_is_not_called_attended():
    r = _read()
    line = A.cause_line(r, "node:substrate.harness_closure")
    assert line.endswith("not judged in this window") and "attended" not in line
    assert A.target_line(r, "node:substrate.harness_closure").endswith("not judged in this window (a candidate only before it)")


def test_drift_carries_the_prior_rate_with_its_numerator_and_denominator():
    r = _read()
    assert re.search(r"drift won [+-]\d+%, eligible [+-]\d+%$", A.target_line(r, "node:substrate.perception"))
    prior = A.prior_line(r, "node:substrate.perception")
    assert re.fullmatch(r"  node:substrate.perception: won \d+% \([\d,]+ of [\d,]+\), eligible [\d.]+% \([\d,]+ of [\d,]+\)", prior), prior
    assert "prior rates each drift is against:" in A.render(r)


def test_eligible_share_is_of_the_frames_the_target_was_judged():
    side = A.Side(frames=100, present={"node:x": 5}, causes={"node:x": {"quiet for itself": 45}})
    e = A.eligibility(side, "node:x", "percentile")
    assert (e.num, e.den) == (5, 50)


def test_pct_never_rounds_a_loss_up_to_100():
    assert A.pct(2690 / 2694) == "99.9%" and A.pct(1.0) == "100%"


def test_the_rollups_own_first_hour_is_partial_and_skipped():
    # the writer was deployed partway through the rollup's first hour: never read it as a full hour
    from datetime import datetime as dt

    assert A.parse_first([{"first": "2026-11-18 10:00:00+00", "origin": "2026-11-18 10:00:00+00"}]) == \
        dt(2026, 11, 18, 11, tzinfo=timezone.utc)
    assert A.parse_first([{"first": "2026-11-18 10:00:00+00", "origin": "2026-11-17 09:00:00+00"}]) == \
        dt(2026, 11, 18, 10, tzinfo=timezone.utc)
    assert A.parse_first([{"first": "", "origin": "2026-11-17 09:00:00+00"}]) is None


def test_an_explicit_end_is_snapped_and_said(tmp_path, monkeypatch):
    text = _baseline_text(tmp_path, monkeypatch)  # asked to 08:23Z
    assert "asked to 2026-11-17T08:23Z: snapped down to whole UTC hours" in text
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    assert "snapped down" not in dig.cmd_attention("24h", "2026-11-17T08:00Z", dig.Out("t"), runner=Runner())
