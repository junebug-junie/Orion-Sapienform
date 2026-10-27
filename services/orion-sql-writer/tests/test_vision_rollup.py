"""Walkway rollups: arithmetic, exclusions, camera-down vs empty, DST and month edges,
the build order, and the whole path against a throwaway Postgres.

Pure part: orion/vision/walkway_rollup.py. Engine: app/vision_rollup.py.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.vision_rollup import MemoryStore, RollupConfig, refresh_stream
from orion.vision import walkway_rollup as wr

TZ = ZoneInfo("America/Denver")
REPO = Path(__file__).resolve().parents[3]


def utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)


def sight(iid, t, kind="person", label=None, **kw) -> wr.SightingRow:
    return wr.SightingRow(sighting_id=f"s-{iid}-{t.isoformat()}", individual_id=iid, started_at=t, kind=kind,
                          label=label, **kw)


def hour(t: datetime, sightings=(), minutes=60, **kw) -> dict:
    p = wr.hour_period(t)
    return wr.build_hour(wr.HourInputs(stream_id="walkway", period=p, sightings=sightings, camera_minutes=minutes,
                                       **kw), TZ)


# --- periods ----------------------------------------------------------------------------------


def test_dst_days_have_25_and_23_hours_and_the_repeated_hour_is_labelled():
    fall = wr.day_period(date(2026, 11, 1), TZ)
    spring = wr.day_period(date(2027, 3, 14), TZ)
    assert len(wr.child_periods(fall, TZ)) == 25
    assert len(wr.child_periods(spring, TZ)) == 23
    assert len(wr.child_periods(wr.day_period(date(2026, 10, 17), TZ), TZ)) == 24
    labels = [wr.hhmm(h.start, TZ) for h in wr.child_periods(fall, TZ)]
    assert "01:00 MDT" in labels and "01:00 MST" in labels and labels.count("02:00") == 1


def test_week_and_month_periods_and_their_children():
    wk = wr.week_period(date(2026, 10, 28), TZ)  # Wednesday
    assert wk.start.astimezone(TZ).date() == date(2026, 10, 26)
    days = wr.child_periods(wk, TZ)
    assert [d.start.astimezone(TZ).date().day for d in days] == [26, 27, 28, 29, 30, 31, 1]
    mo = wr.month_period(date(2026, 10, 31), TZ)
    assert len(wr.child_periods(mo, TZ)) == 31
    assert len(wr.child_periods(wr.month_period(date(2027, 2, 3), TZ), TZ)) == 28
    assert wr.period_label(wk, TZ) == "2026-W44"
    assert wr.period_label(mo, TZ) == "2026-10"


# --- the hour ---------------------------------------------------------------------------------


def test_passing_is_one_individual_per_hour_not_one_crop_or_sighting():
    t = utc(2026, 10, 17, 20, 0)
    r = hour(t, [sight("a", t + timedelta(minutes=5)), sight("a", t + timedelta(minutes=9)),
                 sight("b", t + timedelta(minutes=30), kind="dog")])
    assert {k: r["kinds"]["person"][k] for k in ("passings", "distinct", "sightings")} == {
        "passings": 1, "distinct": 1, "sightings": 2}
    assert r["total"] == {"passings": 2, "distinct": 2, "sightings": 3}


def test_props_stationary_patio_and_parked_repeats_are_excluded_and_tallied():
    t = utc(2026, 10, 17, 20, 0)
    rows = [
        sight("prop", t + timedelta(minutes=1)),
        sight("still", t + timedelta(minutes=2), kind="vehicle", stationary=True),
        sight("parkedind", t + timedelta(minutes=3), kind="vehicle", individual_stationary=True),
        sight("patio", t + timedelta(minutes=4), zone_no_embed=True),
        sight("car", t + timedelta(minutes=5), kind="vehicle"),
        sight("walker", t + timedelta(minutes=6)),
    ]
    r = hour(t, rows, prop_ids=frozenset({"prop"}), parked_ids=frozenset({"car"}))
    assert r["excluded"] == {"stationary": 2, "prop": 1, "patio": 1, "parked_repeat": 1}
    assert r["total"]["passings"] == 1 and list(r["individuals"]) == ["walker"]


def test_parked_repeat_rule_counts_hours_not_sightings():
    base = utc(2026, 10, 17, 0, 0)
    trailing = [sight("car", base + timedelta(hours=h, minutes=m), kind="vehicle") for h in (1, 5, 9) for m in (0, 20)]
    trailing += [sight("dog", base + timedelta(hours=h), kind="dog") for h in range(6)]
    assert wr.parked_vehicle_ids(trailing, 4) == frozenset()
    trailing.append(sight("car", base + timedelta(hours=12), kind="vehicle"))
    assert wr.parked_vehicle_ids(trailing, 4) == frozenset({"car"})


def test_camera_down_hour_never_reads_nobody_came():
    t = utc(2026, 10, 17, 16, 0)
    down = wr.render_line(hour(t, [], minutes=0))
    assert "camera was down" in down and "nobody" not in down
    partial = wr.render_line(hour(t, [], minutes=30))
    assert "nobody seen in the 30 minutes the camera was up" in partial
    empty = wr.render_line(hour(t, [], minutes=60))
    assert "nobody passed" in empty
    # Something seen while the camera was mostly down still is not counted as an hour.
    assert hour(t, [], minutes=14)["camera"]["status"] == "down"
    assert hour(t, [], minutes=15)["camera"]["status"] == "partial"
    assert hour(t, [], minutes=50)["camera"]["status"] == "ok"


def test_named_people_get_their_times_and_unnamed_do_not():
    t = utc(2026, 10, 17, 21, 0)  # 15:00 MDT
    r = hour(t, [sight("m", t + timedelta(minutes=12), label="Alex"), sight("x", t + timedelta(minutes=13))])
    assert r["named"] == {"Alex": {"times": ["15:12"], "sightings": 1, "individual_ids": ["m"]}}
    assert "Alex at 15:12" in wr.render_line(r)


def test_captions_are_distinct_capped_and_stories_hidden_until_the_gate_passes():
    t = utc(2026, 10, 17, 21, 0)
    caps = [wr.CaptionRow(f"e{i}", t + timedelta(minutes=i), txt) for i, txt in
            enumerate(["A dog.", "a dog", "A car.", "A bike.", "A stroller."])]
    stories = [wr.StoryRow("s1", t + timedelta(minutes=3), "person", "A person walked left to right.")]
    closed = {"checked": 8, "gate": {"passed": False}}
    r = hour(t, captions=caps, stories=stories, story_gate=closed)
    assert [c["text"] for c in r["captions"]] == ["A dog.", "A car.", "A bike."]
    assert r["stories"] == {"gate_passed": False, "checked": 8}
    opened = hour(t, captions=caps, stories=stories, story_gate={"checked": 60, "gate": {"passed": True}})
    assert opened["stories"]["count"] == 1 and opened["stories"]["sentences"][0]["text"].startswith("A person")


# --- parents ----------------------------------------------------------------------------------


def _day_rows(d: date, per_hour, minutes=60):
    rows = []
    for i, h in enumerate(wr.child_periods(wr.day_period(d, TZ), TZ)):
        ids = per_hour(i)
        rows.append((h, hour(h.start, [sight(x, h.start + timedelta(minutes=1)) for x in ids], minutes=minutes)))
    return rows


def test_day_is_the_sum_of_its_hours_and_counts_distinct_across_them():
    d = date(2026, 10, 15)
    children = _day_rows(d, lambda i: ["a", f"u{i}"] if i in (8, 17) else [])
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)
    assert day["total"]["passings"] == 4 and day["total"]["distinct"] == 3
    assert day["complete"] is True
    assert day["busiest"]["at"] in ("08:00", "17:00")
    assert sum(e["passings"] for e in day["series"]) == day["total"]["passings"]


def test_dst_fall_back_day_sums_25_hours():
    d = date(2026, 11, 1)
    children = _day_rows(d, lambda i: [f"p{i}"])
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)
    assert day["total"]["passings"] == 25 and day["camera"]["expected_minutes"] == 25 * 60


def test_missing_hours_read_as_no_record_and_make_the_day_incomplete():
    d = date(2026, 10, 15)
    children = [(p, r if i < 20 else None) for i, (p, r) in enumerate(_day_rows(d, lambda i: []))]
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)
    assert day["complete"] is False
    assert day["camera"]["missing_spans"] == ["20:00–00:00"] and day["camera"]["down_spans"] == []
    line = wr.render_line(day)
    assert "No record 20:00–00:00" in line and "so far" not in line


def test_regulars_need_an_earlier_day_and_never_include_named():
    d = date(2026, 10, 16)
    children = _day_rows(d, lambda i: ["r1", "n1", "new"] if i == 9 else [])
    children[9] = (children[9][0], hour(children[9][0].start, [
        sight("r1", children[9][0].start), sight("n1", children[9][0].start, label="Robin"),
        sight("new", children[9][0].start)]))
    yesterday = {"period_start": wr.day_period(date(2026, 10, 15), TZ).start.isoformat(),
                 "individuals": {"r1": {"kind": "person"}, "n1": {"kind": "person", "label": "Robin"}}}
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ, history=[yesterday])
    assert day["regulars"] == {"count": 1, "by_kind": {"person": 1}, "ids": ["r1"]}


def test_week_spanning_a_month_boundary_and_the_month_only_counts_its_own_days():
    tz = TZ
    store = {}
    wk = wr.week_period(date(2026, 10, 28), tz)
    for dp in wr.child_periods(wk, tz):
        d = dp.start.astimezone(tz).date()
        store[d] = wr.build_parent("walkway", dp, _day_rows(d, lambda i, d=d: [f"x{d.day}"] if i == 10 else []), tz)
    week = wr.build_parent("walkway", wk, [(dp, store[dp.start.astimezone(tz).date()])
                                            for dp in wr.child_periods(wk, tz)], tz)
    assert week["total"]["passings"] == 7 and week["complete"] is True
    oct_month = wr.month_period(date(2026, 10, 1), tz)
    children = [(dp, store.get(dp.start.astimezone(tz).date())) for dp in wr.child_periods(oct_month, tz)]
    month = wr.build_parent("walkway", oct_month, children, tz)
    assert month["total"]["passings"] == 6  # Oct 26..31 only; Nov 1 is November's
    assert month["complete"] is False  # Oct 1..25 have no rows


def test_partial_parent_says_so_far_and_skips_the_running_day_as_quietest():
    d = date(2026, 10, 17)
    dp = wr.day_period(d, TZ)
    cut = dp.start + timedelta(hours=18)
    children = _day_rows(d, lambda i: ["a"] if i < 18 else [])
    day = wr.build_parent("walkway", dp, children, TZ, covered_until=cut)
    assert day["complete"] is False and day["total"]["passings"] == 18
    assert "(so far, through 18:00)" in wr.render_line(day)


def test_every_number_in_a_line_is_in_its_rollup():
    d = date(2026, 10, 15)
    children = _day_rows(d, lambda i: [f"p{i}", "a"] if i % 3 == 0 else [])
    children[5] = (children[5][0], hour(children[5][0].start, [], minutes=0))
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)
    for r in [day] + [c for _, c in children]:
        assert wr.unsupported_numbers(wr.render_line(r), r) == [], wr.render_line(r)


def test_guard_catches_an_invented_count():
    facts = {"total": {"passings": 185}, "camera": {"up_minutes": 1212}}
    assert wr.unsupported_numbers("The walkway has been empty on at least five occasions.", facts) == ["five"]
    assert wr.unsupported_numbers("185 passings today; no one stopped.", facts) == []
    assert wr.unsupported_numbers("About 200 people.", facts) == ["200"]


# --- the build order (engine with fakes) ------------------------------------------------------


class FakeReader:
    def __init__(self, sightings, minutes, cursor, gate=None):
        self._s, self._m, self._c = sightings, minutes, cursor
        self._gate = gate or {"checked": 0, "gate": {"passed": False}}
        self._peak_hours = {k for k, v in minutes.items() if v and k.astimezone(TZ).hour in (8, 17)}

    def cursor(self, stream_id):
        return self._c

    def sightings(self, stream_id, lo, hi):
        return [s for s in self._s if lo <= s.started_at < hi]

    def camera_minutes(self, stream_id, lo, hi):
        return {k: v for k, v in self._m.items() if lo <= k < hi}

    def peaks(self, stream_id, lo, hi):
        return {k: {"person": 2} for k in self._m if lo <= k < hi and k in self._peak_hours}

    def captions(self, stream_id, lo, hi):
        return []

    def story_gate(self):
        return self._gate

    def stories(self, *a):
        raise AssertionError("stories must not be read while the gate is closed")


def _fake(day0: date, days: int, now: datetime):
    start = wr.local_midnight(day0, TZ)
    sightings, minutes = [], {}
    t = start
    i = 0
    while t < now:
        minutes[t] = 0 if t.astimezone(TZ).hour == 3 else 60
        if t.astimezone(TZ).hour in (8, 17):
            sightings.append(sight(f"w{i}", t + timedelta(minutes=10)))
            sightings.append(sight("regular", t + timedelta(minutes=20)))
        t += timedelta(hours=1)
        i += 1
    return FakeReader(sightings, minutes, cursor=now)


def test_engine_builds_in_order_and_every_grain_agrees():
    now = utc(2026, 10, 18, 3, 30)  # 21:30 MDT on Sat 17 Oct
    reader = _fake(date(2026, 10, 12), 6, now)
    store = MemoryStore()
    cfg = RollupConfig(settle_sec=60)
    out = refresh_stream(reader, store, "walkway", now, cfg, since=wr.local_midnight(date(2026, 10, 12), TZ))
    assert out["hours"] > 100 and out["story_gate_passed"] is False
    days = {k[2]: v for k, v in store.rows.items() if k[1] == "day"}
    for start, day in days.items():
        hours = [store.rows.get(("walkway", "hour", h.start)) for h in
                 wr.child_periods(wr.Period("day", start, wr._parse(day["period_end"])), TZ)]
        assert day["total"]["passings"] == sum(h["total"]["passings"] for h in hours if h)
    week = store.rows[("walkway", "week", wr.week_period(date(2026, 10, 17), TZ).start)]
    assert week["total"]["passings"] == sum(d["total"]["passings"] for d in days.values())
    assert week["complete"] is False
    sat = days[wr.day_period(date(2026, 10, 17), TZ).start]
    assert sat["complete"] is False and "so far, through 21:00" in store.lines[("walkway", "day", sat and wr.day_period(date(2026, 10, 17), TZ).start)]
    # 03:00 every day: camera down, and the line says so.
    h3 = store.lines[("walkway", "hour", wr.local_midnight(date(2026, 10, 15), TZ) + timedelta(hours=3))]
    assert "camera was down" in h3
    # By Saturday the 08:00 hour has an hourly "usual" from earlier days.
    sat8 = store.rows[("walkway", "hour", wr.local_midnight(date(2026, 10, 17), TZ) + timedelta(hours=8))]
    assert sat8["baseline"]["mean_passings"] == 2.0
    assert sat8["peak_at_once"] == {"person": 2}
    assert sat8["kinds"]["person"]["count_range"]["low"] == 2
    assert "people: at least 2, at most 2" in store.lines[("walkway", "hour", wr._parse(sat8["period_start"]))]
    assert sat["regulars"]["ids"] == ["regular"]


def test_engine_tick_rebuilds_recent_and_fills_missing_only():
    now = utc(2026, 10, 18, 3, 30)
    reader = _fake(date(2026, 10, 16), 2, now)
    store = MemoryStore()
    cfg = RollupConfig(settle_sec=60, recompute_hours=3, catchup_hours=6)
    first = refresh_stream(reader, store, "walkway", now, cfg)
    assert first["hours"] == 6
    second = refresh_stream(reader, store, "walkway", now, cfg)
    assert second["hours"] == 3


def test_engine_waits_for_the_individuals_reducer():
    now = utc(2026, 10, 18, 3, 30)
    reader = _fake(date(2026, 10, 17), 1, now)
    reader._c = utc(2026, 10, 18, 1, 10)  # sightings settled only through 01:09 UTC
    out = refresh_stream(reader, MemoryStore(), "walkway", now, RollupConfig(settle_sec=60))
    assert out["ready_until"] == utc(2026, 10, 18, 1, 0).isoformat()
    reader._c = None
    assert refresh_stream(reader, MemoryStore(), "walkway", now, RollupConfig())["skipped"]


# --- real Postgres ----------------------------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

# Own schema: the CI database is shared with the other walkway Postgres tests, which need
# the real tables in `public`. Every statement here lands in SCHEMA only.
SCHEMA = "walkway_rollup_test"
_RAW_DDL = """
CREATE TABLE vision_individual (individual_id TEXT PRIMARY KEY, stream_id TEXT, kind TEXT, label TEXT,
    stationary BOOLEAN NOT NULL DEFAULT false);
CREATE TABLE vision_individual_sighting (sighting_id TEXT PRIMARY KEY, individual_id TEXT, stream_id TEXT,
    started_at TIMESTAMPTZ, zone_no_embed BOOLEAN NOT NULL DEFAULT false, stationary BOOLEAN NOT NULL DEFAULT false);
CREATE TABLE vision_individuals_cursor (stream_id TEXT PRIMARY KEY, last_created_at TIMESTAMPTZ);
CREATE TABLE vision_scene_inventory (window_id TEXT PRIMARY KEY, stream_id TEXT, observed_at TIMESTAMPTZ,
    counts JSONB NOT NULL DEFAULT '{}'::jsonb);
CREATE TABLE vision_events (event_id TEXT PRIMARY KEY, event_type TEXT, narrative TEXT, tags JSONB,
    created_at TIMESTAMPTZ, stream_id TEXT);
"""


def _schema_uri() -> str:
    sep = "&" if "?" in PG else "?"
    return f"{PG}{sep}options=-csearch_path%3D{SCHEMA}"


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    admin = create_engine(PG)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    engine = create_engine(_schema_uri())
    with engine.begin() as conn:
        for stmt in _RAW_DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    yield engine
    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    admin.dispose()


@needs_pg
def test_postgres_path_writes_rows_reads_them_back_and_is_idempotent(pg):
    from sqlalchemy import text

    from app.vision_rollup import MigrationMissing, run_one_rollup_cycle

    now = utc(2026, 10, 18, 3, 30)
    day0 = wr.local_midnight(date(2026, 10, 17), TZ)
    with pg.begin() as conn:
        conn.execute(text("INSERT INTO vision_individual VALUES ('m','walkway','person','Alex',false), "
                          "('x','walkway','dog',NULL,false), ('car','walkway','vehicle',NULL,true)"))
        for i, (iid, h) in enumerate([("m", 8), ("x", 8), ("m", 17), ("car", 9)]):
            conn.execute(text("INSERT INTO vision_individual_sighting VALUES (:s, :i, 'walkway', :t, false, false)"),
                         {"s": f"s{i}", "i": iid, "t": day0 + timedelta(hours=h, minutes=5)})
        t = day0
        while t < now:
            if t.astimezone(TZ).hour != 3:
                for m in range(0, 60, 1):
                    conn.execute(text("INSERT INTO vision_scene_inventory VALUES (:w, 'walkway', :t, CAST(:c AS jsonb))"),
                                 {"w": f"w{t.isoformat()}{m}", "t": t + timedelta(minutes=m, seconds=7),
                                  "c": '{"person": 2, "dog": 1}' if (t.astimezone(TZ).hour == 8 and m == 5)
                                  else ('{"vehicle": 1}' if m == 30 else '{}')})
            t += timedelta(hours=1)
        conn.execute(text("INSERT INTO vision_individuals_cursor VALUES ('walkway', :t)"), {"t": now})
        conn.execute(text("INSERT INTO vision_events VALUES ('c1','camera_caption','A dog on a leash.', "
                          "'[\"camera_caption\",\"detector:dog\"]', :t, 'walkway')"), {"t": day0 + timedelta(hours=8, minutes=6)})
    cfg = RollupConfig(settle_sec=60, prop_individual_ids=())
    with pytest.raises(MigrationMissing):
        run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0)
    sql = (REPO / "services/orion-sql-db/manual_migration_vision_rollup_v1.sql").read_text()
    with pg.begin() as conn:
        conn.execute(text(sql))
    out = run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0)
    assert out["streams"][0]["hours"] == 21 and out["streams"][0]["story_gate_passed"] is False
    with pg.connect() as conn:
        n1 = conn.execute(text("SELECT count(*) FROM vision_rollup")).scalar()
        day = conn.execute(text("SELECT rollup, line, complete FROM vision_rollup WHERE grain='day'")).one()
    assert day.complete is False
    assert day.rollup["total"]["passings"] == 3 and day.rollup["named"]["Alex"]["times"] == ["08:05", "17:05"]
    assert day.rollup["excluded"]["stationary"] == 1
    assert day.rollup["captions"][0]["text"] == "A dog on a leash."
    assert day.rollup["peak_at_once"] == {"person": 2, "dog": 1, "vehicle": 1}
    assert day.rollup["busiest"]["at"] == "08:00" and day.rollup["busiest"]["peak_at_once"]["person"] == 2
    assert day.rollup["kinds"]["person"]["count_range"] == {
        "low": 2, "high": 3, "method_low": "most_in_one_window", "method_high": "raised_to_low"}
    assert "busiest 08:00 (at most 2 passings; people at least 2, at most 2, most at once 1 dog)" in day.line
    assert "so far, through 21:00" in day.line and "No record" not in day.line
    assert "Camera down" in day.line and "03:00–04:00" in day.line
    run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0)
    with pg.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_rollup")).scalar() == n1
        assert conn.execute(text("SELECT rollup->'total'->>'passings' FROM vision_rollup WHERE grain='day'")
                            ).scalar() == "3"
    # Dry run writes nothing.
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM vision_rollup"))
    mem = MemoryStore()
    run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0, write=False, memory=mem)
    assert mem.rows
    with pg.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_rollup")).scalar() == 0


# --- review findings (2026-10-18) -------------------------------------------------------------


def test_a_day_with_no_rows_is_no_record_never_camera_down():
    d = date(2026, 10, 10)
    dp = wr.day_period(d, TZ)
    day = wr.build_parent("walkway", dp, [(h, None) for h in wr.child_periods(dp, TZ)], TZ)
    line = wr.render_line(day)
    assert day["camera"]["status"] == "missing"
    assert "no record" in line and "camera was down" not in line


def test_status_is_computed_only_over_children_with_rows():
    d = date(2026, 10, 15)
    children = [(p, r if i < 12 else None) for i, (p, r) in enumerate(_day_rows(d, lambda i: []))]
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)
    assert day["camera"]["status"] == "ok"  # 12 watched hours, 12 hours with no row
    assert day["camera"]["missing_spans"] == ["12:00–00:00"]


def test_sightings_without_frame_record_are_unknown_not_down():
    t = utc(2026, 10, 14, 20, 0)
    r = hour(t, [sight("a", t + timedelta(minutes=5))], minutes=0)
    assert r["camera"]["status"] == "unknown" and r["total"]["passings"] == 1
    line = wr.render_line(r)
    assert "camera was down" not in line and "people: at least 1, at most 1" in line and "record incomplete" in line


def test_no_busiest_or_quietest_when_nothing_passed():
    d = date(2026, 10, 15)
    day = wr.build_parent("walkway", wr.day_period(d, TZ), _day_rows(d, lambda i: []), TZ)
    assert day["busiest"] is None and day["quietest"] is None
    assert "busiest" not in wr.render_line(day)


def test_compound_detector_labels_map_to_kinds_and_other_never_names_a_peak():
    assert wr._kind("mail truck") == "vehicle" and wr._kind("vehicle truck") == "vehicle"
    assert wr._kind("bicycle vehicle") == "bicycle" and wr._kind("Person") == "person"
    assert wr._kind("mailbox") == "other"
    assert wr._peaks({"mail truck": 2, "vehicle": 1, "person": 1}) == {"vehicle": 2, "person": 1}
    assert wr._peak_phrase({"other": 4, "vehicle": 3}) == ""
    # People are a range now; their window count is its low end, never a peak phrase.
    assert wr._peak_phrase({"other": 4, "person": 2}) == ""
    assert wr._peak_phrase({"dog": 2, "person": 5}) == "most at once 2 dogs"


def test_guard_dates_and_minutes_cannot_vouch_for_counts_and_times_must_match():
    facts = {"facts": {"total": {"passings": 185}, "covered_until": "2026-10-17T04:00:00-06:00",
                       "busiest": {"at": "16:00", "passings": 36}},
             "lines": ["Walkway Sat 17 Oct 2026: 185 passings; busiest 16:00 (36); counts cover 20.2 h of 24 h."]}
    assert wr.unsupported_numbers("17 dogs today", facts) == ["17"]
    assert wr.unsupported_numbers("it came 4 times", facts) == ["4"]
    assert wr.unsupported_numbers("185 passings on Sat 17 Oct 2026; busiest at 16:00 with 36.", facts) == []
    assert wr.unsupported_numbers("a dog at 09:15", facts) == ["09:15"]


def test_regulars_ignore_day_rows_older_than_the_trailing_week():
    d = date(2026, 10, 16)
    children = _day_rows(d, lambda i: ["r1"] if i == 9 else [])
    old = {"period_start": wr.day_period(date(2026, 10, 1), TZ).start.isoformat(),
           "individuals": {"r1": {"kind": "person"}}}
    recent = {"period_start": wr.day_period(date(2026, 10, 12), TZ).start.isoformat(),
              "individuals": {"r1": {"kind": "person"}}}
    assert wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ, history=[old])["regulars"]["count"] == 0
    assert wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ, history=[recent])["regulars"]["count"] == 1


def test_backfill_never_starts_before_the_first_frame_record():
    now = utc(2026, 10, 18, 3, 30)
    reader = _fake(date(2026, 10, 16), 2, now)
    first = wr.local_midnight(date(2026, 10, 17), TZ) + timedelta(hours=5)
    reader.first_frame = lambda _s: first
    store = MemoryStore()
    out = refresh_stream(reader, store, "walkway", now, RollupConfig(settle_sec=60),
                         since=wr.local_midnight(date(2026, 10, 16), TZ))
    assert out["clamped_to_first_frame"] == first.isoformat()
    assert min(k[2] for k in store.rows if k[1] == "hour") == first


# --- people as a range (rules_version 2, 2026-10-20) ------------------------------------------
#
# Juniper: every Orion-facing people count prints "at least <most at once>, at most <clusters>".
# Live 2026-10-16 12:00 MDT: 79 person clusters, never more than 5 people in one frame.

NOON_1016 = utc(2026, 10, 16, 18, 0)  # 12:00 MDT


def _noon_1016() -> dict:
    t = NOON_1016
    s = [sight(f"p{i}", t + timedelta(seconds=40 * i)) for i in range(79)]
    s += [sight(f"b{i}", t + timedelta(minutes=i), kind="bicycle") for i in range(5)]
    s += [sight(f"v{i}", t + timedelta(minutes=2 * i), kind="vehicle") for i in range(8)]
    return hour(t, s, peak_at_once={"person": 5, "bicycle": 1, "vehicle": 3})


def _v1(row: dict) -> dict:
    """The same row as rules_version 1 wrote it: no count_range anywhere."""
    old = copy.deepcopy(row)
    old["rules_version"] = 1
    for v in (old.get("kinds") or {}).values():
        v.pop("count_range", None)
    for e in old.get("series") or []:
        e.pop("people", None)
    for k in ("busiest", "quietest"):
        if old.get(k):
            old[k].pop("people", None)
    return old


def test_1016_noon_reads_at_least_5_at_most_79_not_79_people():
    r = _noon_1016()
    assert r["rules_version"] == wr.RULES_VERSION
    assert r["kinds"]["person"]["passings"] == 79
    assert r["kinds"]["person"]["count_range"] == {
        "low": 5, "high": 79, "method_low": "most_in_one_window", "method_high": "appearance_clusters"}
    line = wr.render_line(r)
    assert "people: at least 5, at most 79 (the camera splits one person into several)" in line
    assert "79 people" not in line and "92 passing" not in line
    assert "5 bicycles, 8 vehicles passing" in line  # other kinds keep their count
    assert wr.bare_people_counts(line) == [] and wr.unsupported_numbers(line, r) == []


def test_live_1016_noon_row_as_stored_re_renders_with_its_range():
    # The live v1 row (read 2026-10-20): kinds.person 79 passings, peak 5, the old bare line.
    r = _v1(_noon_1016())
    old_line = ("Walkway 12:00–13:00 (Fri 16 Oct): 92 passing (79 people, 5 bicycles, 8 vehicles; "
                "most at once 5 people).")
    shown = wr.display_line(old_line, json.dumps(r))
    assert "people: at least 5, at most 79 (the camera splits one person into several)" in shown
    assert "79 people" not in shown and wr.bare_people_counts(shown) == []
    assert wr.person_count_range(r) == {"low": 5, "high": 79, "method_low": "most_in_one_window_uncorrected",
                                        "method_high": "appearance_clusters"}


def test_hour_range_low_is_most_in_one_window_and_never_above_high():
    t = utc(2026, 10, 17, 18, 0)
    for passings in range(0, 6):
        for peak in range(0, 7):
            s = [sight(f"p{i}", t + timedelta(minutes=i)) for i in range(passings)]
            r = hour(t, s, peak_at_once={"person": peak} if peak else {})
            cr = r["kinds"].get("person", {}).get("count_range")
            if passings == 0:
                assert cr is None and "people" not in wr.render_line(r)
                continue
            assert cr["low"] <= cr["high"]
            assert cr["low"] == max(peak, 1)  # one sighted person is a floor of 1
            assert cr["high"] == max(passings, cr["low"])
            assert cr["method_low"] == ("most_in_one_window" if peak else "one_sighting")
            assert wr.bare_people_counts(wr.render_line(r)) == []


def test_a_prop_seen_as_a_person_comes_off_the_floor():
    # The patio skeleton is a person-kind prop (live 2026-10-20) and the scene inventory
    # counts it in peak_at_once.
    t = utc(2026, 10, 15, 16, 0)
    s = [sight("skeleton", t + timedelta(minutes=1)), sight("walker", t + timedelta(minutes=2))]
    r = hour(t, s, peak_at_once={"person": 2}, prop_ids=frozenset({"skeleton"}))
    assert r["kinds"]["person"]["count_range"] == {
        "low": 1, "high": 1, "method_low": "most_in_one_window_less_props", "method_high": "appearance_clusters"}
    # A stationary person stays in the window count but not in the clusters: high is raised.
    r = hour(t, [sight("walker", t), sight("stander", t, stationary=True)], peak_at_once={"person": 2})
    assert r["kinds"]["person"]["count_range"]["high"] == 2
    assert r["kinds"]["person"]["count_range"]["method_high"] == "raised_to_low"


def test_every_grain_keeps_low_at_most_high_and_low_is_the_largest_hourly_low():
    now = utc(2026, 10, 18, 3, 30)
    reader = _fake(date(2026, 10, 12), 6, now)
    reader.peaks = lambda s, lo, hi: {k: {"person": 3 if k.astimezone(TZ).hour == 8 else 1}
                                      for k in reader._m if lo <= k < hi and k in reader._peak_hours}
    store = MemoryStore()
    refresh_stream(reader, store, "walkway", now, RollupConfig(settle_sec=60),
                   since=wr.local_midnight(date(2026, 10, 12), TZ))
    by_grain: dict = {}
    for (_, grain, _), row in store.rows.items():
        by_grain.setdefault(grain, []).append(row)
    assert set(by_grain) == {"hour", "day", "week", "month"}
    for grain, rows in by_grain.items():
        for row in rows:
            cr = wr.person_count_range(row)
            if cr is None:
                continue
            assert row["kinds"]["person"]["count_range"] == cr  # stored, not derived
            assert cr["low"] <= cr["high"], (grain, row["label"])
            assert cr["high"] >= row["kinds"]["person"]["passings"]
            if grain == "hour":
                assert cr["high"] == max(row["kinds"]["person"]["passings"], cr["low"])
            assert wr.bare_people_counts(wr.render_line(row)) == [], wr.render_line(row)
    for day in by_grain["day"]:
        per = wr.Period("day", wr._parse(day["period_start"]), wr._parse(day["period_end"]))
        hours = [store.rows.get(("walkway", "hour", h.start)) for h in wr.child_periods(per, TZ)]
        lows = [wr.person_count_range(h)["low"] for h in hours if h and wr.person_count_range(h)]
        assert day["kinds"]["person"]["count_range"]["low"] == max(lows) == 3
    for parent in by_grain["week"] + by_grain["month"]:
        assert parent["kinds"]["person"]["count_range"]["low"] == 3


def test_a_bump_in_rules_version_rebuilds_old_hours_inside_the_catchup_window():
    now = utc(2026, 10, 18, 3, 30)
    reader = _fake(date(2026, 10, 16), 2, now)
    store = MemoryStore()
    cfg = RollupConfig(settle_sec=60, recompute_hours=3, catchup_hours=6)
    refresh_stream(reader, store, "walkway", now, cfg)
    for k in [k for k in store.rows if k[1] == "hour"]:
        store.rows[k] = _v1(store.rows[k])
    out = refresh_stream(reader, store, "walkway", now, cfg)
    assert out["hours"] == 6  # every v1 hour in the window, not just the recompute 3
    assert all(r["rules_version"] == wr.RULES_VERSION for k, r in store.rows.items() if k[1] == "hour")
    assert refresh_stream(reader, store, "walkway", now, cfg)["hours"] == 3  # then back to normal


def test_old_parent_rows_re_render_without_a_bare_people_count():
    d = date(2026, 10, 15)
    children = _day_rows(d, lambda i: [f"p{i}", "a"] if i in (8, 9, 17) else [])
    day = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)
    old = _v1(day)
    shown = wr.display_line("Walkway Thu 15 Oct 2026: 6 passings (6 people) by 4 distinct.", old)
    assert "people: at least 1, at most 6" in shown and wr.bare_people_counts(shown) == []
    assert "busiest" in shown and "at most 2 passings" in shown
    assert wr.display_line("x", day) == "x"  # a v2 row's stored line is shown as is
    assert wr.display_line("bare", {"period_start": "not a time", "rules_version": 1}) == ""


def test_a_parent_keeps_a_child_high_that_was_raised_to_its_low():
    # Hour A: 4 people in one window, 1 cluster (3 stood still) -> 4..4. Hour B: 1..1.
    # The day may not say "at most 4": an honest top is 5.
    t = utc(2026, 10, 15, 14, 0)
    a = hour(t, [sight("w", t)] + [sight(f"s{i}", t, stationary=True) for i in range(3)],
             peak_at_once={"person": 4})
    b = hour(t + timedelta(hours=1), [sight("x", t + timedelta(hours=1))], peak_at_once={"person": 1})
    d = date(2026, 10, 15)
    children = [(h, a if h.start == t else b if h.start == t + timedelta(hours=1) else None)
                for h in wr.child_periods(wr.day_period(d, TZ), TZ)]
    cr = wr.build_parent("walkway", wr.day_period(d, TZ), children, TZ)["kinds"]["person"]["count_range"]
    assert (cr["low"], cr["high"], cr["method_high"]) == (4, 5, "raised_to_low")


def test_display_line_never_shows_a_stored_line_it_cannot_vouch_for():
    assert wr.display_line("OLD 79 people", None) == ""
    assert wr.display_line("OLD 79 people", "{not json") == ""
    assert wr.display_line("OLD 79 people", {"rules_version": 1}) == ""  # cannot re-render
    assert wr.display_line("ok", {"rules_version": 2}) == "ok"


def test_bare_people_counts_finds_one_number_of_people_and_passes_ranges():
    # A range word that belongs to another phrase does not excuse the count.
    assert wr.bare_people_counts("at most 3 dogs and 40 people came") == ["40 people"]
    assert wr.bare_people_counts("Up to noon, 79 people walked by") == ["79 people"]
    assert wr.bare_people_counts("2 to 5 people") == []
    assert wr.bare_people_counts("79 people at most") == []
    assert wr.bare_people_counts("79 people walked by.") == ["79 people"]
    assert wr.bare_people_counts("Five different people came.") == ["Five different people"]
    assert wr.bare_people_counts("at least 5, at most 79 people") == []
    assert wr.bare_people_counts("between 5 and 79 people") == []
    assert wr.bare_people_counts("people: at least 5, at most 79 (the camera splits one person into several)") == []


# --- named people from the person record (rules_version 3, 2026-10-24) ------------------------

P_ROBIN = wr.PersonRef("person-robin", "Robin")
P_SAM = wr.PersonRef("person-sam", "Sam")
P_DELIVERY = wr.PersonRef("person-delivery", "Parcel delivery", status="not_a_person")
PERSONS = {
    "r1": P_ROBIN, "r2": P_ROBIN,                       # two clusters, one person
    "d1": P_DELIVERY,                                   # linked to a not_a_person row
    "bike": wr.PersonRef("person-sam", "Sam", source_kind=wr.SOURCE_BELONGING),
}


def _person_hour(t, persons=PERSONS):
    return hour(t, [
        sight("r1", t + timedelta(minutes=5), label="Robin"),
        sight("r2", t + timedelta(minutes=40), label="Rob"),     # typed differently, same person
        sight("d1", t + timedelta(minutes=10), label="Parcel delivery"),
        sight("car", t + timedelta(minutes=12), kind="vehicle", label="no"),
        sight("bike", t + timedelta(minutes=20), kind="bicycle", label="Sam"),
        sight("u1", t + timedelta(minutes=30), label="Kit"),     # typed, no link (yet)
        sight("x", t + timedelta(minutes=31)),
    ], persons=persons)


def test_named_is_by_person_and_clusters_of_one_person_merge():
    t = utc(2026, 10, 17, 21, 0)  # 15:00 MDT
    r = _person_hour(t)
    assert r["rules_version"] == wr.PERSON_RECORD_RULES_VERSION
    assert r["named"] == {
        "Robin": {"times": ["15:05", "15:40"], "sightings": 2, "individual_ids": ["r1", "r2"],
                  "person_ids": ["person-robin"]},
        "Kit": {"times": ["15:30"], "sightings": 1, "individual_ids": ["u1"]},
    }
    assert r["belongings"] == {"Sam": {"times": ["15:20"], "sightings": 1, "individual_ids": ["bike"],
                                       "person_ids": ["person-sam"], "kinds": ["bicycle"]}}
    # Counting is unchanged: the not_a_person cluster and the vehicle still passed.
    assert r["kinds"]["person"]["passings"] == 5 and r["kinds"]["vehicle"]["passings"] == 1
    assert r["person_record"]["available"] is True and r["person_record"]["digest"]
    line = wr.render_line(r)
    assert "Robin at 15:05, 15:40" in line and "Kit at 15:30" in line and "Sam's bicycle at 15:20" in line
    assert "Rob at" not in line and "Parcel delivery" not in line and "no at" not in line


def test_without_the_person_record_named_groups_by_label_as_before():
    t = utc(2026, 10, 17, 21, 0)
    r = _person_hour(t, persons=None)
    assert set(r["named"]) == {"Robin", "Rob", "Parcel delivery", "no", "Sam", "Kit"}
    assert r["belongings"] == {} and r["person_record"] == {"available": False, "digest": None}
    assert "person_ids" not in r["named"]["Robin"]


def test_day_merges_one_person_across_hours_and_week_counts_days():
    d = date(2026, 10, 17)
    dp = wr.day_period(d, TZ)
    children = []
    for h in wr.child_periods(dp, TZ):
        local = h.start.astimezone(TZ).hour
        rows = []
        if local == 9:
            rows = [sight("r1", h.start + timedelta(minutes=3), label="Robin")]
        elif local == 17:
            rows = [sight("r2", h.start + timedelta(minutes=8), label="Rob"),
                    sight("bike", h.start + timedelta(minutes=9), kind="bicycle", label="Sam")]
        children.append((h, hour(h.start, rows, persons=PERSONS)))
    day = wr.build_parent("walkway", dp, children, TZ)
    assert day["named"] == {"Robin": {"sightings": 2, "individual_ids": ["r1", "r2"],
                                      "person_ids": ["person-robin"], "times": ["09:03", "17:08"]}}
    assert day["belongings"]["Sam"]["kinds"] == ["bicycle"]
    line = wr.render_line(day)
    assert "named: Robin (09:03, 17:08), Sam's bicycle (17:09)" in line
    assert wr.unsupported_numbers(line, day) == []
    week = wr.build_parent("walkway", wr.week_period(d, TZ), [(dp, day)], TZ)
    assert week["named"]["Robin"]["days"] == ["2026-10-17"]
    assert "named: Robin on 1 day, Sam's bicycle on 1 day" in wr.render_line(week)


def test_an_old_v2_day_row_still_renders_and_shows_its_stored_line():
    old = {"rules_version": 2, "named": {"Robin": {"times": ["09:03"], "individual_ids": ["r1"]}}}
    assert wr.display_line("stored v2 line", old) == "stored v2 line"


class _PersonFake(FakeReader):
    def __init__(self, *a, persons=None, **kw):
        super().__init__(*a, **kw)
        self.persons = persons

    def person_links(self):
        return self.persons


def test_a_changed_person_link_rebuilds_the_recent_hours_once():
    now = utc(2026, 10, 18, 3, 30)
    base = _fake(date(2026, 10, 16), 2, now)
    reader = _PersonFake(base._s, base._m, base._c, persons={"regular": P_SAM})
    store = MemoryStore()
    cfg = RollupConfig(settle_sec=60, recompute_hours=3, catchup_hours=6)
    assert refresh_stream(reader, store, "walkway", now, cfg)["person_record"] == "ok"
    assert refresh_stream(reader, store, "walkway", now, cfg)["hours"] == 3   # nothing changed
    reader.persons = {"regular": P_ROBIN}                                      # a rename
    assert refresh_stream(reader, store, "walkway", now, cfg)["hours"] == 6
    assert refresh_stream(reader, store, "walkway", now, cfg)["hours"] == 3
    # The record turning unreadable does not rebuild (it would only lose the grouping).
    reader.persons = None
    out = refresh_stream(reader, store, "walkway", now, cfg)
    assert out["hours"] == 3 and out["person_record"] == "unavailable"


@needs_pg
def test_postgres_named_by_person_record_and_fail_open_without_it(pg, caplog):
    from sqlalchemy import text

    import app.vision_rollup as vr
    from app.vision_rollup import DbReader, run_one_rollup_cycle

    now = utc(2026, 10, 18, 3, 30)
    day0 = wr.local_midnight(date(2026, 10, 17), TZ)
    with pg.begin() as conn:
        conn.execute(text((REPO / "services/orion-sql-db/manual_migration_vision_rollup_v1.sql").read_text()))
        conn.execute(text("INSERT INTO vision_individual VALUES ('r1','walkway','person','Robin',false), "
                          "('r2','walkway','person','Rob',false), ('d1','walkway','person','Parcel',false), "
                          "('car','walkway','vehicle','no',false)"))
        for i, (iid, h) in enumerate([("r1", 8), ("r2", 17), ("d1", 9), ("car", 10)]):
            conn.execute(text("INSERT INTO vision_individual_sighting VALUES (:s, :i, 'walkway', :t, false, false)"),
                         {"s": f"s{i}", "i": iid, "t": day0 + timedelta(hours=h, minutes=5)})
        conn.execute(text("INSERT INTO vision_individuals_cursor VALUES ('walkway', :t)"), {"t": now})
    cfg = RollupConfig(settle_sec=60)

    def day_named():
        with pg.connect() as conn:
            return conn.execute(text("SELECT rollup->'named' FROM vision_rollup WHERE grain='day'")).scalar()

    # No person tables: label grouping, logged once, and the rollup transaction survives.
    vr._PERSON_RECORD_WARNED = False
    with caplog.at_level("WARNING", logger="sql-writer.vision_rollup"):
        run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0)
        run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0)
    assert sum("person_record_unavailable" in r.message for r in caplog.records) == 1
    assert set(day_named()) == {"Robin", "Rob", "Parcel", "no"}

    with pg.begin() as conn:
        conn.execute(text((REPO / "services/orion-sql-db/manual_migration_person_record_v1.sql").read_text()))
        conn.execute(text("INSERT INTO person (person_id, display_name, status) VALUES "
                          "('p-robin','Robin','active'), ('p-parcel','Parcel','not_a_person')"))
        for ref, pid, revoked in (("r1", "p-robin", None), ("r2", "p-robin", None), ("d1", "p-parcel", None),
                                  ("r2", "p-parcel", now)):
            conn.execute(text("INSERT INTO person_link (link_id, person_id, source_kind, source_ref, decided_by, "
                              "revoked_at) VALUES (:l, :p, 'vision_individual', :r, 'camera_name', :rv)"),
                         {"l": f"{ref}-{pid}", "p": pid, "r": ref, "rv": revoked})
    with pg.connect() as conn:
        links = DbReader(conn).person_links()
    assert links == {"r1": wr.PersonRef("p-robin", "Robin"), "r2": wr.PersonRef("p-robin", "Robin"),
                     "d1": wr.PersonRef("p-parcel", "Parcel", status="not_a_person")}
    run_one_rollup_cycle(postgres_uri=_schema_uri(), cfg=cfg, now=now, since=day0)
    assert day_named() == {"Robin": {"sightings": 2, "individual_ids": ["r1", "r2"], "person_ids": ["p-robin"],
                                     "times": ["08:05", "17:05"]}}
    with pg.connect() as conn:
        line = conn.execute(text("SELECT line FROM vision_rollup WHERE grain='day'")).scalar()
    assert "named: Robin (08:05, 17:05)" in line and "Parcel" not in line and "no (" not in line
