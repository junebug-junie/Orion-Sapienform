"""Walkway rollups: arithmetic, exclusions, camera-down vs empty, DST and month edges,
the build order, and the whole path against a throwaway Postgres.

Pure part: orion/vision/walkway_rollup.py. Engine: app/vision_rollup.py.
"""

from __future__ import annotations

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
    assert r["kinds"]["person"] == {"passings": 1, "distinct": 1, "sightings": 2}
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
    yesterday = {"individuals": {"r1": {"kind": "person"}, "n1": {"kind": "person", "label": "Robin"}}}
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
    assert "most at once 2 people" in store.lines[("walkway", "hour", sat8 and wr._parse(sat8["period_start"]))]
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

_RAW_DDL = """
DROP TABLE IF EXISTS vision_rollup, vision_individual_sighting, vision_individual, vision_individuals_cursor,
    vision_scene_inventory, vision_events CASCADE;
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


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        for stmt in _RAW_DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    yield engine
    engine.dispose()


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
        run_one_rollup_cycle(postgres_uri=PG, cfg=cfg, now=now, since=day0)
    sql = (REPO / "services/orion-sql-db/manual_migration_vision_rollup_v1.sql").read_text()
    with pg.begin() as conn:
        conn.execute(text(sql))
    out = run_one_rollup_cycle(postgres_uri=PG, cfg=cfg, now=now, since=day0)
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
    assert "busiest 08:00 (2, most at once 2 people)" in day.line
    assert "so far, through 21:00" in day.line and "No record" not in day.line
    assert "Camera down" in day.line and "03:00–04:00" in day.line
    run_one_rollup_cycle(postgres_uri=PG, cfg=cfg, now=now, since=day0)
    with pg.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_rollup")).scalar() == n1
        assert conn.execute(text("SELECT rollup->'total'->>'passings' FROM vision_rollup WHERE grain='day'")
                            ).scalar() == "3"
    # Dry run writes nothing.
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM vision_rollup"))
    mem = MemoryStore()
    run_one_rollup_cycle(postgres_uri=PG, cfg=cfg, now=now, since=day0, write=False, memory=mem)
    assert mem.rows
    with pg.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_rollup")).scalar() == 0
