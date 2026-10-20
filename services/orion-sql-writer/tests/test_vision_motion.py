"""Walkway motion stories (2026-10-16): pure rules, the loop's cursor maths,
config parity, the migration's auto-apply safety, and the whole path against
a REAL Postgres (``ORION_TEST_PG_URI``, a throwaway database: it TRUNCATEs
the walkway tables). docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md.

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 \\
             vision_split_v1 vision_motion_story_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_motion.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_motion as vm  # noqa: E402
from app import vision_motion_loop as vl  # noqa: E402
from orion.vision.zones import load_zones  # noqa: E402

T0 = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
CFG = vm.MotionConfig()
ZONES = load_zones(REPO / "config" / "vision_zones.yaml")["walkway"]
MIGRATION = REPO / "services" / "orion-sql-db" / "manual_migration_vision_motion_story_v1.sql"


def crop(cid: str, sec: float, x: float, y2: float, *, h: float = 200.0, w: float = 80.0, label: str = "person",
         ind: str | None = None, thumb: str | None = None) -> vm.MotionCrop:
    """A box whose foot point is (x, y2)."""
    return vm.MotionCrop(cid, T0 + timedelta(seconds=sec), label, (x - w / 2, y2 - h, x + w / 2, y2),
                         score=0.5, individual_id=ind, thumb_ref=thumb)


def walker(prefix: str = "w", *, start: float = 0.0, x0: float = 600.0, dx: float = 160.0, y2: float = 1350.0,
           n: int = 5, step: float = 2.5, label: str = "person", ind: str | None = None, h: float = 200.0):
    """dx px per step at h=200 px: 160 px / 2.5 s = 0.32 BH/s, a walker."""
    return [crop(f"{prefix}{i}", start + i * step, x0 + i * dx, y2, h=h, label=label, ind=ind) for i in range(n)]


def stories(crops, *, labels=None, stationary=frozenset(), prop_ids=frozenset(), prop_boxes=(), clean_since=None):
    ps = vm.link_passages(crops, CFG)
    return vm.build_stories(ps, stream_id="walkway", zones=ZONES, cfg=CFG, labels=labels or {},
                            stationary_ids=set(stationary), prop_ids=set(prop_ids), prop_boxes=list(prop_boxes),
                            clean_since=clean_since)


# --- linking and the sentence ---------------------------------------------------------------------------


def test_a_walker_walks_left_to_right_and_is_published() -> None:
    [s] = stories(walker())
    assert s.published and s.drop_reason is None
    assert s.speed_class == "walking" and s.dir_lateral == 1 and s.dir_depth == 0
    assert s.sentence == "A person walked left to right on the walkway, over 10 s (5 detections)."
    assert s.detection_count == 5 and s.crop_ids == [f"w{i}" for i in range(5)]
    assert s.story_id == vm.story_id_for("walkway", "w0")


def test_two_walkers_far_apart_are_two_passages_not_ids() -> None:
    a = walker("a", ind="id-1")
    b = walker("b", x0=600, y2=1800, ind="id-1")   # same individual id, different place: two people
    out = stories(a + b)
    assert len(out) == 2 and all(s.published for s in out)


def test_ids_do_not_define_a_passage() -> None:
    """Sample S8: one walker, three individual ids. Geometry links them."""
    cs = walker(n=6)
    cs = [vm.MotionCrop(c.crop_id, c.t, c.label, c.box, individual_id=f"id-{i % 3}") for i, c in enumerate(cs)]
    [s] = stories(cs)
    assert s.detection_count == 6


def test_first_gate_shrinks_with_a_long_gap() -> None:
    """Sample S1: a 3-height jump over 10 s is a guess, not a track; over 5 s it links."""
    assert vm.first_gate(CFG, 2.5) == CFG.first_gate_bh
    assert vm.first_gate(CFG, 10.0) == pytest.approx(CFG.first_gate_bh / 2)
    near = [crop("a", 0, 600, 900), crop("b", 5, 1200, 900)]
    far = [crop("a", 0, 600, 900), crop("b", 10, 1200, 900)]
    assert len(vm.link_passages(near, CFG)) == 1
    assert len(vm.link_passages(far, CFG)) == 2


def test_a_passage_ends_after_the_link_gap() -> None:
    cs = walker(n=3) + [crop("late", 5.0 + 12.0, 920, 900)]
    assert [len(p.crops) for p in vm.link_passages(cs, CFG)] == [3, 1]


def test_height_jump_does_not_link() -> None:
    cs = [crop("a", 0, 600, 900, h=200), crop("b", 2.5, 650, 900, h=450)]
    assert len(vm.link_passages(cs, CFG)) == 2


# --- speed classes --------------------------------------------------------------------------------------


def test_two_box_jitter_is_a_glimpse_never_running() -> None:
    cs = [crop("a", 0, 600, 900), crop("b", 2.5, 1000, 900)]   # 2 BH in 2.5 s = 0.8... but 2 boxes
    [s] = stories(cs)
    assert not s.published and s.drop_reason == "glimpse"
    fast = [crop("a", 0, 600, 900), crop("b", 1.0, 920, 900)]  # 1.6 BH/s on two boxes
    [s] = stories(fast)
    assert s.speed_class == "walking"     # the fast class needs 3 looks
    assert "ran" not in s.sentence


def test_three_fast_detections_run() -> None:
    cs = [crop(f"r{i}", i * 1.0, 600 + i * 320, 900) for i in range(3)]   # 1.6 BH/s
    [s] = stories(cs)
    assert s.speed_class == "running" and " ran left to right" in s.sentence
    # Slice 1 is walker stories only: computed, stored, never published.
    assert not s.published and s.drop_reason == "out_of_scope"


def test_vehicles_are_out_of_scope_in_slice_one() -> None:
    cs = [crop(f"v{i}", i * 2.5, 600 + i * 40, 520, h=80, w=120, label="vehicle") for i in range(4)]
    [s] = stories(cs)
    assert s.speed_class == "creeping" and not s.published and s.drop_reason == "out_of_scope"


def test_fast_needs_three_unclipped_detections() -> None:
    """Three runner boxes, the last touching the right edge: only two usable boxes."""
    cs = [crop("r0", 0, 1900, 900), crop("r1", 1.0, 2220, 900), crop("r2", 2.0, 2530, 900)]
    [s] = stories(cs)
    assert s.used_detections == 2 and s.speed_class == "walking"
    assert not s.published and s.drop_reason == "edge_clipped"


def test_slow_drift_of_a_prop_reads_as_stood_not_walked() -> None:
    """The patio skeleton (fefb24d7) drifted ~109 px at ~270 px tall and was never
    flagged stationary: 0.4 BH net travel is standing, and standing is not published."""
    cs = [crop(f"k{i}", i * 5.0, 2176 + (109 if i % 2 else 0), 1149, h=271) for i in range(8)]
    [s] = stories(cs)
    assert s.speed_class == "standing" and not s.published and s.drop_reason == "standing"
    assert "stood" in s.sentence and s.zone_path == ["patio"]


def test_parked_things_are_not_stories() -> None:
    cs = [crop(f"v{i}", i * 5.0, 1300, 520, h=80, w=120, label="vehicle") for i in range(4)]
    [s] = stories(cs)
    assert s.speed_class == "parked" and s.drop_reason == "parked" and not s.published


# --- edge clipping --------------------------------------------------------------------------------------


def test_edge_clipped_boxes_give_no_zone_and_no_travel() -> None:
    """Sample S10: boxes reaching the bottom edge (y2 1914-1916 of 1920) carry no foot point."""
    cs = [crop(f"c{i}", i * 2.5, 1700 + i * 50, 1916, h=1100, w=600) for i in range(4)]
    [s] = stories(cs)
    assert s.used_detections == 0 and s.edge_clipped_count == 4
    assert s.speed_class is None and s.zone_path == [] and s.net_travel_bh is None
    assert not s.published and s.drop_reason == "edge_clipped"
    assert all(e["clipped"] and e["zone"] is None for e in s.evidence)


def test_a_walker_entering_from_the_edge_counts_only_unclipped_boxes() -> None:
    cs = [crop("e0", 0, 2530, 1350)] + walker("w", start=2.5, x0=2380, dx=-160, n=4)
    [s] = stories(cs)
    assert s.detection_count == 5 and s.used_detections == 4 and s.edge_clipped_count == 1
    assert s.published and s.dir_lateral == -1
    assert s.evidence[0]["clipped"] and s.evidence[0]["zone"] is None


# --- names ----------------------------------------------------------------------------------------------


def test_name_needs_two_thirds_one_named_id_no_other_and_no_edge() -> None:
    labels = {"ind-a": "Alex", "ind-b": "Sam"}
    base = walker(n=6)

    def with_ids(ids):
        return [vm.MotionCrop(c.crop_id, c.t, c.label, c.box, individual_id=i) for c, i in zip(base, ids)]

    [s] = stories(with_ids(["ind-a"] * 4 + [None, None]), labels=labels)
    assert s.individual_id == "ind-a" and s.name_label == "Alex" and s.sentence.startswith("Alex walked")
    [s] = stories(with_ids(["ind-a"] * 3 + [None, None, None]), labels=labels)
    assert s.name_label is None and s.name_withheld == "low_share" and s.sentence.startswith("A person")
    [s] = stories(with_ids(["ind-a"] * 5 + ["ind-b"]), labels=labels)
    assert s.name_label is None and s.name_withheld == "two_named"
    # An UNNAMED id next to a named one may be a second person the linker joined.
    [s] = stories(with_ids(["ind-a"] * 4 + ["x", "x"]), labels=labels)
    assert s.name_label is None and s.name_withheld == "other_id"
    # Two ids that carry the same name are one person.
    [s] = stories(with_ids(["ind-a"] * 3 + ["ind-a2"] * 3), labels={**labels, "ind-a2": " alex "})
    assert s.name_label == "Alex" and s.name_withheld is None
    clipped = with_ids(["ind-a"] * 6)
    clipped[-1] = vm.MotionCrop("w5", clipped[-1].t, "person", (2480, 1150, 2560, 1350), individual_id="ind-a")
    p = vm.Passage("person", clipped)
    assert vm.name_for(p, labels, CFG) == (None, None, "edge_clipped")


def test_unnamed_ids_never_withhold_or_produce_a_name() -> None:
    [s] = stories(walker(ind="anon"), labels={})
    assert s.name_label is None and s.name_withheld is None


# --- zones ----------------------------------------------------------------------------------------------


def test_zone_edge_jitter_does_not_read_as_crossings() -> None:
    """One-frame blips across the patio's left edge (x ~ 0.57) are not crossings."""
    xs = [1400, 1400, 1520, 1400, 1400, 1520, 1520, 1520]
    raw = [crop(f"z{i}", i * 2.5, x, 1500) for i, x in enumerate(xs)]
    assert vm.zone_path(raw, ZONES, CFG) == ["walkway", "patio"]
    assert vm.zone_path(raw[:5], ZONES, CFG, still=True) == ["walkway"]


def test_mailbox_to_patio_path_and_phrase() -> None:
    cs = [crop(f"m{i}", i * 2.5, 600 + 150 * i, 1000 + 69 * i) for i in range(9)]
    [s] = stories(cs)
    assert s.zone_path == ["mailbox", "walkway", "patio"]
    assert "from the mailbox, through the walkway, to the patio" in s.sentence


# --- props ----------------------------------------------------------------------------------------------


def test_props_by_stationary_flag_prop_id_or_prop_box() -> None:
    w = walker(ind="skel")
    assert stories(w, stationary={"skel"})[0].drop_reason == "prop"
    assert stories(w, prop_ids={"skel"})[0].drop_reason == "prop"
    boxes = [("person", list(c.box)) for c in w]
    assert stories(walker(), prop_boxes=boxes)[0].drop_reason == "prop"
    # A parked car's box never swallows a person walking over the same pixels.
    assert stories(walker(), prop_boxes=[("vehicle", b) for _k, b in boxes])[0].published


# --- together -------------------------------------------------------------------------------------------


def test_person_with_a_dog_and_two_people_together() -> None:
    person = walker("p", y2=900)
    dog = walker("d", y2=960, h=90, label="dog")
    out = {s.kind: s for s in stories(person + dog)}
    assert out["person"].sentence.startswith("A person with a dog walked left to right")
    assert out["person"].together_story_ids == [out["dog"].story_id]
    assert out["dog"].companion_kinds == ["person"]
    # One walk-by, one story: the dog's own passage is not published again.
    assert out["person"].published and not out["dog"].published and out["dog"].drop_reason == "companion"

    a, b = walker("a", y2=900), walker("b", x0=640, y2=930)
    sa, sb = stories(a + b)
    assert "together with 1 other person" in sa.sentence and "together with 1 other person" in sb.sentence


def test_opposite_directions_are_not_together() -> None:
    a = walker("a", y2=900)
    b = walker("b", x0=1240, dx=-160, y2=930)
    assert all(not s.together_story_ids for s in stories(a + b))


def test_clean_flag_and_thumb_evidence() -> None:
    cs = walker()
    cs[1] = vm.MotionCrop(cs[1].crop_id, cs[1].t, "person", cs[1].box, thumb_ref="thumb:" + "a" * 64)
    [s] = stories(cs, clean_since=T0 + timedelta(seconds=1))
    assert s.clean is False
    assert [e["thumb"] for e in s.evidence] == [None, "a" * 64, None, None, None]
    [s] = stories(cs, clean_since=T0)
    assert s.clean is True


# --- the loop's cursor maths ----------------------------------------------------------------------------


def _step(crops, pu, horizon, cfg=vl.LoopConfig()):
    ws = pu - timedelta(seconds=cfg.window_sec)
    window = [c for c in crops if ws < c.t <= horizon]
    return vl.close_window(window, stream_id="walkway", processed_until=pu, horizon=horizon, window_start=ws,
                           cfg=cfg, zones=ZONES, context=({}, set(), []))


def test_close_window_emits_each_passage_once_across_ticks() -> None:
    crops = walker("a") + walker("b", start=40.0) + walker("c", start=80.0)
    pu = T0 - timedelta(seconds=30)
    seen = []
    for tick in range(1, 20):
        horizon = T0 + timedelta(seconds=tick * 15)
        out, pu = _step(crops, pu, horizon)
        seen += [s.story_id for s in out]
    assert sorted(seen) == sorted(vm.story_id_for("walkway", f"{p}0") for p in "abc")


def test_an_open_passage_waits_for_the_link_gap() -> None:
    crops = walker("a")          # last box at +10 s
    out, cur = _step(crops, T0 - timedelta(seconds=30), T0 + timedelta(seconds=15))
    assert out == [] and cur == T0 + timedelta(seconds=4)
    out, _ = _step(crops, cur, T0 + timedelta(seconds=30))
    assert [s.story_id for s in out] == [vm.story_id_for("walkway", "a0")]


def test_a_passage_longer_than_the_window_is_not_published() -> None:
    cfg = vl.LoopConfig(window_sec=20.0)
    crops = walker("a", n=20)    # 47.5 s long
    out, _ = _step(crops, T0 + timedelta(seconds=40), T0 + timedelta(seconds=70), cfg)
    assert [s.drop_reason for s in out] == ["too_long"]


# --- config parity --------------------------------------------------------------------------------------


def test_settings_ship_stories_on_and_reach_the_loop_config() -> None:
    from app.settings import Settings

    s = Settings()
    assert s.vision_motion_stories_enabled is True
    cfg = vl.loop_config(s)
    assert cfg.streams == ("walkway", "doorbell") and cfg.motion.min_detections == 3 and cfg.motion.link_gap_sec == 11.0
    assert cfg.clean_since == datetime(2026, 10, 16, 18, 59, tzinfo=timezone.utc)
    assert cfg.prop_individual_ids == ("fefb24d7e4074e9f827ad711874f6edf",)
    assert (cfg.motion.frame_w, cfg.motion.frame_h) == (2560.0, 1920.0)
    env = (SERVICE / ".env_example").read_text()
    for key in ("VISION_MOTION_STORIES_ENABLED=true", "VISION_MOTION_STORY_INTERVAL_SEC=15",
                "VISION_MOTION_STORY_CLOSE_GAP_SEC=11", "VISION_MOTION_STORY_MIN_DETECTIONS=3",
                "VISION_MOTION_CLEAN_SINCE=2026-10-16T18:59:00Z"):
        assert key in env
    assert "VISION_MOTION_STORIES_ENABLED=${VISION_MOTION_STORIES_ENABLED:-true}" in (
        SERVICE / "docker-compose.yml").read_text()


@pytest.mark.asyncio
async def test_kill_switch_off_returns_immediately() -> None:
    from app.settings import Settings

    await asyncio.wait_for(vl.vision_motion_loop(Settings(VISION_MOTION_STORIES_ENABLED=False)), timeout=1.0)


@pytest.mark.asyncio
async def test_cancellation_during_sleep_stops_the_loop() -> None:
    from app.settings import Settings

    s = Settings(VISION_MOTION_STORY_INTERVAL_SEC=3600, POSTGRES_URI="postgresql://x")
    task = asyncio.create_task(vl.vision_motion_loop(s))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)


# --- migration ------------------------------------------------------------------------------------------


def test_migration_passes_the_post_pull_auto_apply_filter() -> None:
    from orion.sql_migration_drift import auto_apply_refusals, parse_required_by

    text = MIGRATION.read_text()
    assert auto_apply_refusals(text, {"vision_crop_observation", "vision_individual"}) == []
    assert auto_apply_refusals(text) == []      # unknown catalog: every table counts as existing
    rb = parse_required_by(text)
    assert rb is not None and rb.services == ("orion-sql-writer", "orion-hub") and not rb.errors
    assert "DO $$" not in text and "CREATE FUNCTION" not in text.upper()


def test_post_pull_applier_would_apply_it() -> None:
    import importlib.util

    from orion import sql_migration_drift as d

    spec = importlib.util.spec_from_file_location("apply_post_pull_migrations",
                                                  REPO / "scripts" / "apply_post_pull_migrations.py")
    ap = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("apply_post_pull_migrations", ap)
    spec.loader.exec_module(ap)
    result = d.FileResult(name=MIGRATION.name, in_window=True, changed_at=T0, status="MISSING")
    item = ap.classify(f"services/orion-sql-db/{MIGRATION.name}", MIGRATION.read_text(), result,
                       {"vision_crop_observation", "vision_individual"})
    assert item.action == "apply", item.reason


# --- the whole path against real Postgres ---------------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE vision_motion_story, vision_motion_story_verdict, vision_motion_story_cursor, "
                          "vision_crop_observation, vision_individual_sighting, vision_individual CASCADE"))
    yield engine
    engine.dispose()


def _insert_crops(engine, crops) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for c in crops:
            conn.execute(text(
                "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
                "box_xyxy, zone, individual_id, thumb_ref, created_at) VALUES (:c, :o, 'walkway', :t, :l, 0.5, :b, "
                "'walkway', :i, :th, :t)"),
                {"c": c.crop_id, "o": "obs-" + c.crop_id, "t": c.t, "l": c.label, "b": list(c.box),
                 "i": c.individual_id, "th": c.thumb_ref})


def _add_individual(engine, iid: str, *, label=None, stationary=False, still_box=None) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, first_seen_at, last_seen_at, "
            "label, stationary, still_box) VALUES (:i, 'walkway', 'person', ARRAY[1.0]::real[], :t, :t, :l, :s, :b)"),
            {"i": iid, "t": T0, "l": label, "s": stationary, "b": still_box})


@needs_pg
def test_cycle_writes_stories_moves_the_cursor_and_is_idempotent(pg) -> None:
    from sqlalchemy import text

    _add_individual(pg, "ind-alex", label="Alex")
    _add_individual(pg, "ind-skel", stationary=True, still_box=[2136, 878, 2216, 1149])
    alex = walker("a", ind="ind-alex")
    alex[2] = vm.MotionCrop(alex[2].crop_id, alex[2].t, "person", alex[2].box, individual_id="ind-alex",
                            thumb_ref="thumb:" + "b" * 64)
    skel = [crop(f"k{i}", i * 5.0, 2176, 1149, h=271, ind="ind-skel") for i in range(6)]
    glimpse = [crop("g0", 60, 600, 1500), crop("g1", 62.5, 700, 1500)]
    _insert_crops(pg, alex + skel + glimpse)
    cfg = vl.LoopConfig(initial_lookback_sec=600, clean_since=T0 - timedelta(hours=1))
    now = T0 + timedelta(seconds=200)

    summary = vl.run_one_motion_cycle(postgres_uri=PG, cfg=cfg, zones_by_stream={"walkway": ZONES}, now=now)
    assert summary["streams"] == 1 and summary["published"] == 1 and summary.get("prop") == 1
    assert summary.get("glimpse") == 1
    with pg.connect() as conn:
        rows = conn.execute(text(
            "SELECT kind, published, drop_reason, sentence, individual_id, name_label, crop_ids, evidence, clean, "
            "rules_version FROM vision_motion_story ORDER BY started_at")).mappings().all()
        cur = conn.execute(text("SELECT processed_until FROM vision_motion_story_cursor")).scalar()
    pub = [r for r in rows if r["published"]]
    assert len(pub) == 1 and pub[0]["sentence"].startswith("Alex walked left to right")
    assert pub[0]["individual_id"] == "ind-alex" and pub[0]["clean"] is True and pub[0]["rules_version"] == 1
    assert pub[0]["crop_ids"] == [c.crop_id for c in alex]
    assert pub[0]["evidence"][2]["thumb"] == "b" * 64 and pub[0]["evidence"][0]["box"] == list(alex[0].box)
    assert {r["drop_reason"] for r in rows if not r["published"]} == {"prop", "glimpse"}
    assert cur == now - timedelta(seconds=cfg.settle_sec + cfg.motion.link_gap_sec)

    # The next tick has nothing new; a replay from scratch writes nothing twice.
    again = vl.run_one_motion_cycle(postgres_uri=PG, cfg=cfg, zones_by_stream={"walkway": ZONES},
                                    now=now + timedelta(seconds=15))
    assert again["written"] == 0
    with pg.begin() as conn:
        conn.execute(text("DELETE FROM vision_motion_story_cursor"))
    replay = vl.run_one_motion_cycle(postgres_uri=PG, cfg=cfg, zones_by_stream={"walkway": ZONES}, now=now)
    assert replay["closed"] == 3 and replay["written"] == 0


@needs_pg
def test_cycle_on_a_missing_table_raises_migration_missing(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        conn.execute(text("ALTER TABLE vision_motion_story_cursor RENAME TO vision_motion_story_cursor_x"))
    try:
        with pytest.raises(vl.MigrationMissing):
            vl.run_one_motion_cycle(postgres_uri=PG, cfg=vl.LoopConfig(), zones_by_stream={}, now=T0)
    finally:
        with pg.begin() as conn:
            conn.execute(text("ALTER TABLE vision_motion_story_cursor_x RENAME TO vision_motion_story_cursor"))


@needs_pg
def test_retention_keeps_stories_juniper_has_checked(pg) -> None:
    from sqlalchemy import text

    old = T0 - timedelta(days=100)
    with pg.begin() as conn:
        for sid in ("a" * 32, "b" * 32):
            conn.execute(text("INSERT INTO vision_motion_story (story_id, stream_id, started_at, ended_at, published) "
                              "VALUES (:s, 'walkway', :t, :t, true)"), {"s": sid, "t": old})
        conn.execute(text("INSERT INTO vision_motion_story_verdict (story_id, verdict) VALUES (:s, 'right')"),
                     {"s": "a" * 32})
    summary = vl.run_one_motion_cycle(postgres_uri=PG, cfg=vl.LoopConfig(streams=()), zones_by_stream={}, now=T0)
    assert summary["pruned"] == 1
    with pg.connect() as conn:
        assert [r[0] for r in conn.execute(text("SELECT story_id FROM vision_motion_story"))] == ["a" * 32]


def test_fill_evidence_links_fills_only_nulls_in_place_order() -> None:
    from app import vision_motion_loop as vl

    ev = [{"crop_id": "c1", "individual_id": None, "thumb": "t"}, {"crop_id": "c2", "individual_id": "ind-a"},
          {"crop_id": "c3", "individual_id": None}, {"individual_id": None}]
    assert vl.unlinked_crop_ids(ev) == ["c1", "c3"]
    out, n = vl.fill_evidence_links(ev, {"c1": "ind-j", "c2": "ind-z"})
    assert n == 1 and [e.get("individual_id") for e in out] == ["ind-j", "ind-a", None, None]
    assert out[0]["thumb"] == "t" and ev[0]["individual_id"] is None      # input untouched
    assert vl.fill_evidence_links(None, {"c1": "x"}) == (None, 0)
    assert vl.unlinked_crop_ids("not a list") == []
