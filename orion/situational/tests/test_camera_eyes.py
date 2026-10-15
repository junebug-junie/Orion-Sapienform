"""Orion's cameras as Orion's own senses (2026-10-15, live turn caa3eb4d).

Pins: the street line says what the camera is; the brief lists every camera
with live/not-seeing state derived from real rows; empty or missing tables
break nothing; the shipped description default equals .env_example.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, text

from orion.schemas.situation import PerceptionContextV1, SituationDiagnosticsV1
from orion.situational import camera_eyes
from orion.situational import context as situation_mod
from orion.situational.camera_eyes import (
    DEFAULT_CAMERA_DESCRIPTIONS,
    CameraEye,
    build_senses,
    camera_descriptions,
    describe_camera,
    fetch_camera_eyes,
    senses_line,
    street_framing,
)
from orion.situational.perception_reader import StreetSummary

REPO = Path(__file__).resolve().parents[3]
TZ = ZoneInfo("America/Denver")
# 09:00 MDT on 2026-10-15.
NOW = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)


# --- descriptions -----------------------------------------------------------


def test_default_walkway_is_outside_and_ours() -> None:
    d = describe_camera("walkway", DEFAULT_CAMERA_DESCRIPTIONS)
    assert d.name == "your outdoor camera"
    assert "community walkway" in d.view and "patio" in d.view
    assert d.outside is True
    assert describe_camera("cam0", DEFAULT_CAMERA_DESCRIPTIONS).outside is False


def test_undescribed_camera_is_plain_and_never_outside() -> None:
    d = describe_camera("porch_eye", {})
    assert d.name == "camera porch_eye" and d.view == "" and d.outside is False


def test_env_overrides_and_bad_env_falls_back() -> None:
    raw = json.dumps({"walkway": {"name": "the street eye", "view": "the road", "outside": True}})
    assert camera_descriptions(raw)["walkway"]["name"] == "the street eye"
    assert camera_descriptions("not json") == DEFAULT_CAMERA_DESCRIPTIONS
    assert camera_descriptions("[1, 2]") == DEFAULT_CAMERA_DESCRIPTIONS
    assert camera_descriptions("") == DEFAULT_CAMERA_DESCRIPTIONS


@pytest.mark.parametrize("service", ["orion-cortex-exec", "orion-hub", "orion-thought"])
def test_env_example_matches_code_default(service: str) -> None:
    lines = (REPO / "services" / service / ".env_example").read_text().splitlines()
    (value,) = [ln.split("=", 1)[1] for ln in lines if ln.startswith(f"{camera_eyes.ENV_KEY}=")]
    assert json.loads(value) == DEFAULT_CAMERA_DESCRIPTIONS


# --- live state from real-shaped tables -------------------------------------


def _engine(*, tables: bool = True, stationary: bool = True):
    eng = create_engine("sqlite://")
    if not tables:
        return eng
    with eng.begin() as c:
        c.execute(text("CREATE TABLE vision_scene_inventory (stream_id TEXT, observed_at TIMESTAMP)"))
        cols = "individual_id TEXT, stream_id TEXT, ended_at TIMESTAMP, zone_no_embed BOOLEAN"
        if stationary:
            cols += ", stationary BOOLEAN"
        c.execute(text(f"CREATE TABLE vision_individual_sighting ({cols})"))
    return eng


def _frames(eng, stream: str, *ages: timedelta) -> None:
    with eng.begin() as c:
        for age in ages:
            c.execute(
                text("INSERT INTO vision_scene_inventory VALUES (:s, :t)"),
                {"s": stream, "t": NOW - age},
            )


def _sighting(eng, ind: str, age: timedelta, *, stationary: bool = False, no_embed: bool = False) -> None:
    with eng.begin() as c:
        c.execute(
            text("INSERT INTO vision_individual_sighting VALUES (:i, 'walkway', :t, :z, :st)"),
            {"i": ind, "t": NOW - age, "z": no_embed, "st": stationary},
        )


def test_empty_tables_read_as_no_frames_not_unread() -> None:
    eyes = fetch_camera_eyes(["cam0", "walkway"], count_individuals_for=["walkway"], engine=_engine(), now=NOW)
    assert [e.stream_id for e in eyes] == ["cam0", "walkway"]
    assert all(e.last_frame_at is None and e.first_frame_at is None for e in eyes)
    assert eyes[1].individuals_today == 0
    senses = build_senses(eyes, now=NOW, tz=TZ, live_max_age_seconds=900)
    assert [s.status for s in senses] == ["no_frames", "no_frames"]
    assert "no frames yet" in senses_line(senses)


def test_missing_tables_are_unread_and_no_engine_is_unread() -> None:
    assert fetch_camera_eyes(["walkway"], engine=_engine(tables=False), now=NOW) is None
    assert fetch_camera_eyes([], engine=_engine(), now=NOW) == []


def test_fresh_walkway_is_live_with_todays_individuals() -> None:
    eng = _engine()
    _frames(eng, "walkway", timedelta(hours=10), timedelta(seconds=9))
    _frames(eng, "cam0", timedelta(days=50), timedelta(seconds=6))
    _frames(eng, "carbon", timedelta(days=50), timedelta(days=36))
    _sighting(eng, "a", timedelta(hours=1))
    _sighting(eng, "a", timedelta(minutes=5))
    _sighting(eng, "b", timedelta(minutes=2))
    _sighting(eng, "fixture", timedelta(minutes=1), stationary=True)
    _sighting(eng, "c", timedelta(hours=12))  # yesterday, local time
    eyes = fetch_camera_eyes(
        ["carbon", "cam0", "walkway"], count_individuals_for=["walkway"], engine=eng, now=NOW, tz_name="America/Denver"
    )
    walk = eyes[2]
    assert walk.individuals_today == 2
    assert eyes[0].individuals_today is None  # room cameras are not counted
    senses = build_senses(eyes, now=NOW, tz=TZ, live_max_age_seconds=900, descriptions=DEFAULT_CAMERA_DESCRIPTIONS)
    assert [s.status for s in senses] == ["not_seeing", "live", "live"]
    line = senses_line(senses)
    assert line.startswith("Your cameras (your own eyes): ")
    assert "your room camera (cam0" in line
    assert (
        "your outdoor camera (walkway, since 2026-10-14; looks at the community walkway, the street"
        " and our patio -- outside and public, not your room): receiving frames (newest within the last 15 min),"
        " 2 individuals counted today (fixtures excluded)"
    ) in line
    assert "the laptop webcam (carbon" in line and "no frames arriving, last frame 36 days ago" in line


def test_stale_walkway_is_not_seeing_and_drops_its_count() -> None:
    eng = _engine()
    _frames(eng, "walkway", timedelta(hours=3))
    _sighting(eng, "a", timedelta(hours=3, minutes=5))
    (eye,) = fetch_camera_eyes(["walkway"], count_individuals_for=["walkway"], engine=eng, now=NOW)
    (sense,) = build_senses([eye], now=NOW, tz=TZ, live_max_age_seconds=900)
    assert sense.status == "not_seeing" and sense.individuals_today is None
    assert senses_line([sense]).endswith("no frames arriving, last frame 3 h ago.")


def test_pre_stationary_schema_still_counts() -> None:
    eng = _engine(stationary=False)
    _frames(eng, "walkway", timedelta(seconds=5))
    with eng.begin() as c:
        c.execute(text("INSERT INTO vision_individual_sighting VALUES ('a', 'walkway', :t, 0)"), {"t": NOW})
    (eye,) = fetch_camera_eyes(["walkway"], count_individuals_for=["walkway"], engine=eng, now=NOW)
    assert eye.individuals_today == 1


def test_street_framing_names_the_eye_and_its_first_day() -> None:
    d = describe_camera("walkway", DEFAULT_CAMERA_DESCRIPTIONS)
    first = datetime(2026, 10, 15, 4, 56, tzinfo=timezone.utc)  # 22:56 MDT on the 14th
    assert street_framing(d, first, TZ) == (
        "Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway,"
        " the street and our patio"
    )
    assert street_framing(d, None, TZ).startswith("Your outdoor camera (walkway), looking over")


# --- the brief ----------------------------------------------------------------


def _cfg(**over):
    cfg = situation_mod.settings_from_runtime(SimpleNamespace())
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _stub(monkeypatch, eyes):
    from orion.situational.perception_reader import PresenceResolution

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: None)
    monkeypatch.setattr(
        situation_mod, "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution(None, None, True),
    )
    monkeypatch.setattr(
        situation_mod, "fetch_street_summary",
        lambda sid, *, tz_name: StreetSummary(sid, ["Around the walkway now: an unfamiliar person (seen 2 min ago)."], True),
    )
    seen = {}

    def _fetch(streams, **kw):
        seen["streams"] = list(streams)
        seen["counted"] = list(kw.get("count_individuals_for") or [])
        return eyes

    monkeypatch.setattr(situation_mod, "fetch_camera_eyes", _fetch)
    monkeypatch.delenv(camera_eyes.ENV_KEY, raising=False)
    return seen


def _brief_text(ctx: PerceptionContextV1) -> str:
    from orion.schemas.situation import SituationBriefV1

    cfg = _cfg(perception_enabled=False)
    diag = SituationDiagnosticsV1()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    brief = SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, NOW)),
        place=situation_mod._build_place_context(cfg),
        perception=ctx,
    )
    return situation_mod._build_prompt_fragment(brief, 7200).compact_text


def test_brief_lists_room_and_walkway_and_the_street_line_says_what_it_is(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    eyes = [
        CameraEye("carbon", now - timedelta(days=50), now - timedelta(days=36)),
        CameraEye("cam0", now - timedelta(days=50), now - timedelta(seconds=6)),
        CameraEye("walkway", datetime(2026, 10, 15, 4, 56, tzinfo=timezone.utc), now - timedelta(seconds=9), 3),
    ]
    seen = _stub(monkeypatch, eyes)
    diag = SituationDiagnosticsV1()
    cfg = _cfg(perception_enabled=True, perception_stream_ids=["carbon", "cam0"])
    ctx = asyncio.run(situation_mod._build_perception_context(cfg, diag))
    assert seen == {"streams": ["carbon", "cam0", "walkway"], "counted": ["walkway"]}
    assert diag.provider_status["perception_cameras"] == "ok"
    assert [c.status for c in ctx.cameras] == ["not_seeing", "live", "live"]
    text = _brief_text(ctx)
    assert "- Your cameras (your own eyes): " in text
    assert "your room camera (cam0" in text
    assert "outside and public, not your room): receiving frames (newest within the last 15 min), 3 individuals counted today (fixtures excluded)" in text
    assert (
        "- Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway, the street"
        " and our patio: Around the walkway now: an unfamiliar person (seen 2 min ago)."
    ) in text


def test_unread_cameras_leave_no_line_and_street_still_framed(monkeypatch) -> None:
    _stub(monkeypatch, None)
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.cameras == []
    assert diag.provider_status["perception_cameras"] == "unread"
    text = _brief_text(ctx)
    assert "Your cameras" not in text
    assert "- Your outdoor camera (walkway), looking over" in text


def test_camera_read_error_is_fail_open(monkeypatch) -> None:
    _stub(monkeypatch, None)

    def _boom(*a, **k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(situation_mod, "fetch_camera_eyes", _boom)
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.cameras == [] and diag.provider_status["perception_cameras"] == "error"
    assert ctx.street_summary.startswith("Your outdoor camera (walkway)")


def test_cache_gate_rebuilds_once_a_live_camera_could_have_aged_out() -> None:
    from orion.schemas.situation import CameraSenseV1

    live = CameraSenseV1(
        stream_id="walkway", name="your outdoor camera", status="live",
        live_window_seconds=900, last_frame_age_seconds=700,
    )
    brief = SimpleNamespace(
        perception=PerceptionContextV1(cameras=[live]),
        cabinet=SimpleNamespace(sound_available=False, sound_age_seconds=None),
    )
    cfg = _cfg()
    assert situation_mod._cached_percept_outlived_gate(brief, 100, cfg) is False
    assert situation_mod._cached_percept_outlived_gate(brief, 250, cfg) is True


def test_outside_must_be_a_real_true() -> None:
    assert describe_camera("cam0", {"cam0": {"name": "x", "outside": "false"}}).outside is False
    assert describe_camera("w", {"w": {"name": "x", "outside": True}}).outside is True


def test_only_a_missing_column_buys_the_fallback_statement() -> None:
    assert camera_eyes._is_missing_column(Exception("no such column: stationary"))
    assert not camera_eyes._is_missing_column(Exception("canceling statement due to statement timeout"))
    assert not camera_eyes._is_missing_column(None)


def test_frames_arriving_without_a_room_description_do_not_claim_sight(monkeypatch) -> None:
    """cam0 frames keep flowing while its captions lag or go blank (live: 9 min).
    The camera clause must say frames arrive, never that Orion sees the room,
    and the Room line keeps its own "do not infer"."""
    now = datetime.now(timezone.utc)
    _stub(monkeypatch, [CameraEye("cam0", now - timedelta(days=5), now - timedelta(seconds=5))])
    cfg = _cfg(perception_enabled=True, perception_stream_ids=["cam0"], street_stream_ids=[])
    ctx = asyncio.run(situation_mod._build_perception_context(cfg, SituationDiagnosticsV1()))
    assert ctx.available is False
    text = _brief_text(ctx)
    assert "your room camera (cam0" in text and "receiving frames (newest within the last 15 min)" in text
    assert "Room: haven't seen anything recently; do not infer." in text
    assert ": live" not in text


def test_slow_camera_read_hits_the_deadline_not_the_turn(monkeypatch) -> None:
    import time as _time

    _stub(monkeypatch, None)
    monkeypatch.setattr(situation_mod, "CAMERA_EYES_DEADLINE_SEC", 0.05)
    monkeypatch.setattr(situation_mod, "fetch_camera_eyes", lambda *a, **k: _time.sleep(0.5))
    diag = SituationDiagnosticsV1()
    t0 = _time.monotonic()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), diag))
    assert diag.provider_status["perception_cameras"] == "timeout" and ctx.cameras == []
    assert ctx.street_summary.startswith("Your outdoor camera (walkway)")
    assert _time.monotonic() - t0 < 2.0
