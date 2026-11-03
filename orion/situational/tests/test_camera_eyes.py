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


def _engine(*, tables: bool = True, rollup: bool = True):
    eng = create_engine("sqlite://")
    if not tables:
        return eng
    with eng.begin() as c:
        c.execute(text("CREATE TABLE vision_scene_inventory (stream_id TEXT, observed_at TIMESTAMP)"))
        # The bare-count source this clause used to read: present and full, so a
        # test proves the clause no longer reads it.
        c.execute(text(
            "CREATE TABLE vision_individual_sighting (individual_id TEXT, stream_id TEXT, ended_at TIMESTAMP, "
            "zone_no_embed BOOLEAN, stationary BOOLEAN)"
        ))
        if rollup:
            c.execute(text(
                "CREATE TABLE vision_rollup (stream_id TEXT, grain TEXT, period_start TIMESTAMP, rollup TEXT)"
            ))
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


# 2026-10-15 local midnight, America/Denver (MDT, UTC-6).
TODAY_START = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)


def _day_rollup(eng, rollup: dict, *, start: datetime = TODAY_START, stream: str = "walkway") -> None:
    with eng.begin() as c:
        c.execute(
            text("INSERT INTO vision_rollup VALUES (:s, 'day', :p, :r)"),
            {"s": stream, "p": start, "r": json.dumps(rollup)},
        )


def _v4_day(low: int, high: int, *, through: str = "2026-10-15T14:00:00+00:00") -> dict:
    """A rules_version 4 day row as orion-sql-writer writes it (live 2026-11-02 shape)."""
    return {
        "schema": "vision.rollup.v1", "rules_version": 4, "grain": "day", "covered_until": through,
        "kinds": {
            "person": {"passings": high, "distinct": high,
                       "count_range": {"low": low, "high": high, "method_low": "peak_at_once",
                                       "method_high": "clusters"}},
            "vehicle": {"passings": 109, "distinct": 70},
        },
        "peak_at_once": {"person": low, "vehicle": 4},
        "total": {"passings": high + 109},
    }


def test_empty_tables_read_as_no_frames_not_unread() -> None:
    eyes = fetch_camera_eyes(["cam0", "walkway"], people_range_for=["walkway"], engine=_engine(), now=NOW)
    assert [e.stream_id for e in eyes] == ["cam0", "walkway"]
    assert all(e.last_frame_at is None and e.first_frame_at is None for e in eyes)
    assert eyes[1].people_today is None  # no day rollup yet: no number at all
    senses = build_senses(eyes, now=NOW, tz=TZ, live_max_age_seconds=900)
    assert [s.status for s in senses] == ["no_frames", "no_frames"]
    assert "no frames yet" in senses_line(senses)


def test_missing_tables_are_unread_and_no_engine_is_unread() -> None:
    assert fetch_camera_eyes(["walkway"], engine=_engine(tables=False), now=NOW) is None
    assert fetch_camera_eyes([], engine=_engine(), now=NOW) == []


def test_fresh_walkway_is_live_with_todays_people_range() -> None:
    eng = _engine()
    _frames(eng, "walkway", timedelta(hours=10), timedelta(seconds=9))
    _frames(eng, "cam0", timedelta(days=50), timedelta(seconds=6))
    _frames(eng, "carbon", timedelta(days=50), timedelta(days=36))
    for i in range(111):  # the clusters the old clause counted bare (live turn 4bbaaf2a)
        _sighting(eng, f"ind{i}", timedelta(minutes=5))
    _day_rollup(eng, _v4_day(2, 15))
    _day_rollup(eng, _v4_day(5, 79), start=TODAY_START - timedelta(days=1))  # yesterday is not today
    eyes = fetch_camera_eyes(
        ["carbon", "cam0", "walkway"], people_range_for=["walkway"], engine=eng, now=NOW, tz_name="America/Denver"
    )
    walk = eyes[2]
    assert walk.people_today == {"low": 2, "high": 15}
    assert eyes[0].people_today is None  # room cameras are not counted
    senses = build_senses(eyes, now=NOW, tz=TZ, live_max_age_seconds=900, descriptions=DEFAULT_CAMERA_DESCRIPTIONS)
    assert [s.status for s in senses] == ["not_seeing", "live", "live"]
    assert (senses[2].people_today_low, senses[2].people_today_high, senses[2].people_today_through) == (2, 15, "08:00")
    line = senses_line(senses)
    assert line.startswith("Your cameras (your own eyes): ")
    assert "your room camera (cam0" in line
    assert (
        "your outdoor camera (walkway, since 2026-10-14; looks at the community walkway, the street"
        " and our patio -- outside and public, not your room): receiving frames (newest within the last 15 min);"
        " today so far (through 08:00), people: at least 2, at most 15 (the camera splits one person into several)"
    ) in line
    assert "111" not in line and "individual" not in line
    assert "the laptop webcam (carbon" in line and "no frames arriving, last frame 36 days ago" in line


def test_no_people_today_drops_the_number_not_says_zero() -> None:
    eng = _engine()
    _frames(eng, "walkway", timedelta(seconds=9))
    for i in range(7):
        _sighting(eng, f"car{i}", timedelta(minutes=5))
    day = _v4_day(0, 0)
    day["kinds"].pop("person")
    _day_rollup(eng, day)
    (eye,) = fetch_camera_eyes(["walkway"], people_range_for=["walkway"], engine=eng, now=NOW)
    assert eye.people_today is None
    line = senses_line(build_senses([eye], now=NOW, tz=TZ, live_max_age_seconds=900))
    assert line.endswith("receiving frames (newest within the last 15 min).")
    assert "people" not in line and "individual" not in line


def test_pre_range_rollup_row_is_derived_into_a_range() -> None:
    """A rules_version 1 day row has no count_range: person_count_range derives it
    (the day letter's rule), so the clause still never prints a bare count."""
    eng = _engine()
    _frames(eng, "walkway", timedelta(seconds=9))
    _day_rollup(eng, {"rules_version": 1, "kinds": {"person": {"passings": 79}}, "peak_at_once": {"person": 5}})
    (eye,) = fetch_camera_eyes(["walkway"], people_range_for=["walkway"], engine=eng, now=NOW)
    (sense,) = build_senses([eye], now=NOW, tz=TZ, live_max_age_seconds=900)
    line = senses_line([sense])
    assert "; today so far, people: at least 5, at most 79 (the camera splits one person into several)." in line


def test_missing_or_bad_rollup_is_no_number_and_cameras_still_read() -> None:
    eng = _engine(rollup=False)
    _frames(eng, "walkway", timedelta(seconds=9))
    (eye,) = fetch_camera_eyes(["walkway"], people_range_for=["walkway"], engine=eng, now=NOW)
    assert eye.last_frame_at is not None and eye.people_today is None
    eng2 = _engine()
    _frames(eng2, "walkway", timedelta(seconds=9))
    with eng2.begin() as c:
        c.execute(text("INSERT INTO vision_rollup VALUES ('walkway', 'day', :p, 'not json')"), {"p": TODAY_START})
    (eye2,) = fetch_camera_eyes(["walkway"], people_range_for=["walkway"], engine=eng2, now=NOW)
    assert eye2.people_today is None


def test_stale_walkway_is_not_seeing_and_drops_its_range() -> None:
    eng = _engine()
    _frames(eng, "walkway", timedelta(hours=3))
    _day_rollup(eng, _v4_day(2, 15))
    (eye,) = fetch_camera_eyes(["walkway"], people_range_for=["walkway"], engine=eng, now=NOW)
    (sense,) = build_senses([eye], now=NOW, tz=TZ, live_max_age_seconds=900)
    assert sense.status == "not_seeing" and sense.people_today_low is None and sense.people_today_high is None
    assert senses_line([sense]).endswith("no frames arriving, last frame 3 h ago.")


def test_quiet_doorbell_reads_waiting_not_broken(monkeypatch) -> None:
    """2026-10-17: the Ring doorbell only sends frames after a visit. Hours
    without one is a quiet door, not a failing eye."""
    monkeypatch.delenv("ORION_CAMERA_DESCRIPTIONS", raising=False)
    eng = _engine()
    _frames(eng, "doorbell", timedelta(hours=3))
    eyes = fetch_camera_eyes(["doorbell", "carbon"], people_range_for=["doorbell"], engine=eng, now=NOW)
    senses = build_senses(eyes, now=NOW, tz=TZ, live_max_age_seconds=900)
    assert [s.status for s in senses] == ["not_seeing", "no_frames"]
    line = senses_line(senses)
    assert "your doorbell camera (doorbell" in line
    assert "sends frames only when something happens there; last time 3 h ago" in line
    assert "the laptop webcam (carbon" in line and "no frames yet" in line  # room cameras unchanged
    assert describe_camera("walkway").event_driven is False


def test_no_bare_people_count_anywhere_in_the_eyes_line() -> None:
    """Spec #2763 decision 5 (Juniper): a people number only ever comes with its range."""
    import re

    from orion.schemas.situation import CameraSenseV1

    assert "individuals_today" not in CameraSenseV1.model_fields
    for low, high in ((1, 1), (2, 15), (0, 3)):
        sense = CameraSenseV1(stream_id="walkway", name="your outdoor camera", status="live", live_window_seconds=900,
                              last_frame_age_seconds=5, people_today_low=low, people_today_high=high)
        line = senses_line([sense])
        assert f"people: at least {low}, at most {high}" in line
        assert not re.search(r"\b\d+ (people|persons|individuals?)\b", line)


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
        seen["counted"] = list(kw.get("people_range_for") or [])
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
        CameraEye("walkway", datetime(2026, 10, 15, 4, 56, tzinfo=timezone.utc), now - timedelta(seconds=9),
                  {"low": 1, "high": 3}),
    ]
    seen = _stub(monkeypatch, eyes)
    diag = SituationDiagnosticsV1()
    cfg = _cfg(perception_enabled=True, perception_stream_ids=["carbon", "cam0"], street_stream_ids=["walkway"])
    ctx = asyncio.run(situation_mod._build_perception_context(cfg, diag))
    assert seen == {"streams": ["carbon", "cam0", "walkway"], "counted": ["walkway"]}
    assert diag.provider_status["perception_cameras"] == "ok"
    assert [c.status for c in ctx.cameras] == ["not_seeing", "live", "live"]
    text = _brief_text(ctx)
    assert "- Your cameras (your own eyes): " in text
    assert "your room camera (cam0" in text
    assert (
        "outside and public, not your room): receiving frames (newest within the last 15 min); today so far,"
        " people: at least 1, at most 3 (the camera splits one person into several)"
    ) in text
    assert "individuals counted today" not in text
    assert (
        "- Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway, the street"
        " and our patio: Around the walkway now: an unfamiliar person (seen 2 min ago)."
    ) in text


def test_unread_cameras_leave_no_line_and_street_still_framed(monkeypatch) -> None:
    _stub(monkeypatch, None)
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True, street_stream_ids=["walkway"]), diag))
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
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True, street_stream_ids=["walkway"]), diag))
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
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True, street_stream_ids=["walkway"]), diag))
    assert diag.provider_status["perception_cameras"] == "timeout" and ctx.cameras == []
    assert ctx.street_summary.startswith("Your outdoor camera (walkway)")
    assert _time.monotonic() - t0 < 2.0
