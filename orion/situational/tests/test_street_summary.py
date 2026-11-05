"""The street in the situation brief (walkway camera spec ideas 5 and 9).

Properties under test: names only when Juniper gave them; the patio is a
count, never a name; an absence claim ("usually here by now") needs evidence;
nothing is said when nothing was read; and the fold into PerceptionContextV1
rides on every room-percept path.
"""

from __future__ import annotations

import os

import pytest

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from orion.schemas.situation import PerceptionContextV1, SituationDiagnosticsV1
from orion.situational import context as situation_mod
from orion.situational.perception_reader import (
    StreetSummary,
    fetch_street_summary,
    summarize_street,
)

TZ = ZoneInfo("America/Denver")
# 07:50 MDT
NOW = datetime(2026, 9, 24, 13, 50, tzinfo=timezone.utc)


def _sighting(iid, kind="dog", label=None, days=1, minutes_ago=2):
    return {"individual_id": iid, "kind": kind, "label": label, "distinct_days": days,
            "first_at": NOW - timedelta(minutes=minutes_ago + 1),
            "last_at": NOW - timedelta(minutes=minutes_ago)}


def _street(**kw):
    base = dict(sightings=[], expectations=[], unresolved=[], patio=None, now=NOW, tz=TZ)
    base.update(kw)
    return summarize_street(**base)


def test_quiet_street_says_nothing() -> None:
    assert _street() == []
    assert _street(sightings=None, expectations=None, unresolved=None) == []


def test_names_only_from_labels_otherwise_unfamiliar_or_counted_days() -> None:
    lines = _street(sightings=[
        _sighting("a", "dog", label="Biscuit"),
        _sighting("b", "person"),
        _sighting("c", "person"),
        _sighting("d", "dog", days=9),
        _sighting("e", "cat", minutes_ago=40),  # outside 15 minutes
    ])
    assert lines == [
        "On the walkway in the last 15 minutes: Biscuit, a dog I have seen on 9 days "
        "but have no name for, and up to 2 unfamiliar people (the camera splits one person into several)."
    ]
    # With the window count read: at most one person stood in one window.
    lines = _street(sightings=[_sighting("b", "person"), _sighting("c", "person")],
                    people_windows=[{"counts": {"person": 1}}])
    assert lines == [
        "On the walkway in the last 15 minutes: at least 1, at most 2 unfamiliar people "
        "(the camera splits one person into several)."
    ]


def test_expectation_outcomes_and_open_windows() -> None:
    exp = [
        {"subject_key": "individual:d1", "subject_label": "the black dog", "status": "missed", "peak_minute": 460},
        {"subject_key": "label:vehicle", "subject_label": "the mail truck", "status": "met", "peak_minute": 440},
        {"subject_key": "individual:d2", "subject_label": "the grey dog", "status": "open", "peak_minute": 465,
         "window_start": NOW - timedelta(minutes=20)},
        {"subject_key": "individual:d3", "subject_label": "the jogger", "status": "open", "peak_minute": 480,
         "window_start": NOW - timedelta(minutes=5)},
    ]
    (line,) = _street(expectations=exp)
    assert line == (
        "Walkway rhythm: the black dog did not come (usually around 07:40); "
        "the mail truck came as expected (usually around 07:20); the grey dog is usually here by now; "
        "the jogger usually comes around 08:00."
    )


def test_open_expectation_already_met_by_a_sighting() -> None:
    exp = [{"subject_key": "individual:d2", "subject_label": "the grey dog", "status": "open",
            "peak_minute": 465, "window_start": NOW - timedelta(minutes=20)}]
    lines = _street(sightings=[_sighting("d2", days=9)], expectations=exp)
    assert "the grey dog is here, as expected" in lines[-1]


def test_no_absence_claim_without_a_sightings_read() -> None:
    exp = [{"subject_key": "individual:d2", "subject_label": "the grey dog", "status": "open",
            "peak_minute": 465, "window_start": NOW - timedelta(minutes=20)}]
    (line,) = _street(sightings=None, expectations=exp)
    assert "usually here by now" not in line
    assert "usually comes around 07:45" in line


def test_unresolved_last_hour() -> None:
    rows = [
        {"observed_at": NOW - timedelta(minutes=3), "description": "something low by the gate"},
        {"observed_at": NOW - timedelta(minutes=30), "description": "a shape"},
    ]
    (line,) = _street(unresolved=rows)
    assert line == ("In the last hour I saw 2 things on the walkway I could not name; "
                    "the latest at 07:47: something low by the gate.")


def test_patio_is_a_count_never_a_name_and_only_when_fresh() -> None:
    fresh = {"state": "present", "count": 3, "subject": "Juniper", "zone": "patio",
             "row_updated_at": datetime.now(timezone.utc)}
    (line,) = _street(patio=fresh)
    assert line == "At least 3 people are on the patio."
    assert "Juniper" not in line
    stale = dict(fresh, row_updated_at=datetime.now(timezone.utc) - timedelta(hours=2))
    assert _street(patio=stale) == []
    absent = dict(fresh, state="absent")
    assert _street(patio=absent) == []
    assert _street(patio=dict(fresh, count=None)) == ["People are on the patio."]


def test_fetch_without_engine_is_unread_not_quiet(monkeypatch) -> None:
    from orion.situational import perception_reader

    monkeypatch.setattr(perception_reader, "_get_engine", lambda: None)
    got = fetch_street_summary("walkway")
    assert got == StreetSummary("walkway", [], False)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return [SimpleNamespace(_mapping=r) for r in self._rows]


class _Conn:
    def __init__(self, tables):
        self.tables = tables
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params):
        sql = str(stmt)
        for needle, rows in self.tables.items():
            if needle in sql:
                if isinstance(rows, Exception):
                    raise rows
                return _Result(rows)
        return _Result([])

    def rollback(self):
        self.rollbacks += 1


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def test_missing_tables_cost_only_their_own_line() -> None:
    missing = RuntimeError('relation "vision_individual_sighting" does not exist')
    conn = _Conn({
        "vision_individual_sighting": missing,
        "vision_percept_expectation": missing,
        "vision_unresolved": [{"observed_at": NOW - timedelta(minutes=3), "description": "a shape"}],
        "substrate_embodied_presence": [],
    })
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.read_ok is True
    assert got.lines == ["In the last hour I saw one thing on the walkway I could not name at 07:47: a shape."]
    # sightings (stationary read + pre-stationary fallback), expectations and
    # the stationary-people read (it joins vision_individual_sighting too).
    assert conn.rollbacks == 4


def test_patio_row_is_read_by_stream_and_stripped_of_identity() -> None:
    seen = {}

    class _PatioConn(_Conn):
        def execute(self, stmt, params):
            if "substrate_embodied_presence" in str(stmt):
                seen.update(params)
            return super().execute(stmt, params)

    conn = _PatioConn({"substrate_embodied_presence": [
        {"presence_id": "walkway:patio",
         "presence_json": '{"state": "present", "count": 1, "subject": "Juniper", "no_embed_zone": true}',
         "updated_at": datetime.now(timezone.utc)},
        # Not flagged as a no-embed zone: never read as one.
        {"presence_id": "walkway:other",
         "presence_json": '{"state": "present", "count": 4}',
         "updated_at": datetime.now(timezone.utc)},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert seen["prefix"] == "walkway:%"
    assert got.lines == ["Someone is on the patio."]


def test_a_second_no_embed_zone_gets_its_own_line_named_from_its_row() -> None:
    now = datetime.now(timezone.utc)
    conn = _Conn({"substrate_embodied_presence": [
        {"presence_id": "walkway:garden",
         "presence_json": '{"state": "present", "subject": {"count": 2}, "no_embed_zone": true}',
         "updated_at": now},
        {"presence_id": "walkway:patio",
         "presence_json": '{"state": "present", "zone": "patio", "subject": {"count": 1}, "no_embed_zone": true}',
         "updated_at": now},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.lines == ["At least 2 people are on the garden.", "Someone is on the patio."]


def test_like_prefix_escapes_wildcards() -> None:
    from orion.situational.perception_reader import _like_prefix

    assert _like_prefix("walk_way%") == "walk\\_way\\%:%"


def test_unresolved_line_names_the_camera_it_came_from() -> None:
    rows = [{"observed_at": NOW - timedelta(minutes=3), "description": "a shape"}]
    (line,) = summarize_street(sightings=[], expectations=[], unresolved=rows, patio=None,
                               now=NOW, tz=TZ, camera="driveway")
    assert "on the driveway" in line and "walkway" not in line


# --- fold into PerceptionContextV1 -------------------------------------------


def _cfg(**over):
    cfg = situation_mod.settings_from_runtime(SimpleNamespace())
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _stub_room(monkeypatch, percept=None):
    from orion.situational.perception_reader import PresenceResolution

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: percept)
    # Camera senses read the database too; tests never reach a real one.
    monkeypatch.setattr(situation_mod, "fetch_camera_eyes", lambda *a, **k: None)
    monkeypatch.setattr(
        situation_mod, "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution(None, None, True),
    )


def test_street_defaults_to_walkway_and_empty_disables() -> None:
    # cam0 (the office camera) joined 2026-11-04: who is in the room, by name.
    assert _cfg().street_stream_ids == ["doorbell", "walkway", "cam0"]
    cfg = situation_mod.settings_from_runtime(SimpleNamespace(orion_situation_street_stream_ids=""))
    assert cfg.street_stream_ids == []


def test_street_rides_on_the_stale_room_path_too(monkeypatch) -> None:
    _stub_room(monkeypatch, percept=None)  # room camera has nothing
    monkeypatch.setattr(
        situation_mod, "fetch_street_summary",
        lambda sid, *, tz_name: StreetSummary(sid, ["Someone is on the patio."], True),
    )
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True, street_stream_ids=["walkway"]), diag))
    assert ctx.available is False
    # Opens with what the camera is (2026-10-15): Orion's own outdoor eye.
    assert ctx.street_summary == (
        "Your outdoor camera (walkway), looking over the community walkway, the street"
        " and our patio: Someone is on the patio."
    )
    assert ctx.street_stream_id == "walkway"
    assert diag.provider_status["perception_street"] == "ok"


def test_every_speaking_street_camera_gets_its_own_part(monkeypatch) -> None:
    """2026-10-17: a visitor at the doorbell must not hide behind a busy
    walkway (first-stream-wins did exactly that)."""
    _stub_room(monkeypatch)
    lines = {"walkway": ["Two people walked by."], "doorbell": ["Someone is at the front door."]}
    monkeypatch.setattr(
        situation_mod, "fetch_street_summary",
        lambda sid, *, tz_name: StreetSummary(sid, lines.get(sid, []), True),
    )
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.street_summary.startswith("Your doorbell camera (doorbell), looking over our front door")
    assert "Someone is at the front door." in ctx.street_summary
    assert ctx.street_summary.endswith("Two people walked by.")
    assert ctx.street_stream_id == "doorbell"
    assert diag.provider_status["perception_street"] == "ok"

    lines["doorbell"] = []  # nobody at the door: the walkway alone speaks
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), SituationDiagnosticsV1()))
    assert ctx.street_stream_id == "walkway"
    assert "doorbell" not in ctx.street_summary


def test_doorbell_line_survives_a_long_walkway_line_cut_at_520(monkeypatch) -> None:
    """chat_stance cuts the joined street text at 520 chars; the doorbell
    goes first so a busy street cannot push it off the end."""
    _stub_room(monkeypatch)
    lines = {"walkway": ["A long walkway caption about the street. " * 20], "doorbell": ["Someone is at the front door."]}
    monkeypatch.setattr(
        situation_mod, "fetch_street_summary",
        lambda sid, *, tz_name: StreetSummary(sid, lines.get(sid, []), True),
    )
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), SituationDiagnosticsV1()))
    assert "Someone is at the front door." in ctx.street_summary[:520]


def test_one_failed_street_read_does_not_hide_the_other(monkeypatch) -> None:
    _stub_room(monkeypatch)

    def _fetch(sid, *, tz_name):
        if sid == "doorbell":
            raise RuntimeError("db hiccup")
        return StreetSummary(sid, ["Two people walked by."], True)

    monkeypatch.setattr(situation_mod, "fetch_street_summary", _fetch)
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.street_stream_id == "walkway"  # first camera after the failed doorbell
    assert "Two people walked by." in (ctx.street_summary or "")
    # The error stays visible; it is not overwritten with "ok".
    assert diag.provider_status["perception_street"] == "error"
    assert diag.provider_errors["perception_street"] == "db hiccup"


def test_office_camera_reads_as_a_word_not_an_id() -> None:
    """2026-11-04: "On the cam0 just now" is not English; the summary's
    camera word for cam0 is "office camera". Other ids are already words."""
    from orion.situational.perception_reader import street_camera_word, summarize_street
    from datetime import datetime, timezone

    assert street_camera_word("cam0") == "office camera"
    assert street_camera_word("walkway") == "walkway"
    now = datetime(2026, 11, 4, 23, 15, tzinfo=timezone.utc)
    lines = summarize_street(
        sightings=[{"individual_id": "i1", "kind": "person", "label": "Juniper", "distinct_days": 3,
                    "first_at": now, "last_at": now}],
        expectations=[], unresolved=[], patio=[], now=now, tz=timezone.utc,
        camera=street_camera_word("cam0"),
    )
    text = " ".join(lines)
    assert "office camera" in text and "Juniper" in text and "cam0" not in text


def test_quiet_or_unread_street_leaves_no_field(monkeypatch) -> None:
    _stub_room(monkeypatch)
    for result in (StreetSummary("walkway", [], True), StreetSummary("walkway", [], False)):
        monkeypatch.setattr(situation_mod, "fetch_street_summary", lambda sid, *, tz_name, r=result: r)
        ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), SituationDiagnosticsV1()))
        assert ctx.street_summary is None


def test_perception_disabled_never_reads_the_street(monkeypatch) -> None:
    def _boom(*a, **k):
        raise AssertionError("must not read")

    monkeypatch.setattr(situation_mod, "fetch_street_summary", _boom)
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=False), SituationDiagnosticsV1()))
    assert ctx.street_summary is None


def _brief(perception: PerceptionContextV1):
    from orion.schemas.situation import SituationBriefV1

    cfg = _cfg(perception_enabled=False)
    diag = SituationDiagnosticsV1()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    return SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, NOW)),
        place=situation_mod._build_place_context(cfg),
        perception=perception,
    )


def test_prompt_line_only_when_there_is_a_summary() -> None:
    street = "Walkway rhythm: the black dog did not come (usually around 07:40)."
    text = situation_mod._build_prompt_fragment(
        _brief(PerceptionContextV1(street_summary=street)), 4000
    ).compact_text
    # The summary carries its own framing (_build_street_fields); the brief
    # renders it as-is rather than under a generic "Street" label.
    assert f"\n- {street}" in text
    assert "Street (walkway camera)" not in text
    quiet = situation_mod._build_prompt_fragment(_brief(PerceptionContextV1()), 4000).compact_text
    assert "Street" not in quiet


def test_midnight_window_is_not_past_peak_before_midnight() -> None:
    # Window opens 23:40 local, peak 00:10; now is 23:50 local.
    now = datetime(2026, 9, 25, 5, 50, tzinfo=timezone.utc)  # 23:50 MDT on the 24th
    exp = [{"subject_key": "individual:o1", "subject_label": "the owl", "status": "open",
            "peak_minute": 10, "window_start": datetime(2026, 9, 25, 5, 40, tzinfo=timezone.utc)}]
    (line,) = summarize_street(sightings=[], expectations=exp, unresolved=[], patio=None, now=now, tz=TZ)
    assert "usually comes around 00:10" in line


def test_window_older_than_the_sightings_lookback_makes_no_absence_claim() -> None:
    exp = [{"subject_key": "individual:d2", "subject_label": "the grey dog", "status": "open",
            "peak_minute": 400, "window_start": NOW - timedelta(minutes=90)}]
    (line,) = _street(expectations=exp)
    assert "usually here by now" not in line


def test_patio_count_zero_says_nothing() -> None:
    fresh = {"state": "present", "count": 0, "zone": "patio", "row_updated_at": datetime.now(timezone.utc)}
    assert _street(patio=fresh) == []


def test_every_read_failing_is_unread() -> None:
    boom = RuntimeError("timeout")
    conn = _Conn({"vision_individual_sighting": boom, "vision_percept_expectation": boom,
                  "vision_unresolved": boom, "substrate_embodied_presence": boom,
                  "FROM vision_events": boom})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.read_ok is False


def test_patio_count_is_read_from_the_reducers_subject_shape() -> None:
    # orion-sql-writer's vision_individuals writes subject={"count": n}
    # (services/orion-sql-writer/app/vision_individuals.py); the count must
    # survive the identity strip, not collapse to "People are on the patio."
    conn = _Conn({"substrate_embodied_presence": [
        {"presence_id": "walkway:patio",
         "presence_json": '{"state": "present", "since_sec": 30, "last_seen_sec": 1, "subject": {"count": 3}, '
                          '"zone": "patio", "no_embed_zone": true}',
         "updated_at": datetime.now(timezone.utc)},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.lines == ["At least 3 people are on the patio."]


# --- Individual windows read back to the window start (2026-11-03) ----------
# Live case: "Rowan" (named by Juniper 10-25) had an open 10:05-13:09 window,
# peak 11:39, no match until 12:48 at the mailbox. At 12:00 the notes said
# "Rowan usually comes around 11:39" and the reply said "Rowan right on her
# usual 11:39"; asked where, the reply invented a vehicle "slot" it "started
# calling Rowan".
_ROWAN_NOW = datetime(2026, 11, 3, 19, 0, tzinfo=timezone.utc)  # 12:00 MST
_ROWAN_ROW = {"subject_key": "individual:am", "subject_label": "Rowan", "status": "open",
               "peak_minute": 699, "window_start": datetime(2026, 11, 3, 17, 5, tzinfo=timezone.utc),
               "window_end": datetime(2026, 11, 3, 20, 9, tzinfo=timezone.utc)}
_NAMED = {"individual_id": "am", "kind": "person", "named_by": "juniper",
          "named_at": datetime(2026, 10, 26, 1, 54, tzinfo=timezone.utc), "matched_at": None, "caught_up": True}


def _rowan(window, *, now=_ROWAN_NOW, row=_ROWAN_ROW):
    (line,) = summarize_street(sightings=[], expectations=[row], unresolved=[], patio=None,
                               now=now, tz=TZ, individual_windows={"am": window})
    return line


def test_open_window_past_peak_with_no_match_says_so() -> None:
    line = _rowan(_NAMED)
    assert "no match for Rowan (a person, name from Juniper, 10-25) yet today (usually around 11:39)" in line
    assert "usually comes" not in line
    assert "A person match is the camera's guess" in line


def test_a_match_says_its_own_time_not_the_peak() -> None:
    line = _rowan({**_NAMED, "matched_at": datetime(2026, 11, 3, 19, 48, tzinfo=timezone.utc)},
                   now=datetime(2026, 11, 3, 19, 50, tzinfo=timezone.utc))
    assert "Rowan (a person, name from Juniper, 10-25) matched at 12:48 (usually around 11:39)" in line
    assert " came " not in line


def test_peak_still_ahead_keeps_the_habit_wording() -> None:
    line = _rowan(_NAMED, now=datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc))  # 11:00
    assert "Rowan (a person, name from Juniper, 10-25) usually comes around 11:39" in line
    assert "no match" not in line


def test_missed_and_met_individuals_say_match_not_came() -> None:
    missed = _rowan(_NAMED, row={**_ROWAN_ROW, "status": "missed"})
    assert "no match for Rowan (a person, name from Juniper, 10-25) in the usual window (usually around 11:39)" in missed
    met = _rowan(_NAMED, row={**_ROWAN_ROW, "status": "met"})
    assert "Rowan (a person, name from Juniper, 10-25) matched inside the usual window" in met


def test_no_name_origin_without_an_answered_ask_and_no_caveat_for_things() -> None:
    line = _rowan({**_NAMED, "kind": "vehicle", "named_by": None, "named_at": None})
    assert "name from" not in line
    assert "person match" not in line


def test_individual_windows_unread_keeps_the_old_guarded_wording() -> None:
    (line,) = summarize_street(sightings=[], expectations=[_ROWAN_ROW], unresolved=[], patio=None,
                               now=_ROWAN_NOW, tz=TZ, individual_windows=None)
    assert line == "Walkway rhythm: Rowan usually comes around 11:39."


def test_individual_windows_are_read_per_window_start_and_failures_degrade() -> None:
    seen = {}

    class _WinConn(_Conn):
        def execute(self, stmt, params):
            if "unnest(" in str(stmt):
                seen.update(params)
            return super().execute(stmt, params)

    conn = _WinConn({
        "unnest(": [{"individual_id": "am", **{k: v for k, v in _NAMED.items() if k != "individual_id"}}],
        "vision_percept_expectation": [_ROWAN_ROW],
    })
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=_ROWAN_NOW)
    assert seen["ids"] == ["am"] and seen["starts"] == [_ROWAN_ROW["window_start"]]
    assert seen["ends"] == [_ROWAN_ROW["window_end"]]
    assert any("no match for Rowan" in line for line in got.lines)

    broken = _Conn({"unnest(": RuntimeError("boom"), "vision_percept_expectation": [_ROWAN_ROW]})
    got = fetch_street_summary("walkway", engine=_Engine(broken), now=_ROWAN_NOW)
    assert "Walkway rhythm: Rowan usually comes around 11:39." in got.lines


def test_no_absence_claim_while_the_individuals_reducer_lags() -> None:
    # Review finding: "no match yet" from missing rows alone could be a crop
    # the individuals reducer has not processed (vision_rhythm._individuals_caught_up).
    for lag in ({**_NAMED, "caught_up": False}, {k: v for k, v in _NAMED.items() if k != "caught_up"}):
        line = _rowan(lag)
        assert "no match" not in line
        assert "Rowan (a person, name from Juniper, 10-25) usually comes around 11:39" in line


def test_person_caveat_only_when_a_person_phrase_is_shown() -> None:
    # Review finding: a fifth outcome is cut by outcomes[:4]; its caveat must go too.
    things = [{"subject_key": f"label:v{i}", "subject_label": f"vehicle {i}", "status": "met", "peak_minute": 600 + i}
              for i in range(4)]
    (line,) = summarize_street(sightings=[], expectations=things + [_ROWAN_ROW], unresolved=[], patio=None,
                               now=_ROWAN_NOW, tz=TZ, individual_windows={"am": _NAMED})
    assert "Rowan" not in line
    assert "person match" not in line


# --- office camera prior guesses (2026-11-04) -------------------------------------------------


def _office_sighting(iid: str, guess, now):
    return {"individual_id": iid, "kind": "person", "label": None, "distinct_days": 1,
            "first_at": now, "last_at": now, "guess": guess}


def test_office_prior_guess_reads_as_a_guess_not_as_unfamiliar_people() -> None:
    from datetime import datetime, timezone

    from orion.situational.perception_reader import street_camera_word, summarize_street

    now = datetime(2026, 11, 5, 1, 30, tzinfo=timezone.utc)
    # One visit split into three records, all guessed Juniper by the office prior.
    rows = [_office_sighting(f"o{k}", "Juniper", now) for k in range(3)]
    lines = summarize_street(sightings=rows, expectations=[], unresolved=[], patio=[], now=now, tz=timezone.utc,
                             camera=street_camera_word("cam0"), people_windows=[{"counts": {"person": 1}}],
                             stationary_people={})
    text_ = " ".join(lines)
    assert "probably Juniper (my guess from who is usually here, not confirmed)" in text_
    assert "unfamiliar" not in text_
    # Two people in one window: one is the guess, the other is still someone Orion does not know.
    lines = summarize_street(sightings=rows, expectations=[], unresolved=[], patio=[], now=now, tz=timezone.utc,
                             camera=street_camera_word("cam0"), people_windows=[{"counts": {"person": 2}}],
                             stationary_people={})
    assert "unfamiliar" in " ".join(lines)


def test_a_name_she_gave_wins_over_the_same_guess() -> None:
    from datetime import datetime, timezone

    from orion.situational.perception_reader import summarize_street

    now = datetime(2026, 11, 5, 1, 30, tzinfo=timezone.utc)
    rows = [{**_office_sighting("o1", None, now), "label": "Juniper"}, _office_sighting("o2", "Juniper", now)]
    text_ = " ".join(summarize_street(sightings=rows, expectations=[], unresolved=[], patio=[], now=now,
                                      tz=timezone.utc, camera="office camera"))
    assert "Juniper" in text_ and "probably Juniper" not in text_


def test_prior_guess_read_fails_open() -> None:
    from orion.situational import perception_reader as pr

    class _Boom:
        def execute(self, *a, **k):
            raise RuntimeError("column basis does not exist")

        def rollback(self):
            pass

    assert pr._read_prior_guesses(_Boom(), [{"individual_id": "o1", "label": None}]) == {}
    assert pr._read_prior_guesses(_Boom(), [{"individual_id": "o1", "label": "Juniper"}]) == {}


_PG = os.getenv("ORION_TEST_PG_URI", "")


@pytest.mark.skipif(not _PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
def test_prior_guess_sql_reads_only_prior_unsure_unrevoked_guesses() -> None:
    from sqlalchemy import create_engine, text as sql

    from orion.situational import perception_reader as pr

    eng = create_engine(_PG)
    with eng.begin() as conn:
        conn.execute(sql("TRUNCATE vision_person_attribution"))
        for iid, band, basis, revoked in (("a", "unsure", "prior", None), ("b", "unsure", "evidence", None),
                                          ("c", "accept", "prior", None), ("d", "unsure", "prior", "2026-11-05")):
            conn.execute(sql("INSERT INTO vision_person_attribution (individual_id, person_label, channel, score, "
                             "band, model_version, decided_at, basis, revoked_at) VALUES (:i, 'Juniper', 'body', "
                             "0.7, :b, 'v', now(), :basis, CAST(:r AS timestamptz))"),
                         {"i": iid, "b": band, "basis": basis, "r": revoked})
    with eng.connect() as conn:
        got = pr._read_prior_guesses(conn, [{"individual_id": i, "label": None} for i in "abcd"])
    eng.dispose()
    assert got == {"a": "Juniper"}
