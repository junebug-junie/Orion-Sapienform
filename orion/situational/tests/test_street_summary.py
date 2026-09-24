"""The street in the situation brief (walkway camera spec ideas 5 and 9).

Properties under test: names only when Juniper gave them; the patio is a
count, never a name; an absence claim ("usually here by now") needs evidence;
nothing is said when nothing was read; and the fold into PerceptionContextV1
rides on every room-percept path.
"""

from __future__ import annotations

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
        "but have no name for, 2 unfamiliar people."
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
        "the mail truck came as expected around 07:20; the grey dog is usually here by now; "
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
    fresh = {"state": "present", "count": 3, "subject": "Juniper", "row_updated_at": datetime.now(timezone.utc)}
    (line,) = _street(patio=fresh)
    assert line == "3 people are on the patio."
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
    assert conn.rollbacks == 2


def test_patio_row_is_read_by_stream_and_stripped_of_identity() -> None:
    seen = {}

    class _PatioConn(_Conn):
        def execute(self, stmt, params):
            if "substrate_embodied_presence" in str(stmt):
                seen.update(params)
            return super().execute(stmt, params)

    conn = _PatioConn({"substrate_embodied_presence": [
        {"presence_json": '{"state": "present", "count": 1, "subject": "Juniper"}',
         "updated_at": datetime.now(timezone.utc)},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert seen["presence_id"] == "walkway:patio"
    assert got.lines == ["Someone is on the patio."]


# --- fold into PerceptionContextV1 -------------------------------------------


def _cfg(**over):
    cfg = situation_mod.settings_from_runtime(SimpleNamespace())
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _stub_room(monkeypatch, percept=None):
    from orion.situational.perception_reader import PresenceResolution

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda: percept)
    monkeypatch.setattr(
        situation_mod, "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution(None, None, True),
    )


def test_street_defaults_to_walkway_and_empty_disables() -> None:
    assert _cfg().street_stream_ids == ["walkway"]
    cfg = situation_mod.settings_from_runtime(SimpleNamespace(orion_situation_street_stream_ids=""))
    assert cfg.street_stream_ids == []


def test_street_rides_on_the_stale_room_path_too(monkeypatch) -> None:
    _stub_room(monkeypatch, percept=None)  # room camera has nothing
    monkeypatch.setattr(
        situation_mod, "fetch_street_summary",
        lambda sid, *, tz_name: StreetSummary(sid, ["Someone is on the patio."], True),
    )
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation_mod._build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.available is False
    assert ctx.street_summary == "Someone is on the patio."
    assert ctx.street_stream_id == "walkway"
    assert diag.provider_status["perception_street"] == "ok"


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
    assert f"Street (walkway camera): {street}" in text
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
    fresh = {"state": "present", "count": 0, "row_updated_at": datetime.now(timezone.utc)}
    assert _street(patio=fresh) == []


def test_every_read_failing_is_unread() -> None:
    boom = RuntimeError("timeout")
    conn = _Conn({"vision_individual_sighting": boom, "vision_percept_expectation": boom,
                  "vision_unresolved": boom, "substrate_embodied_presence": boom})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.read_ok is False


def test_patio_count_is_read_from_the_reducers_subject_shape() -> None:
    # orion-sql-writer's vision_individuals writes subject={"count": n}
    # (services/orion-sql-writer/app/vision_individuals.py); the count must
    # survive the identity strip, not collapse to "People are on the patio."
    conn = _Conn({"substrate_embodied_presence": [
        {"presence_json": '{"state": "present", "since_sec": 30, "last_seen_sec": 1, "subject": {"count": 3}}',
         "updated_at": datetime.now(timezone.utc)},
    ]})
    got = fetch_street_summary("walkway", engine=_Engine(conn), now=NOW)
    assert got.lines == ["3 people are on the patio."]
