"""Walkway forecast + grade journal (walkway spec idea 7).

What must hold: predictions and grades come only from rows; the reverie
vocabulary is used; an ungraded window is never called a miss; and when there
is nothing honest to say, nothing is written (or, when the camera has watched
but no rhythm has enough support, the entry says so with real numbers).
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import app.walkway_forecast as wf
from app.walkway_forecast import (
    VERDICT_BY_STATUS,
    build_forecast_seed,
    build_grade_seed,
    collect_walkway_jobs,
)
from orion.journaler import journal_mode_for_trigger, resolve_policy

TZ = ZoneInfo("America/Denver")
FRI = date(2026, 9, 25)  # weekday
SAT = date(2026, 9, 26)  # weekend


def _exp(key: str, label: str, peak: int, *, day_kind="weekday", support=6, conf=0.8,
         status="open", emitted=datetime(2026, 9, 24, 4, tzinfo=timezone.utc)) -> dict:
    return {
        "expectation_id": f"e-{key}-{peak}",
        "subject_key": key,
        "subject_label": label,
        "day_kind": day_kind,
        "window_start": datetime(2026, 9, 24, 13, 30, tzinfo=timezone.utc),
        "window_end": datetime(2026, 9, 24, 13, 55, tzinfo=timezone.utc),
        "peak_minute": peak,
        "support_days": support,
        "support_sightings": support * 2,
        "confidence": conf,
        "status": status,
        "emitted_at": emitted,
        "scored_at": None,
    }


def test_status_maps_to_reverie_vocabulary_and_open_is_not_mapped() -> None:
    assert VERDICT_BY_STATUS == {"met": "confirmed", "missed": "disconfirmed", "unscorable": "unscored"}
    assert "open" not in VERDICT_BY_STATUS


def test_forecast_lists_supported_expectations_first_person_by_time() -> None:
    rows = [
        _exp("label:vehicle", "the mail truck", 14 * 60 + 10, conf=0.55),
        _exp("individual:d1", "the black dog", 7 * 60 + 40, conf=0.8),
    ]
    seed = build_forecast_seed(rows, tomorrow=FRI, tz=TZ, min_support_days=5, days_watched=9)
    assert seed["enough_days"] is True
    assert seed["lines"][0] == (
        "Tomorrow on the walkway I expect: the black dog around 07:40; the mail truck around 14:10."
    )
    assert seed["least_sure"] == "the mail truck"
    assert "I am least sure about the mail truck" in seed["lines"][1]
    assert "window_from" not in seed["forecasts"][0]  # past-date window, DST-unsafe


def test_forecast_ignores_the_wrong_kind_of_day() -> None:
    rows = [_exp("individual:d1", "the black dog", 460, day_kind="weekday")]
    seed = build_forecast_seed(rows, tomorrow=SAT, tz=TZ, min_support_days=5, days_watched=9)
    assert seed["enough_days"] is False


def test_forecast_keeps_latest_row_per_subject() -> None:
    old = _exp("individual:d1", "the black dog", 460, emitted=datetime(2026, 9, 20, tzinfo=timezone.utc))
    new = _exp("individual:d1", "the black dog", 470, emitted=datetime(2026, 9, 24, tzinfo=timezone.utc))
    seed = build_forecast_seed([new, old], tomorrow=FRI, tz=TZ, min_support_days=5, days_watched=9)
    assert [f["around"] for f in seed["forecasts"]] == ["07:50"]


def test_not_enough_days_is_said_with_real_numbers() -> None:
    rows = [_exp("individual:d1", "the black dog", 460, support=3)]
    seed = build_forecast_seed(rows, tomorrow=FRI, tz=TZ, min_support_days=5, days_watched=4)
    assert seed["enough_days"] is False
    assert seed["expectations_below_support"] == 1
    assert "I do not have enough days yet" in seed["lines"][0]
    assert "4 of the last 14 days" in seed["lines"][0]
    # 2026-10-17: "nothing has come back on at least 5 different days" was composed into
    # "the walkway has been empty on at least five occasions". The line now says what 5 is.
    assert "a prediction needs the same thing to come back on at least 5 different days" in seed["lines"][0]
    assert "the black dog" not in json.dumps(seed["lines"])  # no guessing


def test_no_sightings_means_no_forecast_entry_at_all() -> None:
    assert build_forecast_seed([], tomorrow=FRI, tz=TZ, min_support_days=5, days_watched=0) is None


def test_grade_uses_verdicts_and_separates_pending() -> None:
    rows = [
        _exp("individual:d1", "the black dog", 460, status="met"),
        _exp("label:vehicle", "the mail truck", 850, status="missed"),
        _exp("label:bicycle", "kids on bikes", 930, status="unscorable"),
        _exp("label:person", "the evening walker", 1290, status="open"),
    ]
    calib = [{"subject_key": "individual:d1", "subject_label": "the black dog", "met": 4, "missed": 1, "unscorable": 0},
             {"subject_key": "label:x", "subject_label": "x", "met": 0, "missed": 0, "unscorable": 2}]
    seed = build_grade_seed(rows, calib, day=FRI, tz=TZ)
    assert seed["lines"] == [
        "confirmed: the black dog (around 07:40)",
        "disconfirmed: the mail truck (around 14:10)",
        "unscored: kids on bikes (around 15:30)",
        "not scored yet: the evening walker",
    ]
    assert seed["calibration_last_14_days"][0]["hit_rate"] == 0.8
    # nothing decidable -> None, never 0.0
    assert seed["calibration_last_14_days"][1]["hit_rate"] is None


def test_grade_with_no_rows_is_none() -> None:
    assert build_grade_seed([], [], day=FRI, tz=TZ) is None


def test_trigger_kinds_are_registered_digest_and_not_emailed() -> None:
    seed = {"for_date": "2026-09-25", "lines": ["x"]}
    for kind in ("walkway_forecast", "walkway_grade"):
        trig = wf.build_walkway_trigger(kind, seed, stream_id="walkway")
        assert journal_mode_for_trigger(trig) == "digest"
        assert trig.source_ref == f"{kind}:walkway:2026-09-25"
        assert json.loads(trig.prompt_seed)["lines"] == ["x"]
        policy = resolve_policy(kind)
        assert policy.trigger_kind == kind and policy.email_enabled is False


def _fake_fetch(tables: dict):
    async def _fetch(dsn, sql, params, *, label):
        return tables.get(label)
    return _fetch


NOW = datetime(2026, 9, 25, 4, 45, tzinfo=timezone.utc)  # 22:45 MDT Thu 24 Sep


def test_collect_unreadable_table_writes_nothing(monkeypatch) -> None:
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch({}))
    jobs, skips = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=NOW,
        min_support_days=5, node="athena"))
    assert jobs == [] and skips == ["rollup_unreadable", "expectations_unreadable"]


def test_collect_empty_street_writes_nothing(monkeypatch) -> None:
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch({
        "walkway_rollup_day": [], "walkway_grade": [], "walkway_forecast": [], "walkway_watch_days": [{"days": 0}],
    }))
    jobs, skips = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=NOW,
        min_support_days=5, node="athena"))
    assert jobs == []
    assert skips == ["walkway_day_no_rollup", "grade_no_expectations_today", "forecast_no_sightings"]


def test_collect_builds_both_with_distinct_dedupe_keys(monkeypatch) -> None:
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch({
        "walkway_rollup_day": [],
        "walkway_grade": [_exp("individual:d1", "the black dog", 460, status="met")],
        "walkway_calibration": [],
        "walkway_forecast": [_exp("individual:d1", "the black dog", 460)],
        "walkway_watch_days": [{"days": 7}],
    }))
    jobs, skips = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=NOW,
        min_support_days=5, node="athena"))
    assert skips == ["walkway_day_no_rollup"]
    kinds = [j.trigger.trigger_kind for j in jobs]
    assert kinds == ["walkway_grade", "walkway_forecast"]
    # grade is for local today (Thu 24), forecast for tomorrow (Fri 25)
    assert jobs[0].dedupe_key == "actions:journal:walkway_grade:walkway:2026-09-24:athena"
    assert jobs[1].dedupe_key == "actions:journal:walkway_forecast:walkway:2026-09-25:athena"


def test_malformed_rows_alone_do_not_produce_not_enough_days() -> None:
    rows = [_exp("individual:d1", "", 460)]
    assert build_forecast_seed(rows, tomorrow=FRI, tz=TZ, min_support_days=5, days_watched=9) is None


# --- scheduler tick ----------------------------------------------------------

def _job(kind):
    return wf.WalkwayJournalJob(
        trigger=wf.build_walkway_trigger(kind, {"for_date": "2026-09-24"}, stream_id="walkway"),
        audit_action=f"journal.{kind}", dedupe_key=f"k:{kind}",
    )


def _tick(jobs, skips, dispatch_ok, store, attempts):
    async def collect():
        return list(jobs), list(skips)

    sent = []

    async def dispatch(job):
        sent.append(job.trigger.trigger_kind)
        return dispatch_ok.get(job.trigger.trigger_kind, True)

    done, _ = asyncio.run(wf.run_walkway_tick(
        collect=collect, dispatch=dispatch, job_done_on=store.get,
        mark_job_done=store.__setitem__, local_date="2026-09-24", read_attempts=attempts,
    ))
    return done, sent


def test_tick_skips_complete_the_night() -> None:
    done, sent = _tick([], ["grade_no_expectations_today", "forecast_no_sightings"], {}, {}, {})
    assert done is True and sent == []


def test_tick_partial_failure_retries_only_the_failed_job_even_after_restart() -> None:
    store: dict = {}
    jobs = [_job("walkway_grade"), _job("walkway_forecast")]
    done, sent = _tick(jobs, [], {"walkway_forecast": False}, store, {})
    assert done is False and sent == ["walkway_grade", "walkway_forecast"]
    assert store == {"walkway_grade": "2026-09-24"}
    # "restart": a fresh attempts dict, same durable store
    done, sent = _tick(jobs, [], {}, store, {})
    assert done is True and sent == ["walkway_forecast"]


def test_tick_read_failure_is_retried_a_bounded_number_of_times() -> None:
    attempts: dict = {}
    results = [_tick([], ["expectations_unreadable"], {}, {}, attempts)[0] for _ in range(wf.MAX_READ_ATTEMPTS)]
    assert results == [False] * (wf.MAX_READ_ATTEMPTS - 1) + [True]


def test_tick_collect_exception_is_a_bounded_read_failure() -> None:
    async def collect():
        raise RuntimeError("boom")

    async def dispatch(job):
        raise AssertionError

    done, skips = asyncio.run(wf.run_walkway_tick(
        collect=collect, dispatch=dispatch, job_done_on={}.get, mark_job_done=lambda k, v: None,
        local_date="d", read_attempts={},
    ))
    assert done is False and skips == ["collect_failed"]


# --- day / week / month from the rollup (2026-10-18) ------------------------------------------

def _rollup_row(grain: str = "day", passings: int = 185) -> dict:
    rollup = {
        "rules_version": 2, "label": "2026-09-24", "grain": grain, "local": {"date": "2026-09-24", "year": 2026},
        "complete": False, "covered_until": "2026-09-25T04:00:00+00:00",
        "total": {"passings": passings, "distinct": 93, "sightings": 400},
        "kinds": {"person": {"passings": 105, "distinct": 50, "sightings": 200}},
        "named": {"Alex": {"times": ["15:31"], "sightings": 2, "individual_ids": ["secret-id"]}},
        "individuals": {"secret-id": {"kind": "person"}},
        "regulars": {"count": 3, "by_kind": {"person": 3}, "ids": ["x"]},
        "camera": {"up_minutes": 1212, "expected_minutes": 1320, "status": "ok", "down_spans": ["13:00–14:00"]},
        "captions": [{"at": "14:23", "text": "A dog on a leash.", "detector": ["dog"]}],
        "stories": {"gate_passed": False, "checked": 8},
    }
    return {"grain": grain, "rollup": rollup, "line": f"Walkway Thu 24 Sep 2026: {passings} passings."}


def test_day_job_is_built_from_the_rollup_and_hides_ids_and_closed_gate_stories(monkeypatch) -> None:
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch({
        "walkway_rollup_day": [_rollup_row()], "walkway_grade": [], "walkway_forecast": [],
        "walkway_watch_days": [{"days": 0}],
    }))
    jobs, skips = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=NOW, min_support_days=5, node="athena"))
    assert [j.trigger.trigger_kind for j in jobs] == ["walkway_day"]
    job = jobs[0]
    assert job.dedupe_key == "actions:journal:walkway_day:walkway:day:2026-09-24:athena"
    seed = json.loads(job.trigger.prompt_seed)
    assert seed["lines"] == ["Walkway Thu 24 Sep 2026: 185 passings."]
    assert "secret-id" not in job.trigger.prompt_seed and "stories" not in seed["facts"]
    assert seed["facts"]["captions"][0].startswith("camera caption logged 14:23")
    assert resolve_policy("walkway_day").email_enabled is False
    assert journal_mode_for_trigger(job.trigger) == "digest"


def test_week_on_sunday_and_month_on_the_last_day(monkeypatch) -> None:
    tables = {"walkway_rollup_day": [_rollup_row()], "walkway_rollup_week": [_rollup_row("week")],
              "walkway_rollup_month": [_rollup_row("month")], "walkway_grade": [], "walkway_forecast": [],
              "walkway_watch_days": [{"days": 0}]}
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch(tables))
    sunday_last = datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)  # Sat 31 Oct 22:30 MDT -> last day
    jobs, _ = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=sunday_last, min_support_days=5,
        node="n"))
    assert [j.trigger.trigger_kind for j in jobs] == ["walkway_day", "walkway_month"]
    sunday = datetime(2026, 10, 26, 4, 30, tzinfo=timezone.utc)  # Sun 25 Oct 22:30 MDT
    jobs, _ = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=sunday, min_support_days=5, node="n"))
    assert [j.trigger.trigger_kind for j in jobs] == ["walkway_day", "walkway_week"]
    assert jobs[1].dedupe_key.endswith(":week:2026-10-19:n")


def test_guard_replaces_prose_with_an_invented_number_and_keeps_honest_prose() -> None:
    from orion.journaler.schemas import JournalEntryDraftV1

    seed = wf.build_rollup_seed("walkway_day", _rollup_row())
    trigger = wf.build_walkway_trigger("walkway_day", seed, stream_id="walkway")
    honest = JournalEntryDraftV1(mode="digest", title="The street today",
                                 body="185 passings today, Alex at 15:31; the camera was down 13:00–14:00.")
    kept, bad = wf.guard_walkway_draft(honest, trigger)
    assert bad == [] and kept.body == honest.body
    invented = JournalEntryDraftV1(mode="digest", title="Five quiet stretches",
                                   body="The walkway has been empty on at least five occasions.")
    fixed, bad = wf.guard_walkway_draft(invented, trigger)
    assert "five" in bad
    assert fixed.body == "Walkway Thu 24 Sep 2026: 185 passings."
    assert fixed.title == trigger.summary
    # Other journal kinds are never touched.
    other = trigger.model_copy(update={"trigger_kind": "daily_summary"})
    assert wf.guard_walkway_draft(invented, other) == (invented, [])


# A day row as rules_version 1 wrote it (before people became a range, 2026-10-20).
V1_DAY = {
    "schema": "vision_rollup.v1", "rules_version": 1, "stream_id": "walkway", "grain": "day",
    "period_start": "2026-10-16T06:00:00+00:00", "period_end": "2026-10-17T06:00:00+00:00",
    "label": "2026-10-16", "tz": "America/Denver", "complete": True,
    "covered_until": "2026-10-17T06:00:00+00:00",
    "kinds": {"person": {"passings": 137, "distinct": 89, "sightings": 151},
              "vehicle": {"passings": 40, "distinct": 12, "sightings": 60}},
    "total": {"passings": 177, "distinct": 101, "sightings": 211}, "peak_at_once": {"person": 5, "vehicle": 3},
    "camera": {"up_minutes": 1400, "expected_minutes": 1440, "status": "ok", "down_spans": [], "missing_spans": []},
    "busiest": {"at": "12:00", "passings": 92, "peak_at_once": {"person": 5}}, "quietest": None,
    "baseline": {"mean_passings": None, "periods": 0},
}
V1_LINE = ("Walkway Fri 16 Oct 2026: 177 passings (137 people, 40 vehicles) by 101 distinct; "
           "busiest 12:00 (92, most at once 5 people).")
RANGE_TEXT = "people: at least 5, at most 137 (the camera splits one person into several)"


def test_rollup_seed_gives_people_only_as_a_range() -> None:
    from orion.vision.walkway_rollup import bare_people_counts

    seed = wf.build_rollup_seed("walkway_day", {"rollup": json.dumps(V1_DAY), "line": V1_LINE})
    assert RANGE_TEXT in seed["lines"][0] and bare_people_counts(seed["lines"][0]) == []
    facts = seed["facts"]
    assert facts["kinds"]["person"] == {"at_least": 5, "at_most": 137,
                                        "note": "the camera splits one person into several"}
    assert facts["kinds"]["vehicle"] == {"passings": 40, "distinct": 12}
    assert facts["total"] == {"passings_at_most": 177}
    assert facts["busiest"] == {"at": "12:00", "passings_at_most": 92}
    dumped = json.dumps(facts)
    assert "89" not in dumped and "101" not in dumped  # person / total cluster counts never reach the entry


def test_guard_refuses_a_bare_people_count_even_when_the_number_is_a_fact() -> None:
    from orion.journaler.schemas import JournalEntryDraftV1

    seed = wf.build_rollup_seed("walkway_day", {"rollup": json.dumps(V1_DAY), "line": V1_LINE})
    trigger = wf.build_walkway_trigger("walkway_day", seed, stream_id="walkway")
    bare = JournalEntryDraftV1(mode="digest", title="The street", body="137 people walked by today.")
    fixed, bad = wf.guard_walkway_draft(bare, trigger)
    assert bad == ["137 people"] and fixed.body == seed["lines"][0]
    ranged = JournalEntryDraftV1(mode="digest", title="The street",
                                 body="At least 5, at most 137 people walked by; 40 vehicles.")
    assert wf.guard_walkway_draft(ranged, trigger) == (ranged, [])


def test_days_watched_counts_frame_minutes_not_sightings_or_rollup_rows() -> None:
    sql = " ".join(wf.WATCH_DAYS_SQL.split())
    assert "FROM vision_scene_inventory" in sql and "vision_individual_sighting" not in sql
    assert "vision_rollup" not in sql  # independent of how far the rollups were backfilled
    assert "up_minutes >= %(min_up)s" in sql and wf.WATCHED_MIN_UP_MINUTES == 60


def test_compact_facts_carry_belongings_without_ids() -> None:
    facts = wf.compact_rollup_facts({
        "label": "2026-10-17", "named": {"Robin": {"times": ["09:03"], "individual_ids": ["r1"]}},
        "belongings": {"Sam": {"times": ["17:09"], "sightings": 1, "kinds": ["bicycle"],
                               "individual_ids": ["bike"], "person_ids": ["person-sam"]}}})
    assert facts["belongings"] == {"Sam": {"times": ["17:09"], "sightings": 1, "kinds": ["bicycle"]}}
    assert "bike" not in json.dumps(facts)
