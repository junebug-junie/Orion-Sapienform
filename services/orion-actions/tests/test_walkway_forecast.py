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
    assert jobs == [] and skips == ["expectations_unreadable"]


def test_collect_empty_street_writes_nothing(monkeypatch) -> None:
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch({
        "walkway_grade": [], "walkway_forecast": [], "walkway_watch_days": [{"days": 0}],
    }))
    jobs, skips = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=NOW,
        min_support_days=5, node="athena"))
    assert jobs == []
    assert skips == ["grade_no_expectations_today", "forecast_no_sightings"]


def test_collect_builds_both_with_distinct_dedupe_keys(monkeypatch) -> None:
    monkeypatch.setattr(wf, "fetch_rows", _fake_fetch({
        "walkway_grade": [_exp("individual:d1", "the black dog", 460, status="met")],
        "walkway_calibration": [],
        "walkway_forecast": [_exp("individual:d1", "the black dog", 460)],
        "walkway_watch_days": [{"days": 7}],
    }))
    jobs, skips = asyncio.run(collect_walkway_jobs(
        dsn="x", stream_id="walkway", tz_name="America/Denver", now_utc=NOW,
        min_support_days=5, node="athena"))
    assert skips == []
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
