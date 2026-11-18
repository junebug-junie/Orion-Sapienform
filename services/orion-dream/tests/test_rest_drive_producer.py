"""The rest drive's producer: every dream check publishes one DriveReadingV1 built
from the same values the sleep gate uses, and a publish failure never changes a
sleep decision (Temporal Self rev 4, R2)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from test_dream_cycle_v2 import _Fakes, _all_seen, _rows


def _with_publisher(f, *, fail=False):
    deps = f.deps()
    published = []

    async def publish(reading):
        published.append(reading)
        if fail:
            raise RuntimeError("redis down")

    deps.publish_drive_reading = publish
    return deps, published


def test_tired_check_publishes_due_then_refractory_after_the_sleep():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows())  # pressure 3.6 >= 3.0, idle, never slept
    deps, published = _with_publisher(f)
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle is not None and cycle.status == "completed"
    before, after = published
    assert before.state == "due" and before.due_reason == "threshold"
    assert before.level == cycle.pressure.pressure and before.threshold == cycle.pressure.threshold
    # The sleep just discharged the drive: readers must not keep seeing `due`
    # for another whole check.
    assert after.state == "refractory" and after.last_discharge_at == cycle.ended_at
    assert after.refractory_until == cycle.ended_at + timedelta(hours=6)
    assert after.accumulating_since == cycle.started_at


def test_check_id_joins_the_reading_to_its_saved_observation():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows(), idle=5.0)  # due but Juniper is talking: no sleep
    deps, published = _with_publisher(f)
    saved = []
    deps.persist_pressure_observation = lambda o: saved.append(o) or True
    assert asyncio.run(run_cycle_once(deps)) is None
    (reading,) = published
    assert reading.state == "due" and reading.source_ref == saved[0].check_id


def test_low_pressure_check_is_building_and_empty_window_is_resting():
    from app.cycle import run_cycle_once

    rows = {"compaction_request": [{"request_id": "r1", "theme": "t", "reason": "r"}]}  # weight 0.5
    deps, published = _with_publisher(_Fakes(rows, last_start=datetime.now(timezone.utc) - timedelta(hours=10)))
    assert asyncio.run(run_cycle_once(deps)) is None
    assert published[0].state == "building" and published[0].level == 0.5

    deps, published = _with_publisher(_all_seen(datetime.now(timezone.utc) - timedelta(hours=10)))
    assert asyncio.run(run_cycle_once(deps)) is None
    assert published[0].state == "resting" and published[0].level == 0.0


def test_inside_the_six_hour_minimum_is_refractory_even_when_pressure_is_high():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows(), last_end=datetime.now(timezone.utc) - timedelta(minutes=30))
    deps, published = _with_publisher(f)
    assert asyncio.run(run_cycle_once(deps)) is None
    (reading,) = published
    assert reading.state == "refractory" and reading.level > reading.threshold


def test_overdue_backstop_reads_due_overdue():
    from app.cycle import run_cycle_once
    from app.settings import settings

    last = datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1)
    deps, published = _with_publisher(_all_seen(last, idle=5.0))  # not idle: no sleep
    assert asyncio.run(run_cycle_once(deps)) is None
    assert published[0].state == "due" and published[0].due_reason == "overdue" and published[0].level == 0.0


def test_unreadable_pressure_publishes_no_reading_not_calm():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows())
    deps, published = _with_publisher(f)

    def boom(*a, **k):
        raise OSError("db down")

    deps.load_source_rows = boom
    assert asyncio.run(run_cycle_once(deps)) is None
    (reading,) = published
    assert reading.state == "no_reading" and reading.level is None
    assert reading.no_reading_reason.startswith("pressure_read_failed")


def test_a_degraded_source_read_is_no_reading():
    from app.cycle import run_cycle_once

    class _Partial(dict):
        read_errors = ("metacog",)

    f = _Fakes(_rows())
    deps, published = _with_publisher(f)
    inner = deps.load_source_rows
    deps.load_source_rows = lambda *a, **k: _Partial(inner(*a, **k))
    asyncio.run(run_cycle_once(deps))
    assert published[0].state == "no_reading" and "metacog" in published[0].no_reading_reason


def test_a_publish_failure_never_changes_the_sleep():
    from app.cycle import run_cycle_once

    deps, published = _with_publisher(_Fakes(_rows()), fail=True)
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle is not None and cycle.status == "completed" and len(published) == 2


def test_no_publisher_is_the_old_path():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows())
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle is not None and cycle.status == "completed"


def test_publish_switch_off_wires_no_publisher(monkeypatch):
    from app import main
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_REST_DRIVE_PUBLISH_ENABLED", False)
    assert main.build_cycle_deps().publish_drive_reading is None
    monkeypatch.setattr(settings, "DREAM_REST_DRIVE_PUBLISH_ENABLED", True)
    assert main.build_cycle_deps().publish_drive_reading is not None


def test_publisher_writes_the_redis_key_with_the_staleness_ttl(monkeypatch):
    from app import main
    from app.settings import settings
    from orion.regulation.rest_drive import no_rest_reading
    from orion.schemas.drive_reading import REST_DRIVE_REDIS_KEY, parse_drive_reading

    class _Redis:
        def __init__(self):
            self.calls = []

        async def setex(self, key, ttl, body):
            self.calls.append((key, ttl, body))

    class _Bus:
        redis = _Redis()

    bus = _Bus()

    async def fake_bus():
        return bus

    monkeypatch.setattr(main, "_cycle_bus", fake_bus)
    reading = no_rest_reading(now=datetime.now(timezone.utc), source_ref="dp-x", threshold=3.0, reason="test")
    asyncio.run(main.build_cycle_deps().publish_drive_reading(reading))
    ((key, ttl, body),) = bus.redis.calls
    assert key == REST_DRIVE_REDIS_KEY == "orion:drive:rest:latest"
    assert ttl == int(settings.DREAM_REST_DRIVE_REDIS_TTL_SEC) == 1800
    assert parse_drive_reading(body) == reading


def test_pressure_endpoint_shows_the_same_drive_reading(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "build_cycle_deps", lambda: _Fakes(_rows()).deps())
    out = asyncio.run(main.cycle_pressure_endpoint())
    assert out["rest_drive"]["state"] == "due" and out["rest_drive"]["level"] == out["pressure"]["pressure"]


def test_post_sleep_reading_is_marked_and_comes_after_the_story():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows())
    deps, published = _with_publisher(f)
    order = []

    async def start_story(trigger):
        order.append(("story", len(published)))

    deps.start_story = start_story
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed"
    assert order == [("story", 1)]  # only the pre-sleep reading was out when the story started
    assert published[0].source_ref.startswith("dp-") and not published[0].source_ref.startswith("dp-postsleep-")
    assert published[1].source_ref.startswith("dp-postsleep-")


def test_a_hung_publisher_cannot_hold_up_the_sleep_decision(monkeypatch):
    import app.cycle as cycle_mod
    from app.cycle import run_cycle_once

    monkeypatch.setattr(cycle_mod, "PUBLISH_TIMEOUT_SEC", 0.05)
    f = _Fakes(_rows())
    deps = f.deps()

    async def hang(reading):
        await asyncio.sleep(3600)

    deps.publish_drive_reading = hang
    cycle = asyncio.run(asyncio.wait_for(run_cycle_once(deps), timeout=5))
    assert cycle is not None and cycle.status == "completed"


def test_a_reading_that_fails_to_build_never_kills_the_loop(monkeypatch):
    import app.cycle as cycle_mod
    from app.cycle import run_cycle_once

    def boom(*a, **k):
        raise ValueError("bad reading")

    monkeypatch.setattr(cycle_mod, "read_rest_drive", boom)
    deps, published = _with_publisher(_Fakes(_rows()))
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle is not None and cycle.status == "completed" and published == []


def test_pressure_endpoint_folds_in_source_errors(monkeypatch):
    from app import main

    class _Partial(dict):
        read_errors = ("metacog",)

    f = _Fakes(_rows())
    deps = f.deps()
    inner = deps.load_source_rows
    deps.load_source_rows = lambda *a, **k: _Partial(inner(*a, **k))
    monkeypatch.setattr(main, "build_cycle_deps", lambda: deps)
    out = asyncio.run(main.cycle_pressure_endpoint())
    assert out["rest_drive"]["state"] == "no_reading"
