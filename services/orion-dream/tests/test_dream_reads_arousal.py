"""R3 reader 1: the dream sleeps only when arousal reads `idle` (Temporal Self rev 4).

Veto only. `engaged` / `strained` hold a sleep the Juniper-filtered idle query allowed;
`idle` and every flavour of unknown (absent key, stale, malformed, producer unknown, Redis
error, Redis hang, flag off) leave the idle query alone in charge -- exactly today's decision.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.regulation import ArousalReadingV1, RegulationStateV1
from test_dream_cycle_v2 import _Fakes, _all_seen, _rows


def _state(level, *, age_sec=30.0):
    at = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    reading = ArousalReadingV1(arousal_level=level, since=at - timedelta(minutes=3), observed_at=at,
                               reasons=[f"test_{level}"])
    return RegulationStateV1(generated_at=at, arousal=reading).model_dump_json()


def _deps(f, raw=None, *, raises=None, hangs=False, enabled=True):
    deps = f.deps()
    calls = []

    async def read():
        calls.append(1)
        if raises:
            raise raises
        if hangs:
            await asyncio.sleep(30)
        return raw

    deps.read_arousal = read if enabled else None
    return deps, calls


def _run(deps, **kw):
    from app.cycle import run_cycle_once

    return asyncio.run(run_cycle_once(deps, **kw))


@pytest.mark.parametrize("level", ["engaged", "strained"])
def test_engaged_or_strained_holds_a_due_sleep(level):
    f = _Fakes(_rows())  # pressure 3.6 >= 3.0, idle query 120 min, never slept
    deps, calls = _deps(f, _state(level))
    assert _run(deps) is None
    assert calls == [1] and f.persisted == [] and f.prompts == []


def test_idle_sleeps_exactly_as_today():
    f = _Fakes(_rows())
    deps, _ = _deps(f, _state("idle"))
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed"


@pytest.mark.parametrize(
    "raw",
    [None, "{garbage", _state("unknown"), _state("strained", age_sec=3600)],
    ids=["key_deleted", "malformed", "producer_unknown", "stale"],
)
def test_unknown_falls_back_to_the_idle_query(raw):
    """Spec risk 9b: deleting orion:regulation:latest must not stop Orion sleeping."""
    f = _Fakes(_rows())
    deps, _ = _deps(f, raw)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed"


def test_redis_error_or_hang_is_unknown_not_a_hold(monkeypatch):
    import app.cycle as cycle_mod

    monkeypatch.setattr(cycle_mod, "AROUSAL_READ_TIMEOUT_SEC", 0.05)
    for kw in ({"raises": RuntimeError("redis down")}, {"hangs": True}):
        f = _Fakes(_rows())
        deps, _ = _deps(f, **kw)
        cycle = _run(deps)
        assert cycle is not None and cycle.status == "completed", kw


def test_flag_off_never_reads_and_matches_today():
    f = _Fakes(_rows())
    deps, calls = _deps(f, _state("strained"), enabled=False)
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed" and calls == []


def test_idle_arousal_never_authorizes_a_sleep_the_idle_query_refused():
    """Must-not: arousal only vetoes. Juniper talking (query 5 min) stays not idle."""
    f = _Fakes(_rows(), idle=5.0)
    deps, calls = _deps(f, _state("idle"))
    assert _run(deps) is None
    assert calls == []  # not even read: the dream was not going to sleep
    f = _Fakes(_rows(), idle=None)  # idle query failed
    deps, _ = _deps(f, _state("idle"))
    assert _run(deps) is None


@pytest.mark.parametrize("level", ["engaged", "idle", "strained", "unknown", None])
def test_six_hour_minimum_is_invariant_under_arousal(level):
    """Must-not: the dream six-hour minimum never reads arousal."""
    f = _Fakes(_rows(), last_end=datetime.now(timezone.utc) - timedelta(minutes=30))
    deps, calls = _deps(f, None if level is None else _state(level))
    assert _run(deps) is None and calls == []


def test_strained_also_holds_the_overdue_backstop():
    from app.settings import settings

    f = _all_seen(datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1))
    deps, _ = _deps(f, _state("strained"))
    assert _run(deps) is None
    f = _all_seen(datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1))
    deps, _ = _deps(f, _state("idle"))
    assert _run(deps) is not None


@pytest.mark.parametrize("level", ["engaged", "strained"])
def test_forced_sleep_ignores_arousal(level):
    f = _Fakes(_rows())
    deps, calls = _deps(f, _state(level))
    cycle = _run(deps, force=True)
    assert cycle is not None and calls == []
