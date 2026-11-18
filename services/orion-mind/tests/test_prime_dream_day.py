"""Day-dream Patch 2b in the resident loop (app/prime_loop.py): once a Denver day Prime reads last night's dream_day
row (GET /dream-day/result) and applies its plan once. Fake durable-runs; no network."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace

from orion.prime import day_dream as dd
from orion.prime.state import PrimeConcernV1

from test_prime_loop import T0, runtime

DAY = date(2026, 10, 29)  # T0 is 2026-10-30 12:00Z = 06:00 in Denver: yesterday is 10-29
RESULT = f"http://dr:8121/dream-day/result?day={DAY.isoformat()}"


def dream_rt(page, *, enabled=True):
    rt = runtime(answer={"reading": "ok"}, extra_pages={RESULT: page} if page is not None else {})
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_DREAM_DAY_ENABLED": enabled})
    old = T0 - timedelta(hours=10)
    rt.state = rt.state.model_copy(update={"concerns": [
        PrimeConcernV1(concern_id="held1", text="held one", opened_at=old, last_evidence_at=old),
        PrimeConcernV1(concern_id="held2", text="held two", opened_at=old, last_evidence_at=old)]})
    return rt


PLAN = {"day": DAY.isoformat(), "found": True, "run_id": "dream-day-2026-10-29", "produced": True, "plan": [
    {"action": "drop", "concern_id": "held1", "reason": "settled by the lane's own table", "refs": ["experience:x"]},
    {"action": "keep", "concern_id": "held2", "reason": "still open", "refs": ["experience:y"]},
    {"action": "new", "gist": "check the lane's own table first", "reason": "r", "refs": ["experience:z"]}]}


def test_applies_last_nights_plan_once():
    rt = dream_rt(PLAN)
    asyncio.run(rt._dream_day_upkeep(T0))
    ids = [c.concern_id for c in rt.state.concerns]
    assert "held1" not in ids and "held2" in ids and len(ids) == 2
    assert rt.state.concern_record[-1].note.startswith("dropped: dream 2026-10-29:")
    kept = next(c for c in rt.state.concerns if c.concern_id == "held2")
    assert kept.last_evidence_at == T0 - timedelta(hours=10)  # a dream keep renews nothing
    assert dd.is_applied(rt.dream_day, DAY)
    # Next check (after the throttle): already applied, so the result is not even read again.
    n = len(rt.urls)
    asyncio.run(rt._dream_day_upkeep(T0 + timedelta(seconds=dd.CHECK_INTERVAL_SEC + 1)))
    assert len(rt.urls) == n and [c.concern_id for c in rt.state.concerns] == ids


def test_no_row_yet_changes_nothing_and_is_checked_again():
    rt = dream_rt({"day": DAY.isoformat(), "found": False})
    before = rt.state
    asyncio.run(rt._dream_day_upkeep(T0))
    assert rt.state == before and not dd.is_applied(rt.dream_day, DAY)


def test_unreachable_durable_runs_never_raises_into_the_tick():
    rt = dream_rt(None)  # 404 from the fake
    asyncio.run(rt._dream_day_upkeep(T0))
    assert not dd.is_applied(rt.dream_day, DAY)


def test_throttled_between_checks():
    rt = dream_rt({"day": DAY.isoformat(), "found": False})
    asyncio.run(rt._dream_day_upkeep(T0))
    n = len(rt.urls)
    asyncio.run(rt._dream_day_upkeep(T0 + timedelta(seconds=60)))
    assert len(rt.urls) == n


def test_off_flag_skips_the_step_in_the_tick():
    rt = dream_rt(PLAN, enabled=False)
    assert not rt._dream_day_on()


def test_a_concern_prime_has_taken_on_survives_the_dreams_drop():
    from orion.prime import intention as it

    rt = dream_rt(PLAN)
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_INTENTION_TAKE_ON_ENABLED": True})
    rt.intentions = rt.intentions.model_copy(update={"open": [it.PrimeIntentionOpenV1.model_construct(
        concern_id="held1")]})
    asyncio.run(rt._dream_day_upkeep(T0))
    assert "held1" in [c.concern_id for c in rt.state.concerns]
    assert "drop held1: taken on" in rt.dream_day.applied[-1].skipped
