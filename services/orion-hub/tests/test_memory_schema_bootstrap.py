"""Boot-time memory schema apply: independent, never fatal, retried later.

Live 2026-10-14: hub restarts inside the nightly pg_dump window hit
lock_timeout on memory_cards.sql; the traceback also skipped
memory_crystallizations.sql, and nothing retried until the next restart.
"""
import asyncio

from scripts.memory_schema_bootstrap import ensure_memory_schemas


class _Flaky:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def __call__(self, dsn):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("canceling statement due to lock timeout")
        return False  # fingerprint current, DDL skipped


async def _no_sleep(_):
    return None


def test_a_failing_schema_does_not_skip_the_next_one():
    cards, crys = _Flaky(failures=99), _Flaky(failures=0)

    async def run():
        return await ensure_memory_schemas("dsn", [("memory_cards", cards), ("memory_crystallizations", crys)],
                                           attempts=1, sleep=_no_sleep)

    ready, task = asyncio.run(run())
    assert ready == {"memory_cards": False, "memory_crystallizations": True}
    assert task is None
    assert crys.calls == 1


def test_failed_schema_is_retried_in_background_until_it_succeeds(caplog):
    cards = _Flaky(failures=3)

    async def run():
        ready, task = await ensure_memory_schemas("dsn", [("memory_cards", cards)], attempts=10, sleep=_no_sleep)
        assert ready == {"memory_cards": False}
        return await task

    with caplog.at_level("INFO"):
        outcome = asyncio.run(run())
    assert outcome == {"memory_cards": True}
    assert cards.calls == 4
    assert "memory_schema_apply_failed name=memory_cards" in caplog.text
    assert "memory_schema_ready name=memory_cards" in caplog.text


def test_retry_gives_up_loudly_after_the_budget(caplog):
    cards = _Flaky(failures=99)

    async def run():
        _, task = await ensure_memory_schemas("dsn", [("memory_cards", cards)], attempts=3, sleep=_no_sleep)
        return await task

    with caplog.at_level("INFO"):
        outcome = asyncio.run(run())
    assert outcome == {"memory_cards": False}
    assert cards.calls == 3
    assert "memory_schema_apply_gave_up name=memory_cards attempts=3" in caplog.text


def test_current_schema_needs_no_retry_task():
    async def run():
        return await ensure_memory_schemas("dsn", [("memory_cards", _Flaky(0))], sleep=_no_sleep)

    ready, task = asyncio.run(run())
    assert ready == {"memory_cards": True} and task is None
