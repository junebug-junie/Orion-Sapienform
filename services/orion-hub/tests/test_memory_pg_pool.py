"""Regression: Hub memory pool must retry when Postgres is briefly unavailable.

Live 2026-09-18: Hub + sql-db co-restarted; Hub's one-shot ``asyncpg.create_pool``
failed at 21:13:51 while Postgres only finished recovery at 21:13:56. Pool stayed
``None`` for the rest of the process life; curiosity ticks returned
``stores_not_ready`` forever. Fix is wait/retry at create time, not a health
banner or a curiosity-side workaround.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.memory_pg_pool import create_memory_pg_pool_with_retry


@pytest.mark.asyncio
async def test_create_memory_pg_pool_retries_until_postgres_accepts() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    async def flaky_create_pool(*, dsn: str, min_size: int, max_size: int) -> Any:
        del dsn, min_size, max_size
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("Connection refused")
        return object()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    pool = await create_memory_pg_pool_with_retry(
        dsn="postgresql://postgres:postgres@sql-db:5432/conjourney",
        create_pool=flaky_create_pool,
        sleep=fake_sleep,
        max_attempts=5,
        delay_sec=2.0,
    )

    assert pool is not None
    assert attempts["n"] == 3
    assert sleeps == [2.0, 2.0]


@pytest.mark.asyncio
async def test_create_memory_pg_pool_returns_none_after_exhausted_retries() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    async def always_fail(*, dsn: str, min_size: int, max_size: int) -> Any:
        del dsn, min_size, max_size
        attempts["n"] += 1
        raise OSError("Connection refused")

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    pool = await create_memory_pg_pool_with_retry(
        dsn="postgresql://postgres:postgres@sql-db:5432/conjourney",
        create_pool=always_fail,
        sleep=fake_sleep,
        max_attempts=3,
        delay_sec=0.1,
    )

    assert pool is None
    assert attempts["n"] == 3
    # No sleep after the final failure — that would only delay a dead end.
    assert sleeps == [0.1, 0.1]


def test_main_wires_memory_pg_pool_retry() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "main.py").read_text()
    assert "create_memory_pg_pool_with_retry" in src
    assert "from scripts.memory_pg_pool import create_memory_pg_pool_with_retry" in src
    # One-shot create_pool must not remain as the sole open path.
    startup = src.split("async def startup_event", 1)[1].split("\nasync def ", 1)[0]
    assert "create_memory_pg_pool_with_retry" in startup
    assert "await asyncpg.create_pool(" not in startup


@pytest.mark.asyncio
async def test_create_memory_pg_pool_succeeds_on_first_try_without_sleep() -> None:
    sleeps: list[float] = []

    async def ok_create_pool(*, dsn: str, min_size: int, max_size: int) -> Any:
        del dsn, min_size, max_size
        return "pool"

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    pool = await create_memory_pg_pool_with_retry(
        dsn="postgresql://x",
        create_pool=ok_create_pool,
        sleep=fake_sleep,
        max_attempts=5,
        delay_sec=2.0,
    )

    assert pool == "pool"
    assert sleeps == []
