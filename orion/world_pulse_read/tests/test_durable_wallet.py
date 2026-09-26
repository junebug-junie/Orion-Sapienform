import asyncio
from datetime import datetime, timezone

from fakeredis.aioredis import FakeRedis
from orion.world_pulse_read.wallet_a import (
    settle_durable_turn,
    read_wallet_a_state,
    read_wallet_a_retry_wait,
)


def test_completed_run_is_charged_once_across_concurrent_replay():
    async def run():
        redis = FakeRedis()
        now = datetime.now(timezone.utc)

        async def settle():
            await settle_durable_turn(
                redis,
                run_id="run-1",
                now=now,
                timezone_name="UTC",
                refused=False,
                backoff_base_sec=60,
                backoff_cap_sec=600,
            )

        await asyncio.gather(settle(), settle())
        await settle()
        _, count = await read_wallet_a_state(redis, now=now, timezone_name="UTC")
        assert count == 1
        await redis.aclose()

    asyncio.run(run())


def test_refusal_backoff_is_once_per_run_and_costs_no_slot():
    async def run():
        redis = FakeRedis()
        now = datetime.now(timezone.utc)

        async def settle(run_id):
            await settle_durable_turn(
                redis,
                run_id=run_id,
                now=now,
                timezone_name="UTC",
                refused=True,
                backoff_base_sec=60,
                backoff_cap_sec=600,
            )

        await settle("refusal-1")
        await settle("refusal-1")
        assert 59.99 <= await read_wallet_a_retry_wait(redis, now=now) <= 60.01
        await settle("refusal-2")
        assert 119.99 <= await read_wallet_a_retry_wait(redis, now=now) <= 120.01
        _, count = await read_wallet_a_state(redis, now=now, timezone_name="UTC")
        assert count == 0
        await redis.aclose()

    asyncio.run(run())
