"""Refund semantics for a world-pulse wallet debit (orion/world_pulse_read/wallet_refund.py)."""

import asyncio
from datetime import datetime, timedelta, timezone

from orion.world_pulse_read import wallet_a as wa
from orion.world_pulse_read import wallet_b as wb
from orion.world_pulse_read.retry import is_refused_before_work


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def incr(self, key):
        self.store[key] = str(int(self.store.get(key, "0")) + 1)
        return int(self.store[key])

    async def decr(self, key):
        self.store[key] = str(int(self.store.get(key, "0")) - 1)
        return int(self.store[key])

    async def expire(self, key, ttl):
        return True


T0 = datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)


def _refund(r, receipt, now, *, effective=8400.0, floor=1800.0, refund=wa.refund_wallet_a):
    return asyncio.run(
        refund(r, receipt, now=now, effective_cooldown_sec=effective, retry_floor_sec=floor)
    )


def test_refund_restores_count_and_only_once():
    r = _FakeRedis()
    key = f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-24"
    r.store[key] = "3"
    receipt = asyncio.run(wa.debit_wallet_a(r, now=T0, timezone_name="UTC"))
    assert r.store[key] == "4"

    assert _refund(r, receipt, T0 + timedelta(seconds=30)) is True
    assert r.store[key] == "3"
    assert _refund(r, receipt, T0 + timedelta(seconds=31)) is False
    assert r.store[key] == "3"


def test_refund_never_drives_count_below_zero():
    r = _FakeRedis()
    receipt = asyncio.run(wa.debit_wallet_a(r, now=T0, timezone_name="UTC"))
    r.store[receipt.count_key] = "0"  # e.g. key reset/expired between debit and refund
    _refund(r, receipt, T0)
    assert r.store[receipt.count_key] == "0"

    r2 = _FakeRedis()
    receipt2 = asyncio.run(wa.debit_wallet_a(r2, now=T0, timezone_name="UTC"))
    del r2.store[receipt2.count_key]
    _refund(r2, receipt2, T0)
    assert receipt2.count_key not in r2.store


def test_refund_after_midnight_decrements_the_debited_day():
    r = _FakeRedis()
    late = datetime(2026, 9, 24, 23, 59, tzinfo=timezone.utc)
    receipt = asyncio.run(wa.debit_wallet_a(r, now=late, timezone_name="UTC"))
    day1 = f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-24"
    day2 = f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-25"
    r.store[day2] = "2"

    _refund(r, receipt, late + timedelta(minutes=4))

    assert r.store[day1] == "0"
    assert r.store[day2] == "2"


def test_refund_cooldown_uses_prior_schedule_when_that_is_later_than_the_floor():
    r = _FakeRedis()
    prior = T0 - timedelta(seconds=1000)  # a real read 1000s ago -> due at prior+8400
    r.store[wa.WALLET_A_COOLDOWN_KEY] = prior.isoformat()
    receipt = asyncio.run(wa.debit_wallet_a(r, now=T0, timezone_name="UTC"))
    _refund(r, receipt, T0 + timedelta(seconds=10))
    assert r.store[wa.WALLET_A_COOLDOWN_KEY] == prior.isoformat()


def test_refund_cooldown_floor_after_refusal_when_prior_is_old_or_missing():
    r = _FakeRedis()
    receipt = asyncio.run(wa.debit_wallet_a(r, now=T0, timezone_name="UTC"))
    refused_at = T0 + timedelta(seconds=240)
    _refund(r, receipt, refused_at)
    last = datetime.fromisoformat(r.store[wa.WALLET_A_COOLDOWN_KEY])
    # eligible = last + 8400 == refused_at + 1800
    assert last + timedelta(seconds=8400) == refused_at + timedelta(seconds=1800)


def test_refund_never_lengthens_cooldown():
    r = _FakeRedis()
    receipt = asyncio.run(wa.debit_wallet_a(r, now=T0, timezone_name="UTC"))
    # effective == floor: refund-time + floor would be later than the debit's own schedule.
    _refund(r, receipt, T0 + timedelta(seconds=300), effective=1800.0, floor=1800.0)
    assert r.store[wa.WALLET_A_COOLDOWN_KEY] == T0.isoformat()


def test_refund_does_not_clobber_a_later_debit_cooldown():
    r = _FakeRedis()
    receipt = asyncio.run(wa.debit_wallet_a(r, now=T0, timezone_name="UTC"))
    later = T0 + timedelta(seconds=60)
    r.store[wa.WALLET_A_COOLDOWN_KEY] = later.isoformat()
    _refund(r, receipt, T0 + timedelta(seconds=90))
    assert r.store[wa.WALLET_A_COOLDOWN_KEY] == later.isoformat()


def test_wallet_b_refund_uses_wallet_b_keys_only():
    r = _FakeRedis()
    receipt = asyncio.run(wb.debit_wallet_b(r, now=T0, timezone_name="UTC"))
    _refund(r, receipt, T0, refund=wb.refund_wallet_b)
    assert all(k.startswith("orion:wp_read:wallet_b:") for k in r.store)
    assert r.store[f"{wb.WALLET_B_COUNT_KEY_PREFIX}2026-09-24"] == "0"


def test_refused_before_work_classifier_against_live_reasons():
    refunded = [
        "turn_deferred:stance_react_failed: agent=gateway_capacity_rejected:capacity_wait_budget_exhausted",
        "turn_deferred:stance_react_failed: agent=gpu_pool_unavailable:deadline",
        "turn_deferred:stance_react_failed: stance_react exec result missing thought payload",
        "turn_deferred:empty_imperative",
        "turn_deferred:stance_react_timeout",
        "turn_deferred",
        "bus_unavailable",
    ]
    charged = [
        None,
        "",
        "turn_error:fcc_stream_stalled",
        "turn_error:context_overflow",
        "stage1_turn_timeout",
        "stage2_turn_timeout",
        "turn_exception:boom",
        "empty_generation",
        "journal_bus_unavailable",
        "handoff_invalid:x",
        "interrupted:process_restart",
    ]
    assert all(is_refused_before_work(r) for r in refunded)
    assert not any(is_refused_before_work(r) for r in charged)
