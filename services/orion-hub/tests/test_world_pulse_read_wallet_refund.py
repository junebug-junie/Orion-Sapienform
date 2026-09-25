"""Refund semantics for a world-pulse wallet debit (orion/world_pulse_read/wallet_refund.py).

Lives under services/orion-hub/tests so the orion-reading CI glob
(test_world_pulse_read_*.py) runs it; orion/world_pulse_read/tests is not in CI.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from orion.world_pulse_read import wallet_a as wa
from orion.world_pulse_read import wallet_b as wb
from orion.world_pulse_read.retry import is_refused_before_work
from orion.world_pulse_read.wallet_refund import refund_backoff_sec


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

    async def delete(self, key):
        self.store.pop(key, None)

    async def expire(self, key, ttl):
        return True


T0 = datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)


def _debit(r, now=T0, debit=wa.debit_wallet_a):
    return asyncio.run(debit(r, now=now, timezone_name="UTC"))


def _refund(r, receipt, now, *, base=1800.0, cap=14400.0, refund=wa.refund_wallet_a):
    return asyncio.run(refund(r, receipt, now=now, backoff_base_sec=base, backoff_cap_sec=cap))


def test_refund_restores_count_and_only_once():
    r = _FakeRedis()
    key = f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-24"
    r.store[key] = "3"
    receipt = _debit(r)
    assert r.store[key] == "4"

    assert _refund(r, receipt, T0 + timedelta(seconds=30)) is True
    assert r.store[key] == "3"
    assert _refund(r, receipt, T0 + timedelta(seconds=31)) is False
    assert r.store[key] == "3"
    assert r.store[wa.WALLET_A_REFUND_STREAK_KEY] == "1"


def test_refund_never_drives_count_below_zero():
    r = _FakeRedis()
    receipt = _debit(r)
    r.store[receipt.count_key] = "0"  # e.g. key reset between debit and refund
    _refund(r, receipt, T0)
    assert r.store[receipt.count_key] == "0"

    r2 = _FakeRedis()
    receipt2 = _debit(r2)
    del r2.store[receipt2.count_key]  # expired
    _refund(r2, receipt2, T0)
    assert receipt2.count_key not in r2.store


def test_refund_after_midnight_decrements_the_debited_day():
    r = _FakeRedis()
    late = datetime(2026, 9, 24, 23, 59, tzinfo=timezone.utc)
    receipt = _debit(r, now=late)
    day1 = f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-24"
    day2 = f"{wa.WALLET_A_COUNT_KEY_PREFIX}2026-09-25"
    r.store[day2] = "2"

    _refund(r, receipt, late + timedelta(minutes=4))

    assert r.store[day1] == "0"
    assert r.store[day2] == "2"


def test_refund_restores_prior_cooldown_or_clears_it():
    r = _FakeRedis()
    prior = (T0 - timedelta(hours=3)).isoformat()
    r.store[wa.WALLET_A_COOLDOWN_KEY] = prior
    _refund(r, _debit(r), T0 + timedelta(seconds=10))
    assert r.store[wa.WALLET_A_COOLDOWN_KEY] == prior

    r2 = _FakeRedis()
    _refund(r2, _debit(r2), T0 + timedelta(seconds=10))
    assert wa.WALLET_A_COOLDOWN_KEY not in r2.store


def test_refund_does_not_clobber_a_later_debit_cooldown():
    r = _FakeRedis()
    receipt = _debit(r)
    later = T0 + timedelta(seconds=60)
    r.store[wa.WALLET_A_COOLDOWN_KEY] = later.isoformat()
    _refund(r, receipt, T0 + timedelta(seconds=90))
    assert r.store[wa.WALLET_A_COOLDOWN_KEY] == later.isoformat()


def test_retry_not_before_backs_off_over_consecutive_refunds_and_resets():
    r = _FakeRedis()
    waits = []
    for i in range(6):
        now = T0 + timedelta(hours=i * 5)
        _refund(r, _debit(r, now=now), now)
        waits.append(asyncio.run(wa.read_wallet_a_retry_wait(r, now=now)))
    assert waits == [1800.0, 3600.0, 7200.0, 14400.0, 14400.0, 14400.0]

    asyncio.run(wa.settle_wallet_a(r, _debit(r)))
    assert wa.WALLET_A_REFUND_STREAK_KEY not in r.store
    now = T0 + timedelta(days=2)
    _refund(r, _debit(r, now=now), now)
    assert asyncio.run(wa.read_wallet_a_retry_wait(r, now=now)) == 1800.0


def test_retry_wait_is_none_once_elapsed():
    r = _FakeRedis()
    _refund(r, _debit(r), T0)
    assert asyncio.run(wa.read_wallet_a_retry_wait(r, now=T0 + timedelta(seconds=1801))) is None


def test_backoff_math():
    assert refund_backoff_sec(1, base_sec=0, cap_sec=100) == 0.0
    assert refund_backoff_sec(1, base_sec=600, cap_sec=100) == 600.0  # cap never below base
    assert refund_backoff_sec(99, base_sec=600, cap_sec=4800) == 4800.0


def test_block_reason_refund_backoff_is_last():
    base = dict(enabled=True, done_today=0, daily_cap=6, seconds_since_last=None, min_cooldown_sec=60)
    assert wa.wallet_a_block_reason(wa.WalletAInputs(**base, seconds_until_retry=30)) == "refund_backoff"
    assert wa.wallet_a_block_reason(wa.WalletAInputs(**base, seconds_until_retry=None)) is None
    capped = dict(base, done_today=6)
    assert wa.wallet_a_block_reason(wa.WalletAInputs(**capped, seconds_until_retry=30)) == "daily_cap"
    assert wb.wallet_b_block_reason(wb.WalletBInputs(**base, seconds_until_retry=5)) == "refund_backoff"


def test_wallet_b_refund_uses_wallet_b_keys_only():
    r = _FakeRedis()
    receipt = _debit(r, debit=wb.debit_wallet_b)
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
    ]
    charged = [
        None,
        "",
        "turn_deferredX",
        "turn_error:fcc_stream_stalled",
        "turn_error:bus_unavailable",
        "turn_error:context_overflow",
        "stage1_turn_timeout",
        "stage2_turn_timeout",
        "turn_exception:boom",
        "empty_generation",
        "bus_unavailable",
        "journal_bus_unavailable",
        "handoff_invalid:x",
        "interrupted:process_restart",
    ]
    assert all(is_refused_before_work(r) for r in refunded)
    assert not any(is_refused_before_work(r) for r in charged)
