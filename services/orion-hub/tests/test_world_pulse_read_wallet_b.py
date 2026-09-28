import asyncio
from datetime import datetime, timezone

from orion.world_pulse_read import wallet_b as wb


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

    async def expire(self, key, ttl):
        return True


def _inputs(**over) -> wb.WalletBInputs:
    base = dict(
        enabled=True,
        now_hour=12,
        window_start_hour=8,
        window_end_hour=22,
    )
    base.update(over)
    return wb.WalletBInputs(**base)


def test_debit_uses_wallet_b_keys_only():
    r = _FakeRedis()
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)

    async def _run():
        await wb.debit_wallet_b(r, now=now, timezone_name="UTC")
        return sorted(r.store)

    keys = asyncio.run(_run())
    assert keys == sorted(
        [
            wb.WALLET_B_COOLDOWN_KEY,
            wb.WALLET_B_COUNT_KEY_PREFIX + "2026-09-06",
        ]
    )
    assert keys == sorted(
        [
            "orion:wp_read:wallet_b:last_at",
            "orion:wp_read:wallet_b:count:2026-09-06",
        ]
    )
    assert not any(k.startswith("orion:curiosity:") for k in keys)
    assert not any(k.startswith("orion:wp_read:wallet_a:") for k in keys)


def test_debit_does_not_touch_preexisting_curiosity_keys():
    from scripts.curiosity_investigation import _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX

    r = _FakeRedis()
    r.store[_COOLDOWN_KEY] = "already"
    r.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] = "3"
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)
    asyncio.run(wb.debit_wallet_b(r, now=now, timezone_name="UTC"))
    assert r.store[_COOLDOWN_KEY] == "already"
    assert r.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] == "3"


def test_debit_does_not_touch_preexisting_wallet_a_keys():
    from orion.world_pulse_read import wallet_a as wa

    r = _FakeRedis()
    r.store[wa.WALLET_A_COOLDOWN_KEY] = "already"
    r.store[wa.WALLET_A_COUNT_KEY_PREFIX + "2026-09-06"] = "4"
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)
    asyncio.run(wb.debit_wallet_b(r, now=now, timezone_name="UTC"))
    assert r.store[wa.WALLET_A_COOLDOWN_KEY] == "already"
    assert r.store[wa.WALLET_A_COUNT_KEY_PREFIX + "2026-09-06"] == "4"


def test_gate_order_disabled_beats_outside_window():
    assert wb.wallet_b_block_reason(_inputs(enabled=False, now_hour=3)) == "disabled"


def test_gate_order_outside_window_beats_refund_backoff():
    assert wb.wallet_b_block_reason(_inputs(now_hour=3, seconds_until_retry=30)) == "outside_window"


def test_block_reason_clear():
    assert wb.wallet_b_block_reason(_inputs()) is None


def test_no_daily_cap_or_cooldown_fields():
    """Budgets removed 2026-09-28: the only gates are the switch, the window
    and the refund backoff."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(wb.WalletBInputs)}
    assert fields.isdisjoint({"done_today", "daily_cap", "seconds_since_last", "min_cooldown_sec"})
    assert wb.wallet_b_block_reason(_inputs(window_start_hour=0, window_end_hour=0, now_hour=None)) is None


def test_read_wallet_b_state_after_debit():
    r = _FakeRedis()
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)

    async def _run():
        await wb.debit_wallet_b(r, now=now, timezone_name="UTC")
        later = datetime(2026, 9, 6, 15, 10, tzinfo=timezone.utc)
        return await wb.read_wallet_b_state(r, now=later, timezone_name="UTC")

    since, count = asyncio.run(_run())
    assert since == 600.0
    assert count == 1


def test_wallet_b_module_does_not_import_curiosity():
    import inspect

    source = inspect.getsource(wb)
    assert "curiosity_investigation" not in source
    assert "orion:curiosity:" not in source
    assert "orion:wp_read:wallet_a:" not in source
