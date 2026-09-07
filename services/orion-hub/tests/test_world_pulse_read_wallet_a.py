import asyncio
from datetime import datetime, timezone

from orion.world_pulse_read import wallet_a as wa


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


def _inputs(**over) -> wa.WalletAInputs:
    base = dict(
        enabled=True,
        done_today=0,
        daily_cap=6,
        seconds_since_last=99999,
        min_cooldown_sec=60,
        now_hour=12,
        window_start_hour=8,
        window_end_hour=22,
    )
    base.update(over)
    return wa.WalletAInputs(**base)


def test_block_reason_daily_cap():
    inp = wa.WalletAInputs(
        enabled=True,
        done_today=6,
        daily_cap=6,
        seconds_since_last=99999,
        min_cooldown_sec=60,
        now_hour=12,
        window_start_hour=8,
        window_end_hour=22,
    )
    assert wa.wallet_a_block_reason(inp) == "daily_cap"


def test_debit_uses_wallet_a_keys_only():
    r = _FakeRedis()
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)

    async def _run():
        await wa.debit_wallet_a(r, now=now, timezone_name="UTC")
        return sorted(r.store)

    keys = asyncio.run(_run())
    assert keys == sorted(
        [
            wa.WALLET_A_COOLDOWN_KEY,
            wa.WALLET_A_COUNT_KEY_PREFIX + "2026-09-06",
        ]
    )
    # Curiosity Atlas keys must not appear
    assert not any(k.startswith("orion:curiosity:") for k in keys)


def test_debit_does_not_touch_preexisting_curiosity_keys():
    from scripts.curiosity_investigation import _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX

    r = _FakeRedis()
    r.store[_COOLDOWN_KEY] = "already"
    r.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] = "3"
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)
    asyncio.run(wa.debit_wallet_a(r, now=now, timezone_name="UTC"))
    assert r.store[_COOLDOWN_KEY] == "already"
    assert r.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] == "3"


def test_gate_order_disabled_beats_daily_cap():
    assert wa.wallet_a_block_reason(_inputs(enabled=False, done_today=6)) == "disabled"


def test_gate_order_daily_cap_beats_outside_window():
    assert wa.wallet_a_block_reason(_inputs(done_today=6, now_hour=3)) == "daily_cap"


def test_gate_order_outside_window_beats_cooldown():
    assert wa.wallet_a_block_reason(
        _inputs(now_hour=3, seconds_since_last=1)
    ) == "outside_window"


def test_block_reason_cooldown():
    assert wa.wallet_a_block_reason(_inputs(seconds_since_last=10)) == "cooldown"


def test_block_reason_clear():
    assert wa.wallet_a_block_reason(_inputs()) is None


def test_negative_daily_cap_disables_the_cap():
    assert wa.wallet_a_block_reason(_inputs(daily_cap=-1, done_today=999)) is None


def test_first_debit_is_not_blocked_by_cooldown():
    assert wa.wallet_a_block_reason(_inputs(seconds_since_last=None)) is None


def test_read_wallet_a_state_after_debit():
    r = _FakeRedis()
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)

    async def _run():
        await wa.debit_wallet_a(r, now=now, timezone_name="UTC")
        later = datetime(2026, 9, 6, 15, 10, tzinfo=timezone.utc)
        return await wa.read_wallet_a_state(r, now=later, timezone_name="UTC")

    since, count = asyncio.run(_run())
    assert since == 600.0
    assert count == 1


def test_wallet_a_module_does_not_import_curiosity():
    import inspect

    source = inspect.getsource(wa)
    assert "curiosity_investigation" not in source
    assert "orion:curiosity:" not in source
