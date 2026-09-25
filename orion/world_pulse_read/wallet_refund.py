"""Refund a world-pulse wallet debit for a turn that never did any reading.

Both loops debit their wallet (daily count + cooldown timestamp) right after
claiming a seed, *before* the turn runs. Live 2026-09-23/24 every Stage 1 slot
of the day was spent on turns the stance phase refused for GPU capacity
(``turn_deferred:stance_react_failed: ...capacity...``): six refusals, zero
reads, then ``world_pulse_read_blocked reason=daily_cap`` for the rest of the
day. A refusal before any reading must not cost a reading slot.

A refund does three things:

* gives the day's slot back (the day that was charged, never below zero);
* puts the cooldown timestamp back to the last debit that actually counted, so
  ``last_at`` keeps meaning "last real reading attempt" on the dashboard;
* sets a separate retry-not-before time with exponential backoff over
  consecutive refunds. Without it the daily cap was the only thing stopping a
  capacity outage from retrying every tick -- each retry is a real stance call
  on the saturated GPU and spends one of the seed's ``max_attempts``.

Keys stay owned by ``wallet_a`` / ``wallet_b``: this module only replays the
receipt their ``debit_*`` returned, so it never names a Redis key itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class WalletDebit:
    """What one ``debit_wallet_a`` / ``debit_wallet_b`` call wrote.

    ``count_key`` is the *day's* key the debit incremented, so a refund after
    local midnight decrements the day that was actually charged, not today.
    ``prior_last_at_raw`` is the cooldown value before the debit (``None`` if
    there was none). ``refunded`` makes a second refund attempt of the same
    debit a no-op (at most one attempt, even if Redis errors mid-way).
    """

    cooldown_key: str
    count_key: str
    retry_key: str
    streak_key: str
    debited_at: datetime
    prior_last_at_raw: str | None
    ttl_sec: int
    refunded: bool = field(default=False)


def _decode(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _parse_ts(raw: object) -> datetime | None:
    text = _decode(raw)
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


async def record_debit(
    redis,
    *,
    cooldown_key: str,
    count_key: str,
    retry_key: str,
    streak_key: str,
    now: datetime,
    ttl_sec: int,
) -> WalletDebit:
    """Write one debit (cooldown timestamp + day counter) and return its receipt."""
    prior = _decode(await redis.get(cooldown_key))
    await redis.setex(cooldown_key, ttl_sec, now.isoformat())
    await redis.incr(count_key)
    await redis.expire(count_key, ttl_sec)
    return WalletDebit(
        cooldown_key=cooldown_key,
        count_key=count_key,
        retry_key=retry_key,
        streak_key=streak_key,
        debited_at=now,
        prior_last_at_raw=prior,
        ttl_sec=ttl_sec,
    )


def refund_backoff_sec(streak: int, *, base_sec: float, cap_sec: float) -> float:
    """``base * 2**(streak-1)``, capped. ``streak`` is consecutive refunds incl. this one.

    A zero base (MIN_COOLDOWN_SEC=0) falls back to the cap: the refund has
    already restored the old cooldown, so with no retry time at all a capacity
    outage would retry every tick -- worse than never refunding."""
    if base_sec <= 0:
        base_sec = cap_sec
    if base_sec <= 0 or streak <= 0:
        return 0.0
    return float(min(base_sec * (2 ** min(streak - 1, 20)), max(base_sec, cap_sec)))


async def refund_debit(
    redis,
    receipt: WalletDebit | None,
    *,
    now: datetime,
    backoff_base_sec: float,
    backoff_cap_sec: float,
) -> bool:
    """Undo ``receipt``'s debit. Returns True if the slot or cooldown was refunded.

    Count: decremented on the day the debit hit, never below zero (a key that
    already expired or reads 0 is left alone).

    Cooldown: restored to the pre-debit value (deleted if there was none), but
    only if the key still holds *this* debit's timestamp, so a later real
    debit is never overwritten.

    Retry spacing: ``retry_key`` = now + backoff over the consecutive-refund
    streak (``streak_key``; cleared by :func:`settle_turn_ran` when a turn
    reaches the reader).
    """
    if receipt is None or receipt.refunded or redis is None:
        return False
    receipt.refunded = True
    now = _aware(now)

    count_refunded = False
    raw_count = _decode(await redis.get(receipt.count_key))
    try:
        current = int(raw_count) if raw_count is not None else 0
    except ValueError:
        current = 0
    if current > 0:
        after = await redis.decr(receipt.count_key)
        if after is not None and int(after) < 0:
            # Key expired between get and decr: undo, and keep it from living forever.
            await redis.incr(receipt.count_key)
            await redis.expire(receipt.count_key, receipt.ttl_sec)
        else:
            count_refunded = True

    cooldown_restored = False
    held = _parse_ts(await redis.get(receipt.cooldown_key))
    if held is not None and held == _aware(receipt.debited_at):
        if receipt.prior_last_at_raw:
            await redis.setex(receipt.cooldown_key, receipt.ttl_sec, receipt.prior_last_at_raw)
        else:
            await redis.delete(receipt.cooldown_key)
        cooldown_restored = True

    streak = int(await redis.incr(receipt.streak_key))
    await redis.expire(receipt.streak_key, receipt.ttl_sec)
    backoff = refund_backoff_sec(streak, base_sec=backoff_base_sec, cap_sec=backoff_cap_sec)
    if backoff > 0:
        not_before = now + timedelta(seconds=backoff)
        await redis.setex(receipt.retry_key, receipt.ttl_sec, not_before.isoformat())
    return count_refunded or cooldown_restored


async def settle_turn_ran(redis, receipt: WalletDebit | None) -> None:
    """The turn reached the reader (success or a real failure): the debit
    stands, the consecutive-refund streak resets, and any pending refund
    backoff (possible after a forced tick overrode it) is cleared -- the
    debit's own cooldown now governs spacing."""
    if receipt is None or redis is None:
        return
    await redis.delete(receipt.streak_key)
    await redis.delete(receipt.retry_key)


async def read_retry_wait(redis, retry_key: str, *, now: datetime) -> float | None:
    """Seconds left before a refunded turn may retry, or None if not waiting."""
    not_before = _parse_ts(await redis.get(retry_key))
    if not_before is None:
        return None
    remaining = (not_before - _aware(now)).total_seconds()
    return remaining if remaining > 0 else None
