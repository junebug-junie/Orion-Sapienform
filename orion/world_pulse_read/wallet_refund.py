"""Refund a world-pulse wallet debit for a turn that never did any reading.

Both loops debit their wallet (daily count + cooldown timestamp) right after
claiming a seed, *before* the turn runs. Live 2026-09-23/24 every Stage 1 slot
of the day was spent on turns the stance phase refused for GPU capacity
(``turn_deferred:stance_react_failed: ...capacity...``): six refusals, zero
reads, then ``world_pulse_read_blocked reason=daily_cap`` for the rest of the
day. A refusal before any reading must not cost a reading slot.

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
    ``prior_last_at`` is the cooldown timestamp before the debit (``None`` if
    there was none). ``refunded`` makes a second refund of the same debit a
    no-op.
    """

    cooldown_key: str
    count_key: str
    debited_at: datetime
    prior_last_at: datetime | None
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


async def record_debit(
    redis, *, cooldown_key: str, count_key: str, now: datetime, ttl_sec: int
) -> WalletDebit:
    """Write one debit (cooldown timestamp + day counter) and return its receipt."""
    prior_last_at = _parse_ts(await redis.get(cooldown_key))
    await redis.setex(cooldown_key, ttl_sec, now.isoformat())
    await redis.incr(count_key)
    await redis.expire(count_key, ttl_sec)
    return WalletDebit(
        cooldown_key=cooldown_key,
        count_key=count_key,
        debited_at=now,
        prior_last_at=prior_last_at,
        ttl_sec=ttl_sec,
    )


async def refund_debit(
    redis,
    receipt: WalletDebit | None,
    *,
    now: datetime,
    effective_cooldown_sec: float,
    retry_floor_sec: float,
) -> bool:
    """Undo ``receipt``'s debit. Returns True if anything was refunded.

    Count: decremented on the day the debit hit, never below zero (a key that
    already expired or reads 0 is left alone).

    Cooldown: the next attempt becomes eligible at whichever is later --
    the schedule the wallet was on before this debit (``prior_last_at`` +
    ``effective_cooldown_sec``), or ``now + retry_floor_sec``. The floor keeps a
    capacity outage from turning into a retry on every tick: each retry is a
    real stance call on the saturated GPU and spends one of the seed's
    ``max_attempts``. Never pushed *later* than the debit itself would have
    (a refund only ever shortens the wait). Only rewritten if the cooldown key still holds *this*
    debit's timestamp, so a later real debit is never overwritten.
    """
    if receipt is None or receipt.refunded or redis is None:
        return False
    receipt.refunded = True
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    count_refunded = False
    raw_count = _decode(await redis.get(receipt.count_key))
    try:
        current = int(raw_count) if raw_count is not None else 0
    except ValueError:
        current = 0
    if current > 0:
        after = await redis.decr(receipt.count_key)
        if after is not None and int(after) < 0:
            # Lost a race with an expiry/reset between get and decr: put it back.
            await redis.incr(receipt.count_key)
        else:
            count_refunded = True

    cooldown_restored = False
    held = _parse_ts(await redis.get(receipt.cooldown_key))
    debited_at = receipt.debited_at
    if debited_at.tzinfo is None:
        debited_at = debited_at.replace(tzinfo=timezone.utc)
    if held is not None and held == debited_at:
        effective = timedelta(seconds=max(0.0, float(effective_cooldown_sec)))
        floor = timedelta(seconds=max(0.0, float(retry_floor_sec)))
        eligible_at = now + floor
        if receipt.prior_last_at is not None:
            eligible_at = max(eligible_at, receipt.prior_last_at + effective)
        new_last_at = eligible_at - effective
        if new_last_at < debited_at:
            await redis.setex(receipt.cooldown_key, receipt.ttl_sec, new_last_at.isoformat())
            cooldown_restored = True
    return count_refunded or cooldown_restored
