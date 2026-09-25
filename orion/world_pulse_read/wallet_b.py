"""Wallet B — isolated Redis daily cap + cooldown for world-pulse Stage 2.

Duplicate of Wallet A helpers (same Juniper choice): do not import the
curiosity loop, and do not write Wallet A or Curiosity Atlas Redis keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.world_pulse_read.wallet_refund import (
    WalletDebit,
    read_retry_wait,
    record_debit,
    refund_debit,
    settle_turn_ran,
)

WALLET_B_COOLDOWN_KEY = "orion:wp_read:wallet_b:last_at"
WALLET_B_COUNT_KEY_PREFIX = "orion:wp_read:wallet_b:count:"
# Set only by a refund (a turn refused before reading): earliest retry time,
# with backoff over consecutive refunds. See wallet_refund.py.
WALLET_B_RETRY_NOT_BEFORE_KEY = "orion:wp_read:wallet_b:retry_not_before"
WALLET_B_REFUND_STREAK_KEY = "orion:wp_read:wallet_b:refund_streak"
_STATE_TTL_SEC = 172800


@dataclass(frozen=True)
class WalletBInputs:
    enabled: bool
    done_today: int
    daily_cap: int
    seconds_since_last: float | None
    min_cooldown_sec: float
    now_hour: int | None = None
    window_start_hour: int = 0
    window_end_hour: int = 0
    seconds_until_retry: float | None = None


def window_is_configured(start_hour: int, end_hour: int) -> bool:
    return start_hour >= 0 and end_hour >= 0 and start_hour != end_hour


def window_seconds(start_hour: int, end_hour: int) -> int:
    if not window_is_configured(start_hour, end_hour):
        return 24 * 3600
    if start_hour < end_hour:
        return (end_hour - start_hour) * 3600
    return (24 - start_hour + end_hour) * 3600


def in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    if not window_is_configured(start_hour, end_hour):
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def paced_cooldown_sec(
    *, min_cooldown_sec: float, daily_cap: int, start_hour: int, end_hour: int
) -> float:
    if daily_cap <= 0 or not window_is_configured(start_hour, end_hour):
        return min_cooldown_sec
    spread = window_seconds(start_hour, end_hour) / daily_cap
    return max(min_cooldown_sec, spread)


def wallet_b_block_reason(inp: WalletBInputs) -> str | None:
    if not inp.enabled:
        return "disabled"
    if inp.daily_cap >= 0 and inp.done_today >= inp.daily_cap:
        return "daily_cap"
    if inp.now_hour is not None and not in_window(
        inp.now_hour, inp.window_start_hour, inp.window_end_hour
    ):
        return "outside_window"
    if (
        inp.seconds_since_last is not None
        and inp.seconds_since_last < inp.min_cooldown_sec
    ):
        return "cooldown"
    if inp.seconds_until_retry is not None and inp.seconds_until_retry > 0:
        return "refund_backoff"
    return None


def _local_now(now: datetime, timezone_name: str) -> datetime:
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, KeyError):
        tz = timezone.utc
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(tz)


def _daily_key(now: datetime, timezone_name: str) -> str:
    return f"{WALLET_B_COUNT_KEY_PREFIX}{_local_now(now, timezone_name).date().isoformat()}"


def _decode(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


async def debit_wallet_b(
    redis,
    *,
    now: datetime,
    timezone_name: str,
    ttl_sec: int = _STATE_TTL_SEC,
) -> WalletDebit:
    """Charge one slot. Keep the receipt: :func:`refund_wallet_b` undoes it
    when the turn was refused before any reading happened."""
    return await record_debit(
        redis,
        cooldown_key=WALLET_B_COOLDOWN_KEY,
        count_key=_daily_key(now, timezone_name),
        retry_key=WALLET_B_RETRY_NOT_BEFORE_KEY,
        streak_key=WALLET_B_REFUND_STREAK_KEY,
        now=now,
        ttl_sec=ttl_sec,
    )


async def refund_wallet_b(
    redis,
    receipt: WalletDebit | None,
    *,
    now: datetime,
    backoff_base_sec: float,
    backoff_cap_sec: float,
) -> bool:
    """Undo a Wallet B debit (see :func:`orion.world_pulse_read.wallet_refund.refund_debit`)."""
    return await refund_debit(
        redis,
        receipt,
        now=now,
        backoff_base_sec=backoff_base_sec,
        backoff_cap_sec=backoff_cap_sec,
    )


async def settle_wallet_b(redis, receipt: WalletDebit | None) -> None:
    """The turn reached the reader: keep the charge, reset the refund streak."""
    await settle_turn_ran(redis, receipt)


async def read_wallet_b_retry_wait(redis, *, now: datetime) -> float | None:
    """Seconds until a refunded turn may retry (None when not backing off)."""
    return await read_retry_wait(redis, WALLET_B_RETRY_NOT_BEFORE_KEY, now=now)


async def read_wallet_b_state(
    redis,
    *,
    now: datetime,
    timezone_name: str,
) -> tuple[float | None, int]:
    since: float | None = None
    raw = _decode(await redis.get(WALLET_B_COOLDOWN_KEY))
    if raw:
        last = datetime.fromisoformat(raw)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        since = max(0.0, (now - last).total_seconds())
    count = 0
    raw_count = _decode(await redis.get(_daily_key(now, timezone_name)))
    if raw_count is not None:
        count = int(raw_count)
    return since, count
