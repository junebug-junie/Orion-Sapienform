"""Track substrate cursor tail-seeds that skip unreduced history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

logger = __import__("logging").getLogger("orion.substrate.cursor_gaps")


@dataclass(frozen=True)
class TailSeedRecord:
    cursor_name: str
    reason: str
    at: datetime
    prior_created_at: datetime | None
    prior_event_id: str | None
    seeded_created_at: datetime
    seeded_event_id: str


_lock = Lock()
_tail_seeds: list[TailSeedRecord] = []


def record_tail_seed(record: TailSeedRecord) -> None:
    with _lock:
        _tail_seeds.append(record)
        if len(_tail_seeds) > 200:
            del _tail_seeds[: len(_tail_seeds) - 200]

    logger.error(
        "substrate_data_gap cursor=%s reason=%s prior=(%s,%s) seeded=(%s,%s)",
        record.cursor_name,
        record.reason,
        record.prior_created_at,
        record.prior_event_id,
        record.seeded_created_at,
        record.seeded_event_id,
    )


def tail_seed_snapshot() -> dict:
    with _lock:
        records = list(_tail_seeds)
    latest = records[-1] if records else None
    return {
        "count": len(records),
        "cold_start_count": sum(1 for r in records if r.reason == "cold_start"),
        "lag_exceeded_count": sum(1 for r in records if r.reason == "lag_exceeded"),
        "latest": (
            {
                "cursor_name": latest.cursor_name,
                "reason": latest.reason,
                "at": latest.at.isoformat(),
                "prior_created_at": (
                    latest.prior_created_at.isoformat() if latest.prior_created_at else None
                ),
                "prior_event_id": latest.prior_event_id,
                "seeded_created_at": latest.seeded_created_at.isoformat(),
                "seeded_event_id": latest.seeded_event_id,
            }
            if latest
            else None
        ),
        "recent": [
            {
                "cursor_name": r.cursor_name,
                "reason": r.reason,
                "at": r.at.isoformat(),
            }
            for r in records[-10:]
        ],
    }


def has_recent_tail_seed(within_sec: float = 3600) -> bool:
    cutoff = datetime.now(timezone.utc).timestamp() - within_sec
    with _lock:
        return any(r.at.timestamp() >= cutoff for r in _tail_seeds)


def has_cold_start_tail_seed() -> bool:
    with _lock:
        return any(r.reason == "cold_start" for r in _tail_seeds)


def clear_tail_seeds_for_tests() -> None:
    with _lock:
        _tail_seeds.clear()
