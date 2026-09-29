"""Restart-durable queue of world_pulse_digest journal composes that failed retryably.

Why this exists: the world-news journal is composed once, when
`orion:world_pulse:run:result` arrives. If compose failed (live 2026-09-25..29:
`journal_compose_failed:{'message': 'gpu_pool_unavailable:deadline'}` because the
fast GPU lane is congested at 06:00 local), nothing ever retried it, so the daily
world-news email silently stopped. This store keeps the failed run so the scheduler
loop can retry it with backoff, across process restarts.

It also remembers which run_ids already produced a journal (`completed`), so a
retry never writes a second world_pulse_digest entry for the same run -- the
in-memory journal deduper does not survive a restart; this file does.

Modelled on `scheduler_cursor_store.py` (same directory, same atomic tmp+replace).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List

logger = logging.getLogger("orion-actions.pending_journal")

PENDING_JOURNAL_STORE_FILENAME = "pending_journals.json"
# Backoff between retry attempts, indexed by failures so far (1 -> 5 min, ...,
# 4+ -> 120 min). Give-up is governed by max age, not attempt count.
RETRY_BACKOFF_MINUTES: tuple[int, ...] = (5, 15, 45, 120)
# How long a completed run_id is remembered for duplicate suppression.
COMPLETED_RETENTION = timedelta(days=7)


def pending_journal_store_path_for(scheduler_cursor_path: Path) -> Path:
    """The pending-journal file lives next to scheduler_cursors.json (same bind mount)."""
    return scheduler_cursor_path.parent / PENDING_JOURNAL_STORE_FILENAME


def backoff_for_attempts(attempts: int) -> timedelta:
    idx = max(1, int(attempts)) - 1
    idx = min(idx, len(RETRY_BACKOFF_MINUTES) - 1)
    return timedelta(minutes=RETRY_BACKOFF_MINUTES[idx])


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class PendingJournal:
    run_id: str
    payload: Dict[str, Any]
    correlation_id: str
    attempts: int
    next_at: str
    first_failed_at: str
    last_error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def next_at_dt(self) -> datetime:
        return _parse(self.next_at) or datetime.now(timezone.utc)

    @property
    def first_failed_at_dt(self) -> datetime:
        return _parse(self.first_failed_at) or datetime.now(timezone.utc)


class PendingJournalStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._pending: Dict[str, PendingJournal] = {}
        self._completed: Dict[str, str] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text() or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            # Keep the unreadable file for inspection rather than letting the next
            # _persist silently overwrite it (it holds the completed-run_id list
            # that prevents duplicate journals).
            quarantine = self._path.with_name(
                f"{self._path.name}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
            try:
                self._path.replace(quarantine)
            except OSError:
                quarantine = None
            logger.warning(
                "pending_journal_store_load_failed path=%s error=%s quarantined=%s",
                self._path,
                exc.__class__.__name__,
                quarantine,
            )
            return
        if not isinstance(raw, dict):
            return
        pending_raw = raw.get("pending") if isinstance(raw.get("pending"), dict) else {}
        for run_id, item in pending_raw.items():
            if not isinstance(run_id, str) or not isinstance(item, dict):
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if _parse(item.get("next_at")) is None or _parse(item.get("first_failed_at")) is None:
                continue
            try:
                attempts = int(item.get("attempts") or 0)
            except (TypeError, ValueError):
                attempts = 0
            self._pending[run_id] = PendingJournal(
                run_id=run_id,
                payload=payload,
                correlation_id=str(item.get("correlation_id") or ""),
                attempts=attempts,
                next_at=str(item["next_at"]),
                first_failed_at=str(item["first_failed_at"]),
                last_error=str(item.get("last_error") or ""),
                extra=item.get("extra") if isinstance(item.get("extra"), dict) else {},
            )
        completed_raw = raw.get("completed") if isinstance(raw.get("completed"), dict) else {}
        for run_id, ts in completed_raw.items():
            if isinstance(run_id, str) and _parse(ts) is not None:
                self._completed[run_id] = str(ts)

    def _persist(self) -> None:
        data = {
            "pending": {k: asdict(v) for k, v in sorted(self._pending.items())},
            "completed": dict(sorted(self._completed.items())),
        }
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, sort_keys=True))
        temp.replace(self._path)

    def _prune_completed(self, now: datetime) -> None:
        cutoff = now - COMPLETED_RETENTION
        stale = [k for k, ts in self._completed.items() if (_parse(ts) or now) < cutoff]
        for k in stale:
            self._completed.pop(k, None)

    # --- queries -------------------------------------------------------------

    def get(self, run_id: str) -> PendingJournal | None:
        with self._lock:
            return self._pending.get(run_id)

    def pending(self) -> List[PendingJournal]:
        with self._lock:
            return list(self._pending.values())

    def due(self, now: datetime) -> List[PendingJournal]:
        with self._lock:
            return sorted(
                (p for p in self._pending.values() if p.next_at_dt <= now),
                key=lambda p: p.next_at_dt,
            )

    def is_completed(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._completed

    # --- mutations -----------------------------------------------------------

    def record_failure(
        self,
        *,
        run_id: str,
        payload: Dict[str, Any],
        correlation_id: str,
        error: str,
        now: datetime,
    ) -> PendingJournal | None:
        """Enqueue a first failure, or bump attempts/next_at on an existing entry.

        Returns None (no-op) when the run already produced a journal."""
        with self._lock:
            if run_id in self._completed:
                return None
            existing = self._pending.get(run_id)
            if existing is None:
                entry = PendingJournal(
                    run_id=run_id,
                    payload=payload,
                    correlation_id=correlation_id,
                    attempts=1,
                    next_at=_iso(now + backoff_for_attempts(1)),
                    first_failed_at=_iso(now),
                    last_error=error[:500],
                )
            else:
                attempts = existing.attempts + 1
                entry = PendingJournal(
                    run_id=run_id,
                    payload=existing.payload,
                    correlation_id=existing.correlation_id or correlation_id,
                    attempts=attempts,
                    next_at=_iso(now + backoff_for_attempts(attempts)),
                    first_failed_at=existing.first_failed_at,
                    last_error=error[:500],
                    extra=existing.extra,
                )
            self._pending[run_id] = entry
            self._persist()
            return entry

    def mark_completed(self, run_id: str, *, now: datetime) -> None:
        with self._lock:
            self._pending.pop(run_id, None)
            self._completed[run_id] = _iso(now)
            self._prune_completed(now)
            self._persist()

    def remove(self, run_id: str) -> PendingJournal | None:
        with self._lock:
            entry = self._pending.pop(run_id, None)
            if entry is not None:
                self._persist()
            return entry
