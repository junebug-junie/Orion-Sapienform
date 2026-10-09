"""Orion's learned shed action, pool side: the ``orion_self_shed`` reason on the U4 shed lever.

Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md, "Amendment 2026-09-29"
("A1 as amended"), D5 (kill switches and caps).

What this owns, and nothing else:
- the one RPC-settable reason ``orion_self_shed`` (precedence 1, background only -- see shed.py).
  The reflex's ``cooling_incident`` is never set or cleared here;
- the caps, enforced HERE so no caller can exceed them: max TTL, max shed-seconds per rolling 24 h,
  min gap from the end of one shed to the start of the next;
- the terminal states: ``expired`` (TTL ended), ``cancelled`` (clear verb or kill switch),
  ``preempted_by_reflex`` (a cabinet AC incident opened), ``refused`` (never started);
- the manipulation check: ``drained_at`` (first tick with no background lease granted),
  ``grants_withheld`` (distinct background leases this reason held back), ``delayed_grant_sec``
  (lease-seconds held back);
- the ledger (``gpu_pool_orion_shed``), so a restart can neither reset the daily cap nor orphan an
  active shed. A ledger that cannot be read or written refuses every set (fail closed: this is an
  optional experiment, never worth running blind).

Kill switch ``GPU_POOL_ORION_SHED_ENABLED=false``: every set is refused ``disabled`` and any active
shed found at boot settles ``cancelled``. It never touches ``cooling_incident``; the whole-lever
switch ``GPU_POOL_SHED_ENABLED`` stays the reflex's. With the whole lever off, a set is refused
``lever_disabled`` -- recording a shed that blocks nothing would be a fake treatment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Protocol
from uuid import uuid4

from orion.gpu_pool.shed import ShedBoard, ShedSignal
from orion.schemas.gpu_pool import (
    ORION_SELF_SHED_REASON,
    ORION_SHED_TERMINAL_STATES,
    GpuPoolShedReasonRequestV1,
    GpuPoolShedResultV1,
)

logger = logging.getLogger("orion.gpu_pool.orion_shed")

SHED_DECISION_REASON = f"shed:{ORION_SELF_SHED_REASON}"
DAY_SEC = 86400.0
# Below this much remaining daily budget a shed is refused rather than clipped to a stub.
MIN_USEFUL_TTL_SEC = 300.0
# Persist counters at most this often while a shed is active (state changes persist at once).
PERSIST_EVERY_SEC = 15.0


@dataclass(frozen=True)
class OrionShedCaps:
    max_ttl_sec: float = 900.0
    max_sec_per_day: float = 3600.0
    min_gap_sec: float = 900.0


@dataclass
class OrionShedRecord:
    shed_id: str
    dispatch_id: str
    state: str
    requested_at: datetime
    refusal: str | None = None
    ttl_sec: float | None = None
    started_at: datetime | None = None
    valid_until: datetime | None = None
    ended_at: datetime | None = None
    drained_at: datetime | None = None
    grants_withheld: int = 0
    delayed_grant_sec: float = 0.0
    background_live_at_start: int = 0
    correlation: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    withheld_ids: set[str] = field(default_factory=set)

    def used_sec(self, now: datetime) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at or now
        return max(0.0, (end - self.started_at).total_seconds())

    def to_result(self, ok: bool = True) -> GpuPoolShedResultV1:
        return GpuPoolShedResultV1(
            ok=ok, shed_id=self.shed_id, dispatch_id=self.dispatch_id, state=self.state,  # type: ignore[arg-type]
            refusal=self.refusal, ttl_sec=self.ttl_sec, started_at=self.started_at,
            valid_until=self.valid_until, ended_at=self.ended_at, drained_at=self.drained_at,
            grants_withheld=self.grants_withheld, delayed_grant_sec=round(self.delayed_grant_sec, 1),
            background_live_at_start=self.background_live_at_start,
            detail={**self.detail, "withheld_lease_ids": sorted(self.withheld_ids)[:50]},
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "shed_id": self.shed_id, "dispatch_id": self.dispatch_id, "reason": ORION_SELF_SHED_REASON,
            "state": self.state, "refusal": self.refusal, "ttl_sec": self.ttl_sec,
            "requested_at": self.requested_at, "started_at": self.started_at,
            "valid_until": self.valid_until, "ended_at": self.ended_at, "drained_at": self.drained_at,
            "grants_withheld": self.grants_withheld, "delayed_grant_sec": round(self.delayed_grant_sec, 3),
            "background_live_at_start": self.background_live_at_start,
            "correlation": dict(self.correlation),
            "detail": {**self.detail, "withheld_lease_ids": sorted(self.withheld_ids)[:200]},
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OrionShedRecord":
        detail = dict(row.get("detail") or {})
        withheld = set(detail.pop("withheld_lease_ids", []) or [])
        return cls(
            shed_id=row["shed_id"], dispatch_id=row["dispatch_id"], state=row["state"],
            requested_at=row["requested_at"], refusal=row.get("refusal"), ttl_sec=row.get("ttl_sec"),
            started_at=row.get("started_at"), valid_until=row.get("valid_until"),
            ended_at=row.get("ended_at"), drained_at=row.get("drained_at"),
            grants_withheld=int(row.get("grants_withheld") or 0),
            delayed_grant_sec=float(row.get("delayed_grant_sec") or 0.0),
            background_live_at_start=int(row.get("background_live_at_start") or 0),
            correlation=dict(row.get("correlation") or {}), detail=detail, withheld_ids=withheld,
        )


class OrionShedLedger(Protocol):
    async def ensure(self) -> bool: ...
    async def recent(self, since: datetime) -> list[dict[str, Any]]: ...
    async def by_dispatch(self, dispatch_id: str) -> dict[str, Any] | None: ...
    async def by_id(self, shed_id: str) -> dict[str, Any] | None: ...
    async def upsert(self, row: dict[str, Any]) -> None: ...


class MemoryOrionShedLedger:
    """Same contract, in memory: tests and the eval."""

    def __init__(self, *, available: bool = True) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.available = available

    async def ensure(self) -> bool:
        return self.available

    def _check(self) -> None:
        if not self.available:
            raise RuntimeError("ledger unavailable")

    async def recent(self, since):
        self._check()
        return [dict(r) for r in self.rows.values() if r["requested_at"] >= since]

    async def by_dispatch(self, dispatch_id):
        self._check()
        return next((dict(r) for r in self.rows.values() if r["dispatch_id"] == dispatch_id), None)

    async def by_id(self, shed_id):
        self._check()
        row = self.rows.get(shed_id)
        return dict(row) if row else None

    async def upsert(self, row):
        self._check()
        self.rows[row["shed_id"]] = dict(row)


def refusal_for_set(
    *, now: datetime, records: Iterable[OrionShedRecord], caps: OrionShedCaps, requested_ttl: float | None,
    enabled: bool, lever_enabled: bool, reflex_active: bool,
) -> tuple[str | None, float]:
    """Pure: (refusal or None, granted TTL). Order matters only for which reason is reported."""
    ttl = min(float(requested_ttl or caps.max_ttl_sec), caps.max_ttl_sec)
    if not enabled:
        return "disabled", 0.0
    if not lever_enabled:
        return "lever_disabled", 0.0
    if reflex_active:
        return "reflex_active", 0.0
    started = [r for r in records if r.started_at is not None]
    if any(r.state == "active" for r in started):
        return "already_active", 0.0
    ends = [r.ended_at for r in started if r.ended_at is not None]
    if ends and (now - max(ends)).total_seconds() < caps.min_gap_sec:
        return "min_gap", 0.0
    day_start = now - timedelta(seconds=DAY_SEC)
    used = sum(r.used_sec(now) for r in started if r.started_at >= day_start)
    remaining = caps.max_sec_per_day - used
    if remaining < min(ttl, MIN_USEFUL_TTL_SEC):
        return "daily_cap", 0.0
    return None, min(ttl, remaining)


class OrionShedController:
    """Lives inside the pool runtime; every method is called under the runtime lock."""

    def __init__(self, *, board: ShedBoard, ledger: OrionShedLedger, caps: OrionShedCaps, enabled: bool,
                 lever_enabled: Callable[[], bool], now: Callable[[], datetime]):
        self.board, self.ledger, self.caps, self.enabled = board, ledger, caps, enabled
        self.lever_enabled, self.now = lever_enabled, now
        self.ledger_ok = False
        self.active: OrionShedRecord | None = None
        self._recent: dict[str, OrionShedRecord] = {}
        self._last_persist: datetime | None = None
        self._last_tick: datetime | None = None

    # --- lifecycle ---------------------------------------------------------------------
    async def boot(self) -> list[str]:
        """Load the last day; settle or re-install what a previous process left active.
        Also the retry path: a set that finds the ledger down re-runs this before refusing, so a
        ledger that was unavailable at pool boot does not refuse every set for the process's life."""
        did: list[str] = []
        try:
            self.ledger_ok = await self.ledger.ensure()
            rows = await self.ledger.recent(self.now() - timedelta(seconds=DAY_SEC + self.caps.max_ttl_sec))
        except Exception as exc:  # noqa: BLE001 -- fail closed: no ledger, no sheds
            logger.warning("gpu_pool_orion_shed_ledger_unavailable err=%s", str(exc)[:300])
            self.ledger_ok = False
            return ["ledger_unavailable"]
        if not self.ledger_ok:
            return ["ledger_unavailable"]
        for row in rows:
            rec = OrionShedRecord.from_row(row)
            self._recent[rec.shed_id] = rec
            if rec.state != "active":
                continue
            if not self.enabled:
                await self._end(rec, "cancelled", "kill_switch_at_boot")
                did.append(f"cancelled:{rec.shed_id}")
            elif rec.valid_until is None or rec.valid_until <= self.now():
                await self._end(rec, "expired", "ttl_ended_while_down", at=rec.valid_until)
                did.append(f"expired:{rec.shed_id}")
            else:
                self.active = rec
                self._install(rec)
                did.append(f"restored:{rec.shed_id}")
        return did

    def _install(self, rec: OrionShedRecord) -> None:
        self.board.set(ShedSignal(ORION_SELF_SHED_REASON, rec.shed_id, rec.started_at or self.now(),
                                  rec.valid_until or self.now(),
                                  {"dispatch_id": rec.dispatch_id, "ttl_sec": rec.ttl_sec}))

    async def _persist(self, rec: OrionShedRecord) -> bool:
        try:
            await self.ledger.upsert(rec.to_row())
            self._last_persist = self.now()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("gpu_pool_orion_shed_persist_failed shed_id=%s err=%s", rec.shed_id, str(exc)[:300])
            self.ledger_ok = False
            return False

    async def _end(self, rec: OrionShedRecord, state: str, why: str, *, at: datetime | None = None) -> None:
        rec.state = state
        rec.ended_at = at or self.now()
        rec.detail = {**rec.detail, "ended_because": why}
        self.board.clear(ORION_SELF_SHED_REASON, rec.shed_id)
        if self.active is not None and self.active.shed_id == rec.shed_id:
            self.active = None
        self._recent[rec.shed_id] = rec
        await self._persist(rec)
        logger.warning("gpu_pool_orion_shed_%s shed_id=%s dispatch_id=%s why=%s withheld=%d delayed_sec=%.0f",
                       state, rec.shed_id, rec.dispatch_id, why, rec.grants_withheld, rec.delayed_grant_sec)

    # --- RPC ---------------------------------------------------------------------------
    async def handle(self, req: GpuPoolShedReasonRequestV1, *, background_live: int,
                     reflex_active: bool) -> GpuPoolShedResultV1:
        if req.action == "status":
            rec = await self._find(req)
            if rec is None:
                return GpuPoolShedResultV1(ok=False, dispatch_id=req.dispatch_id, shed_id=req.shed_id,
                                           state="refused", refusal="not_found")
            return rec.to_result()
        if req.action == "clear":
            rec = self.active if self.active and self.active.dispatch_id == req.dispatch_id else None
            if rec is None:
                found = await self._find(req)
                if found is None:
                    return GpuPoolShedResultV1(ok=False, dispatch_id=req.dispatch_id, state="refused",
                                               refusal="not_found")
                return found.to_result()
            await self._end(rec, "cancelled", f"clear_by:{req.actor}")
            return rec.to_result()
        # set
        now = self.now()
        rec = OrionShedRecord(shed_id=f"oshed_{uuid4().hex[:20]}", dispatch_id=req.dispatch_id, state="refused",
                              requested_at=now, correlation=dict(req.correlation),
                              background_live_at_start=background_live)
        if not self.ledger_ok:
            await self.boot()   # retry: ensure the table and reload the 24 h history the caps need
        if not self.ledger_ok:
            rec.refusal = "ledger_unavailable"
            return rec.to_result(ok=False)
        # Idempotent per dispatch_id: a replayed request returns the original record.
        try:
            existing = await self.ledger.by_dispatch(req.dispatch_id)
        except Exception:  # noqa: BLE001
            existing = None
            self.ledger_ok = False
            rec.refusal = "ledger_unavailable"
            return rec.to_result(ok=False)
        if existing is not None:
            return OrionShedRecord.from_row(existing).to_result()
        refusal, ttl = refusal_for_set(now=now, records=self._recent.values(), caps=self.caps,
                                       requested_ttl=req.ttl_sec, enabled=self.enabled,
                                       lever_enabled=self.lever_enabled(), reflex_active=reflex_active)
        if refusal is not None:
            rec.refusal = refusal
            self._recent[rec.shed_id] = rec
            await self._persist(rec)
            logger.info("gpu_pool_orion_shed_refused dispatch_id=%s refusal=%s", req.dispatch_id, refusal)
            return rec.to_result(ok=False)
        rec.state, rec.ttl_sec, rec.started_at = "active", ttl, now
        rec.valid_until = now + timedelta(seconds=ttl)
        if not await self._persist(rec):
            rec.state, rec.refusal, rec.started_at, rec.valid_until = "refused", "ledger_unavailable", None, None
            return rec.to_result(ok=False)
        self._recent[rec.shed_id] = rec
        self.active = rec
        self._install(rec)
        logger.warning("gpu_pool_orion_shed_started shed_id=%s dispatch_id=%s ttl_sec=%.0f background_live=%d",
                       rec.shed_id, rec.dispatch_id, ttl, background_live)
        return rec.to_result()

    async def _find(self, req: GpuPoolShedReasonRequestV1) -> OrionShedRecord | None:
        for rec in self._recent.values():
            if (req.shed_id and rec.shed_id == req.shed_id) or rec.dispatch_id == req.dispatch_id:
                return rec
        try:
            row = (await self.ledger.by_id(req.shed_id)) if req.shed_id else (await self.ledger.by_dispatch(req.dispatch_id))
        except Exception:  # noqa: BLE001
            return None
        return OrionShedRecord.from_row(row) if row else None

    # --- reflex ------------------------------------------------------------------------
    async def preempt_by_reflex(self, source: str) -> bool:
        """The hardware-watch reflex started shedding (``source``: ``<reason>:<source_id>``, e.g.
        ``cabinet_hot:hardware-watch:cabinet`` or v1 ``cooling_incident:<incident_id>``): the reflex
        now covers background, so the learned shed ends."""
        rec = self.active
        if rec is None:
            return False
        rec.detail = {**rec.detail, "preempted_by_reflex": source}
        await self._end(rec, "preempted_by_reflex", f"reflex:{source}"[:200])
        return True

    # --- tick --------------------------------------------------------------------------
    async def expire_due(self) -> None:
        """End a shed whose TTL has passed BEFORE the scheduler reads the board, so a normal TTL end
        is recorded as ``expired`` rather than as a signal that lapsed without a refresh."""
        rec = self.active
        if rec is not None and rec.valid_until is not None and self.now() >= rec.valid_until:
            await self._end(rec, "expired", "ttl", at=rec.valid_until)

    async def on_tick(self, *, live_rows: list[dict[str, Any]], withheld_lease_ids: set[str]) -> None:
        """TTL expiry + manipulation-check bookkeeping. ``withheld_lease_ids``: leases the scheduler
        reported Shed with reason ``shed:orion_self_shed`` this tick."""
        now = self.now()
        dt = 0.0 if self._last_tick is None else max(0.0, (now - self._last_tick).total_seconds())
        self._last_tick = now
        horizon = now - timedelta(seconds=DAY_SEC + self.caps.max_ttl_sec)
        for sid in [k for k, r in self._recent.items() if r.state != "active" and r.requested_at < horizon]:
            self._recent.pop(sid, None)   # bounded: only the window the caps read
        rec = self.active
        if rec is None:
            return
        if rec.valid_until is not None and now >= rec.valid_until:
            await self._end(rec, "expired", "ttl", at=rec.valid_until)
            return
        if not self.enabled:
            await self._end(rec, "cancelled", "kill_switch")
            return
        changed = False
        granted_bg = sum(1 for r in live_rows if r.get("priority") == "background"
                         and r.get("status") in ("granted", "recalling"))
        if rec.drained_at is None and granted_bg == 0:
            rec.drained_at, changed = now, True
        if withheld_lease_ids:
            before = len(rec.withheld_ids)
            rec.withheld_ids |= set(withheld_lease_ids)
            rec.grants_withheld = len(rec.withheld_ids)
            rec.delayed_grant_sec += dt * len(withheld_lease_ids)
            changed = changed or len(rec.withheld_ids) != before
        if changed or self._last_persist is None or (now - self._last_persist).total_seconds() >= PERSIST_EVERY_SEC:
            await self._persist(rec)

    def health(self) -> dict[str, Any]:
        now = self.now()
        day = now - timedelta(seconds=DAY_SEC)
        used = sum(r.used_sec(now) for r in self._recent.values() if r.started_at and r.started_at >= day)
        return {
            "enabled": self.enabled, "ledger_ok": self.ledger_ok,
            "caps": {"max_ttl_sec": self.caps.max_ttl_sec, "max_sec_per_day": self.caps.max_sec_per_day,
                     "min_gap_sec": self.caps.min_gap_sec},
            "used_sec_24h": round(used, 1),
            "active": self.active.to_result().model_dump(mode="json") if self.active else None,
        }


__all__ = [
    "MemoryOrionShedLedger", "OrionShedCaps", "OrionShedController", "OrionShedLedger", "OrionShedRecord",
    "SHED_DECISION_REASON", "refusal_for_set", "ORION_SHED_TERMINAL_STATES",
]
