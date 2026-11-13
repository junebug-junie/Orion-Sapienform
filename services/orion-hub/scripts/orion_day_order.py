"""Orion's Day: Prime's order for a day letter (living Prime design #3224, Part 3).

Hub never decides when a letter is written. Prime (orion-mind) does: when yesterday is closed and
has no letter, Prime opens a ``make`` act (tier 0, no bid) and orders the run here, once per attempt
(``orion-day-<date>-<n>``; a later attempt follows a run that wrote nothing).

``GET  /api/orion-day/letters/{letter_date}`` -> OrionDayLetterStatusV1 (does the day have a letter)
``POST /api/orion-day/letter-run``            OrionDayLetterOrderV1 -> OrionDayLetterOrderResultV1

The order always answers HTTP 200 with a status (422 only for an invalid body):

* ``already_written`` -- an ``orion_day_letter`` row exists (``run_id`` = the row's run).
* ``exists``          -- the durable registry already has ``order.run_id`` (or answered 409).
* ``empty_day``       -- the day holds nothing to write about; no run, no letter.
* ``submitted``       -- Hub gathered the day, built the brief and durable-runs accepted the run.
* ``refused``         -- durable-runs refused the request with a 4xx (``detail`` = its answer), or
                         HUB_ORION_DAY_ENABLED is off.
* ``unavailable``     -- no Postgres pool, the store failed, or durable-runs could not be reached or
                         answered 5xx (transient: Prime sends the order again).

Orders for one letter date are serialized with an asyncio lock, so two orders for the same day
cannot both gather and submit. This module is the only place Hub submits an ``orion_day.letter``
run, and it does so only inside :meth:`OrionDayOrderDesk.order`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

import httpx
from fastapi import APIRouter, HTTPException

from orion.orion_day.brief import OrionDayEmptyError, build_orion_day_brief, build_orion_day_request
from orion.orion_day.store import fetch_letter
from orion.schemas.orion_day import (
    ORION_DAY_ORDER_PATH,
    ORION_DAY_STATUS_PATH,
    MissedLetterV1,
    OrionDayLetterOrderResultV1,
    OrionDayLetterOrderV1,
    OrionDayLetterStatusV1,
)
from orion.world_pulse_read.findings import prior_lookup_from_reader

logger = logging.getLogger("orion-hub.orion_day_order")

# The brief's own cap (OrionDayRunBriefV1.missed_letters max_length).
MAX_MISSED_LETTERS = 7
DETAIL_MAX_CHARS = 500
AUTO_MISSED_REASON = "no letter was written for this day and Prime has no record of why"
LETTER_EXISTS_SQL = "SELECT EXISTS (SELECT 1 FROM orion_day_letter WHERE letter_date = $1)"
_ORDER_RUN_ID = object()


def brief_request_payload(request: Any) -> dict[str, Any]:
    """The durable request as JSON. ``material.corrections`` (added 2026-10-11) is left out when
    empty: orion-durable-runs validates the brief with extra=forbid, so an older durable-runs
    would reject even ``"corrections": []`` and lose the day's letter. Only a day that really
    has corrections needs the newer durable-runs (deploy it before or with Hub)."""
    payload = request.model_dump(mode="json")
    material = (payload.get("brief") or {}).get("material") or {}
    if not material.get("corrections"):
        material.pop("corrections", None)
    return payload


class DurableUnavailable(RuntimeError):
    pass


class DurableRunsClient:
    """The two calls Hub needs from orion-durable-runs' HTTP API (services/orion-durable-runs/app/main.py).
    Also used by the consult loop (scripts/consult_loop.py)."""

    def __init__(self, base_url: str, *, timeout_sec: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_sec) as client:
                resp = await client.get(f"/runs/{run_id}")
        except httpx.HTTPError as exc:
            raise DurableUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise DurableUnavailable(f"http_{resp.status_code}")
        return resp.json()

    async def submit(self, request_json: dict[str, Any]) -> tuple[int, str]:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_sec) as client:
                resp = await client.post("/runs", json=request_json)
        except httpx.HTTPError as exc:
            raise DurableUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return resp.status_code, resp.text[:DETAIL_MAX_CHARS]


def _clip(text: Any) -> str:
    return str(text or "")[:DETAIL_MAX_CHARS]


def missed_letters_for(order: OrionDayLetterOrderV1, *, previous_day_has_letter: bool) -> list[MissedLetterV1]:
    """The order's own list, plus the day before the letter when it has no row and the order did
    not name it (a day Prime was down for is still named). Newest last; at most 7, newest kept."""
    out = list(order.missed_letters)
    previous = order.letter_date - timedelta(days=1)
    if not previous_day_has_letter and all(m.letter_date != previous for m in out):
        out.append(MissedLetterV1(letter_date=previous, reason=AUTO_MISSED_REASON))
    out.sort(key=lambda m: m.letter_date)
    return out[-MAX_MISSED_LETTERS:]


class OrionDayOrderDesk:
    def __init__(
        self,
        *,
        enabled: bool,
        pool_provider: Callable[[], Any],
        durable: Any,
        temporal_day_enabled: bool = True,
        prior_reader_provider: Callable[[], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._pool_provider = pool_provider
        self.durable = durable
        # Temporal Self patch 4 (TEMPORAL_SELF_DAY_BLOCK_ENABLED): read the closed day's own chronology.
        self.temporal_day_enabled = bool(temporal_day_enabled)
        # Orion's graph, read-only: credits a belief a read moved ("What I read").
        self._prior_lookup = prior_lookup_from_reader(prior_reader_provider) if prior_reader_provider else None
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._locks: dict[date, asyncio.Lock] = {}

    def _lock_for(self, letter_date: date) -> asyncio.Lock:
        lock = self._locks.get(letter_date)
        if lock is None:
            lock = self._locks[letter_date] = asyncio.Lock()
        return lock

    # --- status ------------------------------------------------------------------------------

    async def status(self, letter_date: date) -> OrionDayLetterStatusV1:
        """Raises ``LookupError`` when the store cannot answer (never a false "no letter")."""
        pool = self._pool_provider()
        if pool is None:
            raise LookupError("postgres_pool_unavailable")
        try:
            async with pool.acquire() as conn:
                letter = await fetch_letter(conn, letter_date)
        except Exception as exc:  # noqa: BLE001 -- missing table, DB down
            raise LookupError(f"store_unavailable: {type(exc).__name__}: {exc}") from exc
        if letter is None:
            return OrionDayLetterStatusV1(letter_date=letter_date, letter_exists=False)
        return OrionDayLetterStatusV1(letter_date=letter_date, letter_exists=True, run_id=letter.run_id,
                                      emailed=letter.emailed_at is not None)

    # --- order -------------------------------------------------------------------------------

    async def order(self, order: OrionDayLetterOrderV1) -> OrionDayLetterOrderResultV1:
        if not self.enabled:
            return self._result(order, "refused", detail="HUB_ORION_DAY_ENABLED=false")
        async with self._lock_for(order.letter_date):
            result = await self._order_locked(order)
        logger.info("orion_day_order date=%s run=%s act=%s status=%s detail=%s", order.letter_date,
                    result.run_id, order.prime_act_id, result.status, (result.detail or "")[:200])
        return result

    def _result(self, order: OrionDayLetterOrderV1, status: str, *, run_id: Any = _ORDER_RUN_ID,
                detail: Optional[str] = None) -> OrionDayLetterOrderResultV1:
        return OrionDayLetterOrderResultV1(
            letter_date=order.letter_date, run_id=order.run_id if run_id is _ORDER_RUN_ID else run_id,
            status=status, detail=_clip(detail) if detail else None)

    async def _order_locked(self, order: OrionDayLetterOrderV1) -> OrionDayLetterOrderResultV1:
        day = order.letter_date
        pool = self._pool_provider()
        if pool is None:
            return self._result(order, "unavailable", detail="postgres_pool_unavailable")
        try:
            async with pool.acquire() as conn:
                letter = await fetch_letter(conn, day)
                if letter is not None:
                    return self._result(order, "already_written", run_id=letter.run_id)
                previous_has_letter = bool(await conn.fetchval(LETTER_EXISTS_SQL, day - timedelta(days=1)))
        except Exception as exc:  # noqa: BLE001 -- missing table (migration not applied), DB down
            return self._result(order, "unavailable", detail=f"store: {type(exc).__name__}: {exc}")
        try:
            if await self.durable.get_run(order.run_id) is not None:
                return self._result(order, "exists")
        except DurableUnavailable as exc:
            return self._result(order, "unavailable", detail=f"durable_runs: {exc}")
        missed = missed_letters_for(order, previous_day_has_letter=previous_has_letter)
        try:
            async with pool.acquire() as conn:
                brief = await build_orion_day_brief(
                    conn, day, now=self._clock(), timeout_sec=order.timeout_sec,
                    temporal_day=self.temporal_day_enabled, prior_lookup=self._prior_lookup,
                    prime_act_id=order.prime_act_id, missed_letters=missed,
                )
        except OrionDayEmptyError:
            return self._result(order, "empty_day", run_id=None, detail="the day holds nothing to write about")
        except Exception as exc:  # noqa: BLE001
            return self._result(order, "unavailable", detail=f"gather: {type(exc).__name__}: {exc}")
        request = build_orion_day_request(brief, run_id=order.run_id, correlation_id=order.correlation_id,
                                          deadline_at=order.deadline_at)
        try:
            status, body = await self.durable.submit(brief_request_payload(request))
        except DurableUnavailable as exc:
            return self._result(order, "unavailable", detail=f"durable_runs: {exc}")
        if status == 409:
            return self._result(order, "exists", detail=body or "http_409")
        if status >= 500:
            # A DB blip or admission not ready yet: transient. Prime re-sends an unavailable order.
            return self._result(order, "unavailable", detail=f"http_{status}: {body}")
        if status >= 400:
            return self._result(order, "refused", detail=f"http_{status}: {body}")
        logger.info("orion_day_submitted run=%s corr=%s view_tokens=%s missed=%s", order.run_id,
                    order.correlation_id, brief.llm_view.approx_tokens, [str(m.letter_date) for m in missed])
        return self._result(order, "submitted")


def build_orion_day_order_router(desk_provider: Callable[[], Optional[OrionDayOrderDesk]]) -> APIRouter:
    """``desk_provider`` returns the live desk (None while Hub is still starting: 503)."""
    router = APIRouter()

    def _desk() -> OrionDayOrderDesk:
        desk = desk_provider()
        if desk is None:
            raise HTTPException(status_code=503, detail="orion_day_order_not_ready")
        return desk

    @router.get(ORION_DAY_STATUS_PATH, response_model=OrionDayLetterStatusV1)
    async def orion_day_letter_status(letter_date: date) -> OrionDayLetterStatusV1:
        try:
            return await _desk().status(letter_date)
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)[:DETAIL_MAX_CHARS]) from exc

    @router.post(ORION_DAY_ORDER_PATH, response_model=OrionDayLetterOrderResultV1)
    async def orion_day_letter_order(order: OrionDayLetterOrderV1) -> OrionDayLetterOrderResultV1:
        desk = desk_provider()
        if desk is None:
            return OrionDayLetterOrderResultV1(letter_date=order.letter_date, run_id=order.run_id,
                                               status="unavailable", detail="orion_day_order_not_ready")
        return await desk.order(order)

    return router
