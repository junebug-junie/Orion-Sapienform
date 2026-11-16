"""Curiosity reads Orion's now at kickoff and records its cites at the turn's end (integration patch 3).

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A ("Who reads the now",
A3-A4). The organ registry (``config/integration/organs.yaml``) names this module as curiosity's ``reads_now``.

1. **Kickoff** (``read_for_kickoff``). ``GET /temporal-self/now?organ=curiosity&since=<last curiosity run>&
   run_ref=<run_id>`` on orion-durable-runs. ``since`` is when the previous curiosity run (either line) was
   decided (``curiosity_offer_decisions.decided_at``); with no earlier run the endpoint's own default applies.
   The endpoint stores every read as an ``experience_shown`` row and returns its ``shown_id``. The shown_id is
   kept with the run: in this process, and on the run's decision row
   (``curiosity_offer_decisions.constants.now_shown_id``), because the durable turn can run in another
   process attempt, after a Hub restart, hours later.
2. **Turn end** (``record_turn_cites``). The ``[experience:<id>]`` cites in the turn's text, at most 3
   (``orion.curiosity.now_section.extract_cites``), go to ``POST /temporal-self/now/use`` with the run's
   shown_id. The endpoint keeps only ids that were shown (``record_use``); the rest are counted there. No
   cite: nothing is posted (the shown row with no use already says "read the now, cited nothing").

Never fails or delays a kickoff or a turn: every read is bounded by ``NOW_TIMEOUT_SEC`` and every failure is
None plus one log line. Flag: ``HUB_CURIOSITY_NOW_ENABLED`` (default true). Off: no read, no section, no post.
Log lines: ``curiosity_now_read``, ``curiosity_now_unavailable``, ``curiosity_now_cites``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

import httpx

from orion.curiosity.now_section import extract_cites, parse_now
from orion.schemas.temporal_self import TemporalNowV1

logger = logging.getLogger("orion-hub.curiosity_now")

ORGAN = "curiosity"
NOW_ROUTE = "/temporal-self/now"
USE_ROUTE = "/temporal-self/now/use"
# The now read writes its shown row and ranks candidates; the kickoff gathers it with the other
# durable-runs reads, so this is the longest the kickoff waits for it.
NOW_TIMEOUT_SEC = 5.0
CITING_PREFIX = "curiosity_run"
MEMORY_CAP = 500

LAST_RUN_SQL = "SELECT max(decided_at) AS at FROM curiosity_offer_decisions WHERE run_id <> $1"
REMEMBER_SQL = (
    "UPDATE curiosity_offer_decisions SET constants = COALESCE(constants, '{}'::jsonb) || "
    "jsonb_build_object('now_shown_id', $2::text) WHERE run_id = $1"
)
LOAD_SQL = "SELECT constants->>'now_shown_id' AS shown_id FROM curiosity_offer_decisions WHERE run_id = $1"


def citing_ref(run_id: str) -> str:
    return f"{CITING_PREFIX}:{run_id}"


class CuriosityNow:
    def __init__(self, *, enabled: bool, durable_runs_url: str, pool_provider: Callable[[], Any],
                 timeout_sec: float = NOW_TIMEOUT_SEC,
                 client_factory: Optional[Callable[[], Any]] = None) -> None:
        self.enabled = bool(enabled)
        self.root = str(durable_runs_url or "").strip().rstrip("/")
        self._pool_provider = pool_provider
        self.timeout_sec = float(timeout_sec)
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=self.timeout_sec))
        # run_id -> shown_id for turns that run in this process (the decision row is the durable copy).
        self._shown: dict[str, str] = {}

    # --- Postgres (best effort) -------------------------------------------------------------------

    async def _last_run_at(self, run_id: str) -> Optional[datetime]:
        pool = self._pool_provider()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                at = await conn.fetchval(LAST_RUN_SQL, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("curiosity_now_since_unread run=%s err=%s", run_id, exc)
            return None
        return at if isinstance(at, datetime) and at.tzinfo is not None else None

    async def _remember(self, run_id: str, shown_id: str) -> bool:
        if len(self._shown) >= MEMORY_CAP:
            self._shown.pop(next(iter(self._shown)))
        self._shown[run_id] = shown_id
        pool = self._pool_provider()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                status = await conn.execute(REMEMBER_SQL, run_id, shown_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_now_shown_unpersisted run=%s err=%s", run_id, exc)
            return False
        return str(status).endswith(" 1")

    async def _load_shown(self, run_id: str) -> Optional[str]:
        if run_id in self._shown:
            return self._shown[run_id]
        pool = self._pool_provider()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(LOAD_SQL, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_now_shown_unread run=%s err=%s", run_id, exc)
            return None
        return str(row["shown_id"]) if row is not None and row["shown_id"] else None

    # --- the two calls -------------------------------------------------------------------------------

    async def read_for_kickoff(self, run_id: str) -> Optional[TemporalNowV1]:
        """What curiosity is shown for this run, or None (off, unreachable, slow, malformed). The run's
        decision row must already exist for the shown_id to be stored with it."""
        if not (self.enabled and self.root and run_id):
            return None
        since = await self._last_run_at(run_id)
        params = {"organ": ORGAN, "run_ref": run_id}
        if since is not None:
            params["since"] = since.isoformat()
        try:
            async with self._client_factory() as client:
                resp = await client.get(f"{self.root}{NOW_ROUTE}", params=params)
            body = resp.json() if resp.status_code == 200 else None
        except Exception as exc:  # noqa: BLE001
            logger.info("curiosity_now_unavailable run=%s err=%s", run_id, f"{type(exc).__name__}: {exc}"[:200])
            return None
        now = parse_now(body)
        if now is None:
            logger.info("curiosity_now_unavailable run=%s status=%s", run_id, getattr(resp, "status_code", None))
            return None
        persisted = await self._remember(run_id, now.shown_id)
        logger.info("curiosity_now_read run=%s shown_id=%s items=%d candidates=%d since=%s persisted=%s",
                    run_id, now.shown_id, len(now.items), now.candidates, now.since.isoformat(), persisted)
        return now

    async def record_turn_cites(self, run_id: str, text: Optional[str]) -> Optional[dict]:
        """POST the turn's cites (at most 3) against the run's shown row. The endpoint's verdict, or None
        when nothing was posted or the post failed. Never raises."""
        try:
            if not (self.enabled and self.root and run_id):
                return None
            cites = extract_cites(text)
            if not cites:
                logger.info("curiosity_now_cites run=%s cites=0", run_id)
                return None
            shown_id = await self._load_shown(run_id)
            if not shown_id:
                logger.info("curiosity_now_cites run=%s cites=%d posted=false reason=no_shown_id", run_id, len(cites))
                return None
            body = {"shown_id": shown_id, "citing_ref": citing_ref(run_id), "cites": cites}
            async with self._client_factory() as client:
                resp = await client.post(f"{self.root}{USE_ROUTE}", json=body)
            if resp.status_code != 200:
                logger.warning("curiosity_now_cites run=%s cites=%d posted=false status=%s body=%s", run_id,
                               len(cites), resp.status_code, str(getattr(resp, "text", ""))[:200])
                return None
            result = resp.json()
            logger.info("curiosity_now_cites run=%s shown_id=%s cites=%d stored=%d not_shown=%d over_cap=%d",
                        run_id, shown_id, len(cites), len(result.get("uses") or []),
                        len(result.get("rejected_not_shown") or []), len(result.get("rejected_over_cap") or []))
            return result
        except Exception as exc:  # noqa: BLE001 -- a cite is never worth a failed turn
            logger.warning("curiosity_now_cites run=%s posted=false err=%s", run_id,
                           f"{type(exc).__name__}: {exc}"[:200])
            return None


__all__ = ["LAST_RUN_SQL", "LOAD_SQL", "NOW_ROUTE", "REMEMBER_SQL", "USE_ROUTE", "CuriosityNow", "citing_ref"]
