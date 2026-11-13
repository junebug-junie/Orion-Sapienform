"""Intentions Prime takes on: the order and the delivery (choice-as-EFE Step -1, orion/schemas/intention.py).

Hub never decides that Orion takes something on. Prime (orion-mind) does, with its ``take_on`` move, and calls here:

``POST /api/intention/run``     IntentionOrderV1   -> IntentionOrderResultV1
``POST /api/intention/deliver`` IntentionDeliverV1 -> IntentionDeliverResultV1

Order. Hub picks the adapter by the concern's ``event_ref`` prefix (orion/prime/intention.py ``ADAPTERS``), reads the
source rows with it (read-only SQL), lets it build the turn's task, and submits ONE durable run: workflow
``intention.carry``, admission on the agent lane with ``deadline_at`` = the adapter's horizon (the binding family's
window ``latest``). Answers (always HTTP 200, 422 only for an invalid body): ``submitted``, ``exists`` (the run id is
already in the registry), ``refused`` (no adapter, the adapter refused, too close to the horizon, or durable-runs
answered 4xx), ``unavailable`` (no Postgres pool, a read failed, durable-runs down or 5xx: Prime sends it again).

Delivery (route ``chat_thread``). Hub reads the completed run's reply (``GET /runs/{run_id}`` harness_steps) and:

1. tries ONE live send into the originating thread through outreach (``offer_message(session_id=...)``) after the
   body economy admits it (door ``intention``): quiet hours, the shared daily cap, recent chat and strain all apply,
   and a sent message counts on the cap. A send writes the thread's history itself.
2. when the send is held for a TIMING reason (``TIMING_HOLDS``: quiet hours, the cap, recent chat, the economy ...),
   writes the reply into that thread's history anyway (``record_in_thread``): it is there when Juniper opens that
   chat (acceptance S-1.6). When outreach's CONTENT guards refuse it (a PASS, an ungrounded internal-signal claim,
   empty text), nothing is written and the answer is ``failed``: a guard's refusal is never bypassed.

Either way the answer is ``published``, not proof: Prime asks again, and only a ``chat_history_log`` row with this
intention's correlation id in that session (``already_delivered``) closes its act 1. A repeat call after a publish
never sends or writes again (in-process memo; after a restart the row, or ``already_told`` for the ping, stops it).
Other routes (``next_letter``, ``notes_only``) are refused until their delivery ships.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter

from orion.prime.intention import MIN_RUN_ROOM_SEC, adapter_for, carry_prompt
from orion.schemas.intention import (
    INTENTION_CARRY_STEP_KEY,
    INTENTION_CARRY_TAG,
    INTENTION_CARRY_WORKFLOW,
    INTENTION_DELIVER_PATH,
    INTENTION_ORDER_PATH,
    IntentionCarryBriefV1,
    IntentionDeliverResultV1,
    IntentionDeliverV1,
    IntentionOrderResultV1,
    IntentionOrderV1,
    IntentionSourceV1,
)
from scripts.orion_day_order import DurableUnavailable

logger = logging.getLogger("orion-hub.intention_order")

DETAIL_MAX_CHARS = 500
DOOR = "intention"
DELIVERED_SQL = "SELECT 1 FROM chat_history_log WHERE correlation_id = $1 AND session_id = $2 LIMIT 1"
FAILED_RUN = frozenset({"failed", "cancelled", "abandoned"})
# Outreach reasons that only mean "not now": the reply still goes into the thread's history.
TIMING_HOLDS = frozenset({"quiet_hours", "daily_cap", "recent_chat", "turn_in_flight", "already_sending", "cooldown",
                          "disabled", "already_told", "economy_held", "arousal_strained", "rest_drive_cooldown"})


def _clip(text: Any) -> str:
    return str(text or "")[:DETAIL_MAX_CHARS]


class IntentionDesk:
    def __init__(self, *, pool_provider: Callable[[], Any], durable: Any,
                 outreach_provider: Callable[[], Any], clock: Callable[[], datetime] | None = None) -> None:
        self._pool_provider = pool_provider
        self.durable = durable
        self._outreach_provider = outreach_provider
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._locks: dict[str, asyncio.Lock] = {}
        # run_id -> the ``published`` answer already given (no second send or write in this process).
        self._published: dict[str, IntentionDeliverResultV1] = {}

    def _lock_for(self, concern_id: str) -> asyncio.Lock:
        lock = self._locks.get(concern_id)
        if lock is None:
            lock = self._locks[concern_id] = asyncio.Lock()
        return lock

    # --- order -------------------------------------------------------------------------------------------------

    async def order(self, order: IntentionOrderV1) -> IntentionOrderResultV1:
        async with self._lock_for(order.concern_id):
            result = await self._order_locked(order)
        logger.info("intention_order concern=%s run=%s act=%s status=%s detail=%s", order.concern_id, result.run_id,
                    order.prime_act_id, result.status, (result.detail or "")[:200])
        return result

    def _result(self, order: IntentionOrderV1, status: str, *, detail: Optional[str] = None,
                src: Optional[IntentionSourceV1] = None) -> IntentionOrderResultV1:
        return IntentionOrderResultV1(
            concern_id=order.concern_id, run_id=order.run_id if status in ("submitted", "exists") else None,
            status=status, detail=_clip(detail) if detail else None,
            **({"source": src.source, "horizon_at": src.horizon_at, "route": src.route,
                "route_target": src.route_target} if src is not None else {}))

    async def _order_locked(self, order: IntentionOrderV1) -> IntentionOrderResultV1:
        adapter = adapter_for(order.event_ref)
        if adapter is None:
            return self._result(order, "refused", detail=f"no adapter for event_ref {order.event_ref[:80]!r}")
        pool = self._pool_provider()
        if pool is None:
            return self._result(order, "unavailable", detail="postgres_pool_unavailable")
        try:
            async with pool.acquire() as conn:
                row = await adapter.fetch(conn, order.event_ref)
        except Exception as exc:  # noqa: BLE001 -- DB down
            return self._result(order, "unavailable", detail=f"store: {type(exc).__name__}: {exc}")
        src = adapter.adapt(order.event_ref, row, why=order.why)
        if isinstance(src, str):
            return self._result(order, "refused", detail=src)
        try:
            # Before the horizon check: a re-sent order for a run that already exists must follow it (review m3).
            if await self.durable.get_run(order.run_id) is not None:
                return self._result(order, "exists", src=src)
        except DurableUnavailable as exc:
            return self._result(order, "unavailable", detail=f"durable_runs: {exc}")
        if self._clock() >= src.horizon_at - timedelta(seconds=MIN_RUN_ROOM_SEC):
            return self._result(order, "refused", src=src,
                                detail=f"the concern's horizon ({src.horizon_at.isoformat()}) leaves no room for a run")
        request = build_carry_request(order, src)
        try:
            status, body = await self.durable.submit(request.model_dump(mode="json"))
        except DurableUnavailable as exc:
            return self._result(order, "unavailable", detail=f"durable_runs: {exc}")
        if status == 409:
            return self._result(order, "exists", src=src, detail=body or "http_409")
        if status >= 500 or status == 422:
            # 422: an orion-durable-runs older than this contract (deploy-order window): try again, never lose it.
            return self._result(order, "unavailable", detail=f"http_{status}: {body}")
        if status >= 400:
            return self._result(order, "refused", detail=f"http_{status}: {body}")
        logger.info("intention_submitted run=%s corr=%s source=%s horizon=%s route=%s", order.run_id,
                    order.correlation_id, src.source, src.horizon_at.isoformat(), src.route)
        return self._result(order, "submitted", src=src)

    # --- delivery ----------------------------------------------------------------------------------------------

    async def deliver(self, req: IntentionDeliverV1) -> IntentionDeliverResultV1:
        async with self._lock_for(req.concern_id):
            result = await self._deliver_locked(req)
        logger.info("intention_deliver concern=%s run=%s status=%s stored=%s pinged=%s ping_reason=%s detail=%s",
                    req.concern_id, req.run_id, result.status, result.stored, result.pinged, result.ping_reason,
                    (result.detail or "")[:200])
        return result

    def _deliver_result(self, req: IntentionDeliverV1, status: str, **kw: Any) -> IntentionDeliverResultV1:
        if "detail" in kw and kw["detail"]:
            kw["detail"] = _clip(kw["detail"])
        return IntentionDeliverResultV1(concern_id=req.concern_id, status=status, **kw)

    async def _deliver_locked(self, req: IntentionDeliverV1) -> IntentionDeliverResultV1:
        if req.route != "chat_thread" or not req.route_target:
            return self._deliver_result(req, "refused", detail=f"route {req.route} is not delivered by Hub yet")
        try:
            run = await self.durable.get_run(req.run_id)
        except DurableUnavailable as exc:
            return self._deliver_result(req, "unavailable", detail=f"durable_runs: {exc}")
        if run is None:
            return self._deliver_result(req, "failed", detail=f"run {req.run_id} not found")
        status = str(run.get("status") or "")
        if status in FAILED_RUN:
            return self._deliver_result(req, "failed", detail=f"the run ended {status}: {run.get('error') or '-'}")
        if status != "completed":
            return self._deliver_result(req, "not_ready", detail=f"run {status or 'unknown'}")
        step = (run.get("harness_steps") or {}).get(INTENTION_CARRY_STEP_KEY) or {}
        text = str(step.get("text") or "").strip()
        if not text:
            return self._deliver_result(req, "failed", detail="the run completed with no reply text")
        pool = self._pool_provider()
        if pool is None:
            return self._deliver_result(req, "unavailable", detail="postgres_pool_unavailable")
        try:
            async with pool.acquire() as conn:
                if await conn.fetchval(DELIVERED_SQL, req.correlation_id, req.route_target):
                    return self._deliver_result(req, "already_delivered", stored=True, reply_chars=len(text))
        except Exception as exc:  # noqa: BLE001
            return self._deliver_result(req, "unavailable", detail=f"store: {type(exc).__name__}: {exc}")
        if req.run_id in self._published:
            return self._published[req.run_id]
        outreach = self._outreach_provider()
        if outreach is None:
            return self._deliver_result(req, "unavailable", detail="outreach_not_ready")
        meta = {"concern_id": req.concern_id, "run_id": req.run_id, "door_policy": INTENTION_CARRY_TAG}
        pinged, reason = await self._ping(outreach, req, text, meta)
        if pinged:
            return self._remember(req, self._deliver_result(req, "published", pinged=True, ping_reason=reason,
                                                            reply_chars=len(text)))
        if (reason or "") not in TIMING_HOLDS and not (reason or "").startswith("ping_error:"):
            # A content guard refused the reply (PASS, ungrounded signal claim, empty): never written (review M4).
            return self._deliver_result(req, "failed", ping_reason=reason,
                                        detail=f"outreach refused the reply's content: {reason or 'no reason'}")
        try:
            written = await outreach.record_in_thread(text=text, session_id=req.route_target,
                                                      correlation_id=req.correlation_id,
                                                      source_tag=INTENTION_CARRY_TAG, meta=meta)
        except Exception as exc:  # noqa: BLE001
            return self._deliver_result(req, "unavailable", ping_reason=reason,
                                        detail=f"history write: {type(exc).__name__}: {exc}")
        if not written:
            return self._deliver_result(req, "unavailable", ping_reason=reason,
                                        detail="chat history is not published (no bus, or PUBLISH_CHAT_HISTORY_LOG off)")
        return self._remember(req, self._deliver_result(req, "published", pinged=False, ping_reason=reason,
                                                        reply_chars=len(text)))

    def _remember(self, req: IntentionDeliverV1, result: IntentionDeliverResultV1) -> IntentionDeliverResultV1:
        self._published[req.run_id] = result
        while len(self._published) > 500:
            self._published.pop(next(iter(self._published)))
        return result

    async def _ping(self, outreach: Any, req: IntentionDeliverV1, text: str,
                    meta: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """One live send through the normal gates. (sent, the outreach decision's reason)."""
        admit, _economy = await outreach.economy_admit(door=DOOR, key=req.concern_id)
        if not admit:
            return False, "economy_held"
        sent = False
        try:
            res = await outreach.offer_message(
                text=text, correlation_id=req.correlation_id, tag=INTENTION_CARRY_TAG, meta=meta,
                # The pacing gates (cooldown, rest drive) are skipped like reading findings; quiet hours, recent
                # chat, strain and the shared daily cap still hold, and a send counts on the cap.
                skip_rate_gates=True, finding_key=f"intention:{req.concern_id}", session_id=req.route_target)
            sent = bool((res or {}).get("outreach"))
            return sent, str((res or {}).get("reason") or "")[:200] or None
        except Exception as exc:  # noqa: BLE001 -- the history write below still lands the reply
            logger.warning("intention_ping_failed concern=%s err=%s", req.concern_id, exc)
            return False, f"ping_error:{type(exc).__name__}"
        finally:
            await outreach.economy_done(door=DOOR, key=req.concern_id, sent=sent)


def build_carry_request(order: IntentionOrderV1, src: IntentionSourceV1) -> Any:
    from orion.schemas.durable_run import DurableRunRequestV1
    from orion.schemas.resource_admission import ResourceRequirementV1

    return DurableRunRequestV1(
        run_id=order.run_id, workflow=INTENTION_CARRY_WORKFLOW, correlation_id=order.correlation_id,
        brief=IntentionCarryBriefV1(
            intention="intention.carry", concern_id=order.concern_id, source=src.source,
            prime_act_id=order.prime_act_id, prompt=carry_prompt(order.concern_id, src.prompt), route=src.route,
            route_target=src.route_target, quote_ref=src.quote_ref, horizon_at=src.horizon_at,
            retrieval_query=src.retrieval_query, timeout_sec=order.timeout_sec),
        # The concern's horizon is the run's deadline: the binding family's window ``latest`` (hold end = it - 1800 s).
        admission=ResourceRequirementV1(preferred_lane="agent", resource="llm.route.agent", priority="background",
                                        deadline_at=src.horizon_at),
    )


def build_intention_router(desk_provider: Callable[[], Optional[IntentionDesk]]) -> APIRouter:
    """``desk_provider`` returns the live desk (None while Hub is still starting: ``unavailable``)."""
    router = APIRouter()

    @router.post(INTENTION_ORDER_PATH, response_model=IntentionOrderResultV1)
    async def intention_order(order: IntentionOrderV1) -> IntentionOrderResultV1:
        desk = desk_provider()
        if desk is None:
            return IntentionOrderResultV1(concern_id=order.concern_id, status="unavailable",
                                          detail="intention_desk_not_ready")
        return await desk.order(order)

    @router.post(INTENTION_DELIVER_PATH, response_model=IntentionDeliverResultV1)
    async def intention_deliver(req: IntentionDeliverV1) -> IntentionDeliverResultV1:
        desk = desk_provider()
        if desk is None:
            return IntentionDeliverResultV1(concern_id=req.concern_id, status="unavailable",
                                            detail="intention_desk_not_ready")
        return await desk.deliver(req)

    return router


__all__ = ["DELIVERED_SQL", "DOOR", "IntentionDesk", "build_carry_request", "build_intention_router"]
