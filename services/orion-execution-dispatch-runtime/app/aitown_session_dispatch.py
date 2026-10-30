"""Route ``aitown_session``: the motor allocator admits one AI Town session (Juniper, 2026-10-30).

Same shape as ``retro_leisure_dispatch.py``. The allocator admits or refuses; orion-embodiment's
session controller (services/orion-embodiment/app/town_sessions.py) runs every gate (kill switch,
daily cap, gap, one session at a time) and unpauses the town. Mixed into
``ExecutionDispatchRuntimeWorker`` (app/worker.py), which calls two hooks:

- ``_aitown_session_held_reason(item)``: refused before the allocator with a named reason -- the
  kill switch (``aitown_session_disabled``) or the controller's last refusal still holding
  (``doer_refused:<why>``, for the controller's own ``retry_after_sec``: the next local midnight
  after ``daily_cap``).
- ``_send_aitown_session(bus, frame, candidate, now=)``: one start RPC.

The act's cost is its own measured duration (``_act_duration_or_ceiling_cost``, shared with retro
leisure through ``cost_source: act_duration``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.execution_dispatch.policy import AITOWN_SESSION_EXECUTOR_VERB
from orion.schemas.aitown_session import (
    AITOWN_SESSION_START_REPLY_PREFIX,
    AITOWN_SESSION_START_REQUEST_CHANNEL,
    AITOWN_SESSION_START_REQUEST_KIND,
    AITownSessionStartRequestV1,
    AITownSessionStartResultV1,
    aitown_session_object_ref,
)

logger = logging.getLogger("execution-dispatch-runtime.aitown_session")

AITOWN_SESSION_DISABLED = "aitown_session_disabled"
DOER_REFUSED_PREFIX = "doer_refused"
# Hold after a refusal with no retry_after_sec, or an RPC that got no answer (embodiment down).
FAILURE_HOLD_SEC = 300.0


class AITownSessionDispatchMixin:
    """Needs ``self._settings``, ``self._store`` and ``self._record_dispatch_status`` (the worker)."""

    def _aitown_session_enabled(self) -> bool:
        return bool(getattr(self._settings, "orion_dispatch_aitown_session_enabled", False))

    def _aitown_session_held_reason(self, item) -> str | None:
        if (item.cortex_verb or "") != AITOWN_SESSION_EXECUTOR_VERB:
            return None
        if not self._aitown_session_enabled():
            return AITOWN_SESSION_DISABLED
        hold = getattr(self, "_aitown_session_hold", None)
        if hold is not None and datetime.now(timezone.utc) < hold[0]:
            return f"{DOER_REFUSED_PREFIX}:{hold[1]}"
        return None

    def _hold_aitown_session(self, why: str, retry_after_sec: float | None) -> None:
        wait = float(retry_after_sec or 0.0) or FAILURE_HOLD_SEC
        self._aitown_session_hold = (datetime.now(timezone.utc) + timedelta(seconds=wait), why)

    async def _aitown_session_rpc(self, bus, req: AITownSessionStartRequestV1) -> AITownSessionStartResultV1:
        reply_channel = f"{AITOWN_SESSION_START_REPLY_PREFIX}{uuid4().hex}"
        env = BaseEnvelope(kind=AITOWN_SESSION_START_REQUEST_KIND, source=ServiceRef(name=self._settings.service_name),
                           correlation_id=str(uuid4()), reply_to=reply_channel, payload=req.model_dump(mode="json"))
        raw = await bus.rpc_request(AITOWN_SESSION_START_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                    timeout_sec=float(self._settings.orion_dispatch_aitown_session_rpc_timeout_sec))
        decoded = bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
        if not decoded.ok:
            raise RuntimeError(f"aitown_session_reply_decode_failed:{decoded.error}")
        return AITownSessionStartResultV1.model_validate(decoded.envelope.payload)

    def _settle_aitown_session_episode(self, dispatch_id: str, out: AITownSessionStartResultV1 | None) -> None:
        """A proven-path decision was precommitted before the RPC: name its object (the play act's
        key) or void it (the controller refused). With no answer the object is named anyway: it is
        deterministic, so the act's own close, or its expiry, resolves the episode."""
        try:
            if out is not None and out.state == "refused":
                self._store.void_act_episode(dispatch_id, f"{DOER_REFUSED_PREFIX}:{out.refusal}")
            else:
                self._store.set_act_episode_object(dispatch_id, aitown_session_object_ref(dispatch_id))
        except Exception:
            logger.warning("aitown_session_episode_settle_failed dispatch_id=%s", dispatch_id, exc_info=True)

    async def _send_aitown_session(self, bus, frame, candidate, *, now: datetime):
        """One start RPC. Idempotent: a replayed tick finds its result row and sends nothing; the
        controller also answers a repeated dispatch_id with the session it already started."""
        result_id = f"result:{candidate.dispatch_id}"
        if not self._aitown_session_enabled():
            return candidate.model_copy(update={"dispatch_status": "dispatched", "dispatched_at": now,
                                                "dispatch_error": AITOWN_SESSION_DISABLED})
        existing = self._store.load_dispatch_result_by_dispatch_id(candidate.dispatch_id)
        if existing is not None:
            self._record_dispatch_status(existing["status"], live=False)
            return candidate.model_copy(update={"dispatch_status": "dispatched", "dispatched_at": now,
                                                "result_ref": existing["result_id"]})
        started = perf_counter()
        out, error = None, None
        try:
            out = await self._aitown_session_rpc(bus, AITownSessionStartRequestV1(
                dispatch_id=candidate.dispatch_id,
                correlation={"proposal_id": candidate.source_proposal_id,
                             "decision_id": candidate.source_decision_id, "dispatch_frame_id": frame.frame_id}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("aitown_session_start_failed dispatch_id=%s error=%s", candidate.dispatch_id, error)
        rpc_ms = (perf_counter() - started) * 1000.0
        if candidate.dispatch_id in (getattr(self, "_last_proven", None) or {}):
            self._settle_aitown_session_episode(candidate.dispatch_id, out)
        if out is not None and out.state == "refused":
            self._hold_aitown_session(str(out.refusal), out.retry_after_sec)
        elif error:
            self._hold_aitown_session("start_failed", None)
        settlement = {"state": out.state if out else "rpc_error", "kind": "aitown_session",
                      "session_id": out.session_id if out else None,
                      "object_ref": out.object_ref if out else None,
                      "refusal": out.refusal if out else None,
                      "retry_after_sec": out.retry_after_sec if out else None,
                      "rpc_ms": round(rpc_ms, 1)}
        status = "failed" if error else "succeeded"
        self._store.save_dispatch_result(
            result_id=result_id, dispatch_id=candidate.dispatch_id, frame_id=frame.frame_id, status=status,
            result_json={"settlement": settlement, "evidence_refs": [result_id], "latency_ms": None,
                         **({"error": error} if error else {})},
            raw_len=0, latency_ms=None, dispatch_kind=candidate.dispatch_kind, target_id=candidate.target_id)
        self._record_dispatch_status(status)
        logger.info("aitown_session_sent dispatch_id=%s state=%s session=%s refusal=%s rpc_ms=%.0f",
                    candidate.dispatch_id, settlement["state"], settlement["session_id"], settlement["refusal"],
                    rpc_ms)
        update = {"dispatch_status": "dispatched", "dispatched_at": now, "result_ref": result_id}
        if error:
            update["dispatch_error"] = error
        elif out is not None and out.state == "refused":
            update["dispatch_error"] = f"{DOER_REFUSED_PREFIX}:{out.refusal}"
        return candidate.model_copy(update=update)


__all__ = ["AITOWN_SESSION_DISABLED", "AITownSessionDispatchMixin", "DOER_REFUSED_PREFIX", "FAILURE_HOLD_SEC"]
