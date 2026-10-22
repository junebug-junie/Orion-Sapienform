"""Route ``retro_leisure_session``: the motor allocator admits one solo Zelda leisure session.

Orion plays Zelda, section E (docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md). The
allocator admits or refuses; Hub's leisure runner runs every gate and plays. Mixed into
``ExecutionDispatchRuntimeWorker`` (app/worker.py), which calls three hooks:

- ``_retro_leisure_held_reason(item)``: refused before the allocator with a named reason -- the
  kill switch (``retro_leisure_disabled``) or the doer's last refusal still holding
  (``doer_refused:<why>``, for Hub's own ``retry_after_sec``). The worker records it as
  ``allocator:<reason>`` on the dispatch frame like every other refusal.
- ``_act_duration_or_ceiling_cost(ref)`` (any route with ``cost_source: act_duration``): the act's
  own measured open->close seconds (for play, a session's wall time), or the route's
  ``cold_cost_ceiling_sec`` (the runner's session cap) while the act has never closed. This is the
  one exception to "no typed fallback": the ceiling is the doer's enforced maximum, used only when no
  close exists, so a cold act is charged its worst case. A cost that cannot be read is None (refused
  ``no_cost_estimate``), never the ceiling.
- ``_send_retro_leisure(bus, frame, candidate, now=)``: one start RPC. The session's object is minted
  by the dispatch, so there is no control arm and no side-effect-free check first.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.execution_dispatch.policy import RETRO_LEISURE_EXECUTOR_VERB
from orion.schemas.retro_leisure import (
    RETRO_LEISURE_START_REPLY_PREFIX,
    RETRO_LEISURE_START_REQUEST_CHANNEL,
    RETRO_LEISURE_START_REQUEST_KIND,
    RetroLeisureStartRequestV1,
    RetroLeisureStartResultV1,
    retro_leisure_object_ref,
)

logger = logging.getLogger("execution-dispatch-runtime.retro_leisure")

RETRO_LEISURE_DISABLED = "retro_leisure_disabled"
DOER_REFUSED_PREFIX = "doer_refused"
# Hold after a refusal with no retry_after_sec, or an RPC that got no answer (Hub down or old):
# five minutes, the same as the reading pull, so a dead Hub costs one timed-out RPC per hold.
FAILURE_HOLD_SEC = 300.0


class RetroLeisureDispatchMixin:
    """Needs ``self._settings``, ``self._store`` and ``self._record_dispatch_status`` (the worker)."""

    def _retro_leisure_enabled(self) -> bool:
        return bool(getattr(self._settings, "orion_dispatch_retro_leisure_enabled", False))

    def _retro_leisure_held_reason(self, item) -> str | None:
        if (item.cortex_verb or "") != RETRO_LEISURE_EXECUTOR_VERB:
            return None
        if not self._retro_leisure_enabled():
            return RETRO_LEISURE_DISABLED
        hold = getattr(self, "_retro_leisure_hold", None)
        if hold is not None and datetime.now(timezone.utc) < hold[0]:
            return f"{DOER_REFUSED_PREFIX}:{hold[1]}"
        return None

    def _hold_retro_leisure(self, why: str, retry_after_sec: float | None) -> None:
        wait = float(retry_after_sec or 0.0) or FAILURE_HOLD_SEC
        self._retro_leisure_hold = (datetime.now(timezone.utc) + timedelta(seconds=wait), why)

    def _act_duration_or_ceiling_cost(self, ref) -> float | None:
        try:
            measured = self._store.act_duration_p50_sec(
                actor=ref.actor, verb=ref.verb, signal_id=ref.signal_id,
                lookback_hours=int(self._settings.orion_dispatch_act_cost_lookback_hours))
        except Exception:
            logger.warning("act_duration_cost_unreadable act=%s", ref.label(), exc_info=True)
            return None
        if measured is not None:
            return measured
        return getattr(ref, "cold_cost_ceiling_sec", None)

    async def _retro_leisure_rpc(self, bus, req: RetroLeisureStartRequestV1) -> RetroLeisureStartResultV1:
        reply_channel = f"{RETRO_LEISURE_START_REPLY_PREFIX}{uuid4().hex}"
        env = BaseEnvelope(kind=RETRO_LEISURE_START_REQUEST_KIND, source=ServiceRef(name=self._settings.service_name),
                           correlation_id=str(uuid4()), reply_to=reply_channel, payload=req.model_dump(mode="json"))
        raw = await bus.rpc_request(RETRO_LEISURE_START_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                    timeout_sec=float(self._settings.orion_dispatch_retro_leisure_rpc_timeout_sec))
        decoded = bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
        if not decoded.ok:
            raise RuntimeError(f"retro_leisure_reply_decode_failed:{decoded.error}")
        return RetroLeisureStartResultV1.model_validate(decoded.envelope.payload)

    def _settle_retro_leisure_episode(self, dispatch_id: str, out: RetroLeisureStartResultV1 | None) -> None:
        """A proven-path decision was precommitted before the RPC: name its object (the play act's
        key, resolved by the act's own close) or void it (the doer refused). With no answer, Hub may
        still have started the session: the object is named anyway (it is deterministic), so the
        act's own close, or its expiry, resolves the episode."""
        try:
            if out is not None and out.state == "refused":
                self._store.void_act_episode(dispatch_id, f"{DOER_REFUSED_PREFIX}:{out.refusal}")
            else:
                self._store.set_act_episode_object(dispatch_id, retro_leisure_object_ref(dispatch_id))
        except Exception:
            logger.warning("retro_leisure_episode_settle_failed dispatch_id=%s", dispatch_id, exc_info=True)

    async def _send_retro_leisure(self, bus, frame, candidate, *, now: datetime):
        """One start RPC. Idempotent: a replayed tick finds its result row and sends nothing; Hub also
        answers a repeated dispatch_id with the session it already started. The result row carries no
        latency (the RPC is not the work); the allocator's cost is the play act's own duration."""
        result_id = f"result:{candidate.dispatch_id}"
        if not self._retro_leisure_enabled():
            return candidate.model_copy(update={"dispatch_status": "dispatched", "dispatched_at": now,
                                                "dispatch_error": RETRO_LEISURE_DISABLED})
        existing = self._store.load_dispatch_result_by_dispatch_id(candidate.dispatch_id)
        if existing is not None:
            self._record_dispatch_status(existing["status"], live=False)
            return candidate.model_copy(update={"dispatch_status": "dispatched", "dispatched_at": now,
                                                "result_ref": existing["result_id"]})
        started = perf_counter()
        out, error = None, None
        try:
            out = await self._retro_leisure_rpc(bus, RetroLeisureStartRequestV1(
                dispatch_id=candidate.dispatch_id,
                correlation={"proposal_id": candidate.source_proposal_id,
                             "decision_id": candidate.source_decision_id, "dispatch_frame_id": frame.frame_id}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("retro_leisure_start_failed dispatch_id=%s error=%s", candidate.dispatch_id, error)
        rpc_ms = (perf_counter() - started) * 1000.0
        if candidate.dispatch_id in (getattr(self, "_last_proven", None) or {}):
            self._settle_retro_leisure_episode(candidate.dispatch_id, out)
        if out is not None and out.state == "refused":
            self._hold_retro_leisure(str(out.refusal), out.retry_after_sec)
        elif error:
            self._hold_retro_leisure("start_failed", None)
        settlement = {"state": out.state if out else "rpc_error", "kind": "retro_leisure",
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
        logger.info("retro_leisure_sent dispatch_id=%s state=%s session=%s refusal=%s rpc_ms=%.0f",
                    candidate.dispatch_id, settlement["state"], settlement["session_id"], settlement["refusal"],
                    rpc_ms)
        update = {"dispatch_status": "dispatched", "dispatched_at": now, "result_ref": result_id}
        if error:
            update["dispatch_error"] = error
        elif out is not None and out.state == "refused":
            update["dispatch_error"] = f"{DOER_REFUSED_PREFIX}:{out.refusal}"
        return candidate.model_copy(update=update)


__all__ = ["DOER_REFUSED_PREFIX", "FAILURE_HOLD_SEC", "RETRO_LEISURE_DISABLED", "RetroLeisureDispatchMixin"]
