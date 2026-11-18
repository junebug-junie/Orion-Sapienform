"""Orion's consent to a Claude tool call: the orion-mind side of orion:prime:consent:request.

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md section 3. The pure
parts (frame, parse, act) are orion/prime/consent.py; this module does the I/O:

1. the self-monitor act gate on ``act.prime.infer.consent`` (refuse-only; a refusal is a no,
   "Orion's consent faculty is paused");
2. Prime's present (orion/prime/block.py, 600 characters) from Redis;
3. one model call on PRIME_CONSENT_LLM_ROUTE (circe chat lane 0, interactive), thinking off, at most
   PRIME_CONSENT_MAX_TOKENS, seat wait PRIME_CONSENT_SEAT_WAIT_SEC then PRIME_CONSENT_LLM_TIMEOUT_SEC;
4. the act: opened (stamped with the call's start) and closed at once when the model was reached,
   ``artifact_produced`` = the answer parsed. A try that never reached the model opens no act (the same
   rule as a Prime thought, orion/prime/acts.py);
5. the answer on orion:prime:consent:result:<correlation_id>.

Every path answers. Anything but a parsed model answer is a no, and ``path`` names which no.
Each request runs in its own task, so one slow ask never holds another.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

from orion.claude_tools.transport import (
    CONSENT_REQUEST_CHANNEL,
    CONSENT_REQUEST_KIND,
    CONSENT_RESULT_KIND,
    CONSENT_RESULT_PREFIX,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.grammar.acts import act_closed_event, act_opened_event
from orion.grammar.publish import GRAMMAR_EVENT_CHANNEL
from orion.mindloop.backoff import classify_try
from orion.mindloop.gate import build_act_gate
from orion.prime.consent import (
    CONSENT_COMPONENT,
    CONSENT_SYSTEM,
    build_consent_frame,
    consent_act,
    consent_object_ref,
    parse_consent_answer,
)
from orion.schemas.claude_tools import PrimeConsentAnswerV1, PrimeConsentRequestV1

logger = logging.getLogger("orion-mind.prime_consent")

SOURCE_SERVICE = "orion-mind"
GATE_PAUSED_REASON = "Orion's consent faculty is paused"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PrimeConsentResponder:
    def __init__(self, *, settings: Any, bus: Any, llm: Any, gate: Any = None, clock=_utcnow) -> None:
        self.s = settings
        self.bus = bus
        self.llm = llm
        self.gate = gate if gate is not None else build_act_gate(settings)
        self.clock = clock
        self.source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)
        self._task: Optional[asyncio.Task] = None
        self._inflight: set[asyncio.Task] = set()
        self._gate_lock = threading.Lock()

    # --- one ask --------------------------------------------------------------------------------
    async def decide(self, req: PrimeConsentRequestV1) -> PrimeConsentAnswerV1:
        started_mono = time.monotonic()
        started_at = self.clock()

        def answer(path: str, reason: str, *, ans: str = "no", scope: str = "conversation",
                   act_id: Optional[str] = None, error: Optional[str] = None) -> PrimeConsentAnswerV1:
            return PrimeConsentAnswerV1(
                call_id=req.call_id, answer=ans, scope=scope, reason=reason[:200], path=path, act_id=act_id,
                latency_ms=int((time.monotonic() - started_mono) * 1000), error=(error or None) and error[:500],
            )

        object_ref = consent_object_ref(req.call_id)
        decision = await self._gate(object_ref)
        if decision is not None and not decision.admit:
            logger.info("prime_consent_gate_refused call=%s faculty=%s", req.call_id, decision.faculty)
            return answer("gate_refused", GATE_PAUSED_REASON)

        block_text: Optional[str] = None
        try:
            from orion.prime.block import fetch_prime_block

            block_text = (await fetch_prime_block(self.bus.redis)).text
        except Exception:  # noqa: BLE001 -- the frame goes without Prime's present, and says nothing false
            logger.warning("prime_consent_block_failed call=%s", req.call_id, exc_info=True)
        frame = build_consent_frame(req, block_text)

        seat = float(self.s.PRIME_CONSENT_SEAT_WAIT_SEC)
        gen = float(self.s.PRIME_CONSENT_LLM_TIMEOUT_SEC)
        try:
            parsed, err, meta = await asyncio.to_thread(
                self.llm.request_json, system_prompt=CONSENT_SYSTEM, user_prompt=frame,
                route=self.s.PRIME_CONSENT_LLM_ROUTE, max_tokens=int(self.s.PRIME_CONSENT_MAX_TOKENS),
                temperature=0.2, thinking=False, timeout_sec=gen, pool_wait_sec=seat if seat > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            parsed, err, meta = None, f"{type(exc).__name__}: {str(exc)[:160]}", {}
        gw = (meta or {}).get("gateway_error") if isinstance(meta, dict) else None
        kind = "model" if parsed is not None and not err else classify_try(err, gw if isinstance(gw, dict) else None)
        if kind != "model":
            logger.info("prime_consent_no_model call=%s kind=%s err=%s", req.call_id, kind, (err or "-")[:160])
            return answer("no_model", "the model could not be reached, so Orion was not asked", error=err)

        act = consent_act(req.call_id, horizon_sec=seat + gen)
        await self._publish_act(act_opened_event(act, source_service=SOURCE_SERVICE, source_component=CONSENT_COMPONENT,
                                                 correlation_id=req.turn_request_id, at=started_at))
        got, why = parse_consent_answer(parsed) if not err else (None, err)
        if got is None:
            await self._publish_act(act_closed_event(act, produced=False, source_service=SOURCE_SERVICE,
                                                     source_component=CONSENT_COMPONENT,
                                                     correlation_id=req.turn_request_id,
                                                     note=f"failed: unparsed: {str(why)[:160]}"))
            logger.info("prime_consent_unparsed call=%s why=%s", req.call_id, str(why)[:160])
            return answer("unparsed", "Orion's answer did not parse", act_id=act.act_id, error=str(why))
        await self._publish_act(act_closed_event(act, produced=True, source_service=SOURCE_SERVICE,
                                                 source_component=CONSENT_COMPONENT,
                                                 correlation_id=req.turn_request_id,
                                                 note=f"{got['answer']} ({got['scope']}) {req.tool}"))
        logger.info("prime_consent_answered call=%s tool=%s category=%s answer=%s scope=%s ms=%d",
                    req.call_id, req.tool, req.category, got["answer"], got["scope"],
                    int((time.monotonic() - started_mono) * 1000))
        return answer("model", got["reason"], ans=got["answer"], scope=got["scope"], act_id=act.act_id)

    async def _gate(self, object_ref: str):
        """The gate's decision for this ask, read fresh every time (``begin`` re-reads the published view
        and clears last ask's offered probes), under a lock so parallel asks do not share one view. A probe
        is claimed here, before the model call: a probe another ask already holds is a refusal."""
        if not getattr(self.gate, "enabled", False):
            return None
        now = self.clock()

        def run():
            with self._gate_lock:
                self.gate.begin(now)
                decision = self.gate.decide("prime", "infer", object_ref, now=now)
                if decision.admit and decision.probe and not self.gate.claim_probe(decision.faculty, now=now):
                    return replace(decision, admit=False, probe=False)
                return decision

        try:
            return await asyncio.to_thread(run)
        except Exception:  # noqa: BLE001 -- a gate fault changes nothing (refuse-only gate)
            logger.warning("prime_consent_gate_failed", exc_info=True)
            return None

    async def _publish_act(self, event: Any) -> None:
        try:
            await self.bus.publish(GRAMMAR_EVENT_CHANNEL, BaseEnvelope(
                kind="grammar.event.v1", source=self.source, payload=event.model_dump(mode="json")))
        except Exception:  # noqa: BLE001
            logger.warning("prime_consent_act_publish_failed act=%s", getattr(event.act, "act_id", "-"), exc_info=True)

    # --- bus ------------------------------------------------------------------------------------
    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{CONSENT_RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != CONSENT_REQUEST_KIND:
            return
        try:
            req = PrimeConsentRequestV1.model_validate(envelope.payload)
        except ValueError as exc:
            logger.warning("prime_consent_bad_request corr=%s err=%s", envelope.correlation_id, str(exc)[:200])
            call_id = str((envelope.payload or {}).get("call_id") or "unknown-call")[:64]
            result = PrimeConsentAnswerV1(call_id=call_id if len(call_id) >= 1 else "unknown", answer="no",
                                          reason="the ask was malformed", path="error", error=str(exc)[:500])
        else:
            try:
                result = await self.decide(req)
            except Exception as exc:  # noqa: BLE001 -- every ask gets an answer, and it is a no
                logger.exception("prime_consent_failed call=%s", req.call_id)
                result = PrimeConsentAnswerV1(call_id=req.call_id, answer="no", reason="Orion's consent call failed",
                                              path="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        await self.bus.publish(reply, BaseEnvelope(kind=CONSENT_RESULT_KIND, correlation_id=envelope.correlation_id,
                                                   source=self.source, payload=result.model_dump(mode="json")))

    def _spawn(self, envelope: BaseEnvelope) -> None:
        task = asyncio.create_task(self.handle(envelope))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _run(self) -> None:
        delay = 1.0
        while True:
            try:
                await self.bus.connect()
                async with self.bus.subscribe(CONSENT_REQUEST_CHANNEL) as pubsub:
                    logger.info("prime_consent_listening channel=%s route=%s", CONSENT_REQUEST_CHANNEL,
                                self.s.PRIME_CONSENT_LLM_ROUTE)
                    delay = 1.0
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                self._spawn(decoded.envelope)
                        except Exception:  # noqa: BLE001
                            logger.warning("prime_consent_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("prime_consent_reconnecting in %.0fs", delay, exc_info=True)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="orion-prime-consent")

    async def stop(self) -> None:
        for task in [self._task, *self._inflight]:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        self._task = None
        self._inflight.clear()


__all__ = ["GATE_PAUSED_REASON", "PrimeConsentResponder"]
