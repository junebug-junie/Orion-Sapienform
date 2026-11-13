"""The broker: checks Orion's grant, asks Orion when the grant is "ask", runs a fixed reader, caps and
redacts, and audits every call. Runs in orion-harness-governor (``app/claude_tools_listener.py``).

Spec: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md sections 3-4.

Order for one call (each refusal says nothing was shared):

1. ``CLAUDE_TOOLS_ENABLED=false`` -> ``off``. Juniper's switch only takes access away.
2. Unknown tool or bad arguments -> ``invalid``.
3. More than ``max_calls_per_turn`` calls in this Claude turn -> ``limit``.
4. Orion's standing setting for the tool's category (newest grant line; none = ``ask``):
   ``deny`` -> ``standing_deny``; ``allow`` -> ``standing_allow`` (read).
5. ``ask``: a "yes for this conversation" in this Claude session -> ``session_allow`` (read); an
   earlier no in this session -> ``repeat_no`` (the same no, Orion is not asked again); more than
   ``max_asks_per_turn`` asks this turn -> ``limit``; else ask Orion (``asked``). No answer within
   ``consent_deadline_sec`` is a no.

Every call counts toward the per-turn cap, refused ones included (an unknown tool, bad arguments, a
standing deny): the cap bounds how many round trips Claude spends, not how much it saw.

Fail direction: every error path returns "not shared". A Redis failure while looking up a session yes
falls through to asking Orion (never to allowing); a Redis failure while counting refuses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from orion.claude_tools import render
from orion.claude_tools.grants import current_grants, grants_path, load_grant_lines, setting_for
from orion.claude_tools.readers import ReaderUnavailable
from orion.claude_tools.registry import tool_def
from orion.claude_tools.transport import (
    CONSENT_REQUEST_CHANNEL,
    CONSENT_REQUEST_KIND,
    CONSENT_RESULT_KIND,
    CONSENT_RESULT_PREFIX,
    TOOL_AUDIT_CHANNEL,
    TOOL_AUDIT_KIND,
)
from orion.schemas.claude_tools import (
    ORION_REASON_MAX,
    ClaudeToolCallV1,
    ClaudeToolRequestV1,
    ClaudeToolResultV1,
    PrimeConsentAnswerV1,
    PrimeConsentPastAnswerV1,
    PrimeConsentRequestV1,
)

logger = logging.getLogger("orion.claude_tools.broker")

KEY_PREFIX = "orion:claude_tools:"
RECENT_ANSWERS_SHOWN = 5
ANSWER_HISTORY_KEPT = 100
# A consent that cannot vary is not a decision (spec section 3, "Rubber-stamp check").
RUBBER_STAMP_MIN_ASKS = 20
TURN_COUNTER_TTL_SEC = 6 * 3600


class ConsentUnavailable(RuntimeError):
    """The consent call could not be made or its answer could not be read."""


def session_allow_key(room_id: str, claude_session_id: str, category: str) -> str:
    return f"{KEY_PREFIX}session_allow:{room_id}:{claude_session_id}:{category}"


def session_no_key(room_id: str, claude_session_id: str, category: str) -> str:
    return f"{KEY_PREFIX}session_no:{room_id}:{claude_session_id}:{category}"


def turn_count_key(turn_request_id: str, what: str) -> str:
    return f"{KEY_PREFIX}turn:{turn_request_id}:{what}"


def answers_key(category: str) -> str:
    return f"{KEY_PREFIX}answers:{category}"


@dataclass
class BrokerConfig:
    enabled: bool = True
    max_calls_per_turn: int = 6
    max_asks_per_turn: int = 3
    consent_deadline_sec: float = 40.0
    session_ttl_sec: int = 86400
    output_cap: int = render.OUTPUT_CAP
    grants_file: Path = field(default_factory=grants_path)


Reader = Callable[[BaseModel], Awaitable[str]]
AskOrion = Callable[[PrimeConsentRequestV1, float], Awaitable[PrimeConsentAnswerV1]]
PublishAudit = Callable[[ClaudeToolCallV1], Awaitable[None]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clip(text: Optional[str], n: int) -> Optional[str]:
    if text is None:
        return None
    words = " ".join(text.split())
    return words if len(words) <= n else words[: n - 1] + "…"


def history_note(answers: list[PrimeConsentPastAnswerV1]) -> Optional[str]:
    """The rubber-stamp line: 20 or more answers in this category, all the same."""
    if len(answers) < RUBBER_STAMP_MIN_ASKS:
        return None
    kinds = {a.answer for a in answers}
    if len(kinds) != 1:
        return None
    word = kinds.pop()
    n = len(answers)
    return (f"you have said {word} {n} of {n} times in this category; if that is your settled view, a "
            f"standing setting says it honestly")


class ClaudeToolBroker:
    def __init__(
        self,
        *,
        config: BrokerConfig,
        redis: Any,
        ask_orion: AskOrion,
        readers: Mapping[str, Reader],
        publish_audit: PublishAudit,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.config = config
        self.redis = redis
        self.ask_orion = ask_orion
        self.readers = dict(readers)
        self.publish_audit = publish_audit
        self.clock = clock

    # --- audit ---------------------------------------------------------------------------------
    def _audit(self, req: ClaudeToolRequestV1, phase: str, **fields: Any) -> ClaudeToolCallV1:
        b = req.binding
        return ClaudeToolCallV1(
            event_id=f"{req.call_id}:{phase}", phase=phase, call_id=req.call_id,
            turn_request_id=b.turn_request_id, room_id=b.room_id, claude_session_id=b.claude_session_id,
            hub_session_id=b.hub_session_id, trigger=b.trigger, tool=req.tool,
            arguments={k: v for k, v in (req.arguments or {}).items() if k != "reason"},
            reason=req.reason, created_at=self.clock(), **fields,
        )

    async def _emit(self, event: ClaudeToolCallV1) -> None:
        try:
            await self.publish_audit(event)
        except Exception:  # noqa: BLE001 -- the audit never decides a call; a lost row is logged
            logger.warning("claude_tool_audit_publish_failed call=%s phase=%s", event.call_id, event.phase,
                           exc_info=True)

    # --- entry ---------------------------------------------------------------------------------
    async def handle(self, req: ClaudeToolRequestV1) -> ClaudeToolResultV1:
        try:
            result = await self._handle(req)
        except Exception as exc:  # noqa: BLE001 -- never fail open, never leave Claude without an answer
            logger.exception("claude_tool_broker_failed call=%s tool=%s", req.call_id, req.tool)
            result = ClaudeToolResultV1(
                call_id=req.call_id, tool=req.tool, shared=False, decision_path="error",
                text=render.unavailable_text(f"the broker failed ({type(exc).__name__})"),
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        # Every path, refusals and errors included: a reader's stderr or an exception can carry a DSN or token.
        result.text = render.cap_output(render.redact_tool_output(result.text), self.config.output_cap)
        if result.error:
            result.error = render.redact_tool_output(result.error)[:500]
        result.result_chars = len(result.text)
        await self._emit(self._audit(
            req, "result", category=result.category, setting=result.setting,
            decision_path=result.decision_path, shared=result.shared, orion_answer=result.orion_answer,
            orion_scope=result.orion_scope, orion_reason=result.orion_reason,
            consent_latency_ms=result.consent_latency_ms, result_chars=result.result_chars, error=result.error,
        ))
        logger.info(
            "claude_tool_call call=%s turn=%s tool=%s category=%s path=%s shared=%s answer=%s chars=%d",
            req.call_id, req.binding.turn_request_id, req.tool, result.category, result.decision_path,
            result.shared, result.orion_answer or "-", result.result_chars,
        )
        return result

    async def _handle(self, req: ClaudeToolRequestV1) -> ClaudeToolResultV1:
        def refuse(path: str, text: str, **kw: Any) -> ClaudeToolResultV1:
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path=path,
                                      text=text, **kw)

        if not self.config.enabled:
            return refuse("off", render.unavailable_text("Claude's tools are switched off (CLAUDE_TOOLS_ENABLED)"))
        tdef = tool_def(req.tool)
        if tdef is None:
            return refuse("invalid", render.unavailable_text(f"there is no tool named {req.tool[:64]!r}"))
        category = tdef.category
        try:
            args = tdef.arguments.model_validate({**(req.arguments or {}), "reason": req.reason})
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            where = ".".join(str(p) for p in first.get("loc", ())) or "arguments"
            return refuse("invalid", render.unavailable_text(f"bad arguments ({where}: {first.get('msg', 'invalid')})"),
                          category=category)

        try:
            calls = await self._count(req.binding.turn_request_id, "calls")
        except Exception:  # noqa: BLE001
            logger.warning("claude_tool_count_failed call=%s", req.call_id, exc_info=True)
            return refuse("error", render.unavailable_text("the broker could not count this turn's calls"),
                          category=category)
        if calls > self.config.max_calls_per_turn:
            return refuse("limit", render.unavailable_text(
                f"this turn already used its {self.config.max_calls_per_turn} tool calls"), category=category)

        lines, skipped = await asyncio.to_thread(load_grant_lines, self.config.grants_file)
        if skipped:
            logger.warning("claude_grants_unreadable_lines path=%s skipped=%d", self.config.grants_file, skipped)
        setting, line = setting_for(current_grants(lines), category)

        if setting == "deny":
            return refuse("standing_deny", render.deny_text(category, line.at if line else None,
                                                            line.reason if line else None),
                          category=category, setting=setting)
        if setting == "allow":
            return await self._read(req, tdef, args, category=category, setting=setting, path="standing_allow")

        # ask
        b = req.binding
        if await self._get(session_allow_key(b.room_id, b.claude_session_id, category)) is not None:
            return await self._read(req, tdef, args, category=category, setting=setting, path="session_allow")
        earlier_no = await self._get(session_no_key(b.room_id, b.claude_session_id, category))
        if earlier_no is not None:
            reason = _clip(earlier_no, ORION_REASON_MAX)
            return refuse("repeat_no", render.no_text(reason), category=category, setting=setting,
                          orion_answer="no", orion_reason=reason)
        try:
            asks = await self._count(b.turn_request_id, "asks")
        except Exception:  # noqa: BLE001
            logger.warning("claude_tool_count_failed call=%s", req.call_id, exc_info=True)
            return refuse("error", render.unavailable_text("the broker could not count this turn's asks"),
                          category=category, setting=setting)
        if asks > self.config.max_asks_per_turn:
            return refuse("limit", render.not_shared_text(
                f"this turn already asked Orion {self.config.max_asks_per_turn} times"), category=category,
                setting=setting)
        return await self._ask(req, tdef, args, category=category, setting=setting)

    # --- the ask -------------------------------------------------------------------------------
    async def _ask(self, req: ClaudeToolRequestV1, tdef: Any, args: BaseModel, *, category: str,
                   setting: str) -> ClaudeToolResultV1:
        b = req.binding
        past = await self._recent_answers(category)
        creq = PrimeConsentRequestV1(
            call_id=req.call_id, turn_request_id=b.turn_request_id, room_id=b.room_id,
            claude_session_id=b.claude_session_id, tool=req.tool, category=category,
            tool_summary=tdef.summary[:300], reason=req.reason, topic=b.topic, topic_from=b.topic_from,
            setting=setting, recent_answers=past[:RECENT_ANSWERS_SHOWN], history_note=history_note(past),
        )
        await self._emit(self._audit(req, "ask", category=category, setting=setting))
        deadline = float(self.config.consent_deadline_sec)
        started = time.monotonic()
        answer: Optional[PrimeConsentAnswerV1] = None
        failure: Optional[str] = None
        timed_out = False
        try:
            answer = await asyncio.wait_for(self.ask_orion(creq, deadline), timeout=deadline + 1.0)
        except (asyncio.TimeoutError, TimeoutError):
            timed_out = True
        except ConsentUnavailable as exc:
            failure = str(exc) or "Orion's consent call is down"
        except Exception as exc:  # noqa: BLE001
            failure = f"Orion's consent call failed ({type(exc).__name__})"
        latency_ms = int((time.monotonic() - started) * 1000)

        common = dict(category=category, setting=setting, consent_latency_ms=latency_ms)
        if answer is None:
            err = "timeout" if timed_out else failure
            await self._emit(self._audit(req, "answer", error=err, **common))
            text = render.timeout_text(deadline) if timed_out else render.not_shared_text(
                f"{failure}, so Orion could not be asked")
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path="asked",
                                      text=text, error=err, **common)

        reason = _clip(answer.reason, ORION_REASON_MAX) or ""
        await self._emit(self._audit(
            req, "answer", orion_answer=answer.answer, orion_scope=answer.scope, orion_reason=reason,
            consent_path=answer.path, act_id=answer.act_id, error=answer.error, **common,
        ))
        said = dict(orion_answer=answer.answer, orion_scope=answer.scope, orion_reason=reason)
        if answer.path != "model":
            # gate refused, unparsed, no model seat: a no that Orion's model did not write.
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path="asked",
                                      text=render.not_shared_text(reason or answer.path), error=answer.error,
                                      **said, **common)
        await self._remember_answer(category, req.tool, answer, reason)
        if answer.answer == "no":
            await self._set(session_no_key(b.room_id, b.claude_session_id, category), reason)
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path="asked",
                                      text=render.no_text(reason), **said, **common)
        if answer.scope == "conversation":
            await self._set(session_allow_key(b.room_id, b.claude_session_id, category),
                            json.dumps({"reason": reason, "at": self.clock().isoformat(), "call_id": req.call_id}))
        return await self._read(req, tdef, args, path="asked", header=render.yes_header(reason, answer.scope),
                                **said, **common)

    # --- the read ------------------------------------------------------------------------------
    async def _read(self, req: ClaudeToolRequestV1, tdef: Any, args: BaseModel, *, path: str,
                    header: Optional[str] = None, **fields: Any) -> ClaudeToolResultV1:
        reader = self.readers.get(tdef.name)
        try:
            if reader is None:
                raise ReaderUnavailable(f"no reader is wired for {tdef.name}")
            data = await reader(args)
        except ReaderUnavailable as exc:
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path=path,
                                      text=render.unavailable_text(str(exc)), error=str(exc)[:500], **fields)
        except Exception as exc:  # noqa: BLE001
            logger.warning("claude_tool_reader_failed call=%s tool=%s", req.call_id, req.tool, exc_info=True)
            return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=False, decision_path=path,
                                      text=render.unavailable_text(f"the reader failed ({type(exc).__name__})"),
                                      error=f"{type(exc).__name__}: {str(exc)[:300]}", **fields)
        body = render.redact_tool_output(data)
        text = f"{header}\n\n{body}" if header else body
        return ClaudeToolResultV1(call_id=req.call_id, tool=req.tool, shared=True, decision_path=path,
                                  text=render.cap_output(text, self.config.output_cap), **fields)

    # --- redis ---------------------------------------------------------------------------------
    async def _count(self, turn_request_id: str, what: str) -> int:
        key = turn_count_key(turn_request_id, what)
        n = int(await self.redis.incr(key))
        if n == 1:
            await self.redis.expire(key, TURN_COUNTER_TTL_SEC)
        return n

    async def _get(self, key: str) -> Optional[str]:
        try:
            raw = await self.redis.get(key)
        except Exception:  # noqa: BLE001 -- unknown session state: ask Orion again (never allow)
            logger.warning("claude_tool_session_read_failed key=%s", key, exc_info=True)
            return None
        if raw is None:
            return None
        return raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)

    async def _set(self, key: str, value: str) -> None:
        try:
            await self.redis.set(key, value, ex=int(self.config.session_ttl_sec))
        except Exception:  # noqa: BLE001 -- a lost session mark only means Orion is asked again
            logger.warning("claude_tool_session_write_failed key=%s", key, exc_info=True)

    async def _recent_answers(self, category: str) -> list[PrimeConsentPastAnswerV1]:
        try:
            raw = await self.redis.lrange(answers_key(category), 0, ANSWER_HISTORY_KEPT - 1)
        except Exception:  # noqa: BLE001
            return []
        out: list[PrimeConsentPastAnswerV1] = []
        for item in raw or []:
            try:
                out.append(PrimeConsentPastAnswerV1.model_validate_json(item))
            except (ValueError, ValidationError):
                continue
        return out

    async def _remember_answer(self, category: str, tool: str, answer: PrimeConsentAnswerV1, reason: str) -> None:
        past = PrimeConsentPastAnswerV1(at=self.clock(), tool=tool, answer=answer.answer, scope=answer.scope,
                                        reason=reason)
        try:
            await self.redis.lpush(answers_key(category), past.model_dump_json())
            await self.redis.ltrim(answers_key(category), 0, ANSWER_HISTORY_KEPT - 1)
        except Exception:  # noqa: BLE001
            logger.warning("claude_tool_answer_history_failed category=%s", category, exc_info=True)


# --- bus adapters (governor) ----------------------------------------------------------------------

def bus_consent_asker(bus: Any, source: Any) -> AskOrion:
    """The real ``ask_orion``: one bus RPC to orion-mind. A request nobody hears fails at once."""
    from orion.core.bus.async_service import RpcNoListener
    from orion.core.bus.bus_schemas import BaseEnvelope

    async def ask(creq: PrimeConsentRequestV1, deadline_sec: float) -> PrimeConsentAnswerV1:
        correlation_id = uuid4()
        reply = f"{CONSENT_RESULT_PREFIX}{correlation_id}"
        envelope = BaseEnvelope(kind=CONSENT_REQUEST_KIND, correlation_id=correlation_id, reply_to=reply,
                                source=source, payload=creq.model_dump(mode="json"))
        try:
            raw = await bus.rpc_request(CONSENT_REQUEST_CHANNEL, envelope, reply_channel=reply,
                                        timeout_sec=deadline_sec, require_listener=True)
        except RpcNoListener as exc:
            raise ConsentUnavailable("orion-mind is not listening for consent asks") from exc
        decoded = bus.codec.decode(raw.get("data"))
        if (not decoded.ok or str(decoded.envelope.correlation_id) != str(correlation_id)
                or decoded.envelope.kind != CONSENT_RESULT_KIND):
            raise ConsentUnavailable("Orion's consent answer was malformed")
        try:
            answer = PrimeConsentAnswerV1.model_validate(decoded.envelope.payload)
        except ValidationError as exc:
            raise ConsentUnavailable("Orion's consent answer was malformed") from exc
        if answer.call_id != creq.call_id:
            raise ConsentUnavailable("Orion's consent answer was for another call")
        return answer

    return ask


def bus_audit_publisher(bus: Any, source: Any) -> PublishAudit:
    from orion.core.bus.bus_schemas import BaseEnvelope

    async def publish(event: ClaudeToolCallV1) -> None:
        await bus.publish(TOOL_AUDIT_CHANNEL, BaseEnvelope(kind=TOOL_AUDIT_KIND, source=source,
                                                           payload=event.model_dump(mode="json")))

    return publish


__all__ = [
    "BrokerConfig",
    "ClaudeToolBroker",
    "ConsentUnavailable",
    "answers_key",
    "bus_audit_publisher",
    "bus_consent_asker",
    "history_note",
    "session_allow_key",
    "session_no_key",
    "turn_count_key",
]
