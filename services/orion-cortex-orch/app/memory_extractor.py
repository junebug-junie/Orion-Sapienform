from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from jinja2 import Template
from pydantic import ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef
from orion.core.contracts.memory_cards import (
    CardAnnotationV1,
    MemoryCardCreateV1,
    derive_visibility_scope,
)
from orion.core.storage import memory_cards as mc_dal
from orion.core.storage.memory_extraction import CandidateCard, extract_candidates, fingerprint_from_candidate
from orion.schemas.chat_history import ChatHistoryTurnV1

from .settings import get_settings

logger = logging.getLogger("orion.cortex.memory_extractor")

_memory_pool: Optional[Any] = None
_memory_pool_failed: bool = False

# Lazy singleton bus for the write-time card-annotation LLM RPC call (Item 1a
# of docs/superpowers/specs/2026-07-21-memory-cards-self-authored-substrate-spec.md).
# Deliberately separate from cortex-orch's main request-handling bus (svc.bus
# in main.py) so this module has no import-time dependency on main.py --
# handle_memory_history_turn is wired as a Hunter handler that only ever
# receives `env`, not a bus reference.
_annotation_bus: Optional[OrionBusAsync] = None

_ANNOTATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "memory_card_annotation_prompt.j2"
)


async def _get_memory_pool() -> Optional[Any]:
    """Lazy asyncpg pool for memory_cards DAL (same RECALL_PG_DSN as Hub / recall)."""
    global _memory_pool, _memory_pool_failed
    if _memory_pool is not None:
        return _memory_pool
    if _memory_pool_failed:
        return None
    dsn = (get_settings().recall_pg_dsn or "").strip()
    if not dsn:
        return None
    try:
        import asyncpg  # type: ignore[import-untyped]

        _memory_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
        logger.info("memory_extractor_pg_pool_ready")
        return _memory_pool
    except Exception as exc:
        logger.warning("memory_extractor_pool_failed error=%s", exc)
        _memory_pool_failed = True
        return None


def _get_annotation_bus() -> Optional[OrionBusAsync]:
    """Lazy singleton bus for the annotation RPC call. Returns None (never
    raises) if construction fails for any reason -- the caller treats a None
    bus the same as any other annotation failure and falls back to regex
    extraction."""
    global _annotation_bus
    if _annotation_bus is not None:
        return _annotation_bus
    try:
        settings = get_settings()
        _annotation_bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    except Exception as exc:
        logger.warning("memory_extractor_annotation_bus_init_failed error=%s", exc)
        return None
    return _annotation_bus


def _coerce_turn(env: BaseEnvelope) -> Optional[ChatHistoryTurnV1]:
    payload = env.payload
    if not isinstance(payload, dict):
        return None
    try:
        return ChatHistoryTurnV1.model_validate(payload)
    except Exception:
        logger.debug("memory_extractor_turn_parse_failed kind=%s", getattr(env, "kind", ""))
        return None


def _build_annotation_prompt(turn_text: str) -> str:
    template = Template(_ANNOTATION_PROMPT_PATH.read_text(encoding="utf-8"))
    return template.render(turn_text=turn_text)


async def _rpc_annotation_llm(
    bus: OrionBusAsync,
    *,
    prompt: str,
    turn_text: str,
    correlation_id: str,
    source: ServiceRef,
    timeout_sec: float,
) -> CardAnnotationV1:
    """Mirrors decision_router.py's llm_router()/_rpc_llm() RPC-to-llm-gateway
    plumbing exactly: ChatRequestPayload with a JSON-object response format,
    dispatched over the same auto_router_llm_* channels, decoded via the bus
    codec, then validated into a typed model."""
    settings = get_settings()
    await bus.connect()
    payload = ChatRequestPayload(
        route="chat",
        messages=[{"role": "user", "content": prompt}],
        raw_user_text=turn_text,
        options={
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": False,
            "response_format": {"type": "json_object"},
        },
    )
    reply_channel = f"{settings.auto_router_llm_reply_prefix}:{uuid4()}"
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=source,
        correlation_id=correlation_id,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    message = await bus.rpc_request(
        settings.auto_router_llm_request_channel,
        env,
        reply_channel=reply_channel,
        timeout_sec=timeout_sec,
    )
    decoded = bus.codec.decode(message.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error or "llm_decode_failed")
    response_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    text = str(response_payload.get("content") or response_payload.get("text") or "").strip()
    if not text:
        raw = response_payload.get("raw") or {}
        text = str(raw.get("content") or raw.get("text") or "").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("annotation_response_not_object")
    # Hard boundary: CardAnnotationV1 has no sensitivity/visibility_scope
    # field at all, so these would already fail extra="forbid" validation --
    # stripped defensively here too so a stray top-level key from a future
    # prompt regression can never even reach that check.
    data.pop("sensitivity", None)
    data.pop("visibility_scope", None)
    return CardAnnotationV1.model_validate(data)


async def _annotate_via_llm(
    turn_text: str,
    *,
    correlation_id: str,
    source: ServiceRef,
    timeout_sec: float,
) -> Optional[CardAnnotationV1]:
    """Best-effort write-time annotation call. Returns None on ANY failure
    (bus unavailable, timeout, decode error, bad/missing JSON, schema
    mismatch) so the caller always has a safe path to the regex fallback --
    this must never block or suppress card creation on an LLM/bus outage."""
    bus = _get_annotation_bus()
    if bus is None:
        return None
    try:
        prompt = _build_annotation_prompt(turn_text)
        return await asyncio.wait_for(
            _rpc_annotation_llm(
                bus,
                prompt=prompt,
                turn_text=turn_text,
                correlation_id=correlation_id,
                source=source,
                timeout_sec=timeout_sec,
            ),
            timeout=timeout_sec,
        )
    except (asyncio.TimeoutError, ValidationError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("memory_extractor_annotation_llm_failed error=%s", exc)
        return None
    except Exception as exc:  # pragma: no cover -- defensive catch-all, never propagate
        logger.warning("memory_extractor_annotation_llm_unexpected_error error=%s", exc)
        return None


def _card_from_annotation(annotation: CardAnnotationV1) -> tuple[MemoryCardCreateV1, str]:
    """Build the create payload from an LLM annotation. `sensitivity` is
    hardcoded here -- never read from `annotation`, which has no such field
    at all -- and `visibility_scope` is derived from it deterministically.
    See CardAnnotationV1's and derive_visibility_scope()'s docstrings."""
    sensitivity = "private"
    types = list(annotation.types) or ["fact"]
    fp_candidate = CandidateCard(
        summary=annotation.summary,
        types=tuple(types),
        anchor_class=annotation.anchor_class,
    )
    fp = fingerprint_from_candidate(fp_candidate)
    subschema = {"auto_extractor_fingerprint": fp, "auto_extractor_mode": "llm_annotation"}
    create = MemoryCardCreateV1(
        types=types,
        anchor_class=annotation.anchor_class,
        status="pending_review",
        confidence=annotation.confidence,
        sensitivity=sensitivity,
        priority=annotation.priority,
        visibility_scope=derive_visibility_scope(sensitivity),
        time_horizon=annotation.time_horizon,
        provenance="auto_extractor",
        project=annotation.project,
        title=annotation.title,
        summary=annotation.summary,
        still_true=list(annotation.still_true) or None,
        anchors=list(annotation.anchors) or None,
        tags=list(annotation.tags) or None,
        subschema=subschema,
    )
    return create, fp


async def handle_memory_history_turn(env: BaseEnvelope) -> None:
    settings = get_settings()
    if not settings.orion_auto_extractor_enabled:
        return
    if settings.orion_auto_extractor_stage2_enabled:
        raise NotImplementedError("ORION_AUTO_EXTRACTOR_STAGE2_ENABLED is v1.5-only")

    turn = _coerce_turn(env)
    if turn is None or not (turn.prompt or "").strip():
        return

    pool = await _get_memory_pool()
    if pool is None:
        return

    correlation_id = str(getattr(env, "correlation_id", None) or uuid4())
    source = ServiceRef(name=str(getattr(settings, "service_name", "") or "cortex-orch"))
    timeout_sec = float(getattr(settings, "orion_auto_extractor_llm_timeout_sec", 6.0) or 6.0)

    annotation = await _annotate_via_llm(
        turn.prompt, correlation_id=correlation_id, source=source, timeout_sec=timeout_sec
    )

    if annotation is not None:
        if not annotation.worth_saving:
            logger.debug("memory_extractor_llm_gate_worth_saving_false corr=%s", correlation_id)
            return
        create, fp = _card_from_annotation(annotation)
        if await mc_dal.card_exists_by_fingerprint(pool, fp):
            logger.debug("memory_extractor_skip_duplicate fp=%s", fp[:16])
            return
        try:
            await mc_dal.insert_card(pool, create, actor="auto_extractor", op="create")
            logger.info(
                "memory_extractor_card_created fp=%s title=%r mode=llm_annotation",
                fp[:16],
                create.title[:80] if create.title else "",
            )
        except Exception as exc:
            logger.warning("auto_extractor_insert_failed fp=%s err=%s", fp[:16], exc)
        return

    # Fallback: LLM annotation unavailable or failed. Today's regex
    # extraction path, unchanged -- never regresses below pre-existing
    # behavior, never blocks card creation entirely on an LLM/bus outage.
    candidates = extract_candidates(turn.prompt, speaker="user")
    for cand in candidates:
        fp = fingerprint_from_candidate(cand)
        if await mc_dal.card_exists_by_fingerprint(pool, fp):
            logger.debug("memory_extractor_skip_duplicate fp=%s", fp[:16])
            continue
        subschema = {"auto_extractor_fingerprint": fp}
        create = MemoryCardCreateV1(
            types=list(cand.types),
            anchor_class=cand.anchor_class,
            title=cand.summary,
            summary=cand.summary,
            provenance="auto_extractor",
            status="pending_review",
            subschema=subschema,
        )
        try:
            await mc_dal.insert_card(pool, create, actor="auto_extractor", op="create")
            logger.info(
                "memory_extractor_card_created fp=%s title=%r mode=regex_fallback",
                fp[:16],
                cand.summary[:80] if cand.summary else "",
            )
        except Exception as exc:
            logger.warning("auto_extractor_insert_failed fp=%s err=%s", fp[:16], exc)
