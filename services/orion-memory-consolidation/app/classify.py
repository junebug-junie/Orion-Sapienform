from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.memory.consolidation_classify import build_classify_prompt
from orion.memory.turn_change_classify import (
    build_change_only_prompt,
    build_turn_change_appraisal,
    reconcile_novelty_with_shift,
    should_session_reappraise,
)
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1

from app.boundary import scores_from_llm_result

logger = logging.getLogger(__name__)

_MAX_TURN_FIELD_CHARS = 300


def _clip(text: str, *, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _build_window_transcript(turns: list[dict], *, max_turns: int | None = None) -> str:
    cap = max_turns if max_turns is not None else len(turns)
    selected = turns[-cap:] if len(turns) > cap else turns
    lines = []
    for t in selected:
        sig = t.get("memory_significance_score")
        prefix = f"[sig={sig:.2f}] " if isinstance(sig, (int, float)) else ""
        prompt = _clip(str(t.get("prompt") or ""), limit=_MAX_TURN_FIELD_CHARS)
        response = _clip(str(t.get("response") or ""), limit=_MAX_TURN_FIELD_CHARS)
        lines.append(f"{prefix}User: {prompt}\nOrion: {response}\n")
    return "\n".join(lines)


def _prior_turn_baseline(prior_turns: list[dict]) -> tuple[str, str, str | None]:
    if not prior_turns:
        return "none", "", None
    last = prior_turns[-1]
    prompt = _clip(str(last.get("prompt") or ""), limit=_MAX_TURN_FIELD_CHARS)
    response = _clip(str(last.get("response") or ""), limit=_MAX_TURN_FIELD_CHARS)
    text = f"User: {prompt}\nOrion: {response}\n"
    return "prior_turn", text, str(last.get("correlation_id") or "")


def _session_window_baseline(prior_turns: list[dict], *, n: int) -> tuple[str, str]:
    selected = prior_turns[-n:] if len(prior_turns) > n else prior_turns
    if not selected:
        return "none", ""
    return "session_window", _build_window_transcript(selected, max_turns=n)


_CLASSIFY_ROUTES = frozenset({"metacog", "quick"})


def _resolve_classify_route(settings) -> str:
    route = str(getattr(settings, "TURN_CHANGE_CLASSIFY_ROUTE", "metacog") or "metacog").strip().lower()
    if route not in _CLASSIFY_ROUTES:
        logger.warning("invalid TURN_CHANGE_CLASSIFY_ROUTE=%r; falling back to metacog", route)
        return "metacog"
    return route


def _degraded_patch(
    *,
    baseline_mode: str = "none",
    prior_correlation_id: str | None = None,
    classify_route: str | None = None,
) -> dict:
    appraisal = build_turn_change_appraisal(
        baseline_mode=baseline_mode,
        prior_correlation_id=prior_correlation_id,
        novelty_score=None,
        shift_kind=None,
        shift_scores=None,
        confidence=None,
        status="degraded",
    )
    out = {
        "turn_change_appraisal": appraisal,
        "memory_classify_status": "degraded",
        "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
    }
    if classify_route is not None:
        out["turn_change_classify_route"] = classify_route
    return out


async def _llm_classify(
    bus: OrionBusAsync, *, prompt: str, settings, llm_route: str | None = None
) -> dict:
    route = llm_route or _resolve_classify_route(settings)
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=route,
        options={
            "return_logprobs": True,
            "logprobs_top_k": 8,
            "logprob_summary_only": False,
            "max_tokens": 24,
            "llm_route": route,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.NODE_NAME,
        ),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        settings.CHANNEL_LLM_INTAKE,
        env,
        reply_channel=reply_channel,
        timeout_sec=float(settings.MEMORY_CLASSIFY_TIMEOUT_SEC),
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    result_payload = decoded.envelope.payload
    content = str(result_payload.get("content") or result_payload.get("text") or "")
    raw = result_payload.get("raw") if isinstance(result_payload.get("raw"), dict) else {}
    return scores_from_llm_result(content, raw)


async def _classify_scores(
    bus: OrionBusAsync,
    *,
    prompt: str,
    settings,
    primary_route: str,
) -> dict:
    """Try primary classify route, then alternate lane, with short per-route retries."""
    alternate = "quick" if primary_route == "metacog" else "metacog"
    routes = (primary_route, alternate)
    last_error: Exception | None = None
    for route in routes:
        for attempt in range(2):
            try:
                scores = await _llm_classify(
                    bus, prompt=prompt, settings=settings, llm_route=route
                )
                if scores.get("novelty_score") is not None:
                    scores["classify_route_used"] = route
                    return scores
                last_error = RuntimeError("classify_missing_novelty_score")
                logger.warning(
                    "memory_classify_empty_scores route=%s attempt=%s",
                    route,
                    attempt + 1,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "memory_classify_attempt_failed route=%s attempt=%s err=%s",
                    route,
                    attempt + 1,
                    exc,
                )
            await asyncio.sleep(0.2)
    raise last_error or RuntimeError("classify_failed")


async def reappraise_with_session_window(
    bus: OrionBusAsync, *, turn: MemoryTurnPersistedV1, prior_turns: list[dict], settings
) -> dict:
    mode, text = _session_window_baseline(prior_turns, n=settings.TURN_CHANGE_WINDOW_TURNS)
    if mode == "none":
        return {}
    phase = (turn.spark_meta.get("conversation_phase") or {}).get("phase_change") or "unknown"
    prompt = build_change_only_prompt(
        prompt=turn.prompt,
        response=turn.response,
        baseline_text=text,
        phase=str(phase),
    )
    return await _llm_classify(bus, prompt=prompt, settings=settings)


async def classify_turn(
    bus: OrionBusAsync, *, turn: MemoryTurnPersistedV1, prior_turns: list[dict], settings
) -> dict:
    baseline_mode, baseline_text, prior_corr = _prior_turn_baseline(prior_turns)
    classify_route = _resolve_classify_route(settings)
    if baseline_mode == "none":
        appraisal = build_turn_change_appraisal(
            baseline_mode="none",
            prior_correlation_id=None,
            novelty_score=None,
            shift_kind=None,
            shift_scores=None,
            confidence=None,
            status="skipped",
        )
        return {
            "turn_change_appraisal": appraisal,
            "turn_change_classify_route": classify_route,
            "memory_classify_status": "degraded",
            "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
        }

    prompt = build_classify_prompt(
        prompt=turn.prompt,
        response=turn.response,
        spark_meta=turn.spark_meta,
        baseline_mode=baseline_mode,
        baseline_text=baseline_text,
    )

    last_error: Exception | None = None
    scores: dict | None = None
    try:
        scores = await _classify_scores(
            bus, prompt=prompt, settings=settings, primary_route=classify_route
        )
    except Exception as exc:
        last_error = exc
        logger.error("memory_classify_degraded corr=%s err=%s", turn.correlation_id, last_error)

    if scores is None:
        return _degraded_patch(
            baseline_mode=baseline_mode,
            prior_correlation_id=prior_corr,
            classify_route=classify_route,
        )

    route_used = str(scores.pop("classify_route_used", classify_route) or classify_route)

    logger.info(
        "turn_change_classify corr=%s route=%s novelty=%s shift=%s confidence=%s source=%s mem=%s bnd=%s",
        turn.correlation_id,
        route_used,
        scores.get("novelty_score"),
        scores.get("shift_kind"),
        scores.get("confidence"),
        scores.get("scoring_source"),
        scores.get("memory_significance_score"),
        scores.get("conversation_boundary_score"),
    )

    novelty = scores.get("novelty_score")
    if should_session_reappraise(
        scores,
        margin=settings.TURN_CHANGE_CONFIDENCE_MARGIN,
        prior_turn_count=len(prior_turns),
    ):
        try:
            retry = await reappraise_with_session_window(
                bus, turn=turn, prior_turns=prior_turns, settings=settings
            )
            if retry:
                scores["novelty_score"] = retry.get("novelty_score", novelty)
                scores["shift_kind"] = retry.get("shift_kind", scores.get("shift_kind"))
                scores["shift_scores"] = retry.get("shift_scores", scores.get("shift_scores"))
                if retry.get("confidence") is not None:
                    scores["confidence"] = retry["confidence"]
                if retry.get("scoring_source"):
                    scores["scoring_source"] = retry["scoring_source"]
                scores["reappraised_session_window"] = True
                baseline_mode = "session_window"
                _, baseline_text = _session_window_baseline(
                    prior_turns, n=settings.TURN_CHANGE_WINDOW_TURNS
                )
                prior_corr = None
        except Exception as exc:
            logger.warning(
                "turn_change_reappraisal_failed corr=%s err=%s",
                turn.correlation_id,
                exc,
            )

    scores = reconcile_novelty_with_shift(scores)

    status = (
        "ok"
        if scores.get("novelty_score") is not None
        and scores.get("scoring_source") in ("logprobs", "text")
        else "degraded"
    )
    appraisal = build_turn_change_appraisal(
        baseline_mode=baseline_mode,
        prior_correlation_id=prior_corr,
        novelty_score=scores.get("novelty_score"),
        shift_kind=scores.get("shift_kind"),
        shift_scores=scores.get("shift_scores"),
        confidence=scores.get("confidence"),
        status=status,
    )
    if scores.get("novelty_adjusted_from_shift"):
        appraisal["novelty_adjusted_from_shift"] = True
    if scores.get("reappraised_session_window"):
        appraisal["reappraised_session_window"] = True
    return {
        "turn_change_appraisal": appraisal,
        "turn_change_classify_route": route_used,
        "memory_significance_score": scores.get("memory_significance_score"),
        "conversation_boundary_score": scores.get("conversation_boundary_score"),
        "memory_classify_status": "ok" if status == "ok" else "degraded",
        "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
    }
