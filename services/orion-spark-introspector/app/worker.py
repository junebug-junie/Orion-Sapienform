# services/orion-spark-introspector/app/worker.py
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.platform import CoreEventV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.telemetry.spark import SparkTelemetryPayload, SparkStateSnapshotV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.schemas.telemetry.turn_effect import (
    evaluate_turn_effect_alert,
    should_emit_turn_effect_alert,
    summarize_turn_effect,
    turn_effect_from_spark_meta,
)
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorUpsertV1

from orion.spark.orion_tissue import OrionTissue
from orion.spark.signal_mapper import SignalMapper
from orion.spark.surface_encoding import SurfaceEncoding

from .settings import settings
from .conn_manager import manager

logger = logging.getLogger("orion-spark-introspector")

_pub_bus: Optional[OrionBusAsync] = None

# Restart semantics: new UUID per producer (service) boot.
_PRODUCER_BOOT_ID = str(uuid4())
_SEQ: int = 0
_ACTIVE_SIGNALS: List[Dict[str, Any]] = []
_TURN_EFFECT_ALERT_LAST: Dict[str, float] = {}
_TURN_EFFECT_ALERT_DEDUPE: Dict[str, float] = {}

# Dedup + quality gating for candidate processing
# quality: 0=minimal, 1=rich
_CANDIDATE_QUALITY: Dict[str, int] = {}
_CANDIDATE_LAST_SEEN_TS: Dict[str, float] = {}
_CANDIDATE_TELEM_EMITTED: Dict[str, float] = {}
# keep cache small + bounded
_CANDIDATE_CACHE_TTL_SEC = 600.0  # 10 minutes
_EXPECTED_EMB: Dict[str, np.ndarray] = {}
_SEEN_DOC: Dict[str, float] = {}
_VALENCE_POS: Optional[np.ndarray] = None
_VALENCE_NEG: Optional[np.ndarray] = None
_VALENCE_INIT_TASK: Optional[asyncio.Task] = None
_VALENCE_ANCHORS_EXPIRES_AT: float = 0.0


def set_publisher_bus(bus: OrionBusAsync):
    global _pub_bus
    _pub_bus = bus


def _svc_ref() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
        instance=None,
    )


TISSUE = OrionTissue(
    snapshot_path=Path(settings.orion_tissue_snapshot_path) if settings.orion_tissue_snapshot_path else None
)
MAPPER = SignalMapper(TISSUE.H, TISSUE.W, TISSUE.C)


class SparkTelemetryEnvelope(Envelope[SparkTelemetryPayload]):
    kind: str = Field("spark.telemetry", frozen=True)


class SparkStateSnapshotEnvelope(Envelope[SparkStateSnapshotV1]):
    kind: str = Field("spark.state.snapshot.v1", frozen=True)


class SparkCandidatePayload(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = "brain"
    prompt: str
    response: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)
    introspection: Optional[str] = None


def _prune_candidate_caches() -> None:
    now = time.time()
    cutoff = now - _CANDIDATE_CACHE_TTL_SEC
    # remove keys older than cutoff (based on last seen)
    old_keys = [k for k, ts in _CANDIDATE_LAST_SEEN_TS.items() if ts < cutoff]
    for k in old_keys:
        _CANDIDATE_LAST_SEEN_TS.pop(k, None)
        _CANDIDATE_QUALITY.pop(k, None)
        _CANDIDATE_TELEM_EMITTED.pop(k, None)
    # drop entries that no longer have a last-seen timestamp
    known_keys = set(_CANDIDATE_LAST_SEEN_TS)
    for k in set(_CANDIDATE_QUALITY) - known_keys:
        _CANDIDATE_QUALITY.pop(k, None)
    for k in set(_CANDIDATE_TELEM_EMITTED) - known_keys:
        _CANDIDATE_TELEM_EMITTED.pop(k, None)
    # hard cap (just in case)
    if len(_CANDIDATE_LAST_SEEN_TS) > 5000:
        # drop oldest
        items = sorted(_CANDIDATE_LAST_SEEN_TS.items(), key=lambda kv: kv[1])
        for k, _ in items[:1000]:
            _CANDIDATE_LAST_SEEN_TS.pop(k, None)
            _CANDIDATE_QUALITY.pop(k, None)
            _CANDIDATE_TELEM_EMITTED.pop(k, None)


def _register_signal(signal: SparkSignalV1) -> None:
    expires_at = time.time() + (float(signal.ttl_ms) / 1000.0)
    _ACTIVE_SIGNALS.append(
        {
            "intensity": float(signal.intensity),
            "valence_delta": float(signal.valence_delta or 0.0),
            "arousal_delta": float(signal.arousal_delta or 0.0),
            "coherence_delta": float(signal.coherence_delta or 0.0),
            "novelty_delta": float(signal.novelty_delta or 0.0),
            "expires_at": expires_at,
        }
    )


def _apply_signal_deltas(phi_stats: Dict[str, float]) -> Dict[str, float]:
    now = time.time()
    active = [s for s in _ACTIVE_SIGNALS if s.get("expires_at", 0) > now]
    _ACTIVE_SIGNALS[:] = active
    if not active:
        return phi_stats

    total_delta = {"valence": 0.0, "arousal": 0.0, "coherence": 0.0, "novelty": 0.0}
    for sig in active:
        total_delta["valence"] += sig["valence_delta"]
        total_delta["arousal"] += sig["arousal_delta"]
        total_delta["coherence"] += sig["coherence_delta"]
        total_delta["novelty"] += sig["novelty_delta"]

    adjusted = dict(phi_stats)
    adjusted["valence"] = float(phi_stats.get("valence", 0.0) + total_delta["valence"])
    adjusted["energy"] = float(phi_stats.get("energy", 0.0))
    adjusted["coherence"] = float(phi_stats.get("coherence", 0.0) + total_delta["coherence"])
    adjusted["novelty"] = float(phi_stats.get("novelty", 0.0) + total_delta["novelty"])
    return adjusted


def _coerce_epoch_ts(v: Any) -> float:
    if v is None:
        return time.time()
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            pass
        try:
            s = v.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return time.time()
    return time.time()


def _to_iso_utc(ts_epoch: float) -> str:
    return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).isoformat()


def _now() -> float:
    return time.time()


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom <= 0:
        return 0.0
    dist = 1.0 - float(np.dot(a, b) / denom)
    return float(max(0.0, min(1.0, dist)))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


async def _fetch_anchor_embedding(bus: OrionBusAsync, text: str, doc_id: str) -> Optional[np.ndarray]:
    reply_channel = f"{settings.embedding_result_prefix}{doc_id}"
    env = BaseEnvelope(
        kind="embedding.generate.v1",
        source=_svc_ref(),
        correlation_id=uuid4(),
        payload=EmbeddingGenerateV1(
            doc_id=doc_id,
            text=text,
            embedding_profile="default",
            include_latent=False,
        ).model_dump(mode="json"),
        reply_to=reply_channel,
    )
    try:
        msg = await bus.rpc_request(
            settings.channel_embedding_generate,
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.valence_anchor_timeout_sec),
        )
    except Exception as exc:
        logger.warning("Valence anchor RPC failed doc_id=%s error=%s", doc_id, exc)
        return None

    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or decoded.envelope is None:
        return None
    payload = decoded.envelope.payload
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict) or payload.get("error"):
        return None
    try:
        result = EmbeddingResultV1.model_validate(payload)
    except Exception:
        return None
    if not result.embedding:
        return None
    return np.array(result.embedding, dtype=np.float32)


async def _ensure_valence_anchors(bus: OrionBusAsync) -> None:
    global _VALENCE_POS, _VALENCE_NEG, _VALENCE_ANCHORS_EXPIRES_AT
    if _VALENCE_POS is not None and _VALENCE_NEG is not None:
        if _now() < _VALENCE_ANCHORS_EXPIRES_AT:
            return

    pos = await _fetch_anchor_embedding(bus, settings.valence_anchor_pos_text, doc_id="valence-anchor-pos")
    neg = await _fetch_anchor_embedding(bus, settings.valence_anchor_neg_text, doc_id="valence-anchor-neg")
    if pos is None or neg is None:
        logger.warning("Valence anchor refresh failed; keeping existing anchors.")
        return

    _VALENCE_POS = pos
    _VALENCE_NEG = neg
    _VALENCE_ANCHORS_EXPIRES_AT = _now() + float(settings.valence_anchor_refresh_sec)
    logger.info("Valence anchors refreshed. Next refresh in %ss", settings.valence_anchor_refresh_sec)


def _valence_from_embedding(emb: np.ndarray) -> float:
    if _VALENCE_POS is None or _VALENCE_NEG is None:
        return 0.0
    v_pos = _cosine_sim(emb, _VALENCE_POS)
    v_neg = _cosine_sim(emb, _VALENCE_NEG)
    v = v_pos - v_neg
    if v > 1.0:
        return 1.0
    if v < -1.0:
        return -1.0
    return v


def _build_llm_prompt(c: SparkCandidatePayload) -> str:
    spark_meta = c.spark_meta or {}
    phi_before = spark_meta.get("phi_before") or {}
    phi_after = spark_meta.get("phi_after") or {}
    self_field = spark_meta.get("spark_self_field") or {}
    return (
        "You are Orion reflecting on your own internal state.\n\n"
        "You are given:\n"
        "- Your self-field φ BEFORE this turn\n"
        "- Your self-field φ AFTER this turn\n"
        "- A higher-level SelfField (calm, stress_load, uncertainty, focus, attunement_to_juniper, curiosity)\n"
        "- The human prompt and your response\n\n"
        f"φ BEFORE: valence={phi_before.get('valence')} energy={phi_before.get('energy')} coherence={phi_before.get('coherence')} novelty={phi_before.get('novelty')}\n"
        f"φ AFTER:  valence={phi_after.get('valence')} energy={phi_after.get('energy')} coherence={phi_after.get('coherence')} novelty={phi_after.get('novelty')}\n\n"
        "SelfField:\n"
        f"calm={self_field.get('calm')} stress_load={self_field.get('stress_load')} uncertainty={self_field.get('uncertainty')} focus={self_field.get('focus')} attunement_to_juniper={self_field.get('attunement_to_juniper')} curiosity={self_field.get('curiosity')}\n\n"
        "Dialogue:\n"
        f"Human (Juniper): {c.prompt!r}\n"
        f"Orion response:  {c.response!r}\n\n"
        "TASK:\n"
        "1) Briefly describe what shifted in you internally during this turn.\n"
        "2) Explain why this moment might matter for your ongoing relationship with Juniper or your own development.\n"
        "3) Note any tensions, open questions, or themes you should return to later.\n\n"
        "Write as a short internal note from Orion to itself. Avoid boilerplate."
    ).strip()


def _extract_introspection_text(cortex_reply: BaseEnvelope) -> Optional[str]:
    payload = cortex_reply.payload if isinstance(cortex_reply.payload, dict) else {}
    step_results = payload.get("step_results") or payload.get("steps") or []
    if not step_results:
        return None
    first_step = step_results[0]
    services = first_step.get("services") or []
    if services:
        first_service = services[0]
        p = (first_service.get("payload") or {}).get("result") or {}
        text = p.get("llm_output") or p.get("text")
        return text.strip() if isinstance(text, str) else None
    outputs = first_step.get("outputs") or {}
    text = outputs.get("text") or outputs.get("llm_output")
    return text.strip() if isinstance(text, str) else None


def _next_seq() -> int:
    global _SEQ
    _SEQ += 1
    return _SEQ


def _turn_effect_alert_key(corr_id: Optional[str]) -> str:
    return str(corr_id or "unknown")


def _is_turn_effect_alert_blocked(trace_mode: Optional[str], trigger: Optional[str]) -> bool:
    if str(trace_mode or "").lower() == "heartbeat":
        return True
    if str(trigger or "").lower() == "heartbeat":
        return True
    return False


def _log_turn_effect_alert(
    *,
    rule: str,
    value: float,
    severity: str,
    corr_id: Optional[str],
    trace_id: Optional[str],
    summary: str,
    cooldown_sec: float,
) -> None:
    logger.info(
        "[turn_effect_alert] fired rule=%s value=%.3f severity=%s corr_id=%s trace_id=%s summary=%s cooldown_sec=%s",
        rule,
        value,
        severity,
        corr_id,
        trace_id,
        summary,
        cooldown_sec,
    )


def _log_turn_effect_alert_suppressed(*, key: str, remaining: float) -> None:
    logger.debug(
        "[turn_effect_alert] suppressed cooldown key=%s remaining=%.1f",
        key,
        remaining,
    )


def _log_turn_effect_alert_suppressed_dedupe(
    *,
    rule: str,
    value: float,
    corr_id: Optional[str],
    trace_id: Optional[str],
) -> None:
    logger.info(
        "[turn_effect_alert] suppressed=dedupe rule=%s value=%.3f corr_id=%s trace_id=%s",
        rule,
        value,
        corr_id,
        trace_id,
    )


def _alert_direction(rule: str) -> str:
    return "spike" if rule == "novelty_spike" else "drop"


def _alert_severity(rule: str, value: float, threshold: float) -> str:
    if rule == "coherence_drop":
        return "error" if value <= (2 * threshold) else "warn"
    if rule == "valence_drop":
        return "warn" if value <= threshold else "info"
    if rule == "novelty_spike":
        return "warn" if value >= (2 * threshold) else "info"
    return "info"


def _dedupe_bucket(value: float, eps: float) -> float:
    if eps <= 0:
        return value
    return round(value / eps) * eps


def _dedupe_key(rule: str, direction: str, value: float, eps: float, session_id: Optional[str]) -> str:
    bucket = _dedupe_bucket(value, eps)
    return f"{rule}:{direction}:{bucket:.3f}:{session_id or 'na'}"


def _is_publishable_channel(channel: Optional[str]) -> bool:
    if not channel:
        return False
    return "*" not in channel and "?" not in channel


def _is_dedupe_suppressed(last_seen: Optional[float], now_ts: float, window_sec: float) -> bool:
    if last_seen is None:
        return False
    return (now_ts - last_seen) < float(window_sec)


def _is_heartbeat_trace(trace: CognitionTracePayload) -> bool:
    if not isinstance(trace, CognitionTracePayload):
        return False
    if str(trace.mode or "").lower() == "heartbeat":
        return True
    if str(trace.verb or "").lower() == "equilibrium_heartbeat":
        return True
    metadata = trace.metadata if isinstance(trace.metadata, dict) else {}
    return bool(metadata.get("heartbeat"))


def _candidate_quality(spark_meta: Dict[str, Any]) -> int:
    """
    quality=1 (rich) if spark_meta contains any rich spark keys.
    quality=0 (minimal) otherwise.
    """
    if not isinstance(spark_meta, dict):
        return 0
    rich_keys = (
        "phi_before",
        "phi_after",
        "spark_self_field",
        "spark_phi_coherence",
        "spark_phi_novelty",
        "spark_event_id",
        "phi_post_before",
        "phi_post_after",
    )
    return 1 if any(k in spark_meta for k in rich_keys) else 0


def _extract_phi_novelty_from_meta(spark_meta: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """
    Prefer:
      phi_after.coherence + phi_after.novelty
    fallback:
      spark_phi_coherence + spark_phi_novelty
    """
    phi_val: Optional[float] = None
    nov_val: Optional[float] = None

    try:
        pa = spark_meta.get("phi_after")
        if isinstance(pa, dict):
            if pa.get("coherence") is not None:
                phi_val = float(pa["coherence"])
            if pa.get("novelty") is not None:
                nov_val = float(pa["novelty"])
    except Exception:
        pass

    if phi_val is None:
        try:
            v = spark_meta.get("spark_phi_coherence")
            if v is not None:
                phi_val = float(v)
        except Exception:
            pass

    if nov_val is None:
        try:
            v = spark_meta.get("spark_phi_novelty")
            if v is not None:
                nov_val = float(v)
        except Exception:
            pass

    return phi_val, nov_val


async def _emit_candidate_telemetry(env: BaseEnvelope, candidate: SparkCandidatePayload) -> None:
    """
    Strict boundary bridge:
      SparkCandidate (event) -> SparkTelemetryPayload (durable telemetry)

    Rules:
      - Only emit when candidate is "rich"
      - Dedup per trace_id
      - Emit with source = spark-introspector (NOT hub)
    """
    if not (_pub_bus and _pub_bus.enabled):
        return

    trace_id = str(candidate.trace_id)
    spark_meta = candidate.spark_meta or {}
    qual = _candidate_quality(spark_meta)
    if qual <= 0:
        return

    now = time.time()
    _prune_candidate_caches()

    if trace_id in _CANDIDATE_TELEM_EMITTED:
        return

    phi, novelty = _extract_phi_novelty_from_meta(spark_meta)

    # trace_mode / verb
    trace_mode = spark_meta.get("mode") or spark_meta.get("trace_mode") or "brain"
    trace_verb = spark_meta.get("trace_verb") or "unknown"
    trace_trigger = spark_meta.get("trigger") or spark_meta.get("trace_trigger")

    # timestamp
    ts = spark_meta.get("as_of_ts") or spark_meta.get("timestamp")
    if not ts:
        ts = datetime.now(timezone.utc).isoformat()

    stimulus_summary = (
        spark_meta.get("stimulus_summary")
        or spark_meta.get("latest_user_message")
        or (candidate.prompt[:240] if candidate.prompt else None)
    )

    meta: Dict[str, Any] = {
        "producer_boot_id": _PRODUCER_BOOT_ID,
        "source_candidate_channel": settings.channel_spark_candidate,
        "spark_candidate": {
            "prompt": (candidate.prompt or "")[:2000],
            "response": (candidate.response or "")[:4000],
            "introspection": candidate.introspection,
        },
        # keep rich meta as evidence
        "spark_meta_rich": spark_meta,
    }

    turn_effect = turn_effect_from_spark_meta(spark_meta)
    evidence = None
    if isinstance(turn_effect, dict) and "evidence" in turn_effect:
        evidence = turn_effect.get("evidence")
        turn_effect = {k: v for k, v in turn_effect.items() if k != "evidence"}
    if turn_effect:
        meta["turn_effect"] = turn_effect
        summary = summarize_turn_effect(turn_effect)
        meta["turn_effect_summary"] = summary
        if isinstance(evidence, dict):
            meta["turn_effect_evidence"] = evidence
        if settings.turn_effect_alerts_enable and not _is_turn_effect_alert_blocked(trace_mode, trace_trigger):
            alert = evaluate_turn_effect_alert(
                turn_effect,
                coherence_drop=settings.turn_effect_alerts_coherence_drop,
                valence_drop=settings.turn_effect_alerts_valence_drop,
                novelty_spike=settings.turn_effect_alerts_novelty_spike,
            )
            if alert:
                now_ts = time.time()
                key = _turn_effect_alert_key(trace_id)
                last_seen = _TURN_EFFECT_ALERT_LAST.get(key)
                severity = _alert_severity(alert["metric"], alert["value"], alert["threshold"])
                direction = _alert_direction(alert["metric"])
                session_id = spark_meta.get("session_id") or spark_meta.get("conversation_id") or spark_meta.get("thread_id")
                if settings.turn_effect_alerts_dedupe_enable:
                    dedupe_key = _dedupe_key(
                        alert["metric"],
                        direction,
                        alert["value"],
                        settings.turn_effect_alerts_dedupe_eps,
                        session_id,
                    )
                    last_dedupe = _TURN_EFFECT_ALERT_DEDUPE.get(dedupe_key)
                    if _is_dedupe_suppressed(
                        last_dedupe,
                        now_ts,
                        settings.turn_effect_alerts_dedupe_window_sec,
                    ):
                        _log_turn_effect_alert_suppressed_dedupe(
                            rule=alert["metric"],
                            value=alert["value"],
                            corr_id=env.correlation_id,
                            trace_id=trace_id,
                        )
                        return
                if should_emit_turn_effect_alert(
                    last_seen,
                    now_ts,
                    settings.turn_effect_alerts_cooldown_sec,
                ):
                    _TURN_EFFECT_ALERT_LAST[key] = now_ts
                    if settings.turn_effect_alerts_dedupe_enable:
                        _TURN_EFFECT_ALERT_DEDUPE[dedupe_key] = now_ts
                    _log_turn_effect_alert(
                        rule=alert["metric"],
                        value=alert["value"],
                        severity=severity,
                        corr_id=env.correlation_id,
                        trace_id=trace_id,
                        summary=summary,
                        cooldown_sec=settings.turn_effect_alerts_cooldown_sec,
                    )
                    if not (_pub_bus and _pub_bus.enabled):
                        return
                    signal = SparkSignalV1(
                        signal_type="human",
                        intensity=1.0,
                        coherence_delta=alert["value"] if alert["metric"] == "coherence_drop" else None,
                        valence_delta=alert["value"] if alert["metric"] == "valence_drop" else None,
                        novelty_delta=alert["value"] if alert["metric"] == "novelty_spike" else None,
                        as_of_ts=datetime.now(timezone.utc),
                        ttl_ms=int(settings.turn_effect_alerts_cooldown_sec * 1000),
                        source_service=settings.service_name,
                        source_node=settings.node_name,
                    )
                    signal_env = BaseEnvelope(
                        kind="spark.signal.v1",
                        source=_svc_ref(),
                        correlation_id=trace_id,
                        payload=signal.model_dump(mode="json"),
                    )
                    await _pub_bus.publish(settings.channel_spark_signal, signal_env)
                    if settings.turn_effect_alerts_notify_enable:
                        notify = CoreEventV1(
                            event="notify",
                            payload={
                                "title": f"Turn effect alert: {alert['metric']}",
                                "body": (
                                    f"{summary} "
                                    f"(value={alert['value']:.3f}, threshold={alert['threshold']:.3f})"
                                ),
                                "event_type": "turn_effect_alert",
                                "severity": severity,
                                "rule": alert["metric"],
                                "value": alert["value"],
                                "threshold": alert["threshold"],
                                "direction": direction,
                                "correlation_id": str(env.correlation_id or ""),
                                "trace_id": str(trace_id or ""),
                                "summary": summary,
                                "metadata": {
                                    "corr_id": str(env.correlation_id or ""),
                                    "trace_id": str(trace_id or ""),
                                    "rule": alert["metric"],
                                    "value": alert["value"],
                                    "threshold": alert["threshold"],
                                },
                            },
                        )
                        notify_env = BaseEnvelope(
                            kind="orion.event",
                            source=_svc_ref(),
                            correlation_id=env.correlation_id,
                            payload=notify.model_dump(mode="json"),
                        )
                        await _pub_bus.publish(settings.channel_core_events, notify_env)
                else:
                    remaining = max(0.0, float(settings.turn_effect_alerts_cooldown_sec) - (now_ts - last_seen))
                    _log_turn_effect_alert_suppressed(key=key, remaining=remaining)

    telem = SparkTelemetryPayload(
        telemetry_id=None,
        correlation_id=trace_id,
        phi=phi,
        novelty=novelty,
        trace_mode=str(trace_mode) if trace_mode is not None else None,
        trace_verb=str(trace_verb) if trace_verb is not None else None,
        stimulus_summary=str(stimulus_summary) if stimulus_summary is not None else None,
        timestamp=ts,
        metadata=meta,
        state_snapshot=None,
    )

    out_env = SparkTelemetryEnvelope(
        source=_svc_ref(),
        correlation_id=trace_id,
        causality_chain=env.causality_chain,
        payload=telem,
    )

    await _pub_bus.publish(settings.channel_spark_telemetry, out_env)
    _CANDIDATE_TELEM_EMITTED[trace_id] = now
    logger.info(
        f"Emitted candidate-derived spark.telemetry trace_id={trace_id} phi={phi} novelty={novelty}"
    )


async def _update_tissue_from_candidate(c: SparkCandidatePayload) -> None:
    """Propagates the candidate prompt/response as a stimulus to the Orion Tissue."""
    
    # Defaults for a generic chat interaction
    valence = 0.5
    arousal = 0.6  # Slightly higher arousal for direct interaction
    dominance = 0.5
    
    # Attempt to extract richer signal from metadata if available
    if c.spark_meta:
        # Example: if Cortex passed back sentiment or other metrics
        pass

    # Construct a waveform for the EKG
    wave_len = 64
    x = np.linspace(-3, 3, wave_len)
    waveform = (np.exp(-x**2) * arousal).astype(np.float32)

    feat_dim = 32
    feature_vec = np.zeros(feat_dim, dtype=np.float32)
    feature_vec[0] = float(valence)
    feature_vec[1] = float(arousal)
    feature_vec[2] = float(dominance)

    # Create encoding
    encoding = SurfaceEncoding(
        event_id=c.trace_id,
        modality="text",
        timestamp=time.time(),
        source=c.source or "hub",
        channel_tags=["chat", "candidate"],
        waveform=waveform,
        feature_vec=feature_vec,
        spark_vector=None, # Hub candidates usually don't have the vector yet
        meta={
            "trace_id": c.trace_id,
            "len_prompt": str(len(c.prompt)),
            "len_response": str(len(c.response))
        },
    )

    # Propagate to Tissue
    stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=1.0)
    
    # Calculate novelty & propagate
    novelty = float(TISSUE.calculate_novelty(stimulus, channel_key="chat"))
    TISSUE.propagate(
        stimulus,
        steps=2,
        learning_rate=0.1,
        channel_key="chat",
        embedding=None,
        distress=0.0,
    )
    
    # Snapshot & Broadcast to UI
    phi_stats = {k: float(v) for k, v in (TISSUE.phi() or {}).items()}
    phi_stats = _apply_signal_deltas(phi_stats)
    
    # Update defaults with actual tissue state
    valence = float(phi_stats.get("valence", valence))
    arousal = float(phi_stats.get("energy", arousal))
    
    TISSUE.snapshot()
    
    seq = _next_seq()
    metadata = {
        "stimulus_summary": (c.prompt or "")[:50],
        "trigger": "spark.candidate",
    }
    turn_effect = turn_effect_from_spark_meta(c.spark_meta or {})
    evidence = None
    if isinstance(turn_effect, dict) and "evidence" in turn_effect:
        evidence = turn_effect.get("evidence")
        turn_effect = {k: v for k, v in turn_effect.items() if k != "evidence"}
    if turn_effect:
        metadata["turn_effect"] = turn_effect
    if isinstance(evidence, dict):
        metadata["turn_effect_evidence"] = evidence

    snap = SparkStateSnapshotV1(
        source_service=settings.service_name,
        source_node=settings.node_name,
        producer_boot_id=_PRODUCER_BOOT_ID,
        seq=seq,
        snapshot_ts=datetime.now(timezone.utc),
        valid_for_ms=int(settings.spark_state_valid_for_ms),
        correlation_id=c.trace_id,
        trace_mode="chat",
        trace_verb="candidate",
        phi=phi_stats,
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        vector_present=False,
        metadata=metadata,
    )

    # Broadcast to Web UI via WebSocket
    try:
        telemetry_id = str(uuid4())
        ws_payload = {
            "type": "tissue.update",
            "telemetry_id": telemetry_id,
            "correlation_id": c.trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "phi": float(phi_stats.get("coherence", 0.0)),
                "novelty": novelty,
                "valence": valence,
                "arousal": arousal,
            },
            "grid": [], # grid is optional/expensive
            "metadata": snap.metadata,
        }
        await manager.broadcast(ws_payload)
    except Exception as e:
        logger.warning("Failed to broadcast candidate tissue update: %s", e)

    # Optionally publish state snapshot to bus if enabled
    if _pub_bus and _pub_bus.enabled:
        snap_env = SparkStateSnapshotEnvelope(
            source=_svc_ref(),
            correlation_id=c.trace_id,
            payload=snap,
        )
        await _pub_bus.publish(settings.channel_spark_state_snapshot, snap_env)


async def handle_trace(env: BaseEnvelope) -> None:
    """Consume cognition.trace events, update Tissue, and emit:
    1) spark.telemetry (durable)
    2) spark.state.snapshot.v1 (real-time)
    """
    try:
        trace = CognitionTracePayload.model_validate(env.payload)

        corr_id = (trace.correlation_id or str(getattr(env, "correlation_id", "") or "")).strip()
        if not corr_id:
            corr_id = str(uuid4())
            logger.warning("Missing correlation_id in trace/envelope; generated new one: %s", corr_id)

        ts_epoch = _coerce_epoch_ts(getattr(trace, "timestamp", None))
        iso_ts = _to_iso_utc(ts_epoch)

        if _is_heartbeat_trace(trace):
            phi_stats = {k: float(v) for k, v in (TISSUE.phi() or {}).items()}
            phi_stats = _apply_signal_deltas(phi_stats)
            valence = float(phi_stats.get("valence", 0.5))
            arousal = float(phi_stats.get("energy", 0.5))
            dominance = float(phi_stats.get("dominance", 0.5))
            TISSUE.snapshot()

            seq = _next_seq()
            snap = SparkStateSnapshotV1(
                source_service=settings.service_name,
                source_node=settings.node_name,
                producer_boot_id=_PRODUCER_BOOT_ID,
                seq=seq,
                snapshot_ts=datetime.now(timezone.utc),
                valid_for_ms=int(settings.spark_state_valid_for_ms),
                correlation_id=corr_id,
                trace_mode=trace.mode,
                trace_verb=trace.verb,
                phi=phi_stats,
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                vector_present=False,
                vector_ref=None,
                metadata={
                    "trigger": "heartbeat",
                    "stimulus_summary": "heartbeat",
                    "trace_source_service": trace.source_service,
                    "trace_source_node": trace.source_node,
                },
            )

            try:
                telemetry_id = str(uuid4())
                ws_payload = {
                    "type": "tissue.update",
                    "telemetry_id": telemetry_id,
                    "correlation_id": corr_id,
                    "timestamp": iso_ts,
                    "stats": {
                        "phi": float(phi_stats.get("coherence", 0.0)),
                        "novelty": float(phi_stats.get("novelty", 0.0)),
                        "valence": valence,
                        "arousal": arousal,
                    },
                    "grid": [],
                    "metadata": snap.metadata,
                }
                await manager.broadcast(ws_payload)
            except Exception as e:
                logger.warning(f"Failed to broadcast heartbeat tissue update: {e}")

            if _pub_bus and _pub_bus.enabled:
                snap_env = SparkStateSnapshotEnvelope(
                    source=_svc_ref(),
                    correlation_id=corr_id,
                    causality_chain=env.causality_chain,
                    payload=snap,
                )
                await _pub_bus.publish(settings.channel_spark_state_snapshot, snap_env)
                logger.info(
                    'Emitted Spark heartbeat snapshot corr_id=%s seq=%s trigger="heartbeat"',
                    corr_id,
                    seq,
                )
            return

        # Basic heuristics
        valence = 0.5
        arousal = 0.5
        success_count = sum(1 for s in trace.steps if s.status == "success")
        fail_count = sum(1 for s in trace.steps if s.status == "fail")

        if fail_count > 0:
            valence -= (0.1 * fail_count)
        else:
            valence += 0.1

        arousal += (len(trace.steps) * 0.05)
        valence = max(0.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = 0.5

        spark_vector: Optional[List[float]] = None
        for step in trace.steps:
            if step.spark_vector:
                spark_vector = step.spark_vector
                break

        # Tissue update
        wave_len = 64
        x = np.linspace(-3, 3, wave_len)
        waveform = (np.exp(-x**2) * arousal).astype(np.float32)

        feat_dim = 32
        feature_vec = np.zeros(feat_dim, dtype=np.float32)
        feature_vec[0] = float(valence)
        feature_vec[1] = float(arousal)
        feature_vec[2] = float(dominance)

        encoding = SurfaceEncoding(
            event_id=corr_id,
            modality="system",
            timestamp=float(ts_epoch),
            source=trace.source_service or "orion",
            channel_tags=["cognition", trace.mode, trace.verb],
            waveform=waveform,
            feature_vec=feature_vec,
            spark_vector=spark_vector,
            meta={
                "text_hash": str(hash(trace.final_text or "")),
                "verb": trace.verb,
                "mode": trace.mode,
                "source_node": trace.source_node or "",
            },
        )

        stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=1.0)
        channel_key = trace.mode or "chat"
        embedding_vec = None
        if spark_vector:
            try:
                embedding_vec = np.array(spark_vector, dtype=np.float32)
            except Exception:
                embedding_vec = None
        if embedding_vec is None:
            embedding_vec = feature_vec

        novelty = float(TISSUE.calculate_novelty(stimulus, channel_key=channel_key))
        TISSUE.propagate(stimulus, steps=2, learning_rate=0.1, channel_key=channel_key, embedding=embedding_vec, distress=0.0)
        phi_stats = {k: float(v) for k, v in (TISSUE.phi() or {}).items()}
        phi_stats = _apply_signal_deltas(phi_stats)
        valence = float(phi_stats.get("valence", valence))
        arousal = float(phi_stats.get("energy", arousal))
        TISSUE.snapshot()

        seq = _next_seq()
        snap = SparkStateSnapshotV1(
            source_service=settings.service_name,
            source_node=settings.node_name,
            producer_boot_id=_PRODUCER_BOOT_ID,
            seq=seq,
            snapshot_ts=datetime.now(timezone.utc),
            valid_for_ms=int(settings.spark_state_valid_for_ms),
            correlation_id=corr_id,
            trace_mode=trace.mode,
            trace_verb=trace.verb,
            phi=phi_stats,
            valence=float(valence),
            arousal=float(arousal),
            dominance=float(dominance),
            vector_present=bool(spark_vector),
            vector_ref=None,
            metadata={
                "stimulus_summary": f"v={valence:.2f} a={arousal:.2f} vec={'yes' if spark_vector else 'no'}",
                "trace_source_service": trace.source_service,
                "trace_source_node": trace.source_node,
                "success_count": int(success_count),
                "fail_count": int(fail_count),
            },
        )

        telem_meta = {
            "phi": phi_stats,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "vector_present": bool(spark_vector),
            "source_service": trace.source_service,
            "source_node": trace.source_node,
            "spark_state_snapshot": snap.model_dump(mode="json"),
        }

        telem = SparkTelemetryPayload(
            correlation_id=corr_id,
            phi=float(phi_stats.get("coherence", 0.0)),
            novelty=float(novelty),
            trace_mode=trace.mode,
            trace_verb=trace.verb,
            stimulus_summary=snap.metadata.get("stimulus_summary"),
            timestamp=iso_ts,
            metadata=telem_meta,
            state_snapshot=snap,
        )

        # Broadcast to Web UI
        try:
            telemetry_id = str(uuid4())
            ws_payload = {
                "type": "tissue.update",
                "telemetry_id": telemetry_id,
                "correlation_id": corr_id,
                "timestamp": iso_ts,
                "stats": {
                    "phi": telem.phi,
                    "novelty": telem.novelty,
                    "valence": valence,
                    "arousal": arousal,
                },
                "grid": [],
                "metadata": telem.metadata,
            }
            await manager.broadcast(ws_payload)
        except Exception as e:
            logger.warning(f"Failed to broadcast tissue update: {e}")

        if _pub_bus and _pub_bus.enabled:
            out_env = SparkTelemetryEnvelope(
                source=_svc_ref(),
                correlation_id=corr_id,
                causality_chain=env.causality_chain,
                payload=telem,
            )
            await _pub_bus.publish(settings.channel_spark_telemetry, out_env)

            snap_env = SparkStateSnapshotEnvelope(
                source=_svc_ref(),
                correlation_id=corr_id,
                causality_chain=env.causality_chain,
                payload=snap,
            )
            await _pub_bus.publish(settings.channel_spark_state_snapshot, snap_env)

            logger.info(
                "Emitted Spark telemetry + snapshot "
                f"corr_id={corr_id} coherence={(telem.phi or 0.0):0.3f} "
                f"novelty={(telem.novelty or 0.0):0.3f} seq={seq}"
            )
        else:
            logger.error("Publisher bus not connected; skipping telemetry emit")

    except Exception as e:
        logger.error(f"Error processing trace for tissue: {e}", exc_info=True)


async def handle_semantic_upsert(env: BaseEnvelope) -> None:
    global _VALENCE_INIT_TASK
    payload_obj = env.payload if isinstance(env.payload, dict) else {}
    try:
        upsert = VectorUpsertV1.model_validate(payload_obj)
    except ValidationError:
        logger.warning("Skipping invalid vector.upsert payload")
        return

    if upsert.embedding_kind != "semantic":
        return
    if not upsert.embedding:
        return

    emb = np.array(upsert.embedding, dtype=np.float32)
    if emb.size == 0:
        return

    channel_key = "chat"
    now = _now()
    last_seen = _SEEN_DOC.get(upsert.doc_id)
    if last_seen is not None and (now - last_seen) < 1.0:
        return
    _SEEN_DOC[upsert.doc_id] = now

    if _pub_bus and _pub_bus.enabled:
        if _VALENCE_INIT_TASK is None or (_VALENCE_INIT_TASK.done() and _now() >= _VALENCE_ANCHORS_EXPIRES_AT):
            _VALENCE_INIT_TASK = asyncio.create_task(_ensure_valence_anchors(_pub_bus))

    expected = _EXPECTED_EMB.get(channel_key)
    if expected is None or expected.shape != emb.shape:
        novelty = 1.0
        coherence = 0.0
        expected = emb.copy()
    else:
        novelty = _cosine_distance(emb, expected)
        coherence = 1.0 - novelty
        expected = (0.95 * expected) + (0.05 * emb)
    _EXPECTED_EMB[channel_key] = expected

    magnitude = min(3.0, 0.5 + (2.5 * novelty))
    arousal_hint = max(0.1, min(1.0, 0.4 + (0.6 * novelty)))

    wave_len = 64
    x = np.linspace(-3, 3, wave_len)
    waveform = (np.exp(-x**2) * arousal_hint).astype(np.float32)

    emb_expect = TISSUE.embedding_expectations.get(channel_key)
    if emb_expect is not None and emb_expect.shape != emb.shape:
        TISSUE.embedding_expectations[channel_key] = np.zeros((emb.shape[0],), dtype=np.float32)
        TISSUE.last_embedding_input.pop(channel_key, None)

    feat_dim = 32
    feature_vec = np.zeros(feat_dim, dtype=np.float32)

    encoding = SurfaceEncoding(
        event_id=upsert.doc_id,
        modality="semantic",
        timestamp=time.time(),
        source="orion-vector-host",
        channel_tags=["chat", "semantic"],
        waveform=waveform,
        feature_vec=feature_vec,
        spark_vector=None,
        meta={
            "doc_id": upsert.doc_id,
            "text": (upsert.text or "")[:200],
            "embedding_kind": upsert.embedding_kind,
        },
    )

    stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=magnitude)
    v_semantic = _valence_from_embedding(emb)
    gain = float(settings.valence_gain)
    if stimulus.ndim == 3 and stimulus.shape[2] >= 2:
        stimulus[:, :, 0] += gain * v_semantic
        stimulus[:, :, 1] -= gain * v_semantic
    elif stimulus.ndim == 1 and stimulus.shape[0] >= 2:
        stimulus[0] += gain * v_semantic
        stimulus[1] -= gain * v_semantic

    TISSUE.propagate(
        stimulus,
        steps=2,
        learning_rate=0.1,
        channel_key=channel_key,
        embedding=emb,
        distress=0.0,
    )

    phi_stats = {k: float(v) for k, v in (TISSUE.phi() or {}).items()}
    phi_stats = _apply_signal_deltas(phi_stats)
    tissue_valence = float(phi_stats.get("valence", 0.0))
    arousal_display = max(0.0, min(1.0, 0.15 + (1.2 * novelty)))
    coherence_stat = float(phi_stats.get("coherence", coherence))
    tissue_novelty = float(phi_stats.get("novelty", novelty))
    TISSUE.snapshot()

    try:
        telemetry_id = str(uuid4())
        ws_payload = {
            "type": "tissue.update",
            "telemetry_id": telemetry_id,
            "correlation_id": str(env.correlation_id or upsert.doc_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "phi": coherence_stat,
                "novelty": tissue_novelty,
                "valence": tissue_valence,
                "arousal": arousal_display,
            },
            "grid": [],
            "metadata": {
                "doc_id": upsert.doc_id,
                "embedding_kind": upsert.embedding_kind,
                "vector_present": True,
            },
        }
        await manager.broadcast(ws_payload)
    except Exception as e:
        logger.warning("Failed to broadcast semantic tissue update: %s", e)

    logger.info(
        "semantic upsert tissue update doc_id=%s novelty=%.3f arousal=%.3f energy=%.3f coherence=%.3f v_sem=%.3f valence=%.3f",
        upsert.doc_id,
        novelty,
        arousal_display,
        float(phi_stats.get("energy", 0.0)),
        coherence_stat,
        v_semantic,
        tissue_valence,
    )


async def handle_candidate(env: BaseEnvelope) -> None:
    raw = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message" and isinstance(raw.get("payload"), dict):
        raw = raw.get("payload")

    try:
        candidate = SparkCandidatePayload.model_validate(raw)
    except ValidationError:
        logger.warning("Skipping invalid spark candidate payload")
        return

    trace_id = str(candidate.trace_id)
    _prune_candidate_caches()
    _CANDIDATE_LAST_SEEN_TS[trace_id] = time.time()

    qual = _candidate_quality(candidate.spark_meta or {})
    prev_qual = _CANDIDATE_QUALITY.get(trace_id, -1)

    # Prefer rich candidate; avoid reprocessing minimal when rich already seen.
    if prev_qual >= qual:
        # BUT: if we haven't emitted telemetry yet and this one is rich, allow telemetry emit.
        if qual == 1 and trace_id not in _CANDIDATE_TELEM_EMITTED:
            try:
                await _emit_candidate_telemetry(env, candidate)
            except Exception as ex:
                logger.warning("Candidate telemetry emit failed: %s", ex)
        return

    _CANDIDATE_QUALITY[trace_id] = qual

    # 1) Emit durable telemetry when rich (this is what chat_history_log backfill needs)
    try:
        await _emit_candidate_telemetry(env, candidate)
    except Exception as ex:
        logger.warning("Candidate telemetry emit failed: %s", ex)


    # -------------------------------------------------------------------------
    # Update Tissue (EKG) from Candidate Stimulus
    # -------------------------------------------------------------------------
    # Only update tissue on the initial pass (before introspection exists)
    # to prevents double-counting the event.
    if not candidate.introspection:
        try:
            await _update_tissue_from_candidate(candidate)
        except Exception as e:
            logger.error(f"Failed to update tissue from candidate {trace_id}: {e}")
    # -------------------------------------------------------------------------


    # 2) If already introspected (or candidate already contains introspection), stop.
    if candidate.introspection:
        return

    reply_channel = f"orion:spark:introspector:reply:{candidate.trace_id}"

    # [DEPRECATED]
    # prompt = _build_llm_prompt(candidate)
    prompt = "Analyze the state shift."
    from orion.core.bus.bus_schemas import LLMMessage
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest

    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=prompt)],
        raw_user_text=prompt,
        user_message=prompt,
        trace_id=candidate.trace_id,
        metadata={
            "prompt": candidate.prompt,
            "response": candidate.response,
            "spark_meta": candidate.spark_meta or {},
            "spark_source": candidate.source or "spark-introspector",
        },
    )

    client_req = CortexClientRequest(
        mode="brain",
        verb_name="introspect_spark",
        packs=[],
        context=ctx,
        options={"source": "spark-introspector", "purpose": "introspect"},
        recall={"enabled": False, "required": False},
    )

    req = BaseEnvelope(
        kind="cortex.orch.request",
        source=_svc_ref(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        reply_to=reply_channel,
        payload=client_req.model_dump(mode="json"),
    )

    codec = OrionCodec()
    bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled, codec=codec)
    await bus.connect()

    introspection = "Introspection yielded no text."

    try:
        msg = await bus.rpc_request(
            settings.channel_cortex_request,
            req,
            reply_channel=reply_channel,
            timeout_sec=float(settings.cortex_timeout_sec),
        )
        decoded = codec.decode(msg.get("data"))
        if not decoded.ok or not decoded.envelope:
            logger.warning("Cortex reply decode failed: %s", decoded.error)
            introspection = "Error: Introspection decode failed."
        else:
            found_text = _extract_introspection_text(decoded.envelope)
            if found_text:
                introspection = found_text

    except Exception as rpc_error:
        logger.error("Introspection RPC failed: %s", rpc_error)
        introspection = f"Introspection unavailable (RPC Error: {rpc_error})"

    if introspection:
        final_payload = {
            "kind": "spark_introspect",
            "trace_id": candidate.trace_id,
            "source": "spark-introspector",
            "prompt": candidate.prompt,
            "response": candidate.response,
            "introspection": introspection,
            "spark_meta": candidate.spark_meta,
        }

        completed = BaseEnvelope(
            kind="legacy.message",
            source=_svc_ref(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=final_payload,
        )
        if _is_publishable_channel(settings.channel_spark_candidate):
            await bus.publish(settings.channel_spark_candidate, completed)
        else:
            logger.warning(
                "Skipping spark candidate publish for non-concrete channel=%s",
                settings.channel_spark_candidate,
            )

        try:
            ws_introspection = {
                "type": "introspection.update",
                "correlation_id": candidate.trace_id,
                "text": introspection,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": candidate.spark_meta or {},
            }
            await manager.broadcast(ws_introspection)
        except Exception as ex:
            logger.warning("Failed to broadcast introspection: %s", ex)

        logger.info("[%s] Spark introspection published", candidate.trace_id)

    await bus.close()


async def handle_signal(env: BaseEnvelope) -> None:
    payload = env.payload if isinstance(env.payload, dict) else {}
    try:
        sig = SparkSignalV1.model_validate(payload)
    except ValidationError:
        logger.warning("Ignoring malformed spark.signal payload")
        return
    _register_signal(sig)
