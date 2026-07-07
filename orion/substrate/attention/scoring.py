from __future__ import annotations

import re
from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1
from orion.substrate.attention.common import bounded, compact, stable_id
from orion.substrate.attention.salience import SalienceHistory, compute_salience

_EMOTION_RE = re.compile(r"\b(frustrated|pissed|worried|excited|afraid|stuck|blocked|confused|love|hate|urgent)\b", re.I)
_PLAN_RE = re.compile(r"\b(plan|planning|going to|tomorrow|next|later|schedule|deadline|future)\b", re.I)
_ANOMALY_RE = re.compile(r"\b(weird|unexpected|anomaly|regression|broken|failed|mismatch|surprise)\b", re.I)


def known_blob(inputs: dict[str, Any], ctx: dict[str, Any]) -> str:
    pieces: list[str] = [str(ctx.get("memory_digest") or "")]
    for section in ("identity", "concept_induction", "social", "reflective", "reasoning_summary", "autonomy"):
        value = inputs.get(section)
        if isinstance(value, dict):
            pieces.append(str(value))
    return " ".join(pieces).lower()


def merge_signals(signals: list[AttentionSignalV1], *, limit: int) -> list[AttentionSignalV1]:
    by_target: dict[str, AttentionSignalV1] = {}
    for signal in signals:
        key = signal.target_text.strip().lower()
        if not key:
            continue
        existing = by_target.get(key)
        if existing is None or (signal.salience, signal.confidence) > (existing.salience, existing.confidence):
            by_target[key] = signal
    return list(by_target.values())[:limit]


def target_type_for(signal: AttentionSignalV1, user_text: str) -> str:
    hint = signal.target_type_hint
    text = f"{signal.target_text} {user_text} {signal.signal_kind}".lower()
    if hint in {"person", "place", "activity", "plan", "relation", "belief", "object", "concept", "anomaly", "memory_gap", "future_event", "other"}:
        if hint != "other":
            return hint
    if _ANOMALY_RE.search(text):
        return "anomaly"
    if _PLAN_RE.search(text):
        return "plan"
    if any(word in text for word in ("relationship", "with my", "with our", "between")):
        return "relation"
    if any(word in text for word in ("believe", "think", "assumption", "claim")):
        return "belief"
    if any(word in text for word in ("device", "board", "gpu", "file", "service", "object")):
        return "object"
    return "concept" if "concept" in text else "other"


def autonomy_pressure_from_signals(signals: list[AttentionSignalV1]) -> tuple[float, list[str]]:
    autonomy = [s for s in signals if s.source.startswith("autonomy")]
    if not autonomy:
        return 0.0, []
    pressure = max(s.salience for s in autonomy)
    return bounded(max(pressure, min(1.0, len(autonomy) * 0.18))), [s.target_text for s in autonomy[:6]]


def concept_pressure_from_signals(signals: list[AttentionSignalV1], target_text: str) -> float:
    concept_signals = [s for s in signals if s.source.startswith("concept_induction")]
    if any(s.target_text.lower() == target_text.lower() for s in concept_signals):
        return 0.25
    return min(1.0, 0.08 * len(concept_signals))


def build_open_loops(
    *,
    signals: list[AttentionSignalV1],
    ctx: dict[str, Any],
    inputs: dict[str, Any],
    belief_lineage: list[str],
    direct_turn: bool,
    generic_reversal: bool,
    stale_thread_active: bool,
    max_open: int,
) -> list[OpenLoopV1]:
    user_text = compact(ctx.get("user_message") or ctx.get("raw_user_text") or "", 600)
    known = known_blob(inputs, ctx)
    autonomy_value, autonomy_signals = autonomy_pressure_from_signals(signals)
    loops: list[OpenLoopV1] = []
    for signal in merge_signals(signals, limit=max_open):
        phrase = compact(signal.target_text, 120)
        already_known = phrase.lower() in known
        target_type = target_type_for(signal, user_text)
        novelty = 0.18 if already_known else max(0.35, signal.salience)
        continuity = 0.58 if any(token in user_text.lower() for token in ("again", "still", "remember", "continue", "our")) else 0.25
        relational = 0.68 if any(token in user_text.lower() for token in ("my ", "our ", "we ", "juniper", "relationship")) else 0.18
        predictive = 0.72 if target_type in {"plan", "future_event", "anomaly"} or _PLAN_RE.search(user_text) else 0.22
        concept_value = max(concept_pressure_from_signals(signals, phrase), 0.55 if target_type in {"concept", "belief", "anomaly"} else 0.25)
        emotional = 0.65 if _EMOTION_RE.search(user_text) else 0.12
        askability = 0.5 if direct_turn else 0.72
        if generic_reversal or stale_thread_active:
            askability = min(askability, 0.25)
        loop = OpenLoopV1(
            id=stable_id("open-loop", phrase.lower()),
            target_type=target_type,  # type: ignore[arg-type]
            description=phrase,
            source_text=user_text,
            source_refs=list(signal.evidence_refs or ["ctx.user_message"]),
            why_it_matters="novel or unresolved current-turn target with substrate pressure" if not already_known else "current-turn target overlaps known context",
            novelty=bounded(novelty),
            continuity_relevance=bounded(continuity),
            relational_relevance=bounded(relational),
            predictive_value=bounded(predictive),
            concept_value=bounded(concept_value),
            autonomy_value=autonomy_value,
            emotional_charge=bounded(emotional),
            already_known=already_known,
            askability=bounded(askability),
            confidence=bounded(max(signal.confidence, 0.58 if already_known else 0.72)),
            provenance={
                "extractor": "attention_signal_pipeline_v1",
                "signal_id": signal.signal_id,
                "signal_source": signal.source,
                "signal_kind": signal.signal_kind,
                "belief_lineage": list(belief_lineage or [])[:8],
                "autonomy_signals": autonomy_signals,
                **dict(signal.provenance or {}),
            },
        )
        # Always compute + attach the evidence-derived salience features so shadow
        # traces are real even when the v2 flag is off. score_loop decides whether
        # to consume this value or the legacy weighted sum.
        sal, feats = compute_salience(loop=loop, signals=[signal], history=SalienceHistory())
        loop = loop.model_copy(update={"salience": sal, "salience_features": feats.model_dump(mode="json")})
        loops.append(loop)
    return loops


def score_loop(loop: OpenLoopV1) -> float:
    from orion.substrate.attention.salience import salience_v2_enabled

    if salience_v2_enabled():
        return bounded(float(loop.salience))
    raw = (
        loop.novelty * 0.2
        + loop.continuity_relevance * 0.13
        + loop.relational_relevance * 0.12
        + loop.predictive_value * 0.13
        + loop.concept_value * 0.14
        + loop.autonomy_value * 0.16
        + loop.emotional_charge * 0.07
        + loop.askability * 0.05
    )
    if loop.already_known:
        raw *= 0.25
    return bounded(raw * 1.22)
