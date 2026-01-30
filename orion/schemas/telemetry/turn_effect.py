from __future__ import annotations

from typing import Any, Dict, Optional


_TURN_KEYS = ("valence", "energy", "coherence", "novelty")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num > 1.0:
        return 1.0
    if num < -1.0:
        return -1.0
    return num


def _delta_block(before: Dict[str, Any], after: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    delta: Dict[str, float] = {}
    for key in _TURN_KEYS:
        before_val = _coerce_float(before.get(key))
        after_val = _coerce_float(after.get(key))
        if before_val is None or after_val is None:
            continue
        diff = after_val - before_val
        if diff > 1.0:
            diff = 1.0
        if diff < -1.0:
            diff = -1.0
        delta[key] = diff
    return delta or None


def _evidence_block(block: Any) -> Optional[Dict[str, float]]:
    if not isinstance(block, dict):
        return None
    out: Dict[str, float] = {}
    for key in _TURN_KEYS:
        val = _coerce_float(block.get(key))
        if val is None:
            continue
        out[key] = float(val)
    return out or None


def turn_effect_from_spark_meta(spark_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(spark_meta, dict):
        return None

    before = spark_meta.get("phi_before") or {}
    after = spark_meta.get("phi_after") or {}
    post_before = spark_meta.get("phi_post_before") or {}
    post_after = spark_meta.get("phi_post_after") or {}

    effect: Dict[str, Any] = {}

    user_delta = _delta_block(before, after)
    if user_delta:
        effect["user"] = user_delta

    assistant_delta = _delta_block(post_before, post_after)
    if assistant_delta:
        effect["assistant"] = assistant_delta

    turn_delta = _delta_block(before, post_after)
    if turn_delta:
        effect["turn"] = turn_delta

    evidence: Dict[str, Dict[str, float]] = {}
    for label, block in (
        ("phi_before", before),
        ("phi_after", after),
        ("phi_post_before", post_before),
        ("phi_post_after", post_after),
    ):
        dims = _evidence_block(block)
        if dims:
            evidence[label] = dims
    if evidence:
        effect["evidence"] = evidence

    return effect or None


def summarize_turn_effect(effect: Dict[str, Any]) -> str:
    if not isinstance(effect, dict):
        return "none"

    def _fmt(val: float) -> str:
        return f"{val:+.2f}"

    parts = []
    for label in ("user", "assistant", "turn"):
        section = effect.get(label)
        if not isinstance(section, dict):
            continue
        entries = []
        for key, abbrev in (
            ("valence", "v"),
            ("energy", "e"),
            ("coherence", "c"),
            ("novelty", "n"),
        ):
            val = section.get(key)
            if isinstance(val, (int, float)):
                entries.append(f"{abbrev}{_fmt(float(val))}")
        if entries:
            parts.append(f"{label}: " + " ".join(entries))

    return "; ".join(parts) if parts else "none"


def compute_deltas_from_turn_effect(effect: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    if not isinstance(effect, dict):
        return {}

    def _extract(section: Any) -> Dict[str, float]:
        if not isinstance(section, dict):
            return {}
        out: Dict[str, float] = {}
        for key in _TURN_KEYS:
            val = _coerce_float(section.get(key))
            if val is None:
                continue
            out[key] = float(val)
        return out

    deltas: Dict[str, Dict[str, float]] = {}
    for label in ("user", "assistant", "turn"):
        section = _extract(effect.get(label))
        if section:
            deltas[label] = section
    return deltas


def evaluate_turn_effect_alert(
    effect: Dict[str, Any],
    *,
    coherence_drop: float,
    valence_drop: float,
    novelty_spike: float,
) -> Optional[Dict[str, float]]:
    if not isinstance(effect, dict):
        return None

    turn = effect.get("turn")
    if not isinstance(turn, dict):
        return None

    coherence = _coerce_float(turn.get("coherence"))
    if coherence is not None and coherence <= -abs(coherence_drop):
        return {"metric": "coherence_drop", "value": coherence, "threshold": -abs(coherence_drop)}

    valence = _coerce_float(turn.get("valence"))
    if valence is not None and valence <= -abs(valence_drop):
        return {"metric": "valence_drop", "value": valence, "threshold": -abs(valence_drop)}

    novelty = _coerce_float(turn.get("novelty"))
    if novelty is not None and novelty >= abs(novelty_spike):
        return {"metric": "novelty_spike", "value": novelty, "threshold": abs(novelty_spike)}

    return None


def should_emit_turn_effect_alert(
    last_seen_ts: Optional[float],
    now_ts: float,
    cooldown_sec: float,
) -> bool:
    if last_seen_ts is None:
        return True
    return (now_ts - float(last_seen_ts)) >= float(cooldown_sec)
