from __future__ import annotations

from orion.schemas.self_state import SelfStateV1


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def extract_self_state_pressure_snapshot(
    state: SelfStateV1 | None,
    channels: list[str],
) -> dict[str, float]:
    if state is None:
        return {ch: 0.0 for ch in channels}
    out: dict[str, float] = {}
    for ch in channels:
        dim = state.dimensions.get(ch)
        out[ch] = clamp01(dim.score) if dim is not None else 0.0
    return out


def pressure_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}


def classify_pressure_deltas(
    delta: dict[str, float],
    positive_delta_channels: dict[str, str],
) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for channel, direction in positive_delta_channels.items():
        d = delta.get(channel)
        if d is None or abs(d) < 1e-6:
            continue
        if direction == "increase" and d > 0:
            positive.append(f"pressure_delta:{channel}:+{d:.3f}")
        elif direction == "decrease" and d < 0:
            positive.append(f"pressure_delta:{channel}:{d:.3f}")
        elif direction == "increase" and d < 0:
            negative.append(f"pressure_delta:{channel}:{d:.3f}")
        elif direction == "decrease" and d > 0:
            negative.append(f"pressure_delta:{channel}:+{d:.3f}")
    return positive, negative


def normalize_cortex_result_evidence(result: dict[str, object]) -> dict[str, object]:
    """Strip raw blobs; keep status + correlation ids for FeedbackFrameV1."""
    raw_status = result.get("status")
    if raw_status is None and result.get("ok") is not None:
        raw_status = "success" if result.get("ok") else "failed"
    status = str(raw_status or "unknown").lower()
    if status in ("true", "1"):
        status = "success"
    if status in ("false", "0"):
        status = "failed"
    return {
        "result_id": str(result.get("result_id") or result.get("correlation_id") or "unknown"),
        "dispatch_id": str(result.get("dispatch_id") or ""),
        "status": status,
        "evidence_refs": list(result.get("evidence_refs") or []),
    }
