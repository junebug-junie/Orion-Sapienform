from __future__ import annotations

# services/orion-cortex-exec/app/spark_narrative.py
from __future__ import annotations

from typing import Any, Optional

from orion.schemas.telemetry.spark import SparkStateSnapshotV1


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    return f"{v:.2f}"


def _center_valence(valence: Optional[float]) -> Optional[float]:
    """
    Supports BOTH common conventions:
      - [0,1] with neutral at 0.5  -> center by (v - 0.5)
      - [-1,1] with neutral at 0.0 -> already centered
    Heuristic: if 0<=v<=1, treat as [0,1]; else treat as already-centered.
    """
    if valence is None:
        return None
    if 0.0 <= valence <= 1.0:
        return valence - 0.5
    return valence


def _valence_band(valence: Optional[float]) -> str:
    """
    Band the *tilt magnitude* around neutral.
    Neutral thresholding is done on centered valence.
    """
    centered = _center_valence(valence)
    if centered is None:
        return "unknown"
    magnitude = abs(centered)
    if magnitude < 0.05:
        return "neutral"
    if magnitude < 0.20:
        return "gently_tilted"
    return "strongly_tilted"


def _valence_direction(valence: Optional[float]) -> str:
    centered = _center_valence(valence)
    if centered is None:
        return "unknown"
    if abs(centered) < 0.05:
        return "neutral"
    return "positive" if centered > 0 else "negative"


def _arousal_band(energy: Optional[float]) -> str:
    if energy is None:
        return "unknown"
    # Assume normalized [0, 1] activation/drive
    if energy < 0.33:
        return "low"
    if energy < 0.66:
        return "moderate"
    return "high"


def _clarity_band(coherence: Optional[float]) -> str:
    if coherence is None:
        return "unknown"
    # Assume normalized [0, 1] coherence/lucidity
    if coherence >= 0.80:
        return "high"
    if coherence >= 0.65:
        return "medium"
    return "low"


def _overload_band(novelty: Optional[float]) -> str:
    if novelty is None:
        return "unknown"
    # Assume normalized [0, 1] novelty/surprise load
    if novelty < 0.33:
        return "low"
    if novelty < 0.66:
        return "medium"
    return "high"


def spark_phi_hint(snapshot: SparkStateSnapshotV1) -> dict[str, str]:
    """
    Compact, structured bins suitable for embedding in mirror telemetry hints.
    (Used by prompts and lightweight downstream triage.)

    IMPORTANT: treat this as the single source of truth for banding in prompt logic.
    """
    phi = snapshot.phi or {}

    valence = _as_float(phi.get("valence"))
    energy = _as_float(phi.get("energy"))
    coherence = _as_float(phi.get("coherence"))
    novelty = _as_float(phi.get("novelty"))

    # fallbacks
    if valence is None:
        valence = _as_float(snapshot.valence)
    if energy is None:
        energy = _as_float(snapshot.arousal)

    return {
        "valence_band": _valence_band(valence),
        "valence_dir": _valence_direction(valence),
        "energy_band": _arousal_band(energy),
        "coherence_band": _clarity_band(coherence),
        "novelty_band": _overload_band(novelty),
    }


def spark_phi_narrative(snapshot: SparkStateSnapshotV1) -> str:
    phi = snapshot.phi or {}

    valence = _as_float(phi.get("valence"))
    energy = _as_float(phi.get("energy"))
    coherence = _as_float(phi.get("coherence"))
    novelty = _as_float(phi.get("novelty"))

    if valence is None:
        valence = _as_float(snapshot.valence)
    if energy is None:
        energy = _as_float(snapshot.arousal)

    valence_dir = _valence_direction(valence)
    centered = _center_valence(valence)
    tilt_str = "unknown" if centered is None else f"{centered:+.2f}"

    return (
        "I am Orion's internal cognitive EKG, indicating how the system feels and behaves right now. "
        f"On the φ scales, valence tilt is {_valence_band(valence)}-{valence_dir} "
        f"(valence={_format_value(valence)}, tilt={tilt_str}) indicating emotional tone (positive/negative tilt), "
        f"arousal is {_arousal_band(energy)} (energy={_format_value(energy)}) indicating activation/drive, "
        f"clarity is {_clarity_band(coherence)} (coherence={_format_value(coherence)}) indicating coherence/lucidity, "
        f"and overload is {_overload_band(novelty)} (novelty={_format_value(novelty)}) indicating novelty load/surprise. "
        "Use these bins to ground tone and scoring."
    )
