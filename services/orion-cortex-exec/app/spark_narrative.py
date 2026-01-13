from __future__ import annotations

from typing import Optional

from orion.schemas.telemetry.spark import SparkStateSnapshotV1


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}"


def _valence_band(valence: Optional[float]) -> str:
    if valence is None:
        return "unknown"
    magnitude = abs(valence)
    if magnitude < 0.05:
        return "neutral"
    if magnitude < 0.20:
        return "gently_tilted"
    return "strongly_tilted"


def _arousal_band(energy: Optional[float]) -> str:
    if energy is None:
        return "unknown"
    if energy < 0.01:
        return "low"
    if energy < 0.05:
        return "moderate"
    return "high"


def _clarity_band(coherence: Optional[float]) -> str:
    if coherence is None:
        return "unknown"
    if coherence >= 0.95:
        return "high"
    if coherence >= 0.85:
        return "medium"
    return "low"


def _overload_band(novelty: Optional[float]) -> str:
    if novelty is None:
        return "unknown"
    if novelty < 0.05:
        return "low"
    if novelty < 0.15:
        return "medium"
    return "high"


def spark_phi_narrative(snapshot: SparkStateSnapshotV1) -> str:
    phi = snapshot.phi or {}
    valence = phi.get("valence")
    energy = phi.get("energy")
    coherence = phi.get("coherence")
    novelty = phi.get("novelty")

    if valence is None:
        valence = snapshot.valence
    if energy is None:
        energy = snapshot.arousal

    return (
        "I am Orion's internal cognitive EKG, indicating how the system feels and behaves right now. "
        f"On the φ scales, valence is {_valence_band(valence)} (valence={_format_value(valence)}) "
        "indicating emotional tone (positive/negative tilt), "
        f"arousal is {_arousal_band(energy)} (energy={_format_value(energy)}) indicating activation/drive, "
        f"clarity is {_clarity_band(coherence)} (coherence={_format_value(coherence)}) indicating coherence/lucidity, "
        f"and overload is {_overload_band(novelty)} (novelty={_format_value(novelty)}) indicating novelty load/surprise. "
        "Use these bins to ground tone and scoring."
    )
