"""Shared evidence-derived salience — the single source of salience truth.

Pure and import-light (schemas + common only) so the thin `orion-thought`
service may import it without dragging the graph engine. Consumed by coalition
selection (`scoring.score_loop`) AND reverie (`derive_salience`).

Hybrid design (spec decision #1/#7): a deterministic feature vector scored by a
tiny linear combiner with hand-seeded weights. The same code path becomes
learned when `refit_salience_weights.py` emits a new `weights_version`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1, SalienceFeaturesV1
from orion.substrate.attention.common import bounded

logger = logging.getLogger("orion.substrate.attention.salience")

_TRUTHY = {"1", "true", "yes", "on"}

SALIENCE_V2_FLAG = "ORION_ATTENTION_SALIENCE_V2_ENABLED"
HABITUATION_FLAG = "ORION_ATTENTION_HABITUATION_ENABLED"
WEIGHTS_OVERRIDE_ENV = "ORION_ATTENTION_SALIENCE_WEIGHTS"

WEIGHTS_VERSION = "seed-v1"

# Normalizers: turn raw counts into bounded [0,1] features.
BREADTH_NORM = 4.0     # distinct detectors/refs to saturate breadth
RECURRENCE_NORM = 5.0  # recent theme appearances to saturate recurrence
DWELL_NORM = 6.0       # dwell ticks to saturate dwell

SEED_WEIGHTS: dict[str, float] = {
    "evidence_strength": 0.30,
    "novelty_vs_known": 0.20,
    "recency": 0.13,
    "recurrence": 0.15,
    "evidence_breadth": 0.12,
    "dwell": 0.10,
    "habituation": -0.35,
}


def salience_v2_enabled() -> bool:
    return str(os.getenv(SALIENCE_V2_FLAG, "false")).strip().lower() in _TRUTHY


def habituation_enabled() -> bool:
    return str(os.getenv(HABITUATION_FLAG, "false")).strip().lower() in _TRUTHY


@dataclass
class SalienceHistory:
    """Runtime history for the recency/recurrence/dwell/habituation features.

    Empty default → those features are 0 (Phase 1 shadow behavior). The broadcast
    producer fills it in Phase 3. `theme_key` maps to a loop id or theme string.
    """

    dwell_ticks: int = 0
    dwelling_loop_id: str | None = None
    recent_theme_counts: dict[str, int] = field(default_factory=dict)
    resonance_theme_keys: set[str] = field(default_factory=set)
    first_seen_at: dict[str, datetime] = field(default_factory=dict)


class LinearSalienceCombiner:
    """Bounded weighted sum with `habituation` as a subtractive penalty."""

    def __init__(self, weights: dict[str, float] | None = None, weights_version: str = WEIGHTS_VERSION):
        self.weights = dict(weights or SEED_WEIGHTS)
        self.weights_version = weights_version

    def score(self, features: SalienceFeaturesV1) -> float:
        data = features.model_dump()
        total = 0.0
        for name, weight in self.weights.items():
            total += weight * float(data.get(name, 0.0))
        return bounded(total)


def default_combiner() -> LinearSalienceCombiner:
    """Combiner from the seeded weights, with optional JSON env override."""
    raw = os.getenv(WEIGHTS_OVERRIDE_ENV, "").strip()
    if not raw:
        return LinearSalienceCombiner()
    try:
        override = json.loads(raw)
        if isinstance(override, dict):
            merged = dict(SEED_WEIGHTS)
            merged.update({str(k): float(v) for k, v in override.items()})
            return LinearSalienceCombiner(merged, weights_version=f"{WEIGHTS_VERSION}+override")
    except (ValueError, TypeError):
        logger.warning("Ignoring malformed %s override: %r", WEIGHTS_OVERRIDE_ENV, raw)
    return LinearSalienceCombiner()


def _recency(theme_key: str, history: SalienceHistory, now: datetime) -> float:
    first = history.first_seen_at.get(theme_key)
    if first is None:
        return 1.0  # never seen before → maximally fresh
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - first).total_seconds() / 3600.0)
    # Half-life ~6h: fresh≈1.0, ~0.5 at 6h, decays toward 0.
    return bounded(0.5 ** (age_hours / 6.0))


def _loop_dwell(theme_key: str, history: SalienceHistory) -> float:
    """Dwell, scoped to the loop actually dwelling -- not every competitor.

    `history.dwell_ticks` counts how long ONE coalition (`dwelling_loop_id`)
    has held the workspace. Previously this scalar was applied to every
    candidate loop scored in a tick, which cannot demote the specific
    dwelling loop relative to its competitors: a uniform per-tick offset
    changes nothing about who wins. Only the loop that IS the dwelling one
    should carry a dwell signal; everyone else gets 0.
    """
    if theme_key != history.dwelling_loop_id:
        return 0.0
    return min(1.0, history.dwell_ticks / DWELL_NORM)


def _habituation(theme_key: str, history: SalienceHistory) -> float:
    recurrence = min(1.0, history.recent_theme_counts.get(theme_key, 0) / RECURRENCE_NORM)
    dwell = _loop_dwell(theme_key, history)
    resonance = 1.0 if theme_key in history.resonance_theme_keys else 0.0
    return bounded(0.5 * recurrence + 0.3 * dwell + 0.2 * resonance)


def compute_features(
    *,
    loop: OpenLoopV1,
    signals: Sequence[AttentionSignalV1],
    history: SalienceHistory | None = None,
    now: datetime | None = None,
) -> SalienceFeaturesV1:
    history = history or SalienceHistory()
    now = now or datetime.now(timezone.utc)
    theme_key = loop.id

    strengths = [float(s.salience) * float(s.confidence) for s in signals]
    evidence_strength = bounded(max(strengths)) if strengths else 0.0

    distinct: set[str] = set()
    for s in signals:
        distinct.add(str(s.source))
        for ref in (s.evidence_refs or []):
            distinct.add(str(ref))
    evidence_breadth = bounded(len(distinct) / BREADTH_NORM)

    recurrence = bounded(history.recent_theme_counts.get(theme_key, 0) / RECURRENCE_NORM)
    recency = _recency(theme_key, history, now)
    novelty_vs_known = 0.15 if loop.already_known else evidence_strength
    dwell = bounded(_loop_dwell(theme_key, history))
    habituation = _habituation(theme_key, history)

    return SalienceFeaturesV1(
        evidence_strength=evidence_strength,
        evidence_breadth=evidence_breadth,
        recurrence=recurrence,
        recency=recency,
        novelty_vs_known=bounded(novelty_vs_known),
        dwell=dwell,
        habituation=habituation,
    )


def compute_salience(
    *,
    loop: OpenLoopV1,
    signals: Sequence[AttentionSignalV1],
    history: SalienceHistory | None = None,
    now: datetime | None = None,
    combiner: LinearSalienceCombiner | None = None,
    apply_habituation: bool | None = None,
) -> tuple[float, SalienceFeaturesV1]:
    """Return (salience, features). `apply_habituation` None → env flag decides."""
    features = compute_features(loop=loop, signals=signals, history=history, now=now)
    if apply_habituation is None:
        apply_habituation = habituation_enabled()
    scored = features if apply_habituation else features.model_copy(update={"habituation": 0.0})
    combiner = combiner or default_combiner()
    return combiner.score(scored), features
