"""Voluntary attention override (biased competition, Desimone & Duncan).

An active goal projects a *top-down bias* onto attention candidates (open loops)
and competes against bottom-up salience under a bounded *effort budget*. This is
pure math: deterministic, no LLM, no bus, no I/O. It degrades to a no-op (pure
bottom-up selection = current behavior) when there is no goal, and it NEVER
raises — on any internal error it falls back to the pure-bottom-up result.

Contract:

    result = TopDownBiasCombiner(cfg).apply(
        goal=GoalContext(...) | None,
        loops=[OpenLoopV1, ...],
        bottom_up={loop_id: salience_in_[0,1]},
        agency_readiness=0.0..1.0,
    )

`bottom_up` maps loop.id -> salience in [0,1]; a loop missing from `bottom_up`
is treated as 0.0. The winner is the argmax of the RAW combined salience
(s + gain*applied_bias); an override is recorded only when top-down bias flips
the winner away from the pure bottom-up winner (competition, not fiat).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from orion.schemas.attention_frame import OpenLoopV1, VoluntaryOverrideV1

# Proposal-mode master switch + config env (mirrors salience.py flag pattern).
TOPDOWN_FLAG = "ORION_ATTENTION_TOPDOWN_ENABLED"
TOPDOWN_GAIN_ENV = "ATTENTION_TOPDOWN_GAIN"
TOPDOWN_EFFORT_MAX_ENV = "ATTENTION_EFFORT_MAX"
TOPDOWN_SCALE_BY_AGENCY_ENV = "ATTENTION_EFFORT_SCALE_BY_AGENCY"
_TRUTHY = {"1", "true", "yes", "on"}


def top_down_enabled() -> bool:
    """Default OFF (proposal mode): voluntary attention is inert until enabled."""
    return str(os.getenv(TOPDOWN_FLAG, "false")).strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


# Deterministic map: goal drive_origin -> aligned OpenLoopV1 relevance field.
# Unknown / unmapped drive origins fall back to "concept_value".
_RELEVANCE_FIELD_BY_DRIVE: dict[str, str] = {
    "predictive": "predictive_value",
    "relational": "relational_relevance",
    "continuity": "continuity_relevance",
    "autonomy": "autonomy_value",
    "coherence": "concept_value",
    "capability": "concept_value",
}
_RELEVANCE_FALLBACK_FIELD = "concept_value"


def _clamp01(value: float) -> float:
    """Clamp a value into [0, 1]. Treats non-finite/None as 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass
class TopDownConfig:
    gain: float = 0.6           # ATTENTION_TOPDOWN_GAIN
    effort_max: float = 1.0     # ATTENTION_EFFORT_MAX
    scale_by_agency: bool = True

    @classmethod
    def from_env(cls) -> "TopDownConfig":
        scale = str(os.getenv(TOPDOWN_SCALE_BY_AGENCY_ENV, "true")).strip().lower() in _TRUTHY
        return cls(
            gain=_env_float(TOPDOWN_GAIN_ENV, 0.6),
            effort_max=_env_float(TOPDOWN_EFFORT_MAX_ENV, 1.0),
            scale_by_agency=scale,
        )


@dataclass
class GoalContext:
    drive_origin: str
    priority: float             # [0,1]
    goal_artifact_id: Optional[str] = None


@dataclass
class LoopScore:
    top_down_bias: float        # b(c) = priority * relevance
    applied_bias: float         # min(b, remaining_effort) when this loop was processed
    combined_salience: float    # clamp01(s + gain*applied_bias)


@dataclass
class TopDownResult:
    per_loop: dict            # loop_id -> LoopScore
    override: Optional[VoluntaryOverrideV1]
    effort_used: float
    winner_loop_id: Optional[str]   # argmax by RAW combined (s + gain*applied_bias)


def relevance(goal: GoalContext, loop: OpenLoopV1) -> float:
    """Map the goal's drive_origin to the aligned OpenLoopV1 relevance field.

    Returns a value in [0,1]. Returns 0.0 if the field is missing/None. Unknown
    drive origins fall back to ``concept_value`` (documented).
    """
    try:
        field = _RELEVANCE_FIELD_BY_DRIVE.get(
            getattr(goal, "drive_origin", None), _RELEVANCE_FALLBACK_FIELD
        )
        raw = getattr(loop, field, None)
        if raw is None:
            return 0.0
        return _clamp01(raw)
    except Exception:
        return 0.0


class TopDownBiasCombiner:
    def __init__(self, cfg: Optional[TopDownConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else TopDownConfig()

    def _bottom_up(self, bottom_up: dict[str, float], loop_id: str) -> float:
        return _clamp01(bottom_up.get(loop_id, 0.0))

    def _pure_bottom_up(
        self, loops: list[OpenLoopV1], bottom_up: dict[str, float]
    ) -> TopDownResult:
        """Rule 1 result: no top-down bias applied anywhere."""
        per_loop: dict[str, LoopScore] = {}
        for loop in loops:
            s = self._bottom_up(bottom_up, loop.id)
            per_loop[loop.id] = LoopScore(
                top_down_bias=0.0, applied_bias=0.0, combined_salience=s
            )
        winner = self._argmax_bottom_up(loops, bottom_up)
        return TopDownResult(
            per_loop=per_loop, override=None, effort_used=0.0, winner_loop_id=winner
        )

    def _argmax_bottom_up(
        self, loops: list[OpenLoopV1], bottom_up: dict[str, float]
    ) -> Optional[str]:
        """argmax bottom_up salience; tie-break by loop id (ascending)."""
        best_id: Optional[str] = None
        best_s = float("-inf")
        for loop in loops:
            s = self._bottom_up(bottom_up, loop.id)
            if s > best_s or (s == best_s and (best_id is None or loop.id < best_id)):
                best_s = s
                best_id = loop.id
        return best_id

    def apply(
        self,
        *,
        goal: Optional[GoalContext],
        loops: list[OpenLoopV1],
        bottom_up: dict[str, float],
        agency_readiness: float = 1.0,
    ) -> TopDownResult:
        try:
            loops = list(loops or [])
            bottom_up = bottom_up or {}

            # Rule 1: no goal or no loops -> pure bottom-up.
            if goal is None or not loops:
                return self._pure_bottom_up(loops, bottom_up)

            # Rule 2: effort budget.
            agency = _clamp01(agency_readiness)
            effort_max = float(self.cfg.effort_max)
            E = effort_max * (agency if self.cfg.scale_by_agency else 1.0)
            remaining_E = E
            gain = float(self.cfg.gain)
            priority = _clamp01(goal.priority)

            # Rule 3: raw top-down bias per loop.
            bias_by_id: dict[str, float] = {}
            s_by_id: dict[str, float] = {}
            for loop in loops:
                s_by_id[loop.id] = self._bottom_up(bottom_up, loop.id)
                bias_by_id[loop.id] = _clamp01(priority * relevance(goal, loop))

            # Rule 4: iterate loops in DESCENDING b; tie-break bottom_up desc, then id asc.
            order = sorted(
                loops,
                key=lambda lp: (-bias_by_id[lp.id], -s_by_id[lp.id], lp.id),
            )
            per_loop: dict[str, LoopScore] = {}
            combined_raw_by_id: dict[str, float] = {}
            for loop in order:
                b = bias_by_id[loop.id]
                applied = b if b < remaining_E else remaining_E
                if applied < 0.0:
                    applied = 0.0
                remaining_E -= applied
                s = s_by_id[loop.id]
                combined_raw = s + gain * applied
                combined_raw_by_id[loop.id] = combined_raw
                per_loop[loop.id] = LoopScore(
                    top_down_bias=b,
                    applied_bias=applied,
                    combined_salience=_clamp01(combined_raw),
                )

            # Rule 5: winners.
            winner_bottom_up = self._argmax_bottom_up(loops, bottom_up)
            winner_combined = self._argmax_combined(
                loops, combined_raw_by_id, s_by_id
            )

            # Rule 7: total effort spent.
            effort_used = sum(ls.applied_bias for ls in per_loop.values())

            # Rule 6: override only when top-down flipped the winner.
            override: Optional[VoluntaryOverrideV1] = None
            if (
                winner_combined is not None
                and winner_bottom_up is not None
                and winner_combined != winner_bottom_up
            ):
                override = VoluntaryOverrideV1(
                    goal_artifact_id=goal.goal_artifact_id,
                    goal_drive_origin=goal.drive_origin,
                    chosen_loop_id=winner_combined,
                    beat_loop_id=winner_bottom_up,
                    chosen_bottom_up=s_by_id[winner_combined],
                    beat_bottom_up=s_by_id[winner_bottom_up],
                    applied_bias=per_loop[winner_combined].applied_bias,
                    effort_spent=effort_used,
                )

            return TopDownResult(
                per_loop=per_loop,
                override=override,
                effort_used=effort_used,
                winner_loop_id=winner_combined,
            )
        except Exception:
            # Rule 8: never raise. Fall back to pure bottom-up.
            try:
                return self._pure_bottom_up(list(loops or []), bottom_up or {})
            except Exception:
                return TopDownResult(
                    per_loop={}, override=None, effort_used=0.0, winner_loop_id=None
                )

    def _argmax_combined(
        self,
        loops: list[OpenLoopV1],
        combined_raw_by_id: dict[str, float],
        s_by_id: dict[str, float],
    ) -> Optional[str]:
        """argmax combined_raw; tie-break bottom_up desc, then loop id asc."""
        best_id: Optional[str] = None
        best_key: Optional[tuple[float, float, str]] = None
        for loop in loops:
            c = combined_raw_by_id.get(loop.id, 0.0)
            s = s_by_id.get(loop.id, 0.0)
            # Higher combined, then higher bottom_up, then lower id wins.
            if best_id is None:
                best_id = loop.id
                best_key = (c, s)
                continue
            assert best_key is not None
            if (c, s) > best_key or ((c, s) == best_key and loop.id < best_id):
                best_id = loop.id
                best_key = (c, s)
        return best_id
