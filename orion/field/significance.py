"""Level-aware significance: turns real, per-tick channel regimes into one
[0, 1] scalar for a state `orion.attention.tension` structurally cannot see.

WHY THIS EXISTS (docs/superpowers/specs/2026-08-16-level-aware-significance-
design.md; also see docs/superpowers/specs/2026-08-16-tension-driven-
outreach-design.md's own "What this deliberately does NOT claim" section):
`orion.attention.tension.DeviationGate` is a change-detector -- it z-scores
each observation against an EWMA baseline that itself re-centers toward
whatever the channel currently reads. A channel that has been steadily
overloaded for hours is, BY DESIGN, no longer a deviation from its own
adapted baseline -- it reads calm. Juniper named this precisely: "looks
peaceful but is running high load and that is steady state." This module is
the level-aware half: `orion.field.regime.channel_regime()` (PR #1622/#1633,
live, previously unwired to anything) reads level and dispersion as
SEPARATE axes over a declared window, with NO adaptive baseline at all --
`loaded_steady` is exactly "high level, low dispersion," independent of how
long it has held that way.

SCOPED TO `loaded_steady` ONLY, NOT `loaded_volatile` -- deliberately, per
CLAUDE.md 0A's independence check. A channel that is loaded AND volatile is
already the kind of thing a change-detector (DeviationGate) or a
reconstruction-loss anomaly scorer (`app/anomaly_scorer.py`) can plausibly
catch -- it moves. `loaded_steady` is the one cell neither of those
mechanisms can see: high, and not moving. Voting `loaded_volatile` in here
too would blur this metric back into redundancy with what already exists.

Same "scorers rank targets by their own scale, combined by Borda count, no
hand-tuned cross-scorer exchange rate" shape `orion.attention.tension.
competition` already uses -- reused via `orion.attention.rank_aggregation`,
not reinvented. Ballots come from `orion.attention.tension.field_
observations.iter_observations` (same subnormal-coercing per-(node,channel)
extraction the tension package already uses), not a fresh parse of
`field_json`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orion.attention.rank_aggregation import BordaResult, aggregate_borda
from orion.attention.tension.field_observations import iter_observations
from orion.field.regime import MIN_REGIME_SAMPLES, channel_regime

# A channel votes only when its CURRENT regime is exactly this -- see the
# module docstring's independence-check paragraph for why `loaded_volatile`
# is deliberately excluded.
VOTING_REGIME = "loaded_steady"


@dataclass(frozen=True)
class TickResult:
    """One window's regime read, competed across nodes.

    Mirrors `orion.attention.tension.competition.TickResult`'s shape on
    purpose -- same "admitted ballots -> Borda winner" contract, different
    admission rule (level+dispersion regime, not deviation-from-baseline).
    """

    loaded: dict[str, dict[str, float]]
    """channel -> {node_id: pressure_equivalent_level}, `loaded_steady`-regime
    entries only. A channel with no node currently `loaded_steady` is
    omitted from this dict entirely -- same "a quiet channel contributes no
    points to anyone" convention `orion.attention.tension.competition`
    already documents."""

    borda: BordaResult | None
    """None when nothing is loaded-steady this window -- an honest "no
    sustained load right now", not a ranking over an empty competition."""

    channels_evaluated: int
    channels_loaded_steady: int

    @property
    def any_loaded(self) -> bool:
        return self.channels_loaded_steady > 0

    @property
    def winner(self) -> str | None:
        return self.borda.winner if self.borda is not None else None


def compute_tick(
    field_jsons: list[Mapping[str, Any]], *, window_seconds: float
) -> TickResult:
    """Read one window's `loaded_steady` regime state and Borda-rank the
    nodes carrying it.

    `field_jsons` are real `substrate_field_state.field_json` payloads,
    oldest first -- the same shape `orion.attention.tension.competition.
    FieldTensionCompetition.observe_tick` accepts for a single tick, but
    this function takes a whole WINDOW of them (`channel_regime()` needs a
    real series, not one incremental observation; see this module's
    docstring for why that is a structurally different mechanism from the
    EWMA gate, not a redundant one).
    """
    series: dict[tuple[str, str], list[float]] = {}
    for payload in field_jsons:
        for obs in iter_observations(payload):
            series.setdefault((obs.node_id, obs.channel), []).append(obs.value)

    loaded: dict[str, dict[str, float]] = {}
    channels_evaluated = 0
    channels_loaded_steady = 0
    for (node_id, channel), values in series.items():
        if len(values) < MIN_REGIME_SAMPLES:
            continue
        channels_evaluated += 1
        regime = channel_regime(channel, values, window_seconds=window_seconds)
        if regime.regime == VOTING_REGIME:
            channels_loaded_steady += 1
            loaded.setdefault(channel, {})[node_id] = regime.pressure_equivalent_level

    borda = aggregate_borda(loaded) if loaded else None
    return TickResult(
        loaded=loaded,
        borda=borda,
        channels_evaluated=channels_evaluated,
        channels_loaded_steady=channels_loaded_steady,
    )


def sustained_load_pressure(tick: TickResult) -> float:
    """This window's sustained-load scalar, in [0, 1].

    `pressure_equivalent_level` is already a median over already-clamped-to-
    [0,1] channel pressures (see `orion.field.regime.channel_regime`), so
    unlike `orion.attention.tension.competition.deviation_pressure` (which
    needs a disclosed SATURATION constant to compress an unbounded z-excess
    into [0, 1]) this needs no calibration -- the max across `loaded_steady`
    ballots is already on the right scale. 0.0 on a window with nothing
    `loaded_steady` is a real "no sustained load right now" reading, never a
    fabricated absence -- same convention `deviation_pressure` uses for its
    own quiet-tick 0.0.
    """
    if not tick.any_loaded:
        return 0.0
    top = max(v for channel in tick.loaded.values() for v in channel.values())
    return max(0.0, min(1.0, top))
