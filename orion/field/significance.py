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

SCOPED TO `loaded_steady` ONLY, NOT `loaded_volatile`. The reason is
conceptual, stated plainly as reasoned rather than measured: a channel that
is loaded AND volatile is already the kind of thing a change-detector
(DeviationGate) or a reconstruction-loss anomaly scorer (`app/anomaly_
scorer.py`) can plausibly catch -- it moves. `loaded_steady` is the one cell
neither of those mechanisms can see: high, and not moving.

This is NOT an independence-correlation argument, and the real number does
not support dressing it up as one: measured (`scripts/analysis/measure_
sustained_load_pressure.py --include-volatile`, 24h real replay,
2026-08-18), `loaded_steady`-only correlates with `deviation_pressure` at
r=-0.0313; widening to ALSO count `loaded_volatile` gives r=-0.0021 -- both
are essentially zero. The data does not distinguish the two scopes on
independence grounds; scoping to `loaded_steady` is a real design choice
about what this metric should MEAN (sustained, not spiky), not something the
correlation number itself proves. See `orion/proposals/scoring.py`'s
`PRESSURE_DIMENSIONS` comment for the full write-up.

NOT Borda-ranked, unlike the sibling `orion.attention.tension.competition`
module this reuses `channel_regime()`/`iter_observations()` from. That
module ranks nodes by Borda count because it has a real consumer that needs
node IDENTITY (`tension_borda_winner_target_id`, read by Hub's outreach
trigger) -- and even THERE, `deviation_pressure()`'s own scalar is a plain
`max()` over raw ballots, not derived from the Borda ranking either. This
module ships no winner/target field yet (see the design doc's own
Non-goals -- no consumer needs identity today), so building the Borda
machinery here would be real code with zero real callers. `max()` over
`loaded_steady` ballots directly, same as `deviation_pressure()` already
does for its own scalar. If a real consumer for node identity appears
later, Borda-ranking these same ballots by `pressure_equivalent_level` is
the natural mechanism to add THEN -- same staged precedent
`tension_borda_winner_target_id` itself already set.

Ballots come from `orion.attention.tension.field_observations.
iter_observations` (same subnormal-coercing per-(node,channel) extraction
the tension package already uses), not a fresh parse of `field_json`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orion.attention.tension.field_observations import iter_observations
from orion.field.regime import MIN_REGIME_SAMPLES, channel_regime

# A channel votes only when its CURRENT regime is exactly this -- see the
# module docstring's independence-check paragraph for why `loaded_volatile`
# is deliberately excluded.
VOTING_REGIME = "loaded_steady"


@dataclass(frozen=True)
class TickResult:
    """One window's `loaded_steady` regime read."""

    loaded: dict[str, dict[str, float]]
    """channel -> {node_id: pressure_equivalent_level}, `loaded_steady`-regime
    entries only. A channel with no node currently `loaded_steady` is
    omitted from this dict entirely -- same "a quiet channel contributes no
    points to anyone" convention `orion.attention.tension.competition`
    already documents."""

    channels_evaluated: int
    channels_loaded_steady: int

    @property
    def any_loaded(self) -> bool:
        return self.channels_loaded_steady > 0


def compute_tick(
    field_jsons: list[Mapping[str, Any]],
    *,
    window_seconds: float,
    voting_regimes: frozenset[str] = frozenset({VOTING_REGIME}),
) -> TickResult:
    """Read one window's `loaded_steady` regime state.

    `voting_regimes` defaults to `loaded_steady` only (see module docstring's
    independence-check paragraph). Exposed as a real parameter, not a
    module-constant-you-edit-to-test, specifically so `scripts/analysis/
    measure_sustained_load_pressure.py --include-volatile` can measure the
    wider scope's real correlation against `deviation_pressure` rather than
    the exclusion resting on architectural reasoning alone.

    `field_jsons` are real `substrate_field_state.field_json` payloads,
    oldest first -- the same shape `orion.attention.tension.competition.
    FieldTensionCompetition.observe_tick` accepts for a single tick, but
    this function takes a whole WINDOW of them (`channel_regime()` needs a
    real series, not one incremental observation; see this module's
    docstring for why that is a structurally different mechanism from the
    EWMA gate, not a redundant one).

    KNOWN LIMITATION, disclosed not silently accepted: called with no
    `updated_at`/`window_start`, so `channel_regime()` falls back to
    VALUE-based refresh inference (see that function's own docstring). A
    channel held at an EXACTLY flat value for the entire window reads
    `static` -> `no_new_input`, not `loaded_steady`, even if genuinely
    loaded the whole time -- the one case this metric cannot see. Checked
    against real data, 2026-08-18: the actual live driver
    (disk_capacity_pressure on node:athena) does NOT trigger this today (its
    15-minute window carries 7 distinct values, longest identical run 96 of
    347 samples -- real jitter, not a dead-flat read), but a more coarsely
    quantized or more slowly-refreshed channel could. Fixing this properly
    means threading real per-channel producer-write timestamps through
    (`FieldStateV1.node_vector_updated_at`, the same authoritative path
    `field_channel_glossary_routes.py::build_channel_series()` already
    built for the Hub debug panel) -- real, separate follow-up work, not
    done here to keep this patch a sensing-only slice.
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
        if regime.regime in voting_regimes:
            channels_loaded_steady += 1
            loaded.setdefault(channel, {})[node_id] = regime.pressure_equivalent_level

    return TickResult(
        loaded=loaded,
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
