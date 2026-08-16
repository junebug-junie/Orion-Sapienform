"""Field deviation tension sensing.

This package itself is still pure: nothing here publishes to the bus,
registers a schema, feeds a prompt, or takes an action. It turns the live
interoceptive field into a continuously-varying, scale-free tension signal
and ranks it -- no drive taxonomy, no categories, no hand-authored
cross-channel weights.

As of 2026-08-16 it has a live consumer:
`services/orion-field-digester/app/digestion/tension.py` runs
`FieldTensionCompetition` once per digestion tick, persists the gate's
baselines onto `FieldStateV1`, and writes the resulting `deviation_pressure`
scalar (see `competition.deviation_pressure`) into
`orion.proposals.scoring.PRESSURE_DIMENSIONS` -- from there a real proposal
can score on it and dispatch a real, bounded, mutating action. See
`docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md`
(the sensing layer) and
`docs/superpowers/specs/2026-08-16-tension-driven-mutating-dispatch-design.md`
(wiring it to action).
"""
from orion.attention.tension.competition import (
    DEVIATION_PRESSURE_SATURATION,
    FieldTensionCompetition,
    TickResult,
    deviation_pressure,
)
from orion.attention.tension.deviation_gate import DeviationGate
from orion.attention.tension.direction_map import (
    DirectionMap,
    DirectionMapError,
    load_direction_map,
)
from orion.attention.tension.liveness import (
    classify_producer_liveness,
)
from orion.attention.tension.field_observations import (
    Observation,
    geometric_decay_ratio,
    iter_observations,
    subnormal_pinned,
)

__all__ = [
    "DEVIATION_PRESSURE_SATURATION",
    "DeviationGate",
    "DirectionMap",
    "DirectionMapError",
    "FieldTensionCompetition",
    "Observation",
    "classify_producer_liveness",
    "TickResult",
    "deviation_pressure",
    "geometric_decay_ratio",
    "iter_observations",
    "load_direction_map",
    "subnormal_pinned",
]
