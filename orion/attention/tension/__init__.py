"""Field deviation tension sensing.

Read-only: nothing in this package publishes to the bus, registers a schema,
feeds a prompt, or takes an action. It turns the live interoceptive field into a
continuously-varying, scale-free tension signal and ranks it -- no drive
taxonomy, no categories, no hand-authored cross-channel weights.

See `docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md`.
"""
from orion.attention.tension.competition import FieldTensionCompetition, TickResult
from orion.attention.tension.deviation_gate import DeviationGate
from orion.attention.tension.direction_map import (
    DirectionMap,
    DirectionMapError,
    load_direction_map,
)
from orion.attention.tension.field_observations import (
    Observation,
    geometric_decay_ratio,
    iter_observations,
    subnormal_pinned,
)

__all__ = [
    "DeviationGate",
    "DirectionMap",
    "DirectionMapError",
    "FieldTensionCompetition",
    "Observation",
    "TickResult",
    "geometric_decay_ratio",
    "iter_observations",
    "load_direction_map",
    "subnormal_pinned",
]
