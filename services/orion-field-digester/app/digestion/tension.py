"""Producer-side update for `orion.attention.tension`'s deviation gate.

See `orion/schemas/field_state.py`'s `tension_baseline_*`/`tension_deviation_
pressure` docstrings and `docs/superpowers/specs/2026-08-16-tension-driven-
mutating-dispatch-design.md` for the full account. This module only wires the
already-pure `orion.attention.tension.FieldTensionCompetition` into this
service's per-tick digestion pipeline, same shape as
`app/digestion/precision.py::update_dimension_precision_baseline` -- rebuild
a stateful object from this tick's persisted baseline, run one tick through
it, write the result and the new baseline back.
"""

from __future__ import annotations

from functools import lru_cache

from orion.attention.tension import DirectionMap, FieldTensionCompetition, deviation_pressure, load_direction_map
from orion.schemas.field_state import FieldStateV1

# ASCII unit separator: real node_ids already contain colons and dots
# ("node:substrate.route"), ruling those out as a flat-key delimiter. See
# FieldStateV1.tension_baseline_mu's own docstring.
_KEY_SEP = "\x1f"


# 2026-08-16, code review finding: `FieldTensionCompetition()`'s default
# `directions` field re-reads and re-parses `config/attention/
# channel_direction_map.yaml` from disk on every construction. Building a
# fresh competition every ~2s digestion tick (this module's only caller)
# means every tick paid that disk read + YAML parse for no reason -- the
# direction map is static config, not per-tick state, unlike the gate
# baselines below which genuinely must be rehydrated from `state` every
# tick. Cached once per process; `maxsize=1` because this takes no arguments
# and there is exactly one direction map to load.
@lru_cache(maxsize=1)
def _cached_direction_map() -> DirectionMap:
    return load_direction_map()


def _load_gate_state(state: FieldStateV1) -> dict[tuple[str, str], tuple[float, float, int]]:
    out: dict[tuple[str, str], tuple[float, float, int]] = {}
    for flat_key, mu in state.tension_baseline_mu.items():
        node_id, sep, channel = flat_key.partition(_KEY_SEP)
        if not sep:
            # Malformed key (no separator) -- skip rather than raise. A
            # producer restart must degrade to "cold start for this key", not
            # crash the whole digestion tick.
            continue
        var = state.tension_baseline_var.get(flat_key, 0.0)
        count = state.tension_baseline_n.get(flat_key, 0)
        out[(node_id, channel)] = (mu, var, count)
    return out


def _dump_gate_state(
    state: FieldStateV1, baselines: dict[tuple[str, str], tuple[float, float, int]]
) -> None:
    mu: dict[str, float] = {}
    var: dict[str, float] = {}
    n: dict[str, int] = {}
    for (node_id, channel), (b_mu, b_var, b_count) in baselines.items():
        flat_key = f"{node_id}{_KEY_SEP}{channel}"
        mu[flat_key] = b_mu
        var[flat_key] = b_var
        n[flat_key] = b_count
    state.tension_baseline_mu = mu
    state.tension_baseline_var = var
    state.tension_baseline_n = n


def update_tension_pressure(state: FieldStateV1) -> FieldStateV1:
    """Run one real digestion tick through the deviation gate and record the
    resulting `deviation_pressure` scalar (+ the gate's updated baselines) on
    `state`.

    Called from `run_digestion_tick()` AFTER `apply_perturbations()` /
    `apply_decay()` / `apply_diffusion()` / `apply_suppression()` and BEFORE
    `update_dimension_precision_baseline()` -- the latter reads this tick's
    final `field_pressures(state)`, which must already include this tick's
    `deviation_pressure` for its own EWMA baseline to track the real signal
    rather than being permanently one tick stale.

    `state.node_vectors` is fed directly as the tension package's
    `field_json["node_vectors"]` shape -- already `dict[str, dict[str,
    float]]`, no serialization needed for a live tick (unlike the offline
    measurement scripts, which replay `field_json` recovered from Postgres).
    """
    competition = FieldTensionCompetition(directions=_cached_direction_map())
    competition.import_baselines(_load_gate_state(state))
    tick = competition.observe_tick({"node_vectors": state.node_vectors})
    state.tension_deviation_pressure = deviation_pressure(tick)
    # See FieldStateV1.tension_borda_winner_target_id's own docstring for why
    # this is written now (2026-08-16) rather than at the same time as the
    # scalar above -- a real consumer exists.
    state.tension_borda_winner_target_id = tick.winner
    _dump_gate_state(state, competition.export_baselines())
    return state
