"""Field tension competition: admitted deviations -> one ranked ordering of nodes.

The scale-disparity argument this module exists to embody
(`docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md`):

- **Within a channel, over time**, scale is handled by z-scoring against that
  channel's own learned baseline (`DeviationGate`). A thermal rise and a memory
  rise both become "N sigma above this channel's own normal." Dimensionless.
- **Across channels, in one tick**, scale is *not* handled by weights -- because
  there is no exchange rate between thermal degrees and memory fraction, and
  pretending otherwise is what `signal_drive_map.yaml` did before it was deleted.
  Instead each channel ranks the nodes on its own scale and the rankings are
  combined by Borda count (`orion.attention.rank_aggregation`), which is
  commensurable across scorers by construction.

Targets are nodes. Scorers are channels. A channel with nothing admitted this
tick submits an empty ballot, which `aggregate_borda` already treats as
abstention rather than a vote for last place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from orion.attention.rank_aggregation import BordaResult, aggregate_borda
from orion.attention.tension.deviation_gate import DeviationGate
from orion.attention.tension.direction_map import DirectionMap, load_direction_map
from orion.attention.tension.field_observations import iter_observations


@dataclass(frozen=True)
class TickResult:
    """One tick's admission + competition outcome."""

    admitted: dict[str, dict[str, float]]
    """channel -> {node_id: deviation}, admitted entries only."""

    borda: BordaResult | None
    """None when nothing was admitted this tick -- an honest 'nothing is
    happening', not a ranking over an empty competition."""

    observed_count: int
    admitted_count: int
    subnormal_count: int

    @property
    def any_admitted(self) -> bool:
        return self.admitted_count > 0

    @property
    def winner(self) -> str | None:
        return self.borda.winner if self.borda is not None else None


@dataclass
class FieldTensionCompetition:
    """Stateful across ticks -- the gate's baselines are the whole point.

    Feed ticks in chronological order.
    """

    gate: DeviationGate = field(default_factory=DeviationGate)
    directions: DirectionMap = field(default_factory=load_direction_map)

    def observe_tick(self, field_json: Mapping[str, Any]) -> TickResult:
        admitted: dict[str, dict[str, float]] = {}
        observed = 0
        subnormal = 0

        for obs in iter_observations(field_json):
            worse = self.directions.worse_for(obs.channel)
            if worse is None:
                # Unmapped channel: does not train a baseline and does not vote.
                continue
            observed += 1
            if obs.coerced_subnormal:
                subnormal += 1
            deviation = self.gate.observe(
                obs.node_id,
                obs.channel,
                obs.value,
                worse=worse,
            )
            if deviation > 0.0:
                admitted.setdefault(obs.channel, {})[obs.node_id] = deviation

        admitted_count = sum(len(v) for v in admitted.values())
        borda = aggregate_borda(admitted) if admitted_count else None
        return TickResult(
            admitted=admitted,
            borda=borda,
            observed_count=observed,
            admitted_count=admitted_count,
            subnormal_count=subnormal,
        )
