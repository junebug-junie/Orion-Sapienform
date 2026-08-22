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

Targets are nodes. Scorers are channels. **A channel with nothing admitted this
tick is omitted from the ballot set entirely** -- `admitted` only gains a key when
something is admitted -- so a quiet channel contributes no points to anyone.

(An earlier version of this docstring claimed such a channel "submits an empty
ballot, which `aggregate_borda` treats as abstention." Both halves were wrong and
were corrected in review 2026-08-14: no ballot is submitted at all, and an
actually-empty ballot passed to `_borda_points_for_scorer` sorts every target to
`-inf` and hands each the same `(n-1)/2` points -- harmless only because it is a
constant offset, not because it is abstention. Noted for any future caller that
does pass empty ballots explicitly.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from orion.attention.rank_aggregation import BordaResult, aggregate_borda
from orion.attention.tension.deviation_gate import DeviationGate
from orion.attention.tension.direction_map import DirectionMap, load_direction_map
from orion.attention.tension.field_observations import iter_observations

# 2026-08-16 (docs/superpowers/specs/2026-08-16-tension-driven-mutating-
# dispatch-design.md): turns one tick's admitted deviations into the single
# [0, 1] scalar `orion.proposals.scoring.PRESSURE_DIMENSIONS` needs -- Borda
# rank discards magnitude by design (see this module's own docstring), so a
# scalar "how much" has to come from the admission gate, not the ranking.
#
# `top_admitted_deviation` is the largest single admitted (channel, node)
# excess this tick, in z-units past `DeviationGate.z_threshold` (0 at the
# threshold, unbounded above). 10.0 is a disclosed, uncalibrated saturation
# point -- not derived from a distribution fit, chosen the same way
# `RECENT_PERTURBATION_ZSCORE_SATURATION`'s "z>=3.0 is anomalous" convention
# was, but scaled to *excess past threshold* rather than raw z (an excess of
# 10 is z~=11.5 at the default z_threshold=1.5, a genuinely extreme
# admission). Live-checked over 41,973 real substrate_field_state ticks (24h,
# 2026-08-16): 49.9% nonzero, 18,699 distinct values, population variance
# 6.164936e-02, mean 0.1134, 4,357 rest->active rises (recurring refresh, not
# a decayed-unopposed climb) -- clears the CLAUDE.md 0A metric gate's
# non-degenerate-variance and can-read-genuine-calm bars (median exactly
# 0.0). Revisit against live post-deploy admission data, same discipline as
# every other disclosed constant in this arc.
DEVIATION_PRESSURE_SATURATION = 10.0


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

    # Delegate to the gate -- see DeviationGate.export_baselines/import_baselines
    # for the persistence contract. Kept here too so a live producer only needs
    # to know about FieldTensionCompetition, not reach into `.gate` itself.
    def export_baselines(self) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
        return self.gate.export_baselines()

    def import_baselines(self, data: Dict[Tuple[str, str], Tuple[float, float, int]]) -> None:
        self.gate.import_baselines(data)


def deviation_pressure(
    tick: TickResult, *, saturation: float = DEVIATION_PRESSURE_SATURATION
) -> float:
    """This tick's admitted deviation, collapsed to a single [0, 1] scalar.

    0.0 on a quiet tick is a real "nothing is happening" reading (see this
    module's docstring's "honest limit" note), never a fabricated absence.
    """
    if not tick.any_admitted:
        return 0.0
    top = max(v for channel in tick.admitted.values() for v in channel.values())
    return max(0.0, min(1.0, top / saturation))
