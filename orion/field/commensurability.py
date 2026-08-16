"""Merge-domination detector: is a low-resolution input masking a
high-resolution one?

R3 of the phase-5 roadmap, PR #1622 (branch docs/phase5-liveness-scope).
Cited by PR number, not path: that doc is not on main as of this commit.
Depends on R1 (PR #1631) for per-tick merge provenance.

THE FAILURE THIS DETECTS
------------------------
Field pressures are merged twice, and both merges are `max()`:

  source -> channel     (collect_field_channel_pressures, across nodes and
                         capabilities)
  channel -> dimension  (map_channels_to_dimensions, across channels)

`max()` does not select the most informative contender. It selects the
highest-scaled one. When contenders are not commensurable -- different natural
ranges, different resolutions, different producers -- a coarse input can win
permanently and the fine one becomes unreachable, while the merged value still
looks alive because the coarse input does vary a little.

Two confirmed instances, both found by a person reading a diff:

- `thermal_pressure` (a 39-distinct-value quantized reading of one CPU core)
  beat capability `pressure` (a 1,895-distinct composite of five live
  channels) on 91.76% of ticks. Fixed 2026-08-11 by deleting one routing
  entry; `resource_pressure` went from 128 to 1,895 distinct values.
- `node:substrate.codebase` wins the merged `prediction_error` channel on
  54.2% of ticks while contributing 5 distinct values in 20,000 -- an
  effective constant of 0.3357, giving the channel a hard floor it can never
  read calm below -- while `node:substrate.biometrics` (1,188 distinct) and
  `node:substrate.bus_synaptic` (341 distinct) win 0% and contribute nothing
  at all. Still open.

Neither was findable without knowing who won each merge, which is why this
needs R1 rather than raw values.

WHY THE TEST IS RELATIVE, NOT ABSOLUTE
--------------------------------------
"Low resolution" has no absolute meaning -- 5 distinct values is fine for a
genuinely binary-ish signal and catastrophic for a continuous one. So the
detector never asks "is this contender coarse". It asks: **did a contender
that carries substantially more information lose every time?**

That comparison is scale-free, needs no calibrated constant per channel, and
is exactly the shape of both confirmed instances (238x and ~48-73x ratios).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
- It does not fix anything. Changing which contender wins moves proposal
  ranking and feedback credit; that is a cognition-loop change and goes
  through proposal mode.
- It does not flag a walkover (one contender) as a defect. 6 of 7 dimensions
  are single-channel today; that is information, not a bug.
- It does not count TIED wins. R1's `winner_is_unique` exists because a tie
  names a winner by dict insertion order, and tallying those would manufacture
  a domination that never happened -- live, `social_pressure` is tied on 100%
  of ticks.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from orion.field.pressure import (
    HIGHER_IS_BETTER_CHANNELS,
    MERGE_STALENESS_THRESHOLD_SEC,
    PRESSURE_CHANNELS,
    _stale_node_channels,
    clamp01,
    field_pressures_with_provenance,
)
from orion.schemas.field_state import FieldStateV1

# A contender must win at least this share of NON-TIED merges to be considered
# dominant. Declared, not derived: it is the plain-language meaning of
# "usually wins", and both confirmed instances (91.76% and 54.2%) clear it by
# a wide margin, so the exact cut is not load-bearing.
DOMINATION_SHARE: float = 0.5

# A losing contender is "masked" when it carries at least this many times the
# winner's distinct-value count.
#
# Declared with its margin stated, rather than tuned: the two confirmed
# instances sit at 238x (codebase 5 vs biometrics 1,188) and roughly 48-73x
# (thermal 18-39 vs pressure 1,325-1,895), depending on the measurement
# window. 10x is an order of magnitude below the nearer of them, so this is
# not fitted to the examples it must catch. Raising it toward 40x would still
# catch both; lowering it below ~3x would start firing on ordinary
# resolution differences between healthy channels.
MASK_RATIO: float = 10.0

# Below this many observed merges, win shares are noise rather than evidence.
# Same reasoning as channel_glossary's RATCHET_MIN_SAMPLES and regime.py's
# MIN_REGIME_SAMPLES: a short window makes any contender look dominant.
MIN_MERGE_SAMPLES: int = 30

# Decay rate applied by services/orion-field-digester/app/digestion/decay.py.
# Duplicated from regime.py's KNOWN_DECAY_RATES rather than imported so R3
# stays usable without R2; kept honest by
# test_decay_rate_matches_the_regime_module.
DECAY_RATE: float = 0.92
DECAY_RATIO_EPSILON: float = 1e-6

LAYER_SOURCE_TO_CHANNEL = "source_to_channel"
LAYER_CHANNEL_TO_DIMENSION = "channel_to_dimension"

VERDICT_DOMINATED = "dominated"
VERDICT_CONTESTED = "contested"
VERDICT_WALKOVER = "walkover"
VERDICT_ALL_TIED = "all_tied"
VERDICT_INSUFFICIENT = "insufficient_samples"


@dataclass(frozen=True)
class Contender:
    """One participant in a merge, over the observed window."""

    name: str
    wins: int
    win_share: float
    # Distinct values this contender itself contributed -- the information it
    # carries, independent of whether it ever won.
    distinct_values: int
    appearances: int


@dataclass(frozen=True)
class MergeAnalysis:
    """One merge point, analysed over a window of ticks."""

    merge_id: str
    layer: str
    sample_count: int
    # Merges excluded from the win tally because the winner was arbitrary.
    tied_count: int
    contenders: tuple[Contender, ...]
    winner: str | None
    winner_share: float
    winner_distinct_values: int
    # Losing contenders carrying >= MASK_RATIO times the winner's distinct
    # values. Non-empty is the finding.
    masked: tuple[Contender, ...] = field(default_factory=tuple)
    verdict: str = VERDICT_INSUFFICIENT

    @property
    def is_finding(self) -> bool:
        return self.verdict == VERDICT_DOMINATED


def analyse_merge(
    merge_id: str,
    layer: str,
    observations: list[tuple[str | None, dict[str, float]]],
) -> MergeAnalysis:
    """Analyse one merge point.

    `observations` is one entry per tick: `(winner_name, {contender: value})`.
    A `winner_name` of None marks a tick whose winner was arbitrary (a tie);
    those still count toward each contender's distinct-value tally -- the
    information is real -- but are excluded from the win share, because
    crediting a dict-order pick as a win is how a phantom domination gets
    manufactured.
    """
    wins: dict[str, int] = defaultdict(int)
    values: dict[str, set[float]] = defaultdict(set)
    appearances: dict[str, int] = defaultdict(int)
    last_value: dict[str, float] = {}
    tied = 0

    for winner, contributions in observations:
        for name, value in contributions.items():
            appearances[name] += 1
            prev = last_value.get(name)
            last_value[name] = value
            # A value the decay loop produced is not information this
            # contender carries. Counting it was a real false positive:
            # node:circe's reasoning_load showed 943 distinct values live, of
            # which 99.8% of transitions were exact 0.92 decay steps -- one
            # producer write of 0.05 decaying toward zero, a fresh float every
            # tick. Counted raw, that outranked every genuine signal in the
            # merge and manufactured a finding.
            if prev is not None and value != prev:
                if prev > 0.0 and abs(value / prev - DECAY_RATE) < DECAY_RATIO_EPSILON:
                    continue
            values[name].add(value)
        if winner is None:
            tied += 1
        else:
            wins[winner] += 1

    decided = len(observations) - tied
    contenders = tuple(
        Contender(
            name=name,
            wins=wins.get(name, 0),
            win_share=(wins.get(name, 0) / decided) if decided else 0.0,
            distinct_values=len(vals),
            appearances=appearances[name],
        )
        for name, vals in sorted(values.items())
    )

    if len(observations) < MIN_MERGE_SAMPLES:
        return MergeAnalysis(
            merge_id=merge_id,
            layer=layer,
            sample_count=len(observations),
            tied_count=tied,
            contenders=contenders,
            winner=None,
            winner_share=0.0,
            winner_distinct_values=0,
            verdict=VERDICT_INSUFFICIENT,
        )

    if len(contenders) < 2:
        return MergeAnalysis(
            merge_id=merge_id,
            layer=layer,
            sample_count=len(observations),
            tied_count=tied,
            contenders=contenders,
            winner=contenders[0].name if contenders else None,
            winner_share=1.0 if contenders else 0.0,
            winner_distinct_values=(
                contenders[0].distinct_values if contenders else 0
            ),
            verdict=VERDICT_WALKOVER,
        )

    if not decided:
        return MergeAnalysis(
            merge_id=merge_id,
            layer=layer,
            sample_count=len(observations),
            tied_count=tied,
            contenders=contenders,
            winner=None,
            winner_share=0.0,
            winner_distinct_values=0,
            verdict=VERDICT_ALL_TIED,
        )

    top = max(contenders, key=lambda c: c.wins)
    if top.win_share < DOMINATION_SHARE:
        return MergeAnalysis(
            merge_id=merge_id,
            layer=layer,
            sample_count=len(observations),
            tied_count=tied,
            contenders=contenders,
            winner=top.name,
            winner_share=top.win_share,
            winner_distinct_values=top.distinct_values,
            verdict=VERDICT_CONTESTED,
        )

    masked = tuple(
        c
        for c in contenders
        if c.name != top.name
        and top.distinct_values > 0
        and c.distinct_values >= top.distinct_values * MASK_RATIO
    )
    return MergeAnalysis(
        merge_id=merge_id,
        layer=layer,
        sample_count=len(observations),
        tied_count=tied,
        contenders=contenders,
        winner=top.name,
        winner_share=top.win_share,
        winner_distinct_values=top.distinct_values,
        masked=masked,
        verdict=VERDICT_DOMINATED if masked else VERDICT_CONTESTED,
    )


def observe_source_to_channel(
    field: FieldStateV1,
    staleness_threshold_sec: float | None = MERGE_STALENESS_THRESHOLD_SEC,
) -> dict[str, tuple[str | None, dict[str, float]]]:
    """Per-channel: who won this tick's source merge, and what each source
    contributed.

    Mirrors `collect_field_channel_pressures()`'s own rules rather than
    re-deriving them: `clamp01`, `min()` for HIGHER_IS_BETTER_CHANNELS and
    `max()` otherwise, the `channel in PRESSURE_CHANNELS or v > 0` gate that
    decides whether a zero is stored at all, and -- via the shared
    `_stale_node_channels()` -- the staleness exclusion.

    That last one is not optional decoration. The first version of this
    function was written before the staleness rule shipped and kept the old
    semantics, so the gate reported `prediction_error` dominated by
    `node:substrate.codebase` at 72.4% AFTER production had stopped merging
    that way. A detector that has drifted from the thing it detects reports
    fixed defects as live ones, which is worse than not running: it teaches
    the reader to ignore it. Default matches the merge's own default so the
    two move together; pass None to analyse pre-2026-08-14 semantics.

    Winner is None when two or more sources tie at the winning value -- see
    the module docstring on why ties must not be counted as wins.
    """
    stale_pairs = (
        set()
        if staleness_threshold_sec is None
        else _stale_node_channels(field, staleness_threshold_sec)
    )
    per_channel: dict[str, dict[str, float]] = defaultdict(dict)
    for source_id, vector in field.node_vectors.items():
        for channel, value in vector.items():
            if (source_id, channel) in stale_pairs:
                continue
            v = clamp01(float(value))
            if channel in HIGHER_IS_BETTER_CHANNELS or channel in PRESSURE_CHANNELS or v > 0:
                per_channel[channel][source_id] = v
    for capability_id, vector in field.capability_vectors.items():
        for channel, value in vector.items():
            v = clamp01(float(value))
            if channel in HIGHER_IS_BETTER_CHANNELS or channel in PRESSURE_CHANNELS or v > 0:
                resolved = field.capability_provenance.get(capability_id, {}).get(
                    channel, capability_id
                )
                per_channel[channel][resolved] = v

    out: dict[str, tuple[str | None, dict[str, float]]] = {}
    for channel, contributions in per_channel.items():
        if not contributions:
            continue
        best = (
            min(contributions.values())
            if channel in HIGHER_IS_BETTER_CHANNELS
            else max(contributions.values())
        )
        at_best = [name for name, v in contributions.items() if v == best]
        out[channel] = (at_best[0] if len(at_best) == 1 else None, dict(contributions))
    return out


def observe_channel_to_dimension(
    field: FieldStateV1,
) -> dict[str, tuple[str | None, dict[str, float]]]:
    """Per-dimension: which channel won, and what each contributed.

    Reads R1's DimensionProvenance directly rather than recomputing the merge,
    including its `winner_is_unique` flag -- which is precisely the tie signal
    this detector must not tally.
    """
    _dims, detail = field_pressures_with_provenance(field)
    out: dict[str, tuple[str | None, dict[str, float]]] = {}
    for dim_id, prov in detail.items():
        contributions = {c.channel: c.value for c in prov.contributors}
        winner = prov.winning_channel if prov.winner_is_unique else None
        out[dim_id] = (winner, contributions)
    return out


def analyse_field_history(
    states: list[FieldStateV1],
) -> list[MergeAnalysis]:
    """Run both merge layers across a window of ticks, oldest first.

    Returns every merge point analysed, not only the findings -- a consumer
    that wants only the problems filters on `.is_finding`, and one that wants
    to show the whole picture has it. Silent top-N truncation is exactly the
    "reads as covered everything when it didn't" failure this roadmap exists
    to prevent.
    """
    layers = (
        (LAYER_SOURCE_TO_CHANNEL, observe_source_to_channel),
        (LAYER_CHANNEL_TO_DIMENSION, observe_channel_to_dimension),
    )
    results: list[MergeAnalysis] = []
    for layer, observe in layers:
        collected: dict[str, list[tuple[str | None, dict[str, float]]]] = defaultdict(list)
        for state in states:
            for merge_id, obs in observe(state).items():
                collected[merge_id].append(obs)
        for merge_id, observations in sorted(collected.items()):
            results.append(analyse_merge(merge_id, layer, observations))
    return results
