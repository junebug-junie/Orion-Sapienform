"""Competition behaviour, including the scale-freedom property the whole design rests on."""
from __future__ import annotations

import pytest

from orion.attention.tension.competition import FieldTensionCompetition
from orion.attention.tension.deviation_gate import DeviationGate
from orion.attention.tension.direction_map import DirectionMap

# Both channels resolve "up is worse" via the real suffix rule shape.
_DIRECTIONS = DirectionMap(
    exact={"availability": "down"},
    suffixes=(("_pressure", "up"),),
    unmapped=frozenset({"context_gathering_ratio"}),
)

# Three nodes and three channels, because two-by-two cannot produce anything but
# a Borda tie: with one channel ranking a>b and another ranking b>a, both totals
# are 1.0 no matter what the magnitudes are. The first version of this fixture
# was exactly that shape, which made the scale-freedom test below vacuous.
_OSCILLATION = {
    "node:a": {"cpu_pressure": (0.10, 0.20), "memory_pressure": (0.15, 0.25),
               "gpu_pressure": (0.05, 0.15)},
    "node:b": {"cpu_pressure": (0.30, 0.40), "memory_pressure": (0.05, 0.15),
               "gpu_pressure": (0.20, 0.30)},
    "node:c": {"cpu_pressure": (0.02, 0.06), "memory_pressure": (0.40, 0.50),
               "gpu_pressure": (0.10, 0.20)},
}

# Spike expressed as a multiple of each node's own oscillation range, so the
# within-channel ordering is set deliberately rather than falling out of the
# nodes' differing absolute scales.
#   cpu_pressure    ranks a > b > c
#   memory_pressure ranks a > b > c
#   gpu_pressure    ranks c > b > a   (so scorers genuinely disagree on top-1)
# Borda totals: a=4, b=3, c=2 -- all distinct.
_SPIKE_MULTIPLE = {
    "cpu_pressure": {"node:a": 5.0, "node:b": 3.0, "node:c": 1.5},
    "memory_pressure": {"node:a": 5.0, "node:b": 3.0, "node:c": 1.5},
    "gpu_pressure": {"node:a": 1.5, "node:b": 3.0, "node:c": 5.0},
}


def _tick(values: dict[str, dict[str, float]]) -> dict:
    return {"node_vectors": values}


def _history(scale: dict[str, float] | None = None) -> list[dict]:
    """12 oscillating ticks to build real per-channel variance, then one spike tick.

    `scale` multiplies a channel's values on every node -- the transform whose
    invariance is the point of the test below.
    """
    scale = scale or {}

    def s(channel: str, v: float) -> float:
        return v * scale.get(channel, 1.0)

    ticks = []
    for i in range(12):
        ticks.append(
            _tick(
                {
                    node: {ch: s(ch, pair[i % 2]) for ch, pair in channels.items()}
                    for node, channels in _OSCILLATION.items()
                }
            )
        )

    spike = {}
    for node, channels in _OSCILLATION.items():
        spike[node] = {}
        for ch, (lo, hi) in channels.items():
            spike[node][ch] = s(ch, hi + _SPIKE_MULTIPLE[ch][node] * (hi - lo))
    ticks.append(_tick(spike))
    return ticks


def _run(ticks: list[dict]):
    comp = FieldTensionCompetition(
        gate=DeviationGate(alpha=0.1, z_threshold=1.5, relative_sigma_floor=0.01, warmup=5),
        directions=_DIRECTIONS,
    )
    return [comp.observe_tick(t) for t in ticks]


def test_quiet_ticks_produce_no_ranking_at_all():
    """'Nothing is happening' is a representable state, not a low score."""
    comp = FieldTensionCompetition(
        gate=DeviationGate(warmup=5), directions=_DIRECTIONS
    )
    for _ in range(50):
        result = comp.observe_tick(_tick({"node:a": {"cpu_pressure": 0.4}}))
        assert not result.any_admitted
        assert result.borda is None
        assert result.winner is None


def test_spike_is_admitted_and_ranked():
    results = _run(_history())
    final = results[-1]
    assert final.any_admitted, "the spike tick must admit something"
    assert final.borda is not None
    assert final.winner in {"node:a", "node:b"}


def test_the_scale_freedom_fixture_produces_a_genuinely_ordered_ranking():
    """Guards the test below from going vacuous.

    The first version of the scale-freedom test asserted against a fixture whose
    Borda totals were `{node:a: 1.0, node:b: 1.0}` -- a tie, decided entirely by
    `aggregate_borda`'s alphabetical tiebreak, so it would have matched for
    almost any implementation. Caught in review 2026-08-14. If this assertion
    ever fails, the invariance test below has stopped proving anything.
    """
    final = _run(_history())[-1]
    assert final.borda is not None
    totals = final.borda.totals
    assert len(set(totals.values())) > 1, f"fixture degenerated to a tie: {totals}"


@pytest.mark.parametrize("k", [10.0, 0.1, 1000.0, 0.001])
def test_ranking_is_invariant_under_monotonic_rescaling_of_a_channel(k):
    """The property the whole design rests on.

    Multiplying one channel's values by a constant on every node scales that
    channel's own mu and sigma identically, so its z-scores -- and therefore its
    ballot -- are unchanged. A weighted-sum combiner would NOT survive this: a
    10x larger cpu_pressure would swamp memory_pressure purely from scale.

    **Both directions are tested on purpose.** The original absolute
    `sigma_floor` made this false for k<1 -- review verified that at k=0.1,
    `node:b` dropped out of the `cpu_pressure` ballot entirely because the floor
    began to dominate a channel it had not dominated at k=1. The relative floor
    (`DeviationGate.relative_sigma_floor`) scales with the channel, so the
    invariance now holds in both directions rather than only upward.
    """
    base = _run(_history())[-1]
    scaled = _run(_history(scale={"cpu_pressure": k}))[-1]

    assert base.any_admitted and scaled.any_admitted, "test would be vacuous"
    assert base.borda is not None and scaled.borda is not None
    assert scaled.borda.ranking == base.borda.ranking
    assert scaled.borda.winner == base.borda.winner
    assert scaled.borda.totals == pytest.approx(base.borda.totals)
    # The ballots themselves, not just the aggregate, must be untouched.
    assert set(scaled.admitted) == set(base.admitted)
    for channel, per_node in base.admitted.items():
        assert set(scaled.admitted[channel]) == set(per_node)


def test_unmapped_channel_never_votes():
    comp = FieldTensionCompetition(
        gate=DeviationGate(warmup=2), directions=_DIRECTIONS
    )
    for value in (0.1, 0.1, 0.1, 0.99):
        result = comp.observe_tick(_tick({"node:a": {"context_gathering_ratio": value}}))
    assert result.observed_count == 0
    assert not result.any_admitted


def test_subnormal_values_are_coerced_and_counted():
    comp = FieldTensionCompetition(
        gate=DeviationGate(warmup=2), directions=_DIRECTIONS
    )
    result = comp.observe_tick(_tick({"node:circe": {"cpu_pressure": 3e-321}}))
    assert result.subnormal_count == 1
    assert result.observed_count == 1


def test_non_numeric_and_missing_node_vectors_are_survivable():
    comp = FieldTensionCompetition(directions=_DIRECTIONS)
    assert comp.observe_tick({}).observed_count == 0
    assert comp.observe_tick({"node_vectors": None}).observed_count == 0
    assert comp.observe_tick(_tick({"node:a": {"cpu_pressure": "nope"}})).observed_count == 0
