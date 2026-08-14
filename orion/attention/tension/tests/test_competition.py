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

_OSCILLATION = {
    "node:a": {"cpu_pressure": (0.10, 0.20), "memory_pressure": (0.15, 0.25)},
    "node:b": {"cpu_pressure": (0.30, 0.40), "memory_pressure": (0.05, 0.15)},
}
_SPIKE = {
    "node:a": {"cpu_pressure": 0.90, "memory_pressure": 0.20},
    "node:b": {"cpu_pressure": 0.50, "memory_pressure": 0.60},
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
    ticks.append(
        _tick({node: {ch: s(ch, v) for ch, v in chans.items()} for node, chans in _SPIKE.items()})
    )
    return ticks


def _run(ticks: list[dict]):
    comp = FieldTensionCompetition(
        gate=DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, warmup=5),
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


def test_ranking_is_invariant_under_monotonic_rescaling_of_a_channel():
    """The property the whole design rests on.

    Multiplying one channel's values by a constant on every node scales that
    channel's own mu and sigma identically, so its z-scores -- and therefore its
    ballot -- are unchanged. A weighted-sum combiner would NOT survive this: a
    10x larger cpu_pressure would swamp memory_pressure purely from scale.

    (Holds while sigma stays above `sigma_floor` on both runs, which the
    oscillating fixture guarantees: unscaled sigma is ~0.05 against a 0.02 floor.)
    """
    base = _run(_history())[-1]
    scaled = _run(_history(scale={"cpu_pressure": 10.0}))[-1]

    assert base.any_admitted and scaled.any_admitted, "test would be vacuous"
    assert base.borda is not None and scaled.borda is not None
    assert scaled.borda.ranking == base.borda.ranking
    assert scaled.borda.winner == base.borda.winner
    assert scaled.borda.totals == pytest.approx(base.borda.totals)


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
