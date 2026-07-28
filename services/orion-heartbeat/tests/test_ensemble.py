from __future__ import annotations

import pytest

from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate, spread_gate
from app.substrate.reconstruction import compute_h1_ensemble
from app.substrate.routing import route_atom


def _small_config(**overrides) -> EnsembleConfig:
    base = dict(n_trajectories=4, gamma=0.2, base_decay_prob=0.15,
                decay_spread_sensitivity=4.0, reheat_strength=0.08, reheat_prob_scale=0.02)
    base.update(overrides)
    return EnsembleConfig(**base)


def _sample_assignment():
    return route_atom(
        source_service="orion-bus", atom_type="signal", confidence=0.7, salience=0.6, uncertainty=0.3
    )


def test_ensemble_rejects_zero_trajectories() -> None:
    # n_trajectories=0 would otherwise silently build an empty ensemble --
    # tick_count()/ratios()/max_bond()/norm() all index self.trajectories[0]
    # or reduce over an empty list, surfacing as an IndexError on the first
    # /health call instead of a clear startup failure.
    with pytest.raises(ValueError, match="n_trajectories"):
        EnsembleSubstrate(config=_small_config(n_trajectories=0), base_seed=1000)


def test_ensemble_construction_gives_distinct_seeds() -> None:
    ensemble = EnsembleSubstrate(config=_small_config(), base_seed=1000)
    assert ensemble.seeds == [1000, 1001, 1002, 1003]
    assert len(ensemble.trajectories) == 4
    # each trajectory really was constructed with its own distinct seed
    assert [t.seed for t in ensemble.trajectories] == ensemble.seeds


def test_ensemble_absorb_applies_to_every_trajectory() -> None:
    ensemble = EnsembleSubstrate(config=_small_config(), base_seed=2000)
    ensemble.absorb(_sample_assignment())
    assert all(t.tick_count == 1 for t in ensemble.trajectories)
    assert ensemble.tick_count() == 1


def test_ensemble_ratios_length_matches_n_trajectories() -> None:
    ensemble = EnsembleSubstrate(config=_small_config(n_trajectories=6), base_seed=3000)
    ratios = ensemble.ratios()
    assert len(ratios) == 6
    assert all(0.0 <= r <= 1.0 for r in ratios)


def test_decay_reheat_tick_does_not_change_tick_count() -> None:
    """Dissipation is a separate concern from real organ traffic -- tick_count
    must keep meaning "how many real atoms this trajectory has absorbed",
    unchanged by decay_reheat_tick()."""
    ensemble = EnsembleSubstrate(config=_small_config(), base_seed=4000)
    ensemble.absorb(_sample_assignment())
    assert ensemble.tick_count() == 1
    for _ in range(20):
        ensemble.decay_reheat_tick(reheat_prob=0.05)
    assert ensemble.tick_count() == 1


def test_decay_reheat_ensemble_is_deterministic_given_same_seeds() -> None:
    """Acceptance Check 6: replaying the same event stream through two
    separately-constructed ensembles with the same base_seed must reproduce
    bit-identical output, despite genuine per-trajectory stochastic jumps --
    this is what makes forensic replay possible."""
    def run() -> list[float]:
        ensemble = EnsembleSubstrate(config=_small_config(), base_seed=5000)
        ensemble.absorb(_sample_assignment())
        for _ in range(15):
            ensemble.decay_reheat_tick(reheat_prob=0.05)
        ensemble.absorb(_sample_assignment())
        for _ in range(15):
            ensemble.decay_reheat_tick(reheat_prob=0.05)
        return ensemble.ratios()

    first = run()
    second = run()
    assert first == second


def test_spread_gate_suppresses_decay_under_disagreement() -> None:
    settled = spread_gate(recent_spread=0.0, base_decay_prob=0.15, sensitivity=4.0)
    disagreeing = spread_gate(recent_spread=0.3, base_decay_prob=0.15, sensitivity=4.0)
    assert settled == 0.15  # exp(0) == 1 -> full decay when trajectories fully agree
    assert disagreeing < settled  # real disagreement suppresses decay


def test_compute_h1_ensemble_shape_and_verdict_consistency() -> None:
    ensemble = EnsembleSubstrate(config=_small_config(), base_seed=6000)
    result = compute_h1_ensemble(ensemble)

    assert result.tick_count == 0
    assert result.seeds == ensemble.seeds
    assert 0.0 <= result.mean_ratio <= 1.0
    assert result.std_ratio >= 0.0
    if result.mean_ratio >= 0.6:
        assert result.verdict == "redundant"
    elif result.mean_ratio <= 0.2:
        assert result.verdict == "concentrated"
    else:
        assert result.verdict == "mixed"
