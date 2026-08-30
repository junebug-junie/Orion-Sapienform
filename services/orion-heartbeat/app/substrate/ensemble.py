"""EnsembleSubstrate: N independent HeartbeatSubstrate trajectories, sharing
the same real event stream but each with its own seed-derived relaxation
target and stochastic decay/reheat draws.

Design doc: docs/superpowers/specs/2026-07-28-precision-weighted-attention-
organ-and-heartbeat-discrimination-design.md, "Decision: N-trajectory
stochastic dissipation ensemble".

Why an ensemble and not a single trajectory: a single stochastic trajectory
is the physically wrong unit for quantum-trajectory dissipation -- the
ensemble AVERAGE over many independent trajectories is what recovers
correct dissipative behavior (a real rest state) and what the Monte Carlo
wavefunction method this borrows from is actually for. A single trajectory,
seeded or not, is "one arbitrary sample" either way -- see the design doc's
"Why not a single seeded/unseeded stochastic trajectory" section for the
full reasoning trail (this was arrived at after two rounds of course
correction, not the first idea tried).

Bonus, not the primary justification: the ensemble's cross-trajectory
SPREAD is itself a candidate signal (Missing Question 12) -- confirmed live
2026-07-28 that spread spikes exactly at a busy/quiet transition and
collapses toward zero once trajectories agree (either fully busy or fully
settled), a plausibly more physically-grounded confidence/agreement measure
than AttentionSelfModelV1's raw domain-mean. Not wired into anything yet --
`ratios()`/`std_ratio()` below expose it for a future consumer, this module
does not publish it anywhere itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .mps_state import HeartbeatSubstrate
from .routing import BOUNDARY_BULK_CUT, BOND_DIM, SiteAssignment

logger = logging.getLogger("orion-heartbeat.substrate.ensemble")

_MAX_POSSIBLE_ENTROPY = float(np.log2(BOND_DIM))


def spread_gate(recent_spread: float, base_decay_prob: float, sensitivity: float) -> float:
    """Disagreement among cohabitant trajectories suppresses decay (something
    still unresolved); agreement allows full decay (settled, safe to relax).
    Confirmed live 2026-07-28 against a real busy/quiet/busy stream: this
    genuinely reshapes decay to back off at the ambiguous transition and
    speed back up once trajectories agree -- it does NOT by itself prevent
    eventual full convergence under arbitrarily long silence (the gate's own
    signal, spread, decays toward zero as decay proceeds, removing its own
    brake) -- that residual case is what the bus_synaptic-driven reheat
    below is for. An honest, disclosed heuristic (`exp(-sensitivity *
    spread)`), not a derived control law.
    """
    return base_decay_prob * float(np.exp(-sensitivity * recent_spread))


@dataclass
class EnsembleConfig:
    n_trajectories: int = 8
    gamma: float = 0.2
    base_decay_prob: float = 0.15
    decay_spread_sensitivity: float = 4.0
    reheat_strength: float = 0.08
    reheat_prob_scale: float = 0.02


@dataclass
class EnsembleH1ResultV1:
    """Ensemble-level H1 reading. See reconstruction.py's compute_h1_ensemble
    for how this gets populated -- kept in this module (not reconstruction.py)
    because it's specifically the ensemble's own shape, not a single-
    trajectory H1ResultV1 with fields renamed."""

    mean_ratio: float
    std_ratio: float
    verdict: str
    tick_count: int
    seeds: list[int] = field(default_factory=list)
    # Per-trajectory ratios, index-aligned with `seeds`. 2026-07-30: added so
    # a reader can see the actual N samples the mean/std summarize, not just
    # the two summary statistics -- an ensemble whose trajectories have all
    # collapsed to the same value and one whose spread genuinely averages out
    # are indistinguishable from mean+std alone at small N. Computed anyway
    # by compute_h1_ensemble (it already calls ensemble.ratios()), so this
    # carries zero extra compute; it was simply being discarded.
    ratios: list[float] = field(default_factory=list)
    # Mean normalized entropy at bulk-internal cuts (cuts after the
    # boundary/bulk seam) averaged across trajectories -- a distinct scalar
    # from mean_ratio (which reads only the seam cut). Exposed so downstream
    # reducers can discriminate states that share the same seam ratio but
    # differ in how deeply entanglement has penetrated the bulk block.
    bulk_penetration_depth: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnsembleSubstrate:
    """Owns N HeartbeatSubstrate trajectories. Fans the same real atom out to
    every trajectory's absorb(); each trajectory's own decay_reheat_tick()
    stochastic draws (independently seeded, see mps_state.py) are what gives
    the ensemble genuine, non-clone diversity."""

    def __init__(self, *, config: EnsembleConfig, base_seed: int = 1000) -> None:
        # n_trajectories=0 would silently produce an empty ensemble --
        # tick_count()/ratios()/max_bond()/norm() all index/reduce over
        # self.trajectories, so a misconfigured HEARTBEAT_N_TRAJECTORIES=0
        # would surface as an IndexError on the very first /health call
        # rather than a clear startup failure. Fail loud here instead.
        if config.n_trajectories < 1:
            raise ValueError(f"n_trajectories must be >= 1, got {config.n_trajectories}")
        self.config = config
        self.seeds = [base_seed + i for i in range(config.n_trajectories)]
        self.trajectories: list[HeartbeatSubstrate] = []
        for seed in self.seeds:
            traj = HeartbeatSubstrate(seed=seed)
            traj.compute_relaxation_targets(config.gamma)
            self.trajectories.append(traj)
        self._recent_spread = 0.0
        logger.info(
            "ensemble_substrate_initialized n_trajectories=%d seeds=%s gamma=%.3f",
            config.n_trajectories, self.seeds, config.gamma,
        )

    def absorb(self, assignment: SiteAssignment) -> None:
        for traj in self.trajectories:
            traj.absorb(assignment)
        self._recent_spread = self.std_ratio()

    def decay_reheat_tick(self, reheat_prob: float) -> None:
        """One ensemble-wide dissipation tick: spread-gated decay (using the
        ensemble's spread from the PREVIOUS ratio computation, not this
        tick's -- avoids a same-tick circularity) plus bus_synaptic-driven
        two-site reheat, applied independently per trajectory."""
        eff_decay_prob = spread_gate(
            self._recent_spread, self.config.base_decay_prob, self.config.decay_spread_sensitivity
        )
        for traj in self.trajectories:
            traj.decay_reheat_tick(eff_decay_prob, reheat_prob, self.config.reheat_strength)
        self._recent_spread = self.std_ratio()

    def ratios(self) -> list[float]:
        out = []
        for traj in self.trajectories:
            profile = traj.entropy_profile()
            out.append(max(0.0, min(1.0, profile[BOUNDARY_BULK_CUT - 1] / _MAX_POSSIBLE_ENTROPY)))
        return out

    def mean_ratio(self) -> float:
        return float(np.mean(self.ratios()))

    def std_ratio(self) -> float:
        return float(np.std(self.ratios()))

    def tick_count(self) -> int:
        """All trajectories absorb the same real event stream, so their
        tick_count is always identical -- read from the first as a
        representative, not summed (summing would misleadingly suggest
        N times the real event volume)."""
        return self.trajectories[0].tick_count

    def max_bond(self) -> int:
        """Max bond dimension across all trajectories -- a health check
        (should never exceed routing.BOND_DIM given absorb()/
        decay_reheat_tick() both cap it on every gate), not a mean/typical
        value, matching the single-substrate stats field this replaces."""
        return max(t.max_bond() for t in self.trajectories)

    def norm(self) -> float:
        """Mean norm across trajectories -- each should individually sit at
        ~1.0 after normalize(); this is a cheap aggregate health check, not
        claimed to be meaningful beyond confirming none have drifted."""
        return float(np.mean([t.norm() for t in self.trajectories]))
