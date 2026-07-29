"""HeartbeatSubstrate: the actual tensor-network state.

Design doc: docs/superpowers/specs/2026-07-24-spark-field-holographic-lattice-design.md,
"Heartbeat v0 -- scoped design" / "Update dynamics -- resolved".

v0 deliberately does NOT implement the 2026-05-01 heartbeat charter's full
active-inference free-energy minimization (variational sweeps, a generative
model of expected boundary states, precision-weighting) -- that machinery is
real and non-trivial and is mainly justified by H4 (predictive-surprise
dynamics), which v0 doesn't test. What H1 (boundary reconstruction /
entanglement structure) actually needs is simpler: *some* local entangling
update so bulk sites become genuinely coupled to boundary evidence over
ticks. Without that, bulk sites would sit at their random initialization
forever and H1 would measure nothing.

Update rule per absorbed atom (`absorb()`):
  1. A local 1-site unitary at the atom's routed site (routing.SiteAssignment),
     parameterized by confidence*salience -- the "absorption" step.
  2. **Fixed post-review (2026-07-24)**: a *chain* of 2-site entangling
     unitaries from the atom's routed site through every remaining site to
     the end of the chain (site, site+1), (site+1, site+2), ..., not just
     one hop. The original single-hop version left 4 of 5 declared bulk
     sites permanently untouched by any dynamics regardless of how much
     traffic was absorbed -- confirmed empirically by an independent review
     pass (entropy at cuts 6/7/8 froze within ~50 absorptions and never
     moved again through 800). Strength decays geometrically per hop
     (`_HOP_DECAY`) so the effect is strongest near the organ and weaker,
     but never exactly zero, at greater chain-distance -- every absorbed
     atom now touches every bulk site at least a little. Adjacent-site gates
     use `contract='swap+split'`, confirmed working during this session's
     quimb API verification.
  3. Bond dimension is explicitly capped at BOND_DIM on every gate
     (max_bond=BOND_DIM, cutoff=0.0) -- without this an MPS's bond dimension
     grows unboundedly under repeated 2-site gates. Kept at the charter's own
     conservative value: a bigger bond dimension would make the H1
     entanglement measurement trivially larger, which would bias toward a
     false-positive "redundant" finding -- capping it is part of what makes
     the measurement meaningful, not just a performance concern.

Generators (the Hermitian operators exponentiated into unitaries) are fixed
and seeded once at import time, not re-randomized per atom or per process
restart -- this keeps `absorb()` deterministic given the same event stream,
matching this service's own charter requirement
(docs/superpowers/specs/2026-05-01-orion-heartbeat-engineering-spec.md:292,
"Reducers must be deterministic and side-effect-free"). **Correction
2026-07-28**: this docstring previously cited that requirement as coming
from "services/orion-substrate-runtime's reducer contract" -- checked
directly, that service has zero occurrences of "deterministic"/"side-effect"
anywhere in its own code. The requirement is this service's own, not a
cross-service convention.

**2026-07-28: N-trajectory dissipation ensemble added** (see `ensemble.py`
and docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-
and-heartbeat-discrimination-design.md). `absorb()`'s original problem stands
on its own and is unchanged by this addition: it's what makes bulk sites
genuinely couple to boundary evidence. What was missing is a way back down --
confirmed live 2026-07-28 that a single `HeartbeatSubstrate`, absorbing real
`orion:grammar:event` traffic with never any decay, thermalizes to a
permanent near-ceiling boundary/bulk ratio (quantum-chaotic thermalization,
Page's-theorem territory -- not a threshold-tuning bug, a missing mechanism).
`decay_reheat_tick()` below adds that mechanism: a deterministic relaxation
toward this trajectory's own seed-derived initial state (not a shared target
-- preserves inter-trajectory diversity, see `ensemble.py`), balanced against
a small two-site entangling "reheat" driven by real live `bus_synaptic` data
(a single-site reheat gate was tried first and is PROVABLY incapable of
moving entanglement entropy across any cut -- confirmed the hard way with a
null-result spike before correcting to two-site gates). Neither mechanism is
claimed to be rigorously-derived open-quantum-system physics -- both are
honest, disclosed heuristics, same "documented choice, not derivation" spirit
as `_HOP_DECAY`/`_MIN_STRENGTH`/`_MAX_STRENGTH` above.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.linalg as sla
import quimb.tensor as qtn

from .routing import BOND_DIM, N_SITES, PHYS_DIM, OperatorKind, SiteAssignment

logger = logging.getLogger("orion-heartbeat.substrate")

_GENERATOR_SEED = 20260724  # fixed, not re-derived per process -- see module docstring

_OPERATOR_KINDS: tuple[OperatorKind, ...] = ("amplitude", "phase", "rotation", "projection")


def _hermitian(rng: np.random.Generator, dim: int) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (a + a.conj().T) / 2.0


def _build_generators() -> tuple[dict[OperatorKind, np.ndarray], dict[OperatorKind, np.ndarray]]:
    rng = np.random.default_rng(_GENERATOR_SEED)
    one_site = {kind: _hermitian(rng, PHYS_DIM) for kind in _OPERATOR_KINDS}
    two_site = {kind: _hermitian(rng, PHYS_DIM * PHYS_DIM) for kind in _OPERATOR_KINDS}
    return one_site, two_site


_ONE_SITE_GENERATORS, _TWO_SITE_GENERATORS = _build_generators()

# Bounds on gate "strength" (the rotation angle in expm(1j * strength * G)).
# Kept small and bounded so a single high-confidence/high-salience atom can't
# swing the substrate arbitrarily far in one step -- deliberately conservative
# defaults for a v0 that has not been tuned against any real data yet.
_MIN_STRENGTH = 0.05
_MAX_STRENGTH = 0.45

# Per-hop decay applied to the 2-site entangling gate's strength as it
# propagates from the organ's boundary site toward the far end of the chain
# (see absorb()'s post-review fix). 0.7 is a bounded, deterministic default,
# not tuned against any real data -- same "documented choice, not derivation"
# discipline as _MIN_STRENGTH/_MAX_STRENGTH above.
_HOP_DECAY = 0.7


def _strength(a: float, b: float) -> float:
    a = max(0.0, min(1.0, a))
    b = max(0.0, min(1.0, b))
    return _MIN_STRENGTH + (_MAX_STRENGTH - _MIN_STRENGTH) * a * b


def site_reduced_density_matrix(mps: qtn.MatrixProductState, site: int) -> np.ndarray:
    """Dense single-site reduced density matrix, PHYS_DIM x PHYS_DIM. Cheap
    (single-site, not the expensive multi-site dense partial trace
    reconstruction.py's module docstring warns about) -- confirmed live
    2026-07-28 via quimb 1.14.0's `make_reduced_density_matrix([site])`
    (renamed from the deprecated `partial_trace` in this quimb version),
    contracted and read out via `.to_dense(['k{site}'], ['b{site}'])`."""
    rho_tn = mps.make_reduced_density_matrix([site])
    contracted = rho_tn.contract()
    return np.asarray(contracted.to_dense([f"k{site}"], [f"b{site}"]))


class HeartbeatSubstrate:
    """Owns one quimb MatrixProductState and applies the v0 update rule."""

    def __init__(self, *, seed: int = 42) -> None:
        self._mps = qtn.MPS_rand_state(
            L=N_SITES, bond_dim=BOND_DIM, phys_dim=PHYS_DIM, dtype="complex128", seed=seed
        )
        self._mps.normalize()
        self.tick_count = 0
        self.seed = seed
        # Own seeded RNG for decay/reheat stochastic draws -- NOT the global
        # np.random state. Reusing the same seed as the MPS's own
        # initialization keeps this trajectory fully, deterministically
        # self-contained ("this seed" determines both the initial state and
        # every stochastic jump this trajectory will ever make), which is
        # what Acceptance Check 6 (replay with logged seeds reproduces the
        # exact ensemble output) actually requires. Caught in review before
        # this shipped: an earlier draft used bare `np.random.random()`,
        # shared mutable global state whose sequence depends on call order
        # across all N trajectories -- would have silently broken
        # reproducibility for every trajectory after the first.
        self._rng = np.random.default_rng(seed)
        # Relaxation targets are NOT computed here -- __init__ must stay free
        # of any behavior that would make test_compute_h1_on_fresh_substrate_
        # is_not_degenerate's "zero absorptions, zero decay/reheat applied"
        # assumption false. Call compute_relaxation_targets() explicitly once,
        # right after construction, before any decay_reheat_tick() call.
        self._relaxation_ops: list[np.ndarray] | None = None

    def compute_relaxation_targets(self, gamma: float) -> None:
        """Per-site relaxation operator, one per site, each damping THIS
        trajectory's own site toward the dominant eigenvector of that site's
        reduced density matrix AT THE MOMENT THIS IS CALLED (normally right
        after construction, on the fresh random initial state) -- not a
        shared fixed target across trajectories. Confirmed live 2026-07-28:
        damping every trajectory toward the same fixed basis vector collapses
        an ensemble's cross-trajectory diversity to near-zero within a few
        hundred ticks of sustained quiet; per-trajectory (seed-derived)
        targets measurably preserve diversity through the meaningful part of
        any relaxation curve instead.

        Implementation: rotate into the basis where that dominant eigenvector
        is the first basis state, apply the standard "damp everything except
        the surviving component" diagonal form, rotate back -- the same
        mathematical shape as textbook amplitude-damping-toward-|0>, just
        toward an arbitrary reference state instead of a fixed one. An
        honest, disclosed heuristic (see module docstring), not a physically
        rigorous open-system derivation.
        """
        phys_dim = PHYS_DIM
        D = np.diag([1.0] + [1.0 - gamma] * (phys_dim - 1)).astype("complex128")
        ops: list[np.ndarray] = []
        for site in range(N_SITES):
            rho0 = site_reduced_density_matrix(self._mps, site)
            eigvals, eigvecs = np.linalg.eigh(rho0)
            order = np.argsort(eigvals)[::-1]  # dominant eigenvector first
            U = eigvecs[:, order].conj().T
            ops.append(U.conj().T @ D @ U)
        self._relaxation_ops = ops

    def decay_reheat_tick(self, decay_prob: float, reheat_prob: float, reheat_strength: float) -> None:
        """One dissipation-ensemble tick: per-site stochastic relaxation
        toward this trajectory's own target (decay_prob per site), plus
        per-bond stochastic two-site entangling "reheat" (reheat_prob per
        bond) -- see module docstring for why reheat must be two-site.
        Independent of `absorb()` and does NOT increment `tick_count`
        (that counter tracks real organ traffic, not dissipation-ensemble
        housekeeping -- keeps `test_compute_h1_tick_count_reflects_
        absorptions` meaningful unchanged).

        Stochastic draws come from `self._rng` (seeded from this
        trajectory's own construction seed, see `__init__`), not the global
        `np.random` state -- this instance is a fully self-contained,
        deterministically-replayable unit given its seed and the sequence of
        `absorb()`/`decay_reheat_tick()` calls it receives.
        """
        if self._relaxation_ops is None:
            raise RuntimeError("compute_relaxation_targets() must be called before decay_reheat_tick()")

        for site in range(N_SITES):
            if self._rng.random() < decay_prob:
                self._mps.gate_(self._relaxation_ops[site], site, contract=True)
        for left in range(N_SITES - 1):
            if self._rng.random() < reheat_prob:
                G2 = _TWO_SITE_GENERATORS["amplitude"]
                U2 = sla.expm(1j * reheat_strength * G2).reshape(PHYS_DIM, PHYS_DIM, PHYS_DIM, PHYS_DIM)
                self._mps.gate_(
                    U2, (left, left + 1), contract="swap+split", max_bond=BOND_DIM, cutoff=0.0,
                )
        self._mps.normalize()

    def absorb(self, assignment: SiteAssignment) -> None:
        """Apply one atom's absorption-and-entangle update. See module docstring.

        **Fixed post-review (2026-07-24)**: the original version only ever
        gated the pair `(site, site+1)` -- i.e. one hop. For an organ at
        site 4 (route) that reaches bulk site 5, but sites 6-9 were never
        touched by ANY gate, ever, regardless of how many atoms were
        absorbed or from which organ. Confirmed live by an independent
        review pass: cuts 6/7/8 froze at fixed values within ~50 absorptions
        of mixed multi-organ traffic and never moved again through 800
        absorptions -- 4 of 5 declared bulk sites were inert scaffolding,
        directly contradicting this module's own "bulk sites become
        genuinely coupled to boundary evidence" claim. Fixed by propagating
        a full chain of entangling gates from the organ's site through to
        the end of the chain, strength decaying geometrically per hop so
        the effect is strongest near the organ and weaker (but never
        exactly zero) at greater chain-distance -- every absorbed atom now
        touches every site at least a little, not just its nearest
        neighbor.
        """
        site = assignment.site_index
        if not (0 <= site < N_SITES - 1):
            # Every v0 organ site (routing.ORGAN_SITE_MAP) is 0-4, always
            # leaving at least one right-neighbor hop within 0-9 -- this
            # branch is unreachable with the current routing table, kept as
            # a guard against a future routing change rather than silently
            # mis-indexing.
            raise ValueError(f"site {site} has no right-neighbor within N_SITES={N_SITES}")

        strength_1 = _strength(assignment.confidence, assignment.salience)
        G1 = _ONE_SITE_GENERATORS[assignment.operator_kind]
        U1 = sla.expm(1j * strength_1 * G1)
        self._mps.gate_(U1, site, contract=True)

        base_strength = _strength(assignment.confidence, 1.0 - assignment.uncertainty)
        G2 = _TWO_SITE_GENERATORS[assignment.operator_kind]
        hop_strength = base_strength
        for left in range(site, N_SITES - 1):
            U2 = sla.expm(1j * hop_strength * G2).reshape(PHYS_DIM, PHYS_DIM, PHYS_DIM, PHYS_DIM)
            self._mps.gate_(
                U2,
                (left, left + 1),
                contract="swap+split",
                max_bond=BOND_DIM,
                cutoff=0.0,
            )
            hop_strength *= _HOP_DECAY

        self._mps.normalize()
        self.tick_count += 1

    def entropy_profile(self) -> list[float]:
        """Bipartite entanglement entropy at every cut along the chain
        (mps.entropy(k) for k in 1..N_SITES-1), computed from the MPS's own
        Schmidt/singular values at each bond -- O(bond_dim), not a dense
        partial trace. Index 4 in the returned list (cut=5) is the
        boundary/bulk seam -- see reconstruction.py for how this is read as
        the H1 v0 result.
        """
        return [float(self._mps.entropy(k)) for k in range(1, N_SITES)]

    def max_bond(self) -> int:
        return int(self._mps.max_bond())

    def norm(self) -> float:
        return float(abs(self._mps.H @ self._mps))
