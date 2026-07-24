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
matching the reducer-determinism discipline already established elsewhere in
this codebase (services/orion-substrate-runtime's reducer contract:
"Reducers must be deterministic and side-effect-free").
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


class HeartbeatSubstrate:
    """Owns one quimb MatrixProductState and applies the v0 update rule."""

    def __init__(self, *, seed: int = 42) -> None:
        self._mps = qtn.MPS_rand_state(
            L=N_SITES, bond_dim=BOND_DIM, phys_dim=PHYS_DIM, dtype="complex128", seed=seed
        )
        self._mps.normalize()
        self.tick_count = 0

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
