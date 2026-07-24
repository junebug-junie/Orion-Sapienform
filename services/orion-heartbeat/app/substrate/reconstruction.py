"""H1 v0: boundary/bulk entanglement as the holographic-encoding signature.

Design doc: docs/superpowers/specs/2026-07-24-spark-field-holographic-lattice-design.md.

**This deviates from the 2026-05-01 heartbeat charter's literal H1 formula**
(partial trace -> max-entropy completion at fixed bond dimension -> quantum
fidelity F = |Tr(sqrt(sqrt(rho_orig) . rho_recon . sqrt(rho_orig)))|^2), and
that deviation is deliberate, found and confirmed via direct quimb testing
during this session's implementation, not a shortcut taken silently:

1. "Maximum-entropy completion at fixed bond dimension" is named as a concept
   in the charter's engineering spec but never given a concrete algorithm --
   §8 of that document doesn't specify one. There is no standard, off-the-shelf
   quimb routine for it either.
2. For a pure global MPS state (which this substrate always is -- gates are
   unitary, `.normalize()` runs after every absorb()), the boundary and bulk
   reduced density matrices share an IDENTICAL eigenvalue spectrum by basic
   Schmidt-decomposition symmetry, confirmed numerically this session
   (S_boundary == S_bulk to float precision on a random test state). Literal
   fidelity-of-full-bulk-reconstruction-from-full-boundary is close to
   tautological under exact partial trace -- it doesn't test anything the
   entanglement spectrum itself doesn't already show directly.
3. Dense partial traces over larger site subsets (needed for any
   organ-specific "drop this boundary site and see what breaks" test) were
   confirmed this session to be computationally expensive even at N=10 --
   `partial_trace_exact` over a 7-site subset did not complete within 45s
   using quimb's default 'auto-hq' contraction-path optimizer, and remained
   slow even with the faster 'auto' optimizer. Not viable for an always-on,
   frequent tick-loop computation.

What quimb's `MatrixProductState.entropy(cut)` gives instead -- confirmed
this session at ~0.02s for a single cut and ~0.006s for the full 9-cut
profile on N=10/chi=4, computed directly from the MPS's own Schmidt/singular
values at each bond, no dense diagonalization at all -- is the standard,
efficient, textbook way entanglement structure is actually measured in
tensor-network and holographic-code literature (this is the same quantity
Ryu & Takayanagi (2006) relate to minimal-surface area in the physics
literature the charter itself cites). **v0's H1 result is this entropy
profile, read at the boundary/bulk cut (routing.BOUNDARY_BULK_CUT) as the
headline number.**

Deferred, not abandoned: testing whether specific organs are individually
redundant (dropping site 2 only, keeping 0/1/3/4) needs the more expensive
dense partial-trace machinery point 3 ruled out for v0's tick loop. If the
cheap profile below shows non-trivial structure worth investigating further,
that's the natural next increment -- run occasionally/offline, not on every
tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .routing import BOND_DIM, BOUNDARY_BULK_CUT, BOUNDARY_SITES, N_SITES
from .mps_state import HeartbeatSubstrate

# Confirmed via quimb 1.14.0's own source (MatrixProductState.entropy() ->
# `-sum(S * log2(S))` over the Schmidt values at the cut): log base 2
# ("bits"), not natural log. max entropy at bond dimension BOND_DIM is
# therefore log2(BOND_DIM) exactly (Schmidt rank <= bond dimension).
_MAX_POSSIBLE_ENTROPY = math.log2(BOND_DIM)

# Provisional, NOT calibrated against any real baseline -- unlike the
# charter's own H1 thresholds (F>=0.85 success / F<0.7 falsified), which were
# grounded in the active phi encoder's real precedent, v0 has no equivalent
# prior run to calibrate against yet. These are placeholder bands for a first
# reading, not a pre-registered gate; revisit once real ticks have
# accumulated.
_HIGH_RATIO = 0.6
_LOW_RATIO = 0.2


@dataclass(frozen=True)
class H1ResultV1:
    generated_at: datetime
    tick_count: int
    entropy_profile: list[float]  # length N_SITES - 1, index i = entropy at cut (i+1)
    boundary_bulk_entropy: float  # entropy_profile[BOUNDARY_BULK_CUT - 1]
    max_possible_entropy: float
    ratio: float  # boundary_bulk_entropy / max_possible_entropy, in [0, 1]
    verdict: str  # "redundant" | "concentrated" | "mixed" -- never silently omitted
    boundary_subprofile: list[float] = field(default_factory=list)  # entropy_profile
    # restricted to cuts inside the boundary block (1..4) -- how entanglement
    # builds up as more boundary sites are included, a qualitative read on
    # whether the coupling is spread or concentrated within the boundary
    # itself, not just at the boundary/bulk seam.


def compute_h1(substrate: HeartbeatSubstrate) -> H1ResultV1:
    profile = substrate.entropy_profile()
    boundary_bulk_entropy = profile[BOUNDARY_BULK_CUT - 1]
    ratio = max(0.0, min(1.0, boundary_bulk_entropy / _MAX_POSSIBLE_ENTROPY))

    if ratio >= _HIGH_RATIO:
        verdict = "redundant"
    elif ratio <= _LOW_RATIO:
        verdict = "concentrated"
    else:
        verdict = "mixed"

    boundary_subprofile = [profile[i] for i in range(len(BOUNDARY_SITES) - 1)]

    return H1ResultV1(
        generated_at=datetime.now(timezone.utc),
        tick_count=substrate.tick_count,
        entropy_profile=profile,
        boundary_bulk_entropy=boundary_bulk_entropy,
        max_possible_entropy=_MAX_POSSIBLE_ENTROPY,
        ratio=ratio,
        verdict=verdict,
        boundary_subprofile=boundary_subprofile,
    )
