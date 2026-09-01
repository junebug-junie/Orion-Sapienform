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
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .routing import BOND_DIM, BOUNDARY_BULK_CUT, BOUNDARY_SITES, N_SITES
from .mps_state import HeartbeatSubstrate
from .ensemble import EnsembleH1ResultV1, EnsembleSubstrate

# Confirmed via quimb 1.14.0's own source (MatrixProductState.entropy() ->
# `-sum(S * log2(S))` over the Schmidt values at the cut): log base 2
# ("bits"), not natural log. max entropy at bond dimension BOND_DIM is
# therefore log2(BOND_DIM) exactly (Schmidt rank <= bond dimension).
_MAX_POSSIBLE_ENTROPY = math.log2(BOND_DIM)

# Cut-5 mean_ratio bands (legacy single-signal; still exported for Hub band
# gauges on the seam scalar). Ensemble verdict classification no longer uses
# mean_ratio alone -- see classify_ensemble_verdict().
_HIGH_RATIO = 0.6
_LOW_RATIO = 0.2

# Multi-signal ensemble verdict bands. Calibrated from 48h live AST/HOT after
# #1985 (2026-09-01, n=5574 rows with std_ratio+bulk_penetration_depth):
#   std p25/p50/p75/p90 = 0.026 / 0.034 / 0.043 / 0.054
#   bulk p25/p50/p75     = 0.847 / 0.875 / 0.888
# Re-run scripts/analysis/measure_heartbeat_ensemble_calibration.py grammar
# replay before retuning; synthetic quiet phase should still land concentrated.
_STD_MIXED = 0.043  # live p75 -- trajectory disagreement
_STD_REDUNDANT_MAX = 0.030  # below live p50 -- trajectories agree
_BULK_LOW = 0.840  # live ~p25 -- shallow bulk penetration
_BULK_REDUNDANT_MIN = 0.875  # live ~p50 -- settled busy bulk profile


def _normalize_entropy_ratio(entropy: float) -> float:
    return max(0.0, min(1.0, entropy / _MAX_POSSIBLE_ENTROPY))


def classify_ensemble_verdict(
    *,
    mean_ratio: float,
    std_ratio: float,
    bulk_penetration_depth: float,
) -> str:
    """Classify ensemble H1 into redundant / concentrated / mixed.

    Priority (first match wins):
      1. concentrated -- true silence / product-state floor (mean at charter low band)
      2. mixed -- cross-trajectory disagreement (std at or above live p75)
      3. concentrated -- shallow bulk penetration (bulk at or below live p25)
      4. redundant -- high mean + low std + high bulk (settled busy agreement)
      5. mixed -- everything else (middle band; mean alone cannot discriminate)

    mean_ratio at cut-5 is capacity-saturated under real traffic (~0.73-0.95);
    std_ratio and bulk_penetration_depth carry the discriminating structure.
    """
    if mean_ratio <= _LOW_RATIO:
        return "concentrated"
    if std_ratio >= _STD_MIXED:
        return "mixed"
    if bulk_penetration_depth <= _BULK_LOW:
        return "concentrated"
    if (
        mean_ratio >= _HIGH_RATIO
        and std_ratio <= _STD_REDUNDANT_MAX
        and bulk_penetration_depth >= _BULK_REDUNDANT_MIN
    ):
        return "redundant"
    return "mixed"


def bulk_penetration_depth(profile: list[float]) -> float:
    """Mean normalized entropy at cuts strictly inside the bulk block (after
    the boundary/bulk seam at BOUNDARY_BULK_CUT). Pure function over an
    already-computed 9-cut profile -- see compute_h1_ensemble() for how the
    ensemble mean profile is formed."""
    bulk_entropies = profile[BOUNDARY_BULK_CUT:]
    if not bulk_entropies:
        return 0.0
    ratios = [_normalize_entropy_ratio(e) for e in bulk_entropies]
    return float(sum(ratios) / len(ratios))


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


def compute_h1_ensemble(ensemble: EnsembleSubstrate) -> EnsembleH1ResultV1:
    """Ensemble-level H1 reading: mean/std at the boundary/bulk cut, bulk
    penetration depth, and a multi-signal verdict.

    Verdict uses classify_ensemble_verdict() -- mean_ratio alone is
    capacity-saturated under real traffic; std_ratio (trajectory disagreement)
    and bulk_penetration_depth (profile shape) discriminate when cut-5 cannot.
    Bands calibrated from 48h live AST/HOT (2026-09-01, n=5574); re-validate
    via scripts/analysis/measure_heartbeat_ensemble_calibration.py before retuning.
    """
    # Single ratios() call, not ratios() + std_ratio() (std_ratio() would
    # recompute ratios() -- and hence every trajectory's entropy_profile() --
    # a second time for the same tick; caught by review). pstdev matches
    # ensemble.std_ratio()'s own np.std(..., default ddof=0) population
    # convention, not sample stdev.
    ratios = ensemble.ratios()
    mean_ratio = float(sum(ratios) / len(ratios))
    std_ratio = float(statistics.pstdev(ratios)) if len(ratios) > 1 else 0.0

    profiles = [traj.entropy_profile() for traj in ensemble.trajectories]
    n_cuts = len(profiles[0])
    mean_profile = [
        float(sum(p[i] for p in profiles) / len(profiles))
        for i in range(n_cuts)
    ]
    bulk_depth = bulk_penetration_depth(mean_profile)

    verdict = classify_ensemble_verdict(
        mean_ratio=mean_ratio,
        std_ratio=std_ratio,
        bulk_penetration_depth=bulk_depth,
    )

    return EnsembleH1ResultV1(
        mean_ratio=mean_ratio,
        std_ratio=std_ratio,
        verdict=verdict,
        tick_count=ensemble.tick_count(),
        seeds=list(ensemble.seeds),
        ratios=[float(r) for r in ratios],
        bulk_penetration_depth=bulk_depth,
    )


def verdict_thresholds() -> dict[str, float]:
    """The live _HIGH_RATIO/_LOW_RATIO band edges, for read-only surfaces that
    need to draw or explain the bands rather than re-declare them.

    Exists so a consumer (services/orion-hub's attention-organ operator tab)
    renders the SAME thresholds this module actually classifies with, instead
    of mirroring two float constants that would then silently drift apart the
    next time these are retuned -- design doc "Recommended next patch" step 4
    explicitly anticipates retuning them against live data.
    """
    return {
        "high_ratio": _HIGH_RATIO,
        "low_ratio": _LOW_RATIO,
        "std_mixed": _STD_MIXED,
        "std_redundant_max": _STD_REDUNDANT_MAX,
        "bulk_low": _BULK_LOW,
        "bulk_redundant_min": _BULK_REDUNDANT_MIN,
    }
