"""Hop 0 — sustained metacog trend → governed proposal candidate.

This is the piece the stream-of-consciousness arc has repeatedly stopped short
of. `docs/superpowers/specs/2026-07-29-stream-of-consciousness-hop-chain-design.md`
scopes hop 0 as "a deterministic reducer ... registered as a candidate producer
the same way reverie is -- **not built as a bespoke standalone script**", and
that registration is what never got built: `orion/metacog/trend_reducer.py`'s
computation exists, is wired into `orion-equilibrium-service`'s metacog *trigger*
path, and sits flag-off. A trigger is not arena citizenship. This module is the
missing converter.

Deliberately mirrors `orion/reverie/proposal.py::spontaneous_thought_to_candidate`
field-for-field rather than inventing a second shape. Reverie is the live,
working precedent for "a non-deterministic cognitive act competing as a
first-class candidate"; hop 0 is its sibling under the same contract, not a
merge with it and not a new arbitration mechanism.

Deterministic (CLAUDE.md §4): given a trend result, the candidate is a pure
function -- no LLM, no I/O. The reducer decides *whether* there is a trend; this
module only decides how to phrase it as a governed proposal.

## Why `operator_review`, always

The awake organ proposes; the substrate disposes. A hop candidate can enter
Layer 7→8 and compete for a budget slot, but it can never auto-dispatch. This
matters more for a hop than for a reverie thought: the hop-chain design's own
named danger case is "a chain that always self-scores just above the preemption
threshold never actually gets interrupted -- 'interruptible' becomes a claim,
not a verified property." An unconditional `operator_review` gate means that
even if scoring is wrong, the failure mode is a noisy proposal queue, not
autonomous action.

## Chain lineage, and why it is in `reasons` for now

The design doc calls for `chain_id`/`parent_hop_id`/`hop_index` on "whatever
hop-producing schema emerges". Hop 0 is by definition the *start* of a chain --
no parent, index 0 -- so it needs the identity but not the graph. Those go in
`reasons` (already a free-form list on `ProposalCandidateV1`) rather than as new
schema fields, so this patch adds no schema surface for a structure nothing
consumes yet. Hop 1, which actually has a parent to point at, is where real
lineage fields have to exist; inventing them here would be a registry with no
consumer.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING

from orion.schemas.proposal_frame import ProposalCandidateV1

if TYPE_CHECKING:
    from orion.metacog.trend_reducer import MetacogTrendResultV1

COGNITIVE_HOP_PROPOSAL_SOURCE = "cognitive_hop"

# Hop 0 opens a chain; there is no parent to inherit an id from.
HOP_INDEX_ZERO = 0

# Priority scales with HOW LONG this has kept escalating, not with the raw
# z-score. Measured 2026-07-31 against the real reducer: a flat input run drives
# the EWMA variance toward its 1e-6 floor, so the first excursion after one
# scores z=463. The live repair_pressure series is exactly that shape (7 of 21
# confidence-bearing rows sit at one identical modal value), so a z-proportional
# salience would put hop 0 into the arena at priority 1.0 essentially every time
# it fires -- which is the hop-chain design's own named danger case ("a chain
# that always self-scores just above the preemption threshold never actually
# gets interrupted; 'interruptible' becomes a claim, not a verified property").
#
# Run length is robust to that artifact and is the better expression of "trend,
# not spike" anyway. At the reducer's default sustained_hits=3, a just-qualifying
# trend enters at 0.5 and has to keep escalating to earn more.
SUSTAINED_RUN_FOR_FULL_SALIENCE = 6

# Confidence is inverted from |z| against this constant, matching the arena's
# own DIMENSION_PRECISION_ZSCORE_SATURATION (orion/proposals/scoring.py) rather
# than inventing a second calibration.
ZSCORE_CONFIDENCE_SATURATION = 3.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _chain_id(series_name: str, run_started_at: datetime) -> str:
    """Stable, deterministic id for the chain this hop opens.

    Keyed on when the elevated run BEGAN, not on the latest reading. Review
    finding 2026-07-31: the first version hashed the newest reading's timestamp,
    so one continuous escalating trend minted a new chain identity on every new
    row (consec=3/4/5/6 produced 4 distinct chain_ids, reproduced). The
    hop-chain design's Missing Question 5 requires `chain_id` "stable across a
    chain's life" precisely so an eventual action's outcome can be attributed
    back to the observation that started the chain -- an id that changes every
    tick makes one chain look like N unrelated proposals to feedback and
    consolidation, and gives hop 1 nothing stable to set `parent_hop_id` to.

    Still fully deterministic under replay (the original, correct half of the
    reasoning): the run's onset is a stored timestamp, not wall-clock, so
    reconstructing history from stored rows reproduces the same id.
    """
    digest = hashlib.sha256(
        f"{series_name}|{run_started_at.isoformat()}".encode("utf-8")
    ).hexdigest()[:16]
    return f"chain:{series_name}:{digest}"


def trend_result_to_candidate(
    result: "MetacogTrendResultV1",
    *,
    series_name: str,
    run_started_at: datetime,
    fallback_target_id: str,
    evidence_refs: list[str] | None = None,
) -> ProposalCandidateV1 | None:
    """Convert a SUSTAINED metacog trend into a review-gated candidate, or None.

    Returns None (degrades, never raises) for anything that is not a sustained
    trend -- cold start, a single elevated tick, or an ordinary reading. That
    threshold is the reducer's, not this module's: `is_sustained_trend` already
    means `sustained_hits` consecutive elevated ticks, i.e. "has this kept
    happening", which is precisely the "feel across time, not point in time"
    property this hop exists to carry into the arena. A one-tick spike is
    explicitly not a train of thought.

    **Characterization worth knowing before building hop 1 on this** (measured
    2026-07-31 against the real reducer, not inferred): `is_sustained_trend`
    detects sustained ESCALATION, not sustained elevation. A value that jumps
    high and stays high produces one qualifying tick and then goes quiet, because
    the EWMA baseline adopts the new level as normal within a few ticks (held at
    0.55 for 12 ticks: z decays 463 -> 1.79 -> ... -> 0.27, one sustained tick).
    A rising ramp sustains. That is defensible for a prediction-error framing --
    a persistently high value genuinely IS the new normal -- but it means hop 0
    will not open a chain about a condition that has been quietly bad for an
    hour, which is plausibly something a train of thought should notice. Not
    changed here; the reducer is spec'd and shared with the equilibrium trigger
    path, so re-shaping it is its own patch with its own measurement.
    """
    if result is None or not result.is_sustained_trend:
        return None
    # Defensive: is_sustained_trend cannot be True during cold start (the
    # reducer's own guard), but a caller could hand-construct a result.
    if result.is_cold_start or result.latest_zscore is None:
        return None

    zscore = float(result.latest_zscore)
    consecutive = int(result.state.consecutive_elevated)
    salience = _clamp01(consecutive / SUSTAINED_RUN_FOR_FULL_SALIENCE)
    chain_id = _chain_id(series_name, run_started_at)

    return ProposalCandidateV1(
        proposal_id=f"proposal:cognitive_hop:{chain_id}",
        proposal_kind="request_policy_review",
        title="Cognitive hop 0: sustained metacog trend (policy review required)",
        description=(
            f"{series_name} has been elevated for {result.state.consecutive_elevated} "
            f"consecutive readings (latest z={zscore:.2f} against its own running "
            f"baseline of {result.state.count} samples). This is a trend, not a spike."
        )[:500],
        target_id=fallback_target_id,
        target_kind="self_state",
        priority_score=salience,
        # A trend is by construction not urgent -- it is the opposite of a
        # spike. Urgency stays deliberately below priority so a sustained
        # background condition cannot outrank a genuine acute one in the arena.
        urgency_score=_clamp01(salience * 0.4),
        # Precision-weighted against this series' own baseline, matching the
        # arena's native `dimension_confidence()` semantic exactly: confidence
        # is "how well does this reading match its own recent baseline",
        # inverted from the z-score -- NOT "how sure am I there is a trend".
        #
        # Review finding 2026-07-31: the first version used
        # `state.count / 100.0`, and `count` only ever grows (the store replays
        # the whole window each tick), so it pinned to a permanent 1.0 after
        # ~100 rows -- about 4.5 days at this series' ~8.7 rows/day, and never
        # returning. An always-saturated metric is exactly what CLAUDE.md's
        # metric quality gate step 4 forbids, and the cited precedent does not
        # support it: `dimension_confidence()` uses sample count ONLY as a
        # cold-start gate (already handled here by `is_cold_start`) and derives
        # the actual value from an inverted z-score that can move both ways
        # forever. Same 3.0 saturation convention as
        # DIMENSION_PRECISION_ZSCORE_SATURATION.
        confidence_score=_clamp01(1.0 - abs(zscore) / ZSCORE_CONFIDENCE_SATURATION),
        # Low risk to PROPOSE and fully reversible -- the risk lives in whatever
        # action a later hop might reach, which policy must still gate.
        risk_score=0.3,
        reversibility_score=1.0,
        evidence_refs=list(evidence_refs or []),
        reasons=[
            "cognitive_hop",
            f"chain_id:{chain_id}",
            f"hop_index:{HOP_INDEX_ZERO}",
            "parent_hop_id:none",
            f"series:{series_name}",
            f"consecutive_elevated:{result.state.consecutive_elevated}",
            f"zscore:{zscore:.4f}",
            f"baseline_samples:{result.state.count}",
        ],
        proposed_effect="prepare_for_policy_gate",
        required_policy_gate="operator_review",  # never none/read_only
        source=COGNITIVE_HOP_PROPOSAL_SOURCE,
    )
