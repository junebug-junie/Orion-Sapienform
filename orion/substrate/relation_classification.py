"""Layer 3 relation-classification decision logic (Phase 4).

Phase 4 of docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md.

Clustering (topic-foundry, via `orion/substrate/adapters/topic_foundry.py`)
produces free `co_occurs_with` `SubstrateEdgeV1` records by pure counting --
cheap, no inference. Typed relationships (`supports`/`contradicts`/`refines`,
see `SubstrateEdgePredicateV1` in `orion/core/schemas/cognitive_substrate.py`)
require actual semantic judgment that counting cannot produce -- that is
genuinely worth an LLM call, but only for node pairs that have already proven
themselves worth the cost via strong co-occurrence.

This module is the *decision logic* for "is this pair worth classifying" (three
interchangeable threshold strategies) plus the *shape* of the classification
step itself. It does NOT call an LLM: the classification step takes an
injected callable (`RelationClassifier`) supplied by the caller. Wiring a real
classifier and hooking this into a live promotion-state-transition trigger is
explicitly out of scope here -- see the spec's Phase 4 acceptance check and
"Recommended next patch" item 5. Mirrors how the Phase 2 topic-foundry adapter
(`orion/substrate/adapters/topic_foundry.py`) stayed a pure conversion
function with no HTTP/bus I/O.

Three strategies, one entry point:

- **count** (Option A, baseline): raw `edge.metadata["co_occurrence_count"]`
  against a threshold. Deliberately naive -- exists to give the other two
  strategies something to be compared against, not because it is expected to
  win.
- **pmi** (Option B): pointwise mutual information, `log(P(A,B) / (P(A)*P(B)))`,
  using each node's `signals.salience` as the marginal-frequency proxy and the
  edge's own normalized co-occurrence strength (`salience`/`confidence`,
  already normalized in [0, 1] relative to the run's max pair count by
  `map_topic_foundry_run_to_substrate`) as the joint-probability proxy.
- **activation** (Option C): a recency-weighted signal built from
  `SubstrateActivationV1`'s existing `activation`/`decay_half_life_seconds`/
  `decay_floor`/`recency_score` fields on both nodes, reusing the same
  validated half-life decay helper `orion/substrate/dynamics.py` already uses
  in its tick loop (`orion.core.activation_decay.decay_activation`) rather
  than reinventing decay math.

Option D (piggyback purely on the promotion-state transition, with no
co-occurrence signal at all) is explicitly **deferred** per the spec's
Non-goals -- not built here.

All three strategies degrade gracefully on missing/malformed input: they
return `False` (not-worth-classifying) rather than raising. `classify_relation`
never raises either -- an exception from the injected classifier is caught,
not propagated.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

from orion.core.activation_decay import decay_activation
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateEdgePredicateV1,
    SubstrateEdgeV1,
)

from .adapters._common import make_provenance, make_temporal

RelationClassifierStrategy = Literal["count", "pmi", "activation"]

# Injected classifier signature. The caller owns the actual LLM call (or any
# other judgment source); this module only decides *whether* to invoke it and
# *what* to do with the result. Returning `None` means "no confident
# classification" -- treated the same as an exception: no edge is produced.
RelationClassifier = Callable[
    [ConceptNodeV1, ConceptNodeV1, SubstrateEdgeV1], Optional[SubstrateEdgePredicateV1]
]

DEFAULT_COUNT_THRESHOLD = 5
# PMI > 0 means the pair co-occurs more than chance would predict given each
# concept's individual salience -- the natural "worth it" cutoff for Option B.
DEFAULT_PMI_THRESHOLD = 0.0
DEFAULT_ACTIVATION_THRESHOLD = 0.3


# --------------------------------------------------------------------------
# Option A -- raw count baseline.
# --------------------------------------------------------------------------


def count_score(edge: Optional[SubstrateEdgeV1]) -> Optional[int]:
    """Read `edge.metadata["co_occurrence_count"]`. `None` if missing/malformed."""
    if edge is None:
        return None
    metadata = getattr(edge, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("co_occurrence_count")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def worth_classifying_count(
    edge: Optional[SubstrateEdgeV1], *, threshold: int = DEFAULT_COUNT_THRESHOLD
) -> bool:
    """Option A: worth classifying iff the raw co-occurrence count >= threshold."""
    count = count_score(edge)
    if count is None:
        return False
    return count >= threshold


# --------------------------------------------------------------------------
# Option B -- PMI-style normalized association.
# --------------------------------------------------------------------------


def pmi_score(
    node_a: Optional[ConceptNodeV1],
    node_b: Optional[ConceptNodeV1],
    edge: Optional[SubstrateEdgeV1],
) -> Optional[float]:
    """log(P(A,B) / (P(A) * P(B))), using salience as the marginal proxy.

    `P(A)`/`P(B)` come from each node's `signals.salience` (topic-foundry's
    adapter already sets this to `count / total_docs` -- a real marginal-
    frequency proxy, not an invented one). `P(A,B)` comes from the edge's own
    normalized co-occurrence strength (`salience`, falling back to
    `confidence` if `salience` is unset), which `map_topic_foundry_run_to_substrate`
    already normalizes to [0, 1] relative to the run's max pair count.

    Returns `None` (never raises) on any missing field, non-positive marginal,
    non-positive joint term, or a `math.log` domain error -- all sentinel for
    "not worth classifying" under this strategy.
    """
    try:
        marginal_a = node_a.signals.salience
        marginal_b = node_b.signals.salience
        joint = edge.salience if edge.salience > 0.0 else edge.confidence
    except AttributeError:
        return None

    if marginal_a is None or marginal_b is None or joint is None:
        return None
    if marginal_a <= 0.0 or marginal_b <= 0.0 or joint <= 0.0:
        return None

    try:
        return math.log(joint / (marginal_a * marginal_b))
    except (ValueError, ZeroDivisionError):
        return None


def worth_classifying_pmi(
    node_a: Optional[ConceptNodeV1],
    node_b: Optional[ConceptNodeV1],
    edge: Optional[SubstrateEdgeV1],
    *,
    threshold: float = DEFAULT_PMI_THRESHOLD,
) -> bool:
    """Option B: worth classifying iff PMI(A, B) >= threshold."""
    score = pmi_score(node_a, node_b, edge)
    if score is None:
        return False
    return score >= threshold


# --------------------------------------------------------------------------
# Option C -- decayed activation crossing a threshold.
# --------------------------------------------------------------------------


def decayed_activation_score(
    node: Optional[ConceptNodeV1], *, now: Optional[datetime] = None
) -> Optional[float]:
    """Recency-weighted activation for one node.

    Combines the half-life-decayed `activation` value (via the same
    `orion.core.activation_decay.decay_activation` helper
    `orion/substrate/dynamics.py`'s tick loop already uses -- reusing
    validated decay math rather than reinventing it) with the node's own
    `recency_score`, averaged equally.

    Returns `None` (degrade, never raise/guess) if the node's activation
    bundle is missing or at its schema default (`activation <= 0.0` is
    indistinguishable from "never seeded" under this schema, so it is treated
    as unset rather than as a real zero-activation reading).
    """
    if node is None:
        return None
    try:
        activation_signal = node.signals.activation
        observed_at = node.temporal.observed_at
    except AttributeError:
        return None
    if activation_signal is None or observed_at is None:
        return None
    if activation_signal.activation <= 0.0:
        return None

    reference = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    elapsed_seconds = max(0.0, (reference - observed_at).total_seconds())

    try:
        decayed = decay_activation(
            current=activation_signal.activation,
            elapsed_seconds=elapsed_seconds,
            half_life_seconds=activation_signal.decay_half_life_seconds,
            floor=activation_signal.decay_floor,
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    combined = (0.5 * decayed) + (0.5 * activation_signal.recency_score)
    return max(activation_signal.decay_floor, min(1.0, combined))


def worth_classifying_activation(
    node_a: Optional[ConceptNodeV1],
    node_b: Optional[ConceptNodeV1],
    edge: Optional[SubstrateEdgeV1] = None,  # noqa: ARG001 -- kept for a uniform 3-strategy signature
    *,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    now: Optional[datetime] = None,
) -> bool:
    """Option C: worth classifying iff both nodes' decayed activation crosses threshold.

    Uses `min()` across the pair -- one node showing recent activity does not
    make an otherwise-dormant partner worth an LLM call.
    """
    score_a = decayed_activation_score(node_a, now=now)
    score_b = decayed_activation_score(node_b, now=now)
    if score_a is None or score_b is None:
        return False
    return min(score_a, score_b) >= threshold


# --------------------------------------------------------------------------
# Single dispatch entry point (A/B/C behind one flag, for empirical
# comparison -- Option D is explicitly deferred, see module docstring).
# --------------------------------------------------------------------------


def is_worth_classifying(
    node_a: Optional[ConceptNodeV1],
    node_b: Optional[ConceptNodeV1],
    edge: Optional[SubstrateEdgeV1],
    *,
    strategy: RelationClassifierStrategy = "count",
    count_threshold: int = DEFAULT_COUNT_THRESHOLD,
    pmi_threshold: float = DEFAULT_PMI_THRESHOLD,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    now: Optional[datetime] = None,
) -> bool:
    """Dispatch to the requested strategy. Unknown strategy or any internal
    error degrades to `False` -- never raises."""
    try:
        if strategy == "count":
            return worth_classifying_count(edge, threshold=count_threshold)
        if strategy == "pmi":
            return worth_classifying_pmi(node_a, node_b, edge, threshold=pmi_threshold)
        if strategy == "activation":
            return worth_classifying_activation(
                node_a, node_b, edge, threshold=activation_threshold, now=now
            )
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------
# The classification step's shape (no live LLM call -- classifier injected).
# --------------------------------------------------------------------------


def classify_relation(
    node_a: ConceptNodeV1,
    node_b: ConceptNodeV1,
    edge: SubstrateEdgeV1,
    *,
    classifier: RelationClassifier,
    strategy: RelationClassifierStrategy = "count",
    count_threshold: int = DEFAULT_COUNT_THRESHOLD,
    pmi_threshold: float = DEFAULT_PMI_THRESHOLD,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    now: Optional[datetime] = None,
) -> Optional[SubstrateEdgeV1]:
    """Decide whether `edge` is worth an LLM classification call, and if so,
    call the caller-injected `classifier` and build a typed `SubstrateEdgeV1`
    from its result.

    `classifier` is invoked with `(node_a, node_b, edge)` and must return a
    `SubstrateEdgePredicateV1` (e.g. "supports"/"contradicts"/"refines") or
    `None`. This function never calls a real LLM itself -- that is live
    infrastructure wiring, out of scope for this phase.

    Returns `None` (no new edge) if:
    - the pair is not worth classifying under `strategy` (see `is_worth_classifying`);
    - `classifier` returns `None`;
    - `classifier` raises (the exception is caught, never propagated);
    - the returned predicate fails schema validation when building the edge.
    """
    if not is_worth_classifying(
        node_a,
        node_b,
        edge,
        strategy=strategy,
        count_threshold=count_threshold,
        pmi_threshold=pmi_threshold,
        activation_threshold=activation_threshold,
        now=now,
    ):
        return None

    try:
        predicate = classifier(node_a, node_b, edge)
    except Exception:
        return None

    if predicate is None:
        return None

    try:
        return SubstrateEdgeV1(
            source=NodeRefV1(node_id=node_a.node_id, node_kind=node_a.node_kind),
            target=NodeRefV1(node_id=node_b.node_id, node_kind=node_b.node_kind),
            predicate=predicate,
            temporal=make_temporal(observed_at=now),
            confidence=edge.confidence,
            salience=edge.salience,
            provenance=make_provenance(
                source_kind="relation_classification.llm_judgment",
                source_channel=f"orion:substrate:relation_classification:{strategy}",
                producer="relation_classification",
                evidence_refs=[edge.edge_id],
            ),
            metadata={"strategy": strategy, "source_edge_id": edge.edge_id},
        )
    except Exception:
        # Malformed predicate (e.g. a stub classifier returning garbage) or
        # any other schema-validation failure -- degrade, never raise.
        return None
