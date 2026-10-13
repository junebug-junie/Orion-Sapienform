"""Retrieval scoring for reading resonance: a recall net, never evidence.

Why the raw number was misleading (measured 2026-10-13 on the 29 reads whose Stage 2
finished in the last 14 days; report:
docs/superpowers/pr-reports/2026-10-13-reading-resonance-crosswalk-pr.md).

vector-host serves bge-large-en-v1.5 with mean pooling and no query instruction. bge's
raw cosines are anisotropic: every vector sits in one narrow cone, so a belief and an
unrelated belief still score 0.66 on median and a related pair 0.83. Every read's nearest
prior lands at 0.56-0.71 raw, whatever the read is about; the Utah board-appointment read's
nearest prior scored 0.59-0.62 (offline probe: a prior about ``governor.reject`` calls,
one shared surface word).

What fixes what:

- Mean-centering (subtract the corpus mean vector, renormalize; Mu & Viswanath 2018,
  "All-but-the-Top") moves the unrelated rest point from 0.66 to 0.08, so a printed score
  finally has a calm state: the Utah read's nearest prior drops to 0.22 (below the offer
  floor, so no belief is offered for it) while the CoreWeave GPU reads keep 0.27-0.35.
  It does NOT add separation: AUC on the labelled-by-construction sets is 0.95-0.99
  either way, and the unrelated p95 (0.31) still sits above the related p10 (0.25).
  With ~3,000 candidates per read, the unrelated tail outnumbers the handful of real
  neighbours, so top-k is mostly adjacency.
- Hub correction (CSLS, Conneau et al. 2018): subtract half of each document's mean
  similarity to its K nearest reference reads. One crystallization was in the top-2 of
  8 of 29 reads raw; after centering + hub correction the worst attractor is in 3 of 29.
- The query instruction prefix and CLS pooling did not help (offline probe, same model
  files: AUC 0.978 with prefix vs 0.979 without; CLS 0.955), so vector-host is unchanged
  and nothing needs re-embedding.
- The query is the article's content (Stage 1 what_i_learned + Stage 2 claims), not the
  Stage 2 summary, which narrates the fetch ("Stage 1 handoff", "SPA", "scraped").

So scores here only ORDER candidates for a fixed per-kind budget. The judge decides,
and code checks its citations (resonance.check_judgment).
"""

from __future__ import annotations

import math
from typing import Sequence

Vector = Sequence[float]

# Mean of the top-K reference-read similarities a document gets (its "hubness").
HUB_K = 10
# Centered-cosine floor below which a candidate is not offered at all. Set to the related
# p10 of the 2026-10-13 replay's belief sanity sets (a prior vs its own Findings, a lived
# question vs its own answers; n=301, centered against the replay pool's mean): related
# p10 = 0.246, unrelated p50 = 0.078, unrelated p95 = 0.310. It keeps ~90% of related pairs
# and drops the bulk of unrelated ones, but its tail still overlaps (unrelated p95 > floor):
# a recall net, not a judgment. The packet budget still caps what is shown. The value is
# tied to the mean vector it was measured against; a different pool mean needs a re-run.
CENTERED_OFFER_FLOOR = 0.25


def _norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def mean_vector(vectors: Sequence[Vector]) -> list[float]:
    if not vectors:
        raise ValueError("mean_vector needs at least one vector")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise ValueError("vectors differ in dimension")
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


def center(v: Vector, mu: Vector) -> list[float]:
    """v minus the corpus mean, renormalized. A zero result stays zero."""
    if len(v) != len(mu):
        raise ValueError("vector and mean differ in dimension")
    d = [a - b for a, b in zip(v, mu)]
    n = _norm(d)
    return [x / n for x in d] if n > 0 else d


def dot(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors differ in dimension")
    return sum(x * y for x, y in zip(a, b))


def hub_from_sims(sims: Sequence[float], k: int = HUB_K) -> float:
    """Mean of the k highest similarities a document has to the reference reads. The caller
    leaves the query read itself out of ``sims``: a document near ONLY this read must not be
    penalised for being near it."""
    if not sims:
        return 0.0
    top = sorted(sims, reverse=True)[: max(1, min(k, len(sims)))]
    return sum(top) / len(top)


def hub_scores(doc_vecs: Sequence[Vector], reference_vecs: Sequence[Vector], k: int = HUB_K) -> list[float]:
    """Per document: mean of its k highest similarities to the reference reads.

    Inputs are already centered. With fewer than k references, all are used.
    """
    return [hub_from_sims([dot(d, r) for r in reference_vecs], k) for d in doc_vecs]


def rank_score(query_vec: Vector, doc_vec: Vector, hub: float) -> float:
    """CSLS-style rank score on centered vectors: cos - hub/2 (the query's own hub
    term is constant per query and does not change its ranking)."""
    return dot(query_vec, doc_vec) - hub / 2.0
