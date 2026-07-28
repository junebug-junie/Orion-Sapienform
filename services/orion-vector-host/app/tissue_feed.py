"""Feeds OrionTissue's real decay/diffusion physics from real per-turn embeddings.

Extraction, not a port (2026-07-28, spark-introspector retirement --
docs/superpowers/specs/2026-07-28-spark-introspector-retirement-and-honest-
substrate-convergence.md). `orion/spark/orion_tissue.py`'s `OrionTissue` class
was confirmed real (16x16x8 decay+diffusion tensor, zero torch/cuda
dependency) while everything around it in the old
`services/orion-spark-introspector` was confirmed theater (SelfState-anchored
phi formulas, a fallback-default masquerading as a genuine "calm" reading on
61% of ticks). This module is orion-vector-host's new home for the one real
piece: `TISSUE.propagate()` still gets fed, but from the honest signal this
service already produces (the raw embedding it just computed for every
chat/social/GPT-log message), not from the old worker.py's
SurfaceEncoding/SignalMapper/synthetic-Gaussian-waveform/cosine-novelty
pipeline. That old pipeline is NOT ported here -- it painted a
`np.exp(-x**2) * arousal_hint` waveform and tag-hash bumps into the tissue,
none of which were derived from the embedding's actual content. Relocating
that machinery would relocate fake math, not remove it.

Stimulus formula (new code, documented per CLAUDE.md 0A's metric-quality
gate -- this is not a port, so it gets its own trace):

1. Normalize the incoming embedding to a unit vector (`emb / ||emb||`, or the
   embedding unchanged if it is exactly zero -- degenerate input, treated as
   no signal rather than dividing by zero).
2. Maintain a per-channel-key exponential moving average (EMA) of recent unit
   embeddings (`_EMBEDDING_EMA`, alpha=`TISSUE_STIMULUS_EMA_ALPHA`, default
   0.05 -- the same blend rate the old pipeline used for its own expectation
   tracking). The first observation for a channel key after process start
   seeds the EMA from `TISSUE.embedding_expectations[channel_key]` when a
   real one survived a restart (persisted in the tissue's own .npz snapshot);
   only a genuinely first-ever observation for that channel key is treated
   as maximally novel (no prior reference to compare against at all).
3. `delta = unit_embedding - ema` -- how different this turn's embedding is
   from the recent running average, in the embedding's own real coordinate
   space. `magnitude = ||delta||_2` (bounded in [0, 2] for unit vectors).
4. Project `delta` onto the tissue's H*W grid with a fixed, unlearned
   Gaussian random projection matrix (seed=42, columns unit-normalized) --
   deterministic and content-derived (varies with what the embedding
   actually contains), not hand-tuned to any particular "feeling" semantics.
   This is the same style of projection `orion/spark/signal_mapper.py`'s
   "Path A: Neural Projection" already used when a `spark_vector` was
   supplied, self-contained here rather than importing SignalMapper's
   larger surface (legacy tag/waveform Path B included), per the retirement
   spec's explicit "no SurfaceEncoding/MAPPER indirection" instruction.
5. Split the projected grid's sign into stimulus channels 0 (positive) and 1
   (negative), scaled by `magnitude` -- matches `OrionTissue.phi()`'s own
   polarity_diff definition (`T[...,0].mean() - T[...,1].mean()`). All other
   channels stay at 0; this module does not invent per-channel semantics
   beyond what the tissue itself already assigns meaning to.
6. `TISSUE.calculate_novelty(stimulus, channel_key=...)` runs before
   `propagate()` -- the old worker.py called `TISSUE.propagate()` directly
   for `handle_semantic_upsert` without ever calling `calculate_novelty()`
   or `inject_surface()` first, which meant `TISSUE.phi()["novelty"]` was
   structurally dead (permanently 0.0 for the "chat" channel) for every
   real chat turn -- a second, previously-undocumented instance of the same
   fallback-default-masquerading-as-calm failure mode the retirement was
   triggered by. Calling `calculate_novelty()` here is a genuine, in-scope
   fix: it uses the tissue's own already-real rolling z-score mechanism
   (`orion_tissue.py`'s `calculate_novelty`), not a new formula.
7. `TISSUE.propagate(stimulus, steps=2, learning_rate=0.1, embedding=...,
   distress=0.0)` -- same physics tuning constants (steps/learning_rate) the
   old call site used; these are OrionTissue's own decay/diffusion
   parameters, not cognition-shaped dressing, so they are kept for
   continuity rather than re-derived without cause.
8. `TISSUE.snapshot()` persists the tensor to disk, same cadence as before
   (every feed call) so restart-amnesia protection is unchanged.

Dropped from the old pipeline, deliberately, not carried forward:
- The valence-anchor RPC pipeline (`_ensure_valence_anchors`/
  `_valence_from_embedding`, cosine similarity against two hand-picked
  anchor texts fetched over the bus). Same species of hand-tuned signal
  shaping as the waveform/arousal_hint math the retirement spec named
  explicitly -- an extra RPC round-trip and a hand-picked pair of anchor
  sentences are not simpler or more honest than the embedding-delta signal
  this module already has in-process.
- Per-doc-id 1-second debounce (`_SEEN_DOC` in the old worker.py). That
  existed to protect against bus redelivery of the same `vector.upsert.v1`
  message; this module is called in-process, once per successful embed, so
  the failure mode it guarded against does not apply here.

Failure mode: every call is wrapped by the caller (`main.py`'s
`_publish_semantic_upsert`) in a try/except that logs and continues -- a
tissue-feed failure must never break the real embedding pipeline it rides on.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from orion.spark.orion_tissue import OrionTissue

from .conn_manager import manager
from .settings import settings

logger = logging.getLogger("orion-vector-host.tissue")

# Same env var OrionTissue itself already reads directly (os.environ) if no
# explicit snapshot_path is passed -- kept explicit here so .env_example
# documents the contract rather than relying on an implicit fallback deep in
# orion/spark/orion_tissue.py.
_snapshot_env = os.environ.get("ORION_TISSUE_SNAPSHOT_PATH")
TISSUE = OrionTissue(snapshot_path=Path(_snapshot_env) if _snapshot_env else None)

_EMBEDDING_EMA: Dict[str, np.ndarray] = {}
_PROJECTION: Optional[np.ndarray] = None
_PROJECTION_DIM: Optional[int] = None


def _projection_matrix(input_dim: int) -> np.ndarray:
    """Fixed, unlearned Gaussian random projection: input_dim -> H*W.

    Lazily built (and rebuilt only if the embedding dimension changes, e.g. a
    backend swap), seeded for determinism -- not retrained, not tuned.
    """
    global _PROJECTION, _PROJECTION_DIM
    if _PROJECTION is None or _PROJECTION_DIM != input_dim:
        rng = np.random.RandomState(42)
        output_dim = TISSUE.H * TISSUE.W
        matrix = rng.randn(input_dim, output_dim).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = 1.0
        matrix /= norms
        _PROJECTION = matrix
        _PROJECTION_DIM = input_dim
    return _PROJECTION


def _build_stimulus(delta: np.ndarray, magnitude: float) -> np.ndarray:
    projection = _projection_matrix(delta.shape[0])
    activations = delta @ projection
    std = float(np.std(activations))
    if std > 1e-6:
        activations = activations / std
    grid = activations.reshape((TISSUE.H, TISSUE.W))

    stimulus = np.zeros((TISSUE.H, TISSUE.W, TISSUE.C), dtype=np.float32)
    stimulus[:, :, 0] = np.maximum(grid, 0.0) * magnitude
    if TISSUE.C >= 2:
        stimulus[:, :, 1] = np.abs(np.minimum(grid, 0.0)) * magnitude
    return stimulus


def get_phi_stats() -> Dict[str, float]:
    """Read-model accessor: polarity_diff/mean_abs_activation/
    embedding_similarity/novelty_zscore, all derived directly from TISSUE's
    real internal tensor state (`OrionTissue.phi()`), never injected from
    self-state or any other source."""
    return {k: float(v) for k, v in (TISSUE.phi() or {}).items()}


async def feed_tissue(
    embedding: Any,
    *,
    doc_id: str = "",
    channel_key: str = "chat",
    correlation_id: str = "",
    broadcast: bool = True,
) -> Dict[str, float]:
    """Feed one real embedding into TISSUE.propagate() and return phi_stats.

    Safe to call for every embedding this service produces (chat history,
    chat turns, social-room turns, ChatGPT log rows, ad hoc embedding.generate
    requests) -- matches the old service's behavior, which fed the tissue
    from every `vector.upsert.v1` message regardless of which handler
    produced it (they all set `embedding_kind="semantic"`).
    """
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.size == 0:
        return get_phi_stats()

    norm = float(np.linalg.norm(emb))
    unit = emb / norm if norm > 0 else emb

    prior = _EMBEDDING_EMA.get(channel_key)
    if prior is None:
        # Restart-amnesia guard: TISSUE.embedding_expectations is persisted
        # across restarts (loaded from the .npz snapshot in
        # OrionTissue.__init__), but this module's own _EMBEDDING_EMA is a
        # separate, process-local dict (different alpha than propagate()'s
        # own adaptive blend, so it can't just reuse that dict directly) --
        # without this seed, every restart would treat the first post-
        # restart embedding per channel_key as maximally novel even though
        # the tissue itself remembers a real prior expectation. Found by
        # code review (2026-07-28): the module docstring claimed unchanged
        # restart-amnesia protection without this seed actually being wired.
        seed = TISSUE.embedding_expectations.get(channel_key)
        if seed is not None and seed.shape == unit.shape and float(np.linalg.norm(seed)) > 0:
            prior = seed.astype(np.float32)
    if prior is None or prior.shape != unit.shape:
        delta = unit.copy()
        new_ema = unit.copy()
    else:
        delta = unit - prior
        alpha = float(settings.TISSUE_STIMULUS_EMA_ALPHA)
        new_ema = ((1.0 - alpha) * prior) + (alpha * unit)
    _EMBEDDING_EMA[channel_key] = new_ema

    magnitude = float(np.linalg.norm(delta))
    stimulus = _build_stimulus(delta, magnitude)

    TISSUE.calculate_novelty(stimulus, channel_key=channel_key)
    TISSUE.propagate(
        stimulus,
        steps=int(settings.TISSUE_STIMULUS_STEPS),
        learning_rate=float(settings.TISSUE_STIMULUS_LEARNING_RATE),
        channel_key=channel_key,
        embedding=emb,
        distress=0.0,
    )
    TISSUE.snapshot()

    phi_stats = get_phi_stats()

    if broadcast:
        try:
            await manager.broadcast(
                {
                    "type": "tissue.update",
                    "correlation_id": correlation_id or doc_id,
                    "timestamp": _now_iso(),
                    "stats": {
                        "embedding_similarity": phi_stats.get("embedding_similarity", 0.0),
                        "novelty_zscore": phi_stats.get("novelty_zscore", 0.0),
                        "polarity_diff": phi_stats.get("polarity_diff", 0.0),
                        "mean_abs_activation": phi_stats.get("mean_abs_activation", 0.0),
                    },
                    "metadata": {
                        "doc_id": doc_id,
                        "channel_key": channel_key,
                    },
                }
            )
        except Exception as exc:
            logger.warning("tissue ws broadcast failed doc_id=%s error=%s", doc_id, exc)

    return phi_stats


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
