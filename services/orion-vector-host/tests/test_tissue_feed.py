from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app import tissue_feed


def _rand_embedding(dim: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


@pytest.fixture(autouse=True)
def _reset_tissue_state():
    """Each test gets a clean TISSUE tensor + EMA store so tests don't leak
    state into each other via the module-level singleton."""
    tissue_feed.TISSUE.T[:] = 0.0
    tissue_feed.TISSUE.expectations.clear()
    tissue_feed.TISSUE.expectations["chat"] = tissue_feed.TISSUE.expectation.copy()
    tissue_feed.TISSUE.embedding_expectations.clear()
    tissue_feed.TISSUE.last_embedding_input.clear()
    tissue_feed.TISSUE.last_novelty_per_channel.clear()
    tissue_feed.TISSUE.novelty_stats.clear()
    tissue_feed.TISSUE.coherence_stats.clear()
    tissue_feed._EMBEDDING_EMA.clear()
    yield


def test_feed_tissue_returns_real_phi_stats():
    stats = asyncio.run(
        tissue_feed.feed_tissue(_rand_embedding(seed=1), doc_id="doc-1", broadcast=False)
    )
    assert set(stats.keys()) == {"polarity_diff", "mean_abs_activation", "embedding_similarity", "novelty_zscore"}
    assert all(isinstance(v, float) for v in stats.values())


def test_feed_tissue_first_call_is_maximally_novel():
    """No prior EMA for this channel -> delta == the (unit) embedding itself,
    the same "no reference yet, treat as new" convention the old
    OrionTissue.calculate_novelty used for an empty expectation."""
    stats = asyncio.run(
        tissue_feed.feed_tissue(_rand_embedding(seed=2), doc_id="doc-1", broadcast=False)
    )
    # calculate_novelty() ran (not dead/never-called, the bug this module's
    # docstring documents fixing) -- mean_abs_activation/novelty_zscore must
    # not all be the zero-initialized tensor's default.
    assert stats["mean_abs_activation"] != 0.0 or stats["novelty_zscore"] != 0.0


def test_feed_tissue_seeds_ema_from_persisted_tissue_expectation():
    """Restart-amnesia guard (found by code review, 2026-07-28): if
    TISSUE.embedding_expectations already has a real (persisted, survived a
    restart) entry for a channel key, the first feed_tissue() call for that
    key in this process must seed _EMBEDDING_EMA from it rather than treating
    the turn as maximally novel."""
    emb = _rand_embedding(seed=42)
    persisted = emb / np.linalg.norm(emb)
    tissue_feed.TISSUE.embedding_expectations["chat"] = persisted.astype(np.float32)

    assert "chat" not in tissue_feed._EMBEDDING_EMA
    asyncio.run(tissue_feed.feed_tissue(emb, doc_id="doc-seed", broadcast=False))

    # Same embedding as the "persisted prior" -> delta should be ~0, i.e. the
    # EMA moved only slightly from the seed, not from a zero/empty start.
    ema_after = tissue_feed._EMBEDDING_EMA["chat"]
    assert np.linalg.norm(ema_after - persisted) < np.linalg.norm(persisted)


def test_feed_tissue_repeated_identical_embedding_reduces_novelty_signal():
    """Feeding the same embedding repeatedly should make the running EMA
    converge toward it, shrinking delta magnitude over time -- confirms this
    is live, responsive physics rather than a fixed/degenerate reading."""
    emb = _rand_embedding(seed=3)
    magnitudes = []
    for _ in range(8):
        prior_ema = tissue_feed._EMBEDDING_EMA.get("chat")
        asyncio.run(tissue_feed.feed_tissue(emb, doc_id="doc-repeat", broadcast=False))
        new_ema = tissue_feed._EMBEDDING_EMA["chat"]
        if prior_ema is not None:
            magnitudes.append(float(np.linalg.norm(new_ema - prior_ema)))
    # EMA step size should shrink tick over tick as it converges on emb.
    assert magnitudes[-1] < magnitudes[0]


def test_feed_tissue_distinct_embeddings_produce_distinct_stimulus_direction():
    """Two very different embeddings should not collapse to the same
    projected stimulus -- confirms the projection is content-derived, not a
    constant/degenerate broadcast."""
    a = _rand_embedding(dim=32, seed=10)
    b = -_rand_embedding(dim=32, seed=10)  # maximally different direction

    stimulus_a = tissue_feed._build_stimulus(a, magnitude=1.0)
    stimulus_b = tissue_feed._build_stimulus(b, magnitude=1.0)
    assert not np.allclose(stimulus_a, stimulus_b)


def test_feed_tissue_zero_embedding_does_not_crash():
    stats = asyncio.run(
        tissue_feed.feed_tissue(np.zeros(16, dtype=np.float32), doc_id="doc-zero", broadcast=False)
    )
    assert set(stats.keys()) == {"polarity_diff", "mean_abs_activation", "embedding_similarity", "novelty_zscore"}


def test_feed_tissue_empty_embedding_short_circuits():
    stats = asyncio.run(tissue_feed.feed_tissue([], doc_id="doc-empty", broadcast=False))
    assert set(stats.keys()) == {"polarity_diff", "mean_abs_activation", "embedding_similarity", "novelty_zscore"}
    assert "chat" not in tissue_feed._EMBEDDING_EMA or tissue_feed._EMBEDDING_EMA.get("chat") is None


def test_projection_matrix_is_deterministic_for_same_dim():
    m1 = tissue_feed._projection_matrix(64)
    m2 = tissue_feed._projection_matrix(64)
    assert m1 is m2  # cached, not rebuilt


def test_get_phi_stats_matches_tissue_phi():
    asyncio.run(tissue_feed.feed_tissue(_rand_embedding(seed=5), doc_id="doc-5", broadcast=False))
    assert tissue_feed.get_phi_stats() == {
        k: float(v) for k, v in tissue_feed.TISSUE.phi().items()
    }
