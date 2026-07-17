from __future__ import annotations

import numpy as np
import pytest

from app.models import ModelSpec
from app.services.training import _build_clusterer, _build_reducer


def _model_spec(**overrides) -> ModelSpec:
    base = dict(
        algorithm="hdbscan",
        embedding_source_url="http://fake-embedding:8320/embedding",
        min_cluster_size=5,
        metric="euclidean",
        params={},
    )
    base.update(overrides)
    return ModelSpec(**base)


def test_build_reducer_returns_none_for_tiny_sample_count():
    spec = _model_spec()
    assert _build_reducer(spec, n_samples=3) is None


def test_build_reducer_clamps_n_neighbors_below_sample_count():
    # Default umap_n_neighbors (service setting) is 15 -- with only 10 samples,
    # UMAP would raise if n_neighbors were left at 15 (must be < n_samples).
    spec = _model_spec()
    reducer = _build_reducer(spec, n_samples=10)
    assert reducer is not None
    assert reducer.n_neighbors <= 9
    assert reducer.n_neighbors >= 2


def test_build_reducer_respects_model_spec_params_override():
    spec = _model_spec(params={"umap_n_neighbors": 4, "umap_n_components": 2, "umap_metric": "euclidean"})
    reducer = _build_reducer(spec, n_samples=50)
    assert reducer.n_neighbors == 4
    assert reducer.n_components == 2
    assert reducer.metric == "euclidean"


def test_build_reducer_uses_service_defaults_when_params_empty():
    from app.settings import settings

    spec = _model_spec(params={})
    reducer = _build_reducer(spec, n_samples=100)
    assert reducer.n_neighbors == settings.topic_foundry_umap_n_neighbors
    assert reducer.n_components == settings.topic_foundry_umap_n_components
    assert reducer.metric == settings.topic_foundry_umap_metric


def test_build_clusterer_excludes_umap_reserved_keys_from_hdbscan_kwargs():
    """Regression: _build_clusterer used to forward every key in spec.params
    straight into HDBSCAN(**params) -- including the umap_* override keys
    and `random_state` that _build_reducer reads from that SAME dict.
    HDBSCAN's constructor accepts **kwargs and silently stores unrecognized
    ones, which then get forwarded into the actual distance computation at
    fit time and raise TypeError there. This must not happen for any of the
    reserved keys."""
    spec = _model_spec(
        params={
            "umap_n_neighbors": 4,
            "umap_n_components": 2,
            "umap_min_dist": 0.1,
            "umap_metric": "cosine",
            "random_state": 7,
        }
    )
    clusterer = _build_clusterer(spec)
    # Must not raise when actually fit -- this is what reproduces the bug
    # (a TypeError at fit time, not construction time).
    embeddings = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
    labels = clusterer.fit_predict(embeddings)
    assert len(labels) == 30


def test_reduce_then_cluster_with_umap_param_overrides_does_not_crash():
    """End-to-end proof that the documented umap_* override mechanism
    (exercised by test_build_reducer_respects_model_spec_params_override)
    actually works all the way through _build_clusterer too, not just
    _build_reducer in isolation."""
    spec = _model_spec(
        min_cluster_size=5,
        params={"umap_n_neighbors": 5, "umap_n_components": 2, "random_state": 7},
    )
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(30, 16)).astype(np.float32)

    reducer = _build_reducer(spec, n_samples=len(embeddings))
    assert reducer is not None
    reduced = reducer.fit_transform(embeddings)

    clusterer = _build_clusterer(spec)
    labels = clusterer.fit_predict(reduced)
    assert len(labels) == 30


def test_reduce_then_cluster_end_to_end_on_synthetic_two_blob_embeddings():
    """Proves the actual fix: UMAP-reduced input separates two well-formed
    synthetic clusters that HDBSCAN, run directly on raw high-dimensional
    embeddings with a small min_cluster_size mismatch, would otherwise blur."""
    rng = np.random.default_rng(42)
    # Two well-separated blobs in a high-dimensional space (64-D, mimicking a
    # real embedding model's output size), 40 points each.
    blob_a = rng.normal(loc=0.0, scale=0.05, size=(40, 64)).astype(np.float32)
    blob_b = rng.normal(loc=5.0, scale=0.05, size=(40, 64)).astype(np.float32)
    embeddings = np.vstack([blob_a, blob_b])

    spec = _model_spec(min_cluster_size=10, metric="euclidean")
    reducer = _build_reducer(spec, n_samples=len(embeddings))
    assert reducer is not None

    reduced = reducer.fit_transform(embeddings)
    assert reduced.shape[1] == reducer.n_components
    assert reduced.shape[1] < embeddings.shape[1]

    clusterer = _build_clusterer(spec)
    labels = clusterer.fit_predict(reduced)

    unique_real_labels = set(int(l) for l in labels if l >= 0)
    # The two well-separated blobs must resolve to at least 2 real clusters
    # (not collapse into 1, and not all-outlier).
    assert len(unique_real_labels) >= 2
    outlier_pct = float(sum(1 for l in labels if l < 0)) / len(labels)
    assert outlier_pct < 0.5
