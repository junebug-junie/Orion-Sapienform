from __future__ import annotations

import numpy as np
import pytest

from app.models import ModelSpec
from app.services.training import _build_clusterer, _build_reducer, _resolve_keyword_stop_words


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


def test_build_clusterer_uses_settings_fallback_for_min_samples_and_selection_method(monkeypatch):
    """Live incident 2026-07-21: TOPIC_FOUNDRY_HDBSCAN_MIN_SAMPLES/
    _CLUSTER_SELECTION_METHOD had no fallback at all in _build_clusterer --
    only app/topic_engine.py (dead code for the real training path) ever
    read them. Mirrors _build_reducer's existing settings-fallback pattern
    for UMAP params, applied here for the first time."""
    from app.services.training import settings as training_settings

    monkeypatch.setattr(training_settings, "topic_foundry_hdbscan_min_samples", 3)
    monkeypatch.setattr(training_settings, "topic_foundry_hdbscan_cluster_selection_method", "leaf")
    spec = _model_spec(params={})
    clusterer = _build_clusterer(spec)
    assert clusterer.min_samples == 3
    assert clusterer.cluster_selection_method == "leaf"


def test_build_clusterer_per_model_params_override_settings_fallback(monkeypatch):
    """A model explicitly setting min_samples/cluster_selection_method in
    its own spec.params must win over the service-wide settings fallback --
    matches the existing override precedent for UMAP params."""
    from app.services.training import settings as training_settings

    monkeypatch.setattr(training_settings, "topic_foundry_hdbscan_min_samples", 5)
    monkeypatch.setattr(training_settings, "topic_foundry_hdbscan_cluster_selection_method", "eom")
    spec = _model_spec(params={"min_samples": 2, "cluster_selection_method": "leaf"})
    clusterer = _build_clusterer(spec)
    assert clusterer.min_samples == 2
    assert clusterer.cluster_selection_method == "leaf"


def test_build_clusterer_excludes_vectorizer_reserved_keys_from_hdbscan_kwargs():
    """Live incident 2026-07-21: setting stop_words_extra/vectorizer_stop_words
    in model_spec.params (the only documented way to reach
    _resolve_keyword_stop_words) crashed HDBSCAN's constructor directly --
    TypeError: __init__() got an unexpected keyword argument
    'vectorizer_stop_words' -- unlike the umap_* keys above, which HDBSCAN's
    constructor silently accepts and only fails on later at fit time. Both
    failure shapes are covered: this one at construction, the umap one above
    at fit."""
    spec = _model_spec(
        params={
            "vectorizer_stop_words": "english",
            "stop_words_extra": "hi,hey,juniper,orion",
        }
    )
    clusterer = _build_clusterer(spec)  # must not raise
    embeddings = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
    labels = clusterer.fit_predict(embeddings)
    assert len(labels) == 30


def test_resolve_keyword_stop_words_defaults_to_english():
    assert _resolve_keyword_stop_words({}) == "english"


def test_resolve_keyword_stop_words_none_disables_filtering():
    for off in ("none", "off", "false", "0", ""):
        assert _resolve_keyword_stop_words({"vectorizer_stop_words": off}) is None


def test_resolve_keyword_stop_words_extends_english_list():
    """The actual live defect this fixes: sklearn's 'english' list does not
    cover conversational filler at all -- confirmed live, none of
    hi/like/just/let/know/need are in ENGLISH_STOP_WORDS."""
    result = _resolve_keyword_stop_words({"stop_words_extra": "hi, Juniper, orion"})
    assert isinstance(result, list)
    assert "hi" in result
    assert "juniper" in result  # lowercased
    assert "the" in result  # base english list still present


def test_resolve_keyword_stop_words_custom_list_replaces_english():
    result = _resolve_keyword_stop_words({"vectorizer_stop_words": "foo,bar"})
    assert result == ["foo", "bar"]


def test_llm_label_topics_disabled_returns_empty(monkeypatch):
    from app.services import training

    monkeypatch.setattr(training.settings, "topic_foundry_llm_enable", False)
    result = training._llm_label_topics({"0": ["meow", "cat"]}, {"0": ["hi cat"]})
    assert result == {}


def test_llm_label_topics_parses_real_response(monkeypatch):
    from app.services import training

    monkeypatch.setattr(training.settings, "topic_foundry_llm_enable", True)
    monkeypatch.setattr(training, "_LLM_LABEL_BATCH_SIZE", 1)

    per_topic_label = {"0": "Cat Persona", "1": "Hardware Setup"}

    class _FakeClient:
        def request_json(self, **kwargs):
            # One call per topic at batch size 1 -- each call's prompt names
            # exactly the one topic it's labeling.
            for tid, label in per_topic_label.items():
                if f"Topic {tid}:" in kwargs["user_prompt"]:
                    return {"labels": {tid: label}}
            raise AssertionError(f"unexpected prompt: {kwargs['user_prompt']!r}")

    monkeypatch.setattr(training, "get_llm_client", lambda: _FakeClient())
    result = training._llm_label_topics(
        {"0": ["meow", "cat", "purrrr"], "1": ["gpu", "server"]},
        {"0": ["meow meow"], "1": ["setting up the gpu"]},
    )
    assert result == {"0": "Cat Persona", "1": "Hardware Setup"}


def test_llm_label_topics_batches_to_respect_llm_route_token_cap(monkeypatch):
    """Live incident 2026-07-21: a single call listing all ~29 topics at once
    always came back truncated -- confirmed via orion-llm-gateway's own logs
    that the "quick" bus route hard-caps completion at ~146 tokens
    regardless of the requested max_tokens (tried None, then 1740 -- same
    cutoff both times). This is a route-level constraint, not something the
    caller's max_tokens can raise. The real fix is batching small enough
    that each call's response comfortably fits -- this locks in that
    _llm_label_topics makes multiple small calls rather than one big one,
    and that a partial per-batch failure doesn't blank out labels for
    topics in other, successful batches."""
    from app.services import training

    monkeypatch.setattr(training.settings, "topic_foundry_llm_enable", True)
    monkeypatch.setattr(training, "_LLM_LABEL_BATCH_SIZE", 2)

    call_prompts = []

    class _FlakyClient:
        def request_json(self, **kwargs):
            call_prompts.append(kwargs["user_prompt"])
            # Simulate the real truncation failure mode for one specific
            # batch (topics 2/3) while others succeed -- proves partial
            # failure doesn't wipe out the rest.
            if "Topic 2:" in kwargs["user_prompt"]:
                return None  # truncated response -> unparseable JSON upstream
            labels = {}
            for tid in ("0", "1", "3", "4"):
                if f"Topic {tid}:" in kwargs["user_prompt"]:
                    labels[tid] = f"Label {tid}"
            return {"labels": labels} if labels else None

    monkeypatch.setattr(training, "get_llm_client", lambda: _FlakyClient())
    keywords = {str(i): [f"kw{i}"] for i in range(5)}
    samples = {str(i): [f"sample {i}"] for i in range(5)}
    result = training._llm_label_topics(keywords, samples)

    # 5 topics at batch size 2 -> 3 calls, not 1.
    assert len(call_prompts) == 3
    # Topic 2's batch failed -- topic 2 (and its batch-mate 3, if bundled
    # together) may be missing, but topics from the OTHER, successful
    # batches must still be present.
    assert "0" in result and result["0"] == "Label 0"
    assert "1" in result and result["1"] == "Label 1"
    assert "4" in result and result["4"] == "Label 4"
    assert "2" not in result


def test_llm_label_topics_fails_open_on_bad_response(monkeypatch):
    from app.services import training

    monkeypatch.setattr(training.settings, "topic_foundry_llm_enable", True)

    class _FakeClient:
        def request_json(self, **kwargs):
            return {"not_labels": "oops"}

    monkeypatch.setattr(training, "get_llm_client", lambda: _FakeClient())
    result = training._llm_label_topics({"0": ["meow"]}, {"0": ["hi"]})
    assert result == {}


def test_llm_label_topics_fails_open_on_exception(monkeypatch):
    from app.services import training

    monkeypatch.setattr(training.settings, "topic_foundry_llm_enable", True)

    class _FakeClient:
        def request_json(self, **kwargs):
            raise RuntimeError("bus down")

    monkeypatch.setattr(training, "get_llm_client", lambda: _FakeClient())
    result = training._llm_label_topics({"0": ["meow"]}, {"0": ["hi"]})
    assert result == {}


def test_compute_topic_artifacts_applies_llm_labels_when_enabled(monkeypatch):
    from app.services.training import _compute_topic_artifacts
    from app.services import training
    from app.services.types import RowBlock

    monkeypatch.setattr(training.settings, "topic_foundry_llm_enable", True)

    class _FakeClient:
        def request_json(self, **kwargs):
            return {"labels": {"0": "Cat Persona"}}

    monkeypatch.setattr(training, "get_llm_client", lambda: _FakeClient())

    segments = [
        RowBlock(doc_id="a", text="meow meow purr cat", row_ids=["r1"], timestamps=["t1"]),
        RowBlock(doc_id="b", text="meow cat playful whiskers", row_ids=["r2"], timestamps=["t2"]),
    ]
    labels = np.array([0, 0])
    summary, keywords = _compute_topic_artifacts(segments, labels)
    assert summary[0]["topic_id"] == 0
    assert summary[0]["label"] == "Cat Persona"
    assert "meow" in keywords["0"] or "cat" in keywords["0"]


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
