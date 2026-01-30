from app.topic_rail.persistence.model_store import ModelStore
from app.topic_rail.embeddings.vector_host import VectorHostEmbeddingProvider
from app.main import TopicRailService
from app.settings import settings


def test_manifest_written_and_loaded(tmp_path, monkeypatch):
    store = ModelStore(str(tmp_path))
    manifest = {
        "model_version": "v1",
        "created_at": "2025-01-01T00:00:00+00:00",
        "embedding_endpoint_url": "http://embed",
        "embedding_model_name": "test-model",
        "embedding_dim": 384,
        "bertopic_params": {"n_neighbors": 15},
    }
    store.write_manifest("v1", manifest)
    loaded = store.load_manifest("v1")
    assert loaded["embedding_dim"] == 384


def test_manifest_validation_dim_mismatch(monkeypatch):
    service = TopicRailService()
    service.embedder._last_dim = 128
    manifest = {"embedding_dim": 384}
    try:
        service._validate_manifest(manifest)
    except RuntimeError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("Expected dimension mismatch error")


def test_manifest_validation_model_mismatch_requires_flag(monkeypatch):
    service = TopicRailService()
    service.embedder._last_model = "current"
    manifest = {"embedding_model_name": "other"}
    monkeypatch.setattr(settings, "topic_rail_allow_embed_model_mismatch", False)
    try:
        service._validate_manifest(manifest)
    except RuntimeError as exc:
        assert "model mismatch" in str(exc)
    else:
        raise AssertionError("Expected model mismatch error")
