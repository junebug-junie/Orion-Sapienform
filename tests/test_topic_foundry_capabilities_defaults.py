import sys

sys.path.insert(0, "services/orion-topic-foundry")

from fastapi.testclient import TestClient

from app.main import app


def test_capabilities_backward_compatible_with_defaults_block():
    client = TestClient(app)
    response = client.get("/capabilities")
    assert response.status_code == 200
    payload = response.json()

    # Backward-compatible keys
    assert "capabilities" in payload
    assert "backends" in payload
    assert "topic_modeling" in payload["capabilities"]
    assert "representations" in payload["backends"]

    # Optional new defaults block
    assert "defaults" in payload
    vec = payload["defaults"]["vectorizer"]
    assert isinstance(vec["ngram_range"], list) and len(vec["ngram_range"]) == 2
    assert isinstance(payload["defaults"]["ctfidf"]["reduce_frequent_words"], bool)
    assert "supported_metrics" in payload
    assert "cosine" in payload["supported_metrics"]
    assert payload.get("default_metric")
