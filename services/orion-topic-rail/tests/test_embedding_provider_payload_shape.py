from app.topic_rail.embeddings.vector_host import VectorHostEmbeddingProvider


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_embedding_provider_payload_shape(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"embedding": [0.1, 0.2], "embedding_model": "test", "embedding_dim": 2})

    monkeypatch.setattr("requests.post", fake_post)

    provider = VectorHostEmbeddingProvider("http://vector-host/embedding")
    embeddings = provider.embed_texts(["hello"])

    assert embeddings == [[0.1, 0.2]]
    assert captured["json"]["doc_id"].startswith("topic-rail-")
    assert captured["json"]["text"] == "hello"
    assert captured["json"]["embedding_profile"] == "default"
    assert captured["json"]["include_latent"] is False
