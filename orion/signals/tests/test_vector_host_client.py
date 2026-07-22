import pytest

from orion.signals import vector_host_client


class _FakeResponse:
    def __init__(self, json_data, status: int = 200):
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self._json


def test_embed_text_returns_none_for_blank_text():
    assert vector_host_client.embed_text("") is None
    assert vector_host_client.embed_text("   ") is None


def test_embed_text_returns_vector_on_success(monkeypatch):
    def fake_post(url, json, timeout):
        assert url.endswith("/embedding")
        assert json["text"] == "hello"
        return _FakeResponse({"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(vector_host_client.requests, "post", fake_post)
    result = vector_host_client.embed_text("hello", base_url="http://fake-vector-host:8320")
    assert result == [0.1, 0.2, 0.3]


def test_embed_text_returns_none_on_http_error(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({}, status=500)

    monkeypatch.setattr(vector_host_client.requests, "post", fake_post)
    assert vector_host_client.embed_text("hello", base_url="http://fake-vector-host:8320") is None


def test_embed_text_returns_none_on_transport_exception(monkeypatch):
    def fake_post(url, json, timeout):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(vector_host_client.requests, "post", fake_post)
    assert vector_host_client.embed_text("hello", base_url="http://fake-vector-host:8320") is None


def test_embed_text_returns_none_on_malformed_embedding(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({"embedding": "not-a-list"})

    monkeypatch.setattr(vector_host_client.requests, "post", fake_post)
    assert vector_host_client.embed_text("hello", base_url="http://fake-vector-host:8320") is None


def test_vector_host_url_uses_env_override(monkeypatch):
    monkeypatch.setenv(vector_host_client.ORION_VECTOR_HOST_URL_ENV, "http://custom-host:1234")
    assert vector_host_client.vector_host_url() == "http://custom-host:1234"


def test_vector_host_url_default_when_unset(monkeypatch):
    monkeypatch.delenv(vector_host_client.ORION_VECTOR_HOST_URL_ENV, raising=False)
    assert vector_host_client.vector_host_url() == vector_host_client._DEFAULT_VECTOR_HOST_URL
