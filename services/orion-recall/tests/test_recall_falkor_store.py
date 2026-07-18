"""get_recall_falkor_client: lazy singleton, never raises, self-heals on retry."""

from __future__ import annotations

from app import recall_falkor_store


def _reset():
    recall_falkor_store._CLIENT = None


def test_no_falkordb_uri_returns_none(monkeypatch) -> None:
    _reset()
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    assert recall_falkor_store.get_recall_falkor_client() is None
    _reset()


def test_construction_failure_returns_none_and_does_not_raise(monkeypatch) -> None:
    _reset()
    monkeypatch.setenv("FALKORDB_URI", "redis://example.test:6379")

    def _raise(*, uri, graph_name):
        raise RuntimeError("boom")

    monkeypatch.setattr(recall_falkor_store, "RedisGraphQueryClient", _raise)
    assert recall_falkor_store.get_recall_falkor_client() is None
    _reset()


def test_successful_construction_is_cached_across_calls(monkeypatch) -> None:
    _reset()
    monkeypatch.setenv("FALKORDB_URI", "redis://example.test:6379")
    monkeypatch.setenv("FALKORDB_RECALL_GRAPH", "orion_recall")

    build_calls = []

    class _FakeClient:
        def __init__(self, *, uri, graph_name):
            build_calls.append((uri, graph_name))

    monkeypatch.setattr(recall_falkor_store, "RedisGraphQueryClient", _FakeClient)

    first = recall_falkor_store.get_recall_falkor_client()
    second = recall_falkor_store.get_recall_falkor_client()
    assert first is second
    assert len(build_calls) == 1
    assert build_calls[0] == ("redis://example.test:6379", "orion_recall")
    _reset()


def test_retries_construction_on_next_call_after_a_failure(monkeypatch) -> None:
    """Transient failure self-heals -- unlike a permanent-failure flag, this
    module retries construction every call while still uninitialized."""
    _reset()
    monkeypatch.setenv("FALKORDB_URI", "redis://example.test:6379")

    attempts = {"n": 0}

    class _FlakyClient:
        def __init__(self, *, uri, graph_name):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("first attempt fails")

    monkeypatch.setattr(recall_falkor_store, "RedisGraphQueryClient", _FlakyClient)

    assert recall_falkor_store.get_recall_falkor_client() is None
    second = recall_falkor_store.get_recall_falkor_client()
    assert second is not None
    assert attempts["n"] == 2
    _reset()
