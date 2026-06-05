from __future__ import annotations

import pytest
import requests

from orion.graph.fuseki_http import (
    call_with_fuseki_retry,
    is_fuseki_lock_exhaustion,
    is_fuseki_retryable_http_error,
)


def test_is_fuseki_lock_exhaustion() -> None:
    assert is_fuseki_lock_exhaustion(500, "Maximum lock count exceeded")
    assert not is_fuseki_lock_exhaustion(400, "Maximum lock count exceeded")
    assert not is_fuseki_lock_exhaustion(500, "bad request")


def test_is_fuseki_retryable_http_error() -> None:
    assert is_fuseki_retryable_http_error(503, "")
    assert is_fuseki_retryable_http_error(500, "Maximum lock count exceeded")
    assert not is_fuseki_retryable_http_error(400, "")


def test_call_with_fuseki_retry_recovers_from_lock_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _response(status: int, text: str = "") -> requests.Response:
        resp = requests.Response()
        resp.status_code = status
        resp._content = text.encode("utf-8")
        return resp

    def _fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.HTTPError(response=_response(500, "Maximum lock count exceeded"))
        return "ok"

    monkeypatch.setenv("FUSEKI_HTTP_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("FUSEKI_HTTP_RETRY_BASE_DELAY_SEC", "0.01")
    assert call_with_fuseki_retry(_fn) == "ok"
    assert calls["n"] == 2


def test_call_with_fuseki_retry_does_not_retry_400(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fn():
        resp = requests.Response()
        resp.status_code = 400
        resp._content = b"bad"
        raise requests.HTTPError(response=resp)

    monkeypatch.setenv("FUSEKI_HTTP_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("FUSEKI_HTTP_RETRY_BASE_DELAY_SEC", "0.01")
    with pytest.raises(requests.HTTPError):
        call_with_fuseki_retry(_fn)
