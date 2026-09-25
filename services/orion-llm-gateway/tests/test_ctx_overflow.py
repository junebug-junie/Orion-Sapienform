"""Context-overflow DETECTION (the escalation ladder is gone: main.py re-leases from the GPU pool
with a larger min_ctx_tokens instead -- see test_pool_dispatch.py).

Historical note kept below: why a retry exists at all.

Orion's journaling ran on circe's 131k-token lane to carry a 1,749-token median prompt. Moving it
to atlas is right for 97.8% of it -- and would silently fail the 2.2% tail (max observed 4,966
tokens against a 4,096 window) without this.

The ladder is DISCOVERED from each lane's real `n_ctx`, not hardcoded, because `metacog` -- the
lane we actually want as the fallback (atlas, 4 slots, 0.00% all-busy over 27.74 h, better quant)
-- is currently the same 4,096 as `quick`. Hardcoding `quick -> chat` would be right today and
wrong the moment metacog's context is raised.

Real sizes, measured 2026-08-16:
    quick 4,096    metacog 4,096    agent 32,768    chat 131,072
"""
from __future__ import annotations

import pytest

from app.ctx_overflow import CONTEXT_OVERFLOW_ERROR, is_context_overflow
from app.llm_backend import _context_overflow_result


# ------------------------------------------------------------ what counts as an overflow

@pytest.mark.parametrize("msg", [
    "the request exceeds the available context size",
    "Context size exceeded",
    "prompt is too long",
    "n_ctx too small for this request",
    "too many tokens in prompt",
])
def test_real_overflow_messages_are_recognised(msg):
    assert is_context_overflow(400, {"error": {"message": msg}}) is True


@pytest.mark.parametrize("status", [400, 413, 422, 500])
def test_the_statuses_llamacpp_actually_uses_are_covered(status):
    assert is_context_overflow(status, {"error": {"message": "exceeds context size"}}) is True


def test_a_generic_bad_request_is_not_an_overflow():
    """THE LOAD-BEARING NEGATIVE. Treating every 400 as an overflow would retry a malformed
    request up every lane in the fleet -- burning the biggest model on a bug."""
    assert is_context_overflow(400, {"error": {"message": "invalid model parameter"}}) is False
    assert is_context_overflow(400, {"error": "unknown field 'temperatur'"}) is False


@pytest.mark.parametrize("status", [200, 404, 429, 503])
def test_unrelated_statuses_are_never_overflows(status):
    """404 is a missing route, 429 is rate limiting, 503 is a dead upstream. None is fixed by a
    bigger context window, and all three have their own handling."""
    assert is_context_overflow(status, {"error": {"message": "exceeds context size"}}) is False


def test_a_plain_string_body_still_parses():
    assert is_context_overflow(400, "prompt is too long for n_ctx") is True


def test_an_empty_or_odd_body_is_not_an_overflow():
    for body in (None, {}, [], 0, {"error": None}):
        assert is_context_overflow(400, body) is False


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_overflow_response_becomes_a_typed_error_result():
    result = _context_overflow_result(
        _Resp(400, {"error": {"message": "the request exceeds the available context size"}}),
        route="quick", served_by="circe-worker-fast", spark_meta={},
    )
    assert result is not None
    assert result["raw"]["error"] == CONTEXT_OVERFLOW_ERROR
    assert result["raw"]["details"]["served_by"] == "circe-worker-fast"


@pytest.mark.parametrize("status,body", [
    (200, {"choices": []}),
    (400, {"error": {"message": "invalid model parameter"}}),
    (400, ValueError("not json")),
])
def test_non_overflow_responses_are_left_alone(status, body):
    assert _context_overflow_result(_Resp(status, body), route="quick", served_by=None, spark_meta={}) is None
