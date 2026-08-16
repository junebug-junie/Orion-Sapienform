"""A context-overflowed request retries on a lane that can hold it.

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

from app.ctx_overflow import (
    MAX_ATTEMPTS,
    build_escalation_ladder,
    is_context_overflow,
    next_larger_route,
)

# The live fleet.
LADDER = [("metacog", 4096), ("quick", 4096), ("agent", 32768), ("chat", 131072)]


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


# ------------------------------------------------------------ escalation

def test_it_skips_a_same_sized_lane():
    """THE REASON THE LADDER IS DISCOVERED. quick and metacog are both 4,096 today, so
    quick -> metacog would fail identically and spend one of only three attempts."""
    assert next_larger_route(LADDER, "quick") == "agent"
    assert next_larger_route(LADDER, "metacog") == "agent"


def test_metacog_becomes_the_fallback_automatically_once_its_context_is_raised():
    """Juniper's actual preference. Raise metacog's n_ctx and it starts winning with NO code
    change -- and circe drops out of the journaling path entirely."""
    raised = [("quick", 4096), ("metacog", 16384), ("agent", 32768), ("chat", 131072)]
    assert next_larger_route(raised, "quick") == "metacog"


def test_it_escalates_to_the_smallest_lane_that_fits_not_the_biggest():
    """Jumping straight to circe's 131k lane for a 5k prompt would re-create the problem this
    whole change exists to fix."""
    assert next_larger_route(LADDER, "agent") == "chat"


def test_the_largest_lane_has_nowhere_to_escalate():
    assert next_larger_route(LADDER, "chat") is None


def test_an_unknown_route_escalates_to_the_largest_rather_than_giving_up():
    """A lane whose /props could not be read is not in the ladder. Something beats a silent
    failure."""
    assert next_larger_route(LADDER, "mystery") == "chat"


def test_an_empty_ladder_yields_nothing():
    assert next_larger_route([], "quick") is None


# ------------------------------------------------------------ the cap

def test_the_attempt_cap_is_three():
    """Juniper's cap. TOTAL attempts, not retries after the first."""
    assert MAX_ATTEMPTS == 3


def test_walking_the_ladder_terminates_within_the_cap():
    """Even from the bottom of a 4-lane ladder, escalation cannot exceed the cap."""
    route, attempts = "metacog", 1
    while attempts < MAX_ATTEMPTS:
        nxt = next_larger_route(LADDER, route)
        if nxt is None:
            break
        route, attempts = nxt, attempts + 1
    assert attempts <= MAX_ATTEMPTS
    assert route == "chat"      # metacog -> agent -> chat, exactly 3 attempts


# ------------------------------------------------------------ building the ladder

def test_the_ladder_is_ordered_by_real_context_size(monkeypatch):
    sizes = {"http://q": 4096, "http://c": 131072, "http://m": 4096}
    monkeypatch.setattr("app.ctx_overflow.probe_context_size", lambda url, **kw: sizes.get(url))
    ladder = build_escalation_ladder({"quick": "http://q", "chat": "http://c", "metacog": "http://m"})
    assert [n for _, n in ladder] == sorted(n for _, n in ladder)
    assert ladder[-1][0] == "chat"


def test_an_unprobeable_lane_is_omitted_rather_than_guessed(monkeypatch):
    """A guessed context window is how a lane ends up carrying prompts it cannot hold. Absent
    beats assumed -- the same rule the rest of this arc enforces on measurements."""
    monkeypatch.setattr("app.ctx_overflow.probe_context_size",
                        lambda url, **kw: 4096 if url == "http://q" else None)
    ladder = build_escalation_ladder({"quick": "http://q", "dead": "http://d"})
    assert ladder == [("quick", 4096)]


# ------------------------------------------------------------ the wiring, not just the parts

class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
    def json(self):
        return self._body


class _Client:
    """Records every URL posted to, so escalation is observable."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.posted = []
    def post(self, url, json=None):
        self.posted.append(url)
        return self._responses.pop(0)


OVERFLOW = _Resp(400, {"error": {"message": "the request exceeds the available context size"}})
OK = _Resp(200, {"choices": [{"message": {"content": "hi"}}]})


def _wire(monkeypatch, targets, ladder):
    import app.llm_backend as b
    monkeypatch.setattr(b, "get_route_targets", lambda: targets)
    monkeypatch.setattr(b, "_CTX_LADDER", ladder)
    return b


class _T:
    def __init__(self, url): self.url = url


def test_an_overflow_escalates_and_the_path_is_preserved(monkeypatch):
    """The whole point end to end: a 4k lane rejects, a bigger lane serves it, and the endpoint
    path (/v1/chat/completions) survives the host swap."""
    b = _wire(monkeypatch,
              {"quick": _T("http://atlas:8013"), "chat": _T("http://circe:8011")},
              [("quick", 4096), ("chat", 131072)])
    c = _Client([OVERFLOW, OK])
    r, route, url = b._post_with_ctx_escalation(
        c, "http://atlas:8013/v1/chat/completions", {}, "quick", "corr-1")
    assert r.status_code == 200
    assert route == "chat"
    assert c.posted == ["http://atlas:8013/v1/chat/completions",
                        "http://circe:8011/v1/chat/completions"]


def test_a_request_that_fits_is_never_retried(monkeypatch):
    """97.8% of journal traffic. One post, no probing, no behaviour change."""
    b = _wire(monkeypatch, {"quick": _T("http://atlas:8013")}, [("quick", 4096)])
    c = _Client([OK])
    r, route, _ = b._post_with_ctx_escalation(c, "http://atlas:8013/x", {}, "quick", "c")
    assert r.status_code == 200 and route == "quick" and len(c.posted) == 1


def test_a_non_overflow_error_is_returned_untouched(monkeypatch):
    """A malformed request must not be retried up every lane in the fleet."""
    b = _wire(monkeypatch,
              {"quick": _T("http://atlas:8013"), "chat": _T("http://circe:8011")},
              [("quick", 4096), ("chat", 131072)])
    c = _Client([_Resp(400, {"error": {"message": "invalid model parameter"}})])
    r, route, _ = b._post_with_ctx_escalation(c, "http://atlas:8013/x", {}, "quick", "c")
    assert r.status_code == 400 and route == "quick" and len(c.posted) == 1


def test_attempts_are_capped_at_three(monkeypatch):
    """Juniper's cap, enforced against a ladder deep enough to exceed it."""
    b = _wire(monkeypatch,
              {"a": _T("http://h:1"), "b": _T("http://h:2"), "c": _T("http://h:3"), "d": _T("http://h:4")},
              [("a", 1000), ("b", 2000), ("c", 3000), ("d", 4000)])
    c = _Client([OVERFLOW, OVERFLOW, OVERFLOW])
    r, _, _ = b._post_with_ctx_escalation(c, "http://h:1/x", {}, "a", "c")
    assert len(c.posted) == 3
    assert r.status_code == 400


def test_overflow_on_the_largest_lane_returns_the_error(monkeypatch):
    """Nowhere left to go. The error surfaces rather than looping."""
    b = _wire(monkeypatch, {"chat": _T("http://circe:8011")}, [("chat", 131072)])
    c = _Client([OVERFLOW])
    r, route, _ = b._post_with_ctx_escalation(c, "http://circe:8011/x", {}, "chat", "c")
    assert r.status_code == 400 and route == "chat" and len(c.posted) == 1


def test_a_non_json_body_is_not_treated_as_an_overflow(monkeypatch):
    class _Bad:
        status_code = 500
        def json(self): raise ValueError("not json")
    b = _wire(monkeypatch, {"quick": _T("http://a:1"), "chat": _T("http://c:1")},
              [("quick", 4096), ("chat", 131072)])
    c = _Client([_Bad()])
    r, route, _ = b._post_with_ctx_escalation(c, "http://a:1/x", {}, "quick", "c")
    assert route == "quick" and len(c.posted) == 1
