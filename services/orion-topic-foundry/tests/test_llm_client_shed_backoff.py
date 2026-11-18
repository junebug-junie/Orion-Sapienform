"""A topic-foundry batch makes one LLM call per topic/boundary. While the GPU pool sheds the route,
calls are skipped (None, exactly what a refused call returned) instead of hitting the gateway;
the first call after the heat clears goes through (orion/gpu_pool/shed_backoff.py)."""
from __future__ import annotations

import pytest

from orion.core.bus.bus_schemas import ChatResultPayload
from orion.gpu_pool.shed_backoff import BASE_SEC, ShedBackoff

from app.services import llm_client as llm_mod
from app.settings import settings

SHED = ChatResultPayload(content="", raw={"error": "gpu_pool_unavailable",
                                          "details": {"reason": "shed:cabinet_hot", "retry_after_sec": 45.0}})
OK = ChatResultPayload(content='{"labels": {"1": "x"}}')
OTHER = ChatResultPayload(content="", raw={"error": "gpu_pool_unavailable", "details": {"reason": "deadline"}})


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "topic_foundry_llm_use_bus", True)
    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    c = llm_mod.TopicFoundryLLMClient()
    clock = Clock()
    c._shed_backoff = ShedBackoff(name="topic-foundry", clock=clock, rng=lambda: 0.5, log=lambda m: None)
    c.calls = 0
    c.reply = SHED

    def fake(**kw):
        c.calls += 1
        return c.reply

    monkeypatch.setattr(c, "_bus_chat_result", fake)
    return c, clock


def _ask(c):
    return c.request_json(system_prompt="s", user_prompt="u", temperature=0)


def test_batch_during_shed_reaches_the_gateway_once(client):
    c, clock = client
    assert [_ask(c) for _ in range(400)] == [None] * 400
    assert c.calls == 1
    clock.t += 44.0                              # the gateway's hint (45 s) is longer than BASE_SEC
    assert _ask(c) is None and c.calls == 1
    c.reply = OK                                 # heat clears
    clock.t += 2.0
    assert _ask(c) == {"labels": {"1": "x"}}
    assert _ask(c) == {"labels": {"1": "x"}}
    assert c.calls == 3


def test_non_shed_refusal_unchanged(client):
    c, _ = client
    c.reply = OTHER
    assert [_ask(c) for _ in range(3)] == [None] * 3
    assert c.calls == 3


def test_base_window_without_hint(client):
    c, clock = client
    c.reply = ChatResultPayload(content="", raw={"error": "gpu_pool_unavailable",
                                                 "details": {"reason": "shed:cabinet_hot"}})
    _ask(c)
    clock.t += BASE_SEC - 1
    _ask(c)
    assert c.calls == 1
    clock.t += 1
    _ask(c)
    assert c.calls == 2


def test_worker_thread_returning_none_is_a_failed_call_not_a_crash(client):
    c, clock = client
    c.reply = None
    assert _ask(c) is None


def test_gateway_deadline_releases_the_probe_without_ending_the_episode(client):
    c, clock = client
    c.reply = ChatResultPayload(content="", raw={"error": "gpu_pool_unavailable",
                                                 "details": {"reason": "shed:cabinet_hot"}})
    _ask(c)
    clock.t += BASE_SEC
    c.reply = ChatResultPayload(content="", raw={"error": "gpu_pool_unavailable", "details": {"reason": "deadline"}})
    _ask(c)                                       # the probe: neutral
    assert c._shed_backoff.reason(c._shed_key()) == "shed:cabinet_hot"
    _ask(c)                                       # released, so the next call probes at once
    assert c.calls == 3
