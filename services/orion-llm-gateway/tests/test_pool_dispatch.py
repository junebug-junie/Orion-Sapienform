"""Every gateway LLM call is placed by orion-gpu-pool (stage 3 of the GPU pool spec).

The bus path, the compat GET /routes view, per-role executors and the min_ctx estimate. The
HTTP passthroughs are covered in test_openai_passthrough.py / test_anthropic_passthrough.py.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from app import main as gateway
from app import pool_placement
from app.models import ChatBody
from app.settings import settings

OVERFLOW = {"text": "[Error: context overflow]", "raw": {"error": "context_overflow"}}


def _ok(body, plan) -> Dict[str, Any]:
    return {"text": "hello", "raw": {}, "route": plan.route, "served_by": plan.route_target.served_by,
            "url": plan.route_target.url}


def _body(route: str = "metacog", content: str = "x" * 40, **options: Any) -> ChatBody:
    return ChatBody(route=route, messages=[{"role": "user", "content": content}], options=options)


@pytest.fixture(autouse=True)
def _no_lane_routing(monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)


# ── bus path ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_call_runs_on_the_granted_url_and_reports_its_served_by(fake_pool, monkeypatch):
    seen = {}

    def run(body, plan):
        seen["active"] = fake_pool.active
        return _ok(body, plan)

    monkeypatch.setattr(gateway, "run_llm_chat", run)
    fake_pool.choose = lambda kw: "agent"  # metacog spilled onto the agent card
    result = await gateway._dispatch_chat(_body("metacog"), correlation_id="turn-1", holder="cortex-exec")
    assert result["url"] == "http://pool-agent:8015"
    assert result["served_by"] == "circe-worker-agent"
    assert seen["active"] == 1
    call = fake_pool.calls[0]
    assert (call["work_class"], call["priority"], call["kind"]) == ("metacog", "system", "request")
    assert call["turn_correlation_id"] == "turn-1"
    assert fake_pool.releases == ["ok"]


@pytest.mark.asyncio
async def test_handle_chat_leases_as_the_calling_service_and_carries_served_by(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    corr = str(uuid.uuid4())
    env = BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"), correlation_id=corr,
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route="chat").model_dump(mode="json"),
    )
    out = await gateway.handle_chat(env)
    assert fake_pool.calls[0]["holder"] == "cortex-exec"
    assert fake_pool.calls[0]["turn_correlation_id"] == corr
    assert (fake_pool.calls[0]["work_class"], fake_pool.calls[0]["priority"]) == ("chat", "interactive")
    assert out.payload.meta["served_by"] == "circe-worker-chat"
    assert out.payload.content == "hello"


def test_passthrough_holders_are_http_prefixed():
    # cortex-exec's first-person "I was made to wait" cue skips holders starting with "http:".
    assert pool_placement.HOLDER_OPENAI == "http:openai"
    assert pool_placement.HOLDER_ANTHROPIC == "http:anthropic"


@pytest.mark.asyncio
async def test_lease_unavailable_returns_gpu_pool_unavailable(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run without a grant"))
    fake_pool.unavailable = "no_serviceable_role"
    result = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert result["text"] == "" and result["content"] == ""
    assert result["raw"]["error"] == "gpu_pool_unavailable"
    assert result["raw"]["details"] == {"reason": "no_serviceable_role", "route": "quick", "work_class": "fast"}


@pytest.mark.asyncio
async def test_no_bus_means_unavailable_never_a_guessed_url(monkeypatch):
    monkeypatch.setattr(pool_placement, "_bus", None)
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    result = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert result["raw"]["details"]["reason"] == "pool_bus_unavailable"


@pytest.mark.asyncio
async def test_route_not_in_gpu_pool_yaml_is_refused(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    result = await gateway._dispatch_chat(_body("latents"), correlation_id="c")
    assert result["raw"]["error"] == "route_not_in_gpu_pool"
    assert fake_pool.calls == []


@pytest.mark.asyncio
async def test_min_ctx_estimate_is_chars_over_four_plus_max_tokens(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    body = ChatBody(route="quick", messages=[{"role": "system", "content": "a" * 9},
                                             {"role": "user", "content": "b" * 30}],
                    options={"max_tokens": 100})
    await gateway._dispatch_chat(body, correlation_id="c")
    assert fake_pool.calls[0]["min_ctx_tokens"] == 10 + 100  # ceil(39 / 4) + 100


@pytest.mark.asyncio
async def test_context_overflow_re_leases_once_with_a_bigger_min_ctx(fake_pool, monkeypatch):
    results = iter([OVERFLOW, None])
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: next(results) or _ok(body, plan))
    result = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    # fast (4096/slot) overflowed; the first fast-class role with ctx >= 4097 is agent (131072).
    assert result["text"] == "hello" and result["url"] == "http://pool-agent:8015"
    assert fake_pool.calls[0]["min_ctx_tokens"] == 10
    assert fake_pool.calls[1]["min_ctx_tokens"] == 4096 + 1  # the overflowed role's ctx_per_slot + 1
    assert fake_pool.releases == ["ok", "ok"]  # an overflow is not the GPU's failure


@pytest.mark.asyncio
async def test_a_second_overflow_returns_the_overflow_error(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: dict(OVERFLOW))
    result = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert result["raw"]["error"] == "context_overflow"
    assert len(fake_pool.calls) == 2
    assert fake_pool.releases == ["ok", "ok"]


@pytest.mark.asyncio
async def test_nothing_big_enough_after_overflow_returns_the_overflow(fake_pool, monkeypatch):
    # chat's class has one role (chat, 65536/slot). It overflows; the re-lease at 65537 is refused
    # at once with min_ctx_exceeds_class:65536 -- the answer is the overflow, not
    # gpu_pool_unavailable, and there is no clamp back down to 65536 (no loop).
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: dict(OVERFLOW))
    result = await gateway._dispatch_chat(_body("chat"), correlation_id="c")
    assert result["raw"]["error"] == "context_overflow"
    assert [c["min_ctx_tokens"] for c in fake_pool.calls] == [10, 65536 + 1]
    assert fake_pool.releases == ["ok"]


@pytest.mark.asyncio
async def test_an_upstream_error_result_releases_as_upstream_error(fake_pool, monkeypatch):
    failed = {"text": "[Error: llamacpp failed: 500]", "raw": {}}
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: dict(failed))
    result = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert result["text"].startswith("[Error:")
    assert fake_pool.releases == ["upstream_error"]
    assert len(fake_pool.calls) == 1  # not an overflow: no re-lease


@pytest.mark.asyncio
async def test_a_vision_refusal_is_not_the_gpus_failure(fake_pool, monkeypatch):
    refused = {"text": "[Error: route cannot accept images]", "raw": {"vision": {"status": "refused"}}}
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: dict(refused))
    await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert fake_pool.releases == ["ok"]


@pytest.mark.asyncio
async def test_pool_wait_is_capped_by_the_callers_budget_and_priority(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    await gateway._dispatch_chat(_body("quick", gateway_read_timeout_sec=8), correlation_id="c")
    assert 7.0 < fake_pool.calls[0]["deadline_sec"] <= 8.0
    monkeypatch.setattr(settings, "read_timeout_sec", 900.0)
    await gateway._dispatch_chat(_body("quick_background"), correlation_id="c")
    assert fake_pool.calls[1]["priority"] == "background"
    assert fake_pool.calls[1]["deadline_sec"] <= settings.llm_gateway_pool_background_wait_sec
    await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert fake_pool.calls[2]["deadline_sec"] <= settings.llm_gateway_pool_wait_sec


# ── estimate helper ───────────────────────────────────────────────────────────────────────


def test_estimate_counts_block_content_and_tool_inputs():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a" * 8}, {"type": "image_url", "image_url": {}}]},
        {"role": "assistant", "content": [{"type": "tool_use", "input": {"k": "v"}}]},
    ]
    # 8 text chars + len('{"k": "v"}') == 10 -> 18 chars -> 5 tokens
    assert pool_placement.estimate_min_ctx_tokens(messages, None) == 5
    assert pool_placement.estimate_min_ctx_tokens([], "12", extra="abcd") == 1 + 12


# ── executors ─────────────────────────────────────────────────────────────────────────────


def test_each_granted_role_gets_its_own_executor():
    pool_placement.shutdown_executors()
    try:
        a = pool_placement.executor_for("http://pool-agent:8015")
        b = pool_placement.executor_for("http://pool-fast:8013/")
        assert a is not b
        assert pool_placement.executor_for("http://pool-agent:8015/") is a
        assert a._max_workers == settings.llm_gateway_executor_workers_per_role
    finally:
        pool_placement.shutdown_executors()


@pytest.mark.asyncio
async def test_bus_calls_run_on_the_granted_roles_executor(fake_pool, monkeypatch):
    import threading

    names = []
    monkeypatch.setattr(gateway, "run_llm_chat",
                        lambda body, plan: names.append(threading.current_thread().name) or _ok(body, plan))
    fake_pool.choose = lambda kw: "fast" if kw["work_class"] == "fast" else "chat"
    await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    await gateway._dispatch_chat(_body("chat"), correlation_id="c")
    assert names[0].startswith("llm-gw-8013") and names[1].startswith("llm-gw-8011")


# ── GET /routes compatibility view ────────────────────────────────────────────────────────


def _role(role: str, status: str = "confirmed", **kw: Any) -> Dict[str, Any]:
    port = pool_placement.pool_config().roles[role].port
    return {"role": role, "kind": "llm", "cards": ["x"], "url": f"http://100.112.254.99:{port}", "status": status,
            "model_file": f"{role}.gguf", "ctx_per_slot": 8192, "vision": False, "checked_at": "2026-09-24T00:00:00Z",
            **kw}


def _state(roles, *, gpu0_lent=False):
    return {"generated_at": "2026-09-24T00:00:00Z", "cards": [
        {"card": "gpu0", "vram_gb": 32, "lendable": True, "lent": gpu0_lent},
        {"card": "gpu1", "vram_gb": 32}, {"card": "gpu2", "vram_gb": 32}, {"card": "gpu3", "vram_gb": 32},
    ], "roles": roles}


def _by_id(payload):
    return {r["id"]: r for r in payload["routes"]}


def test_routes_compat_view_statuses_from_pool_state():
    state = _state([_role("chat"), _role("agent"), _role("agent-gpu2", "unloaded"), _role("metacog", "down"),
                    _role("fast", vision=True)])
    routes = _by_id(pool_placement.build_routes_compat(state))
    assert set(routes) == set(pool_placement.pool_routes())
    chat = routes["chat"]
    assert chat["status"] == "up" and chat["upstream"] == "http://100.112.254.99:8011"
    assert (chat["n_ctx"], chat["model"], chat["served_by"], chat["backend"]) == (8192, "chat.gguf", "circe-worker-chat", "llamacpp")
    assert chat["latency_ms"] is None and chat["reserved_free_slots"] is None
    # metacog's own role is down: the first up role of its class that it may use (fast)
    assert routes["metacog"]["status"] == "up" and routes["metacog"]["upstream"].endswith(":8013")
    assert routes["quick"]["vision"] is True
    # burst compat: chat-burst closed while gpu0 is not lent; agent-burst follows agent-gpu2 only
    assert routes["chat-burst"]["status"] == "operator_closed" and routes["chat-burst"]["gate_open"] is False
    assert routes["agent-burst"]["status"] == "down" and routes["agent-burst"]["upstream"].endswith(":8016")
    # the Hub picker's definitional priority, not the pool's queue priority
    assert routes["quick"]["priority"] is None
    assert routes["quick_background"]["priority"] == "background"
    assert routes["harness"]["priority"] == "system"


def test_routes_compat_chat_burst_opens_when_gpu0_is_lent():
    routes = _by_id(pool_placement.build_routes_compat(_state([_role("chat"), _role("agent-gpu2")], gpu0_lent=True)))
    assert routes["chat-burst"]["status"] == "up" and routes["chat-burst"]["gate_open"] is True
    assert routes["agent-burst"]["status"] == "up"


def test_routes_compat_class_with_no_live_role_is_down_with_its_first_roles_url():
    routes = _by_id(pool_placement.build_routes_compat(_state([_role("chat", "down")])))
    assert routes["chat"]["status"] == "down"
    assert routes["chat"]["upstream"] == "http://100.112.254.99:8011"
    assert routes["chat"]["n_ctx"] is None and routes["chat"]["model"] is None


def test_routes_compat_a_borrower_does_not_count_an_unlent_chat_card():
    # Only chat is up: agent work may not use it unless gpu0 is lent.
    routes = _by_id(pool_placement.build_routes_compat(_state([_role("chat")])))
    assert routes["agent"]["status"] == "down"
    routes = _by_id(pool_placement.build_routes_compat(_state([_role("chat")], gpu0_lent=True)))
    assert routes["agent"]["status"] == "up" and routes["agent"]["upstream"].endswith(":8011")


def test_routes_compat_pool_unreachable_is_unknown_never_up():
    payload = pool_placement.build_routes_compat(None)
    assert payload["routes"] and all(r["status"] == "unknown" for r in payload["routes"])
    assert all(r["n_ctx"] is None for r in payload["routes"])


def test_get_routes_endpoint_uses_pool_state(monkeypatch):
    async def state():
        return _state([_role("fast")])

    monkeypatch.setattr(pool_placement, "fetch_pool_state", state)
    payload = TestClient(gateway.app).get("/routes").json()
    assert payload["default_route"] == settings.llm_route_default
    assert _by_id(payload)["quick"]["status"] == "up"


@pytest.mark.asyncio
async def test_fetch_pool_state_rpc_shape_and_cache(monkeypatch):
    pool_placement.reset_pool_state_cache()
    sent = []

    class _Decoded:
        ok = True

        class envelope:
            payload = {"roles": [], "cards": []}

    class _Codec:
        def decode(self, data):
            assert data == b"raw"
            return _Decoded()

    class _Bus:
        codec = _Codec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec, health_label):
            sent.append((channel, env, reply_channel))
            return {"data": b"raw"}

    monkeypatch.setattr(pool_placement, "_bus", _Bus())
    try:
        assert await pool_placement.fetch_pool_state() == {"roles": [], "cards": []}
        await pool_placement.fetch_pool_state()
        assert len(sent) == 1  # cached
        channel, env, reply_channel = sent[0]
        assert channel == "orion:gpu_pool:state:request"
        assert env.kind == "gpu_pool.state.request.v1"
        assert env.reply_to == reply_channel and reply_channel.startswith("orion:gpu_pool:state:reply:")
        assert env.payload == {"include_leases": False}
    finally:
        pool_placement.reset_pool_state_cache()


@pytest.mark.asyncio
async def test_fetch_pool_state_failure_is_none(monkeypatch):
    pool_placement.reset_pool_state_cache()

    class _Bus:
        async def rpc_request(self, *a, **k):
            raise TimeoutError("no reply")

    monkeypatch.setattr(pool_placement, "_bus", _Bus())
    try:
        assert await pool_placement.fetch_pool_state() is None
    finally:
        pool_placement.reset_pool_state_cache()
