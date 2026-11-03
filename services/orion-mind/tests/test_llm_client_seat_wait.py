"""MindLLMClient seat wait and gateway errors (Orion Prime reliability, 2026-11-03).

Prime's calls may wait for a GPU pool seat on top of their generation budget; the bus RPC must
wait for both, and an empty reply must say WHY (no seat vs. the model failing)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def _client(monkeypatch, reply: tuple[str, dict, str, dict]):
    _mind_prep()
    from app.llm_client import MindLLMClient

    seen: dict[str, Any] = {}

    def fake_bus_chat(self, *, system_prompt, user_prompt, route, options, context, timeout_sec):
        seen["options"], seen["timeout_sec"] = options, timeout_sec
        return reply

    monkeypatch.setattr(MindLLMClient, "_bus_chat", fake_bus_chat)
    monkeypatch.setattr("app.llm_client.settings.MIND_LLM_USE_BUS", True)
    monkeypatch.setattr("app.llm_client.settings.ORION_BUS_ENABLED", True)
    return MindLLMClient(rpc_health=False), seen


def test_a_seat_wait_is_sent_to_the_gateway_and_the_rpc_waits_for_both(monkeypatch) -> None:
    from_client, seen = _client(monkeypatch, ('{"ok": true}', {}, "chat", {}))
    from app.llm_client import POOL_WAIT_RPC_MARGIN_SEC

    parsed, err, _ = from_client.request_json(system_prompt="s", user_prompt="u", route="prime", max_tokens=10,
                                              timeout_sec=60, pool_wait_sec=600)
    assert parsed == {"ok": True} and err is None
    assert seen["options"]["gateway_read_timeout_sec"] == 60
    assert seen["options"]["gateway_pool_wait_sec"] == 600
    assert seen["timeout_sec"] == 60 + 600 + POOL_WAIT_RPC_MARGIN_SEC


def test_without_a_seat_wait_nothing_changes(monkeypatch) -> None:
    client, seen = _client(monkeypatch, ('{"ok": true}', {}, "chat", {}))
    client.request_json(system_prompt="s", user_prompt="u", route="metacog", max_tokens=10, timeout_sec=30)
    assert "gateway_pool_wait_sec" not in seen["options"] and seen["timeout_sec"] == 30


def test_an_empty_reply_carries_the_gateways_typed_error(monkeypatch) -> None:
    gw = {"error": "gpu_pool_unavailable", "reason": "deadline", "retry_after_sec": None}
    client, _ = _client(monkeypatch, ("", {}, None, {"gateway_error": gw}))
    parsed, err, meta = client.request_json(system_prompt="s", user_prompt="u", route="prime", max_tokens=10)
    assert parsed is None and err == "empty_llm_response" and meta["gateway_error"] == gw


def test_gateway_error_of_reads_raw_error_and_details() -> None:
    _mind_prep()
    from app.llm_client import gateway_error_of

    assert gateway_error_of({}) is None and gateway_error_of(None) is None
    assert gateway_error_of({"error": "gpu_pool_unavailable",
                             "details": {"reason": "shed:cabinet_hot", "retry_after_sec": 42.0}}) == {
        "error": "gpu_pool_unavailable", "reason": "shed:cabinet_hot", "retry_after_sec": 42.0}
    assert gateway_error_of({"error": "upstream_http_5xx", "details": "x"}) == {
        "error": "upstream_http_5xx", "reason": None, "retry_after_sec": None}


def test_the_gateways_seat_wait_is_passed_up(monkeypatch) -> None:
    client, _ = _client(monkeypatch, ('{"ok": true}', {}, "chat", {"pool_wait_ms": 4200}))
    _, _, meta = client.request_json(system_prompt="s", user_prompt="u", route="prime", max_tokens=10)
    assert meta["pool_wait_ms"] == 4200.0
