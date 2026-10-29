"""Per-call token ledger, harness side.

* A per-turn FCC spawn sends ``X-Orion-Correlation-Id`` on every model call, so the gateway's
  ledger row carries the turn id (before 2026-10-29 every harness call logged ``corr=-``).
* The warm relay records an FCC-path call (turn bound, no lease) itself, with the turn id, and
  does NOT record a lease-path call (the gateway records that one -- no double count).
"""
from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")

from orion.harness import fcc_warm_relay as relay  # noqa: E402
from orion.harness.fcc_motor import _build_subprocess_env  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseRefV1  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "llm/tests/fixtures/call_usage"
SSE = (FIX / "anthropic_stream.sse").read_bytes()


@pytest.fixture(autouse=True)
def _recording_on():
    """The governor turns relay recording on at boot (HARNESS_LLM_CALL_USAGE_ENABLED + bus)."""
    relay.set_usage_recording(True)
    relay.get_usage_sink().drain(limit=10_000)
    yield
    relay.set_usage_recording(False)
    relay.get_usage_sink().drain(limit=10_000)


def test_spawn_env_carries_the_turn_correlation_id(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Trace: t\nX-Orion-Correlation-Id: stale-turn")
    env = _build_subprocess_env(fcc_server_url="http://fcc:8082", auth_token="tok", correlation_id="turn-1")
    assert env["ANTHROPIC_CUSTOM_HEADERS"].splitlines() == ["X-Trace: t", "X-Orion-Correlation-Id: turn-1"]
    ref = GpuLeaseRefV1(lease_id="h", generation=1, role="agent", holder="durable-runs:r")
    env = _build_subprocess_env(fcc_server_url="http://fcc:8082", auth_token="tok", correlation_id="turn-2",
                                gpu_lease=ref.model_dump(mode="json"))
    assert "X-Orion-Correlation-Id: turn-2" in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()


def test_spawn_env_without_a_correlation_id_drops_an_inherited_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Orion-Correlation-Id: stale-turn")
    assert "ANTHROPIC_CUSTOM_HEADERS" not in _build_subprocess_env(fcc_server_url="http://f", auth_token="t")
    # A header-injection attempt is refused, not forwarded.
    env = _build_subprocess_env(fcc_server_url="http://f", auth_token="t", correlation_id="a\nEvil: 1")
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


class _Upstream:
    def __init__(self, *, gzip_body: bool = False) -> None:
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                body = gzip.compress(SSE) if stub.gzip_body else SSE
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                if stub.gzip_body:
                    self.send_header("content-encoding", "gzip")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.gzip_body = gzip_body
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _relay_call(*, lease: bool, gzip_body: bool = False):
    from starlette.testclient import TestClient

    up = _Upstream(gzip_body=gzip_body)
    sink = relay.get_usage_sink()
    sink.drain(limit=10_000)
    try:
        registry = relay.RelayRegistry()
        secret = registry.register_slot("s0")
        target = relay.RelayUpstream(base_url=up.base, auth_token="real")
        registry.bind("s0", relay.RelayTurnBinding(correlation_id="turn-42", upstream=target,
                                                   gpu_lease_header="abc" if lease else None))
        app = relay.build_relay_app(registry, target)
        with TestClient(app) as client:
            r = client.post("/slot/s0/v1/messages", headers={"authorization": f"Bearer {secret}"},
                            content=json.dumps({"model": "llamacpp/chat", "stream": True, "messages": []}))
            assert r.status_code == 200
            assert b"message_delta" in (gzip.decompress(r.content) if gzip_body and r.content[:2] == b"\x1f\x8b" else r.content)
        return sink.drain(limit=10_000)
    finally:
        up.close()


def test_relay_records_an_fcc_path_call_with_the_turn_id():
    (ev,) = _relay_call(lease=False)
    assert ev.recorder == "harness_relay" and ev.path == "relay_anthropic"
    assert ev.correlation_id == "turn-42" and ev.model == "llamacpp/chat" and ev.stream
    assert (ev.prompt_tokens, ev.cached_prompt_tokens, ev.completion_tokens) == (14, 10, 8)
    assert ev.outcome == "served" and ev.duration_ms is not None


def test_relay_does_not_record_a_lease_path_call():
    assert _relay_call(lease=True) == []


def test_relay_reads_a_gzip_encoded_reply():
    (ev,) = _relay_call(lease=False, gzip_body=True)
    assert (ev.prompt_tokens, ev.completion_tokens) == (14, 8)


def test_record_relay_call_never_raises_on_garbage():
    ev = relay.record_relay_call(correlation_id="t", request_body=b"not json", status=500, completed=True,
                                 head=b"\x00\x01", tail=b"", duration_ms=5)
    assert ev is not None and ev.outcome == "http_500" and ev.token_source == "none"
    relay.get_usage_sink().drain(limit=10_000)


def test_relay_records_nothing_until_the_governor_turns_recording_on():
    relay.set_usage_recording(False)
    assert _relay_call(lease=False) == []
    assert relay.record_relay_call(correlation_id="t", request_body=b"{}", status=200, completed=True,
                                   head=b"", tail=b"", duration_ms=1) is None
