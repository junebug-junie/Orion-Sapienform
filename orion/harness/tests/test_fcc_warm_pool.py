"""Warm FCC chat pool (spec L5): per-turn values, context isolation, lifecycle, fallback.

Every model call goes to a local stub HTTP server. The ``claude`` binary is
``fake_claude_stream.py``, which speaks the stream-json shapes the real CLI
2.1.291 produced (PR report). Real processes, a real relay, a real stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest
import pytest_asyncio

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")

from orion.harness import fcc_motor as motor  # noqa: E402
from orion.harness import fcc_warm_pool as wp  # noqa: E402
from orion.harness.fcc_warm_relay import CORRELATION_HEADER  # noqa: E402
from orion.llm.resource_lease import GPU_LEASE_HEADER, decode_gpu_lease_header  # noqa: E402

FAKE = Path(__file__).with_name("fake_claude_stream.py")


def _lease(lease_id: str) -> Dict[str, Any]:
    return {"lease_id": lease_id, "generation": 1, "role": "chat-gpu0", "holder": "test"}


class _Stub:
    """Records every upstream request; answers like a minimal Messages API."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # noqa: D401
                pass

            def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _record(self, body: Any) -> None:
                with stub.lock:
                    stub.requests.append(
                        {"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                    )

            def do_GET(self):
                self._record(None)
                if self.path.endswith("/health"):
                    return self._send(200, b'{"ok":true}')
                return self._send(200, b'{"data":[],"has_more":false}')

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw) if raw else None
                self._record(body)
                if "/sse/" in self.path:
                    payload = b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                    return self._send(200, payload, "text/event-stream")
                msgs = (body or {}).get("messages") or []
                last = str(msgs[-1]["content"]) if msgs else ""
                if "SLOW" in last:
                    time.sleep(0.8)
                text = f"echo:{last[:60]}"
                out = {"id": "m", "type": "message", "role": "assistant", "model": "stub-model",
                       "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 1, "output_tokens": 1}}
                return self._send(200, json.dumps(out).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def messages_requests(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [r for r in self.requests if r["method"] == "POST" and r["path"].endswith("/v1/messages")]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = _Stub()
    yield s
    s.close()


@pytest.fixture
def env_setup(tmp_path, monkeypatch, stub):
    fcc_env = tmp_path / "fcc.env"
    fcc_env.write_text("MODEL_SONNET=llamacpp/chat\nMODEL_AGENT=llamacpp/agent\n", encoding="utf-8")
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    spawn_log = tmp_path / "spawns.jsonl"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HARNESS_FCC_ENV_PATH", str(fcc_env))
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "false")
    monkeypatch.setenv("HARNESS_FCC_REPEAT_FAILURE_THRESHOLD", "0")
    monkeypatch.setenv("HARNESS_FCC_SKIP_PERMISSIONS", "false")
    monkeypatch.setenv("HARNESS_LLM_GATEWAY_URL", stub.base + "/gw")
    monkeypatch.setenv("FAKE_CLAUDE_SPAWN_LOG", str(spawn_log))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS", *motor.TURN_CLOCK_ENV_KEYS):
        monkeypatch.delenv(key, raising=False)
    return {"wrapper": str(wrapper), "spawn_log": spawn_log, "workspace": str(workspace), "tmp": tmp_path}


def _spawns(env_setup) -> List[Dict[str, Any]]:
    path = env_setup["spawn_log"]
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _start_pool(env_setup, stub, **overrides) -> wp.WarmPool:
    cfg = dict(
        size=1,
        relay_port=0,
        claude_bin=env_setup["wrapper"],
        workspace=env_setup["workspace"],
        fcc_server_url=stub.base + "/fcc",
        auth_token="fcc-token",
        state_dir=str(env_setup["tmp"] / "state"),
        health_interval_sec=3600.0,
        spawn_timeout_sec=20.0,
        clear_timeout_sec=5.0,
    )
    cfg.update(overrides)
    pool = await wp.start_warm_pool(wp.WarmPoolConfig(**cfg))
    await _wait_idle(pool, cfg["size"])
    return pool


async def _wait_idle(pool: wp.WarmPool, n: int, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if sum(1 for s in pool._slots if s.state == "idle") >= n:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"pool never reached {n} idle slot(s): {pool.status()}")


async def _turn(env_setup, stub, prompt: str, corr: str, **kw) -> List[Dict[str, Any]]:
    kwargs = dict(
        prompt=prompt,
        correlation_id=corr,
        fcc_model_label="MODEL_SONNET",
        workspace=env_setup["workspace"],
        fcc_server_url=stub.base + "/fcc",
        auth_token="fcc-token",
        claude_bin=env_setup["wrapper"],
        timeout_sec=15.0,
        chat_reply=True,
    )
    kwargs.update(kw)
    return [frame async for frame in motor.run_fcc_turn(**kwargs)]


def _terminal(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    return [f for f in frames if f["type"] in ("final", "error")][-1]


def _pid(pool: wp.WarmPool, idx: int = 0):
    return getattr(pool._slots[idx].proc, "pid", None)


@pytest_asyncio.fixture
async def pool_cleanup():
    yield
    await wp.stop_warm_pool()


# ---------------------------------------------------------------- per-turn values


@pytest.mark.asyncio
async def test_lease_header_and_upstream_follow_the_current_turn(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    pool = await _start_pool(env_setup, stub)
    pid = _pid(pool)

    a = _terminal(await _turn(env_setup, stub, "turn A", "corr-A", gpu_lease=_lease("lease-A")))
    b = _terminal(await _turn(env_setup, stub, "turn B", "corr-B"))
    c = _terminal(await _turn(env_setup, stub, "turn C", "corr-C", gpu_lease=_lease("lease-C")))

    for frame, text in ((a, "turn A"), (b, "turn B"), (c, "turn C")):
        assert frame["type"] == "final", frame
        assert frame["llm_response"] == f"echo:{text}"
        assert frame["metadata"]["fcc_spawn_mode"] == "warm"
        assert isinstance(frame["metadata"]["fcc_spawn_or_acquire_ms"], int)
        assert isinstance(frame["metadata"]["fcc_first_event_ms"], int)

    reqs = {r["headers"][CORRELATION_HEADER.lower()]: r for r in stub.messages_requests()}
    assert set(reqs) == {"corr-A", "corr-B", "corr-C"}
    # Leased turns go to the gateway with THEIR lease; the unleased one to FCC with none.
    assert reqs["corr-A"]["path"].startswith("/gw/")
    assert decode_gpu_lease_header(reqs["corr-A"]["headers"][GPU_LEASE_HEADER.lower()]).lease_id == "lease-A"
    assert reqs["corr-A"]["headers"]["authorization"] == "Bearer orion-resource-lease"
    assert reqs["corr-B"]["path"].startswith("/fcc/")
    assert GPU_LEASE_HEADER.lower() not in reqs["corr-B"]["headers"]
    assert reqs["corr-B"]["headers"]["authorization"] == "Bearer fcc-token"
    assert decode_gpu_lease_header(reqs["corr-C"]["headers"][GPU_LEASE_HEADER.lower()]).lease_id == "lease-C"

    # One process served all three, and it was never respawned.
    assert [s for s in _spawns(env_setup) if s["streaming"]] == [{"pid": pid, "streaming": True, "model": "llamacpp/chat"}]
    assert not [s for s in _spawns(env_setup) if not s["streaming"]]
    assert _pid(pool) == pid
    # Telemetry line per turn, with the correlation id.
    timing = [r.getMessage() for r in caplog.records if "fcc_turn_start_timing" in r.getMessage()]
    assert any("corr=corr-B mode=warm spawn_or_acquire_ms=" in m for m in timing)
    # The relay is unbound between turns.
    assert pool.registry.binding(pool._slots[0].slot_id) is None


@pytest.mark.asyncio
async def test_clear_isolates_context_between_turns(env_setup, stub, pool_cleanup):
    await _start_pool(env_setup, stub)
    first = _terminal(await _turn(env_setup, stub, "SECRET_ALPHA remember this", "corr-1"))
    second = _terminal(await _turn(env_setup, stub, "what did I say?", "corr-2"))
    assert first["type"] == second["type"] == "final"
    reqs = {r["headers"][CORRELATION_HEADER.lower()]: r for r in stub.messages_requests()}
    second_body = json.dumps(reqs["corr-2"]["body"])
    assert "SECRET_ALPHA" not in second_body
    assert reqs["corr-2"]["body"]["messages"] == [{"role": "user", "content": "what did I say?"}]
    assert first["metadata"]["claude_session_id"] != second["metadata"]["claude_session_id"]


def test_fake_cli_would_leak_without_clear(env_setup, stub, monkeypatch):
    """Control: without /clear the fake sends the previous prompt upstream, so the
    isolation test above can actually fail."""
    import subprocess

    env = dict(os.environ, ANTHROPIC_BASE_URL=stub.base + "/fcc", ANTHROPIC_AUTH_TOKEN="t")
    lines = "".join(
        json.dumps({"type": "user", "message": {"role": "user", "content": t}}) + "\n"
        for t in ("SECRET_ALPHA one", "two")
    )
    subprocess.run([env_setup["wrapper"], "-p", "--input-format", "stream-json"], input=lines,
                   text=True, env=env, capture_output=True, timeout=30, check=True)
    assert "SECRET_ALPHA" in json.dumps(stub.messages_requests()[-1]["body"])


@pytest.mark.asyncio
async def test_turn_clock_reaches_bash_through_env_file_not_process_env(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub)
    before = time.time()
    one = _terminal(await _turn(env_setup, stub, "ENVFILE", "corr-1", timeout_sec=111.0))
    two = _terminal(await _turn(env_setup, stub, "ENVFILE", "corr-2", timeout_sec=222.0))
    seen1 = json.loads(one["llm_response"].removeprefix("CLOCK "))
    seen2 = json.loads(two["llm_response"].removeprefix("CLOCK "))
    assert seen1["ORION_TURN_BUDGET_SEC"] == "111"
    assert seen2["ORION_TURN_BUDGET_SEC"] == "222"
    assert abs(int(seen2["ORION_TURN_DEADLINE_EPOCH"]) - (before + 222)) < 5
    assert seen2["ORION_TURN_STEP_STALL_SEC"] == "222"  # stall cap clamps to the turn budget
    # Frozen process env never carries a turn's clock.
    environ = Path(f"/proc/{_pid(pool)}/environ")
    if environ.exists():
        raw = environ.read_bytes().decode(errors="replace")
        assert "ORION_TURN_DEADLINE_EPOCH" not in raw
        assert "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1" in raw
        assert GPU_LEASE_HEADER not in raw


# ---------------------------------------------------------------- lifecycle


@pytest.mark.asyncio
async def test_deadline_overrun_kills_and_respawns(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub)
    old_pid = _pid(pool)
    frames = await _turn(env_setup, stub, "HANG forever", "corr-hang", timeout_sec=1.5)
    assert _terminal(frames)["error_code"] == "fcc_timeout"
    await _wait_idle(pool, 1)
    assert _pid(pool) != old_pid
    assert pool.counters["respawn:killed"] == 1
    after = _terminal(await _turn(env_setup, stub, "still alive", "corr-next"))
    assert after["type"] == "final" and after["metadata"]["fcc_spawn_mode"] == "warm"


@pytest.mark.asyncio
async def test_idle_crash_falls_back_to_spawn_then_respawns(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    pool = await _start_pool(env_setup, stub)
    old_pid = _pid(pool)
    os.kill(old_pid, signal.SIGKILL)
    for _ in range(100):
        if pool._slots[0].proc.returncode is not None:
            break
        await asyncio.sleep(0.02)
    frame = _terminal(await _turn(env_setup, stub, "after crash", "corr-fb"))
    assert frame["type"] == "final"
    assert frame["metadata"]["fcc_spawn_mode"] == "spawn"
    assert any("fcc_warm_pool_fallback corr=corr-fb" in r.getMessage() for r in caplog.records)
    assert [s for s in _spawns(env_setup) if not s["streaming"]], "fallback must spawn a per-turn process"
    await _wait_idle(pool, 1)
    assert _pid(pool) != old_pid
    assert _terminal(await _turn(env_setup, stub, "warm again", "corr-w"))["metadata"]["fcc_spawn_mode"] == "warm"


@pytest.mark.asyncio
async def test_crash_mid_turn_is_an_error_not_a_retry(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub)
    frame = _terminal(await _turn(env_setup, stub, "CRASH now", "corr-crash"))
    assert frame["type"] == "error" and frame["error_code"] == "fcc_nonzero_exit"
    # It had already spoken (init), so no silent re-run as a spawn.
    assert not [s for s in _spawns(env_setup) if not s["streaming"]]
    await _wait_idle(pool, 1)
    assert pool.counters["respawn:process_died"] == 1


@pytest.mark.asyncio
async def test_silent_death_after_prompt_retries_as_spawn(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    await _start_pool(env_setup, stub)
    frame = _terminal(await _turn(env_setup, stub, "CRASH_SILENT", "corr-silent"))
    # The spawn retry gets the same prompt and dies too -- reported as a spawn error.
    assert frame["error_code"] == "fcc_nonzero_exit"
    assert frame["metadata"]["fcc_spawn_mode"] == "spawn"
    assert any("reason=warm_process_died_before_first_event" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_recycle_after_max_turns(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub, max_turns=2)
    first_pid = _pid(pool)
    await _turn(env_setup, stub, "one", "c1")
    assert _pid(pool) == first_pid
    await _turn(env_setup, stub, "two", "c2")
    await _wait_idle(pool, 1)
    assert _pid(pool) != first_pid
    assert pool.counters["respawn:recycle_max_turns"] == 1
    assert pool._slots[0].turns == 0


@pytest.mark.asyncio
async def test_recycle_after_max_age_from_health_tick(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub, max_age_sec=60.0)
    first_pid = _pid(pool)
    pool._slots[0].spawned_at -= 120.0
    await pool.health_tick()
    await _wait_idle(pool, 1)
    assert _pid(pool) != first_pid
    assert pool.counters["respawn:recycle_max_age"] == 1


@pytest.mark.asyncio
async def test_two_concurrent_turns_use_two_processes(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub, size=2)
    pids = {_pid(pool, 0), _pid(pool, 1)}
    assert len(pids) == 2
    a, b = await asyncio.gather(
        _turn(env_setup, stub, "SLOW a", "corr-a", gpu_lease=_lease("lease-a")),
        _turn(env_setup, stub, "SLOW b", "corr-b", gpu_lease=_lease("lease-b")),
    )
    assert _terminal(a)["llm_response"] == "echo:SLOW a"
    assert _terminal(b)["llm_response"] == "echo:SLOW b"
    assert _terminal(a)["metadata"]["fcc_spawn_mode"] == _terminal(b)["metadata"]["fcc_spawn_mode"] == "warm"
    reqs = {r["headers"][CORRELATION_HEADER.lower()]: r for r in stub.messages_requests()}
    assert decode_gpu_lease_header(reqs["corr-a"]["headers"][GPU_LEASE_HEADER.lower()]).lease_id == "lease-a"
    assert decode_gpu_lease_header(reqs["corr-b"]["headers"][GPU_LEASE_HEADER.lower()]).lease_id == "lease-b"
    # Each turn ran in its own warm process.
    assert {int(reqs["corr-a"]["headers"]["x-fake-pid"]), int(reqs["corr-b"]["headers"]["x-fake-pid"])} == pids
    assert {_pid(pool, 0), _pid(pool, 1)} == pids


@pytest.mark.asyncio
async def test_model_change_falls_back_and_retargets_the_slot(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    pool = await _start_pool(env_setup, stub)
    frame = _terminal(await _turn(env_setup, stub, "agent please", "corr-m1", fcc_model_label="MODEL_AGENT"))
    assert frame["metadata"]["fcc_spawn_mode"] == "spawn"
    assert any("reason=signature_mismatch" in r.getMessage() for r in caplog.records)
    await _wait_idle(pool, 1)
    assert pool._slots[0].model_id == "llamacpp/agent"
    again = _terminal(await _turn(env_setup, stub, "agent again", "corr-m2", fcc_model_label="MODEL_AGENT"))
    assert again["metadata"]["fcc_spawn_mode"] == "warm"


@pytest.mark.asyncio
async def test_investigation_turns_never_touch_the_pool(env_setup, stub, pool_cleanup, caplog):
    caplog.set_level(logging.INFO)
    pool = await _start_pool(env_setup, stub)
    frame = _terminal(await _turn(env_setup, stub, "curious", "corr-inv", chat_reply=False))
    assert frame["metadata"]["fcc_spawn_mode"] == "spawn"
    assert not any("fcc_warm_pool_fallback" in r.getMessage() for r in caplog.records)
    assert pool.counters["hit"] == 0


@pytest.mark.asyncio
async def test_no_pool_means_plain_spawn(env_setup, stub):
    assert wp.get_warm_pool() is None
    frame = _terminal(await _turn(env_setup, stub, "hello", "corr-nopool"))
    assert frame["type"] == "final" and frame["metadata"]["fcc_spawn_mode"] == "spawn"


@pytest.mark.asyncio
async def test_warm_turn_skips_the_startup_cost(env_setup, stub, pool_cleanup, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_BOOT_SEC", "1.0")
    await _start_pool(env_setup, stub)
    warm = _terminal(await _turn(env_setup, stub, "warm", "corr-warm"))
    cold = _terminal(await _turn(env_setup, stub, "cold", "corr-cold", chat_reply=False))
    assert warm["metadata"]["fcc_spawn_mode"] == "warm"
    assert cold["metadata"]["fcc_spawn_mode"] == "spawn"
    assert warm["metadata"]["fcc_spawn_or_acquire_ms"] < 500
    assert cold["metadata"]["fcc_spawn_or_acquire_ms"] >= 1000


# ---------------------------------------------------------------- relay


@pytest.mark.asyncio
async def test_relay_refuses_bad_secret_and_unbound_model_calls(env_setup, stub, pool_cleanup):
    pool = await _start_pool(env_setup, stub)
    slot_id = pool._slots[0].slot_id
    url = pool.relay.slot_base_url(slot_id)
    secret = pool.registry._secrets[slot_id]
    async with httpx.AsyncClient() as client:
        r = await client.post(url + "/v1/messages", json={"messages": []})
        assert r.status_code == 401
        r = await client.post(url + "/v1/messages", json={"messages": []}, headers={"authorization": "Bearer nope"})
        assert r.status_code == 401
        r = await client.post(url + "/v1/messages", json={"messages": []}, headers={"authorization": f"Bearer {secret}"})
        assert r.status_code == 503
        before = len(stub.requests)
        r = await client.get(url + "/v1/models", headers={"authorization": f"Bearer {secret}"})
        assert r.status_code == 200
        forwarded = stub.requests[before]
        assert forwarded["path"] == "/fcc/v1/models"
        assert forwarded["headers"]["authorization"] == "Bearer fcc-token"


@pytest.mark.asyncio
async def test_relay_streams_sse_and_strips_client_lease(env_setup, stub, pool_cleanup):
    from orion.harness.fcc_warm_relay import RelayTurnBinding, RelayUpstream

    pool = await _start_pool(env_setup, stub)
    slot_id = pool._slots[0].slot_id
    secret = pool.registry._secrets[slot_id]
    pool.registry.bind(
        slot_id,
        RelayTurnBinding(
            correlation_id="corr-sse",
            upstream=RelayUpstream(base_url=stub.base + "/sse", auth_token="gw-token"),
            gpu_lease_header=None,
        ),
    )
    async with httpx.AsyncClient() as client:
        r = await client.post(
            pool.relay.slot_base_url(slot_id) + "/v1/messages",
            json={"stream": True, "messages": []},
            headers={"authorization": f"Bearer {secret}", GPU_LEASE_HEADER: "forged"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"message_stop" in r.content
    sent = stub.messages_requests()[-1]["headers"]
    assert GPU_LEASE_HEADER.lower() not in sent
    assert sent["authorization"] == "Bearer gw-token"
    pool.registry.unbind(slot_id)
