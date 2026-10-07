#!/usr/bin/env python3
"""Spec L5 before/after probe: the real ``claude`` CLI, spawned per turn vs borrowed from the warm pool.

No model is ever called. A local stub on 127.0.0.1 plays both the FCC server
and the llm-gateway (Anthropic Messages, streamed). Every turn runs through the
real ``orion.harness.fcc_motor.run_fcc_turn``, so the numbers are the motor's
own ``fcc_spawn_or_acquire_ms`` (turn start -> the turn's first ``system/init``)
and ``fcc_first_event_ms`` (-> first assistant event).

It also checks, on the real CLI, what the unit tests check on a fake one:
- the relay delivers each turn's own GPU lease header and correlation id;
- the turn after ``/clear`` sends nothing from the turn before it.

Run inside orion-harness-governor to include its real MCP servers (the
container's env drives MCP rendering exactly as for a live turn), or on a host
for the bare-CLI floor. Uses a throwaway CLAUDE_CONFIG_DIR unless
``--config-dir`` is given.

    python scripts/probe_fcc_warm_pool.py --turns 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REQUESTS: List[Dict[str, Any]] = []


def _sse(text: str) -> bytes:
    def ev(kind: str, data: dict) -> str:
        return f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return "".join(
        [
            ev("message_start", {"type": "message_start", "message": {"id": "msg_probe", "type": "message", "role": "assistant", "model": "probe-stub", "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
            ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}),
            ev("message_stop", {"type": "message_stop"}),
        ]
    ).encode()


class _Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.endswith("/health"):
            return self._send(200, b'{"ok":true}', "application/json")
        return self._send(200, b'{"data":[],"has_more":false}', "application/json")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        REQUESTS.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if "count_tokens" in self.path:
            return self._send(200, b'{"input_tokens":10}', "application/json")
        if body and body.get("stream"):
            return self._send(200, _sse("probe reply"), "text/event-stream")
        out = {"id": "m", "type": "message", "role": "assistant", "model": "probe-stub", "content": [{"type": "text", "text": "probe reply"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        return self._send(200, json.dumps(out).encode(), "application/json")


async def _turn(motor, *, prompt: str, corr: str, chat: bool, base: str, workspace: str, claude_bin: str, lease: dict | None) -> Dict[str, Any]:
    frames = [
        f
        async for f in motor.run_fcc_turn(
            prompt=prompt,
            correlation_id=corr,
            fcc_model_label="MODEL_SONNET",
            workspace=workspace,
            fcc_server_url=base + "/fcc",
            auth_token="probe-token",
            claude_bin=claude_bin,
            timeout_sec=120.0,
            chat_reply=chat,
            gpu_lease=lease,
        )
    ]
    return [f for f in frames if f["type"] in ("final", "error")][-1]


def _summary(xs: List[int]) -> str:
    xs = [x for x in xs if isinstance(x, int)]
    if not xs:
        return "n=0"
    return f"n={len(xs)} median={statistics.median(xs):.0f}ms min={min(xs)}ms max={max(xs)}ms all={xs}"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--claude-bin", default=os.environ.get("HARNESS_FCC_CLAUDE_BIN", "claude"))
    ap.add_argument("--config-dir", default=None, help="CLAUDE_CONFIG_DIR (default: a throwaway temp dir)")
    args = ap.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    tmp = Path(tempfile.mkdtemp(prefix="probe-fcc-warm-"))
    fcc_env = tmp / "fcc.env"
    fcc_env.write_text("MODEL_SONNET=llamacpp/chat\n", encoding="utf-8")
    workspace = os.environ.get("HARNESS_FCC_WORKSPACE") or str(tmp / "ws")
    Path(workspace).mkdir(parents=True, exist_ok=True)
    os.environ["HARNESS_FCC_ENV_PATH"] = str(fcc_env)
    os.environ["HARNESS_LLM_GATEWAY_URL"] = base + "/gw"
    os.environ["CLAUDE_CONFIG_DIR"] = args.config_dir or str(tmp / "claude-config")
    # Never let an inherited real endpoint or key leak into a probe turn.
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS"):
        os.environ.pop(key, None)

    from orion.harness import fcc_motor as motor
    from orion.harness import fcc_warm_pool as wp
    from orion.llm.resource_lease import GPU_LEASE_HEADER, decode_gpu_lease_header

    spawn_ms: List[int] = []
    spawn_first: List[int] = []
    for i in range(args.turns):
        fr = await _turn(motor, prompt=f"spawn turn {i}", corr=f"probe-spawn-{i}", chat=False, base=base, workspace=workspace, claude_bin=args.claude_bin, lease=None)
        spawn_ms.append(fr.get("metadata", {}).get("fcc_spawn_or_acquire_ms"))
        spawn_first.append(fr.get("metadata", {}).get("fcc_first_event_ms"))

    pool = await wp.start_warm_pool(
        wp.WarmPoolConfig(
            size=1, relay_port=0, claude_bin=args.claude_bin, workspace=workspace,
            fcc_server_url=base + "/fcc", auth_token="probe-token", health_interval_sec=3600,
        )
    )
    for _ in range(600):
        if any(s.state == "idle" for s in pool._slots):
            break
        await asyncio.sleep(0.1)
    warm_ms: List[int] = []
    warm_first: List[int] = []
    modes: List[str] = []
    start = len(REQUESTS)
    for i in range(args.turns):
        lease = {"lease_id": f"probe-lease-{i}", "generation": 1, "role": "chat-gpu0", "holder": "probe"}
        fr = await _turn(motor, prompt=f"MARKER_{i} warm turn", corr=f"probe-warm-{i}", chat=True, base=base, workspace=workspace, claude_bin=args.claude_bin, lease=lease)
        md = fr.get("metadata", {})
        modes.append(md.get("fcc_spawn_mode"))
        warm_ms.append(md.get("fcc_spawn_or_acquire_ms"))
        warm_first.append(md.get("fcc_first_event_ms"))
    status = pool.status()
    await wp.stop_warm_pool()
    server.shutdown()

    leak = False
    lease_ok = True
    checked = set()
    for r in REQUESTS[start:]:
        if "/v1/messages" not in r["path"] or "count_tokens" in r["path"]:
            continue
        if not r["path"].startswith("/gw/"):
            lease_ok = False  # a leased turn's model call went somewhere else
            continue
        corr = r["headers"].get("x-orion-correlation-id", "")
        i = int(corr.rsplit("-", 1)[-1]) if corr.startswith("probe-warm-") else -1
        lease_hdr = r["headers"].get(GPU_LEASE_HEADER.lower())
        if i < 0 or not lease_hdr or decode_gpu_lease_header(lease_hdr).lease_id != f"probe-lease-{i}":
            lease_ok = False
        checked.add(i)
        text = json.dumps(r["body"])
        if any(f"MARKER_{j}" in text for j in range(args.turns) if j != i):
            leak = True

    print(f"spawn  spawn_or_acquire: {_summary(spawn_ms)}")
    print(f"spawn  first_event:      {_summary(spawn_first)}")
    print(f"warm   spawn_or_acquire: {_summary(warm_ms)}  modes={modes}")
    print(f"warm   first_event:      {_summary(warm_first)}")
    lease_ok = lease_ok and checked == set(range(args.turns))
    print(f"per-turn lease+corr via relay: {'OK' if lease_ok else 'MISMATCH'} (turns with a checked model call: {sorted(checked)})")
    print(f"prior-turn content leaked after /clear: {'YES' if leak else 'no'}")
    print(f"pool counters: {status['counters']}")
    return 0 if (lease_ok and not leak and all(m == 'warm' for m in modes)) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
