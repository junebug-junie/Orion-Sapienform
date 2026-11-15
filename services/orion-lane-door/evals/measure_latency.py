#!/usr/bin/env python3
"""How much time the lane door adds to a call: direct to an engine vs through a door in front of it.

The spec's rule for the chat path is "no per-call pool check in the door" (a pool RPC would sit on
Juniper's chat path). This measures what the door does instead costs: one more TCP connect on the docker
bridge, a header parse, and a relay.

It starts a real door process (``python -m app.main``, shadow mode, logging on) in front of an engine and
times the same request both ways, a new connection per request (the gateway opens a new client per call).

  * default: a local engine stand-in answering instantly -- isolates the door's own cost;
  * ``--engine host:port``: a REAL engine, with ``GET /props`` only (runs no model: safe on a live seat).
    The door runs on this machine, so both paths cross the same network hop.

Also times the first byte of a stream (the stand-in streams 20 chunks, 20 ms apart).

    python3 services/orion-lane-door/evals/measure_latency.py [--n 400] [--engine 100.112.254.99:8013]

Prints p50/p95/p99 per path and the added time. Exit 1 if the door adds more than --budget-ms at p50
(default 2 ms) on the stand-in.
"""
from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from urllib.parse import urlsplit

SERVICE = Path(__file__).resolve().parents[1]


class _Engine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_GET(self):
        data = b'{"model_path":"stand-in","total_slots":10}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/v1/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(20):
                chunk = f"data: {i}\n\n".encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"0\r\n\r\n")
            return
        data = b'{"choices":[{"message":{"content":"ok"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port: int) -> None:
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError(f"nothing listening on {port}")


def _conn(url: str) -> tuple[http.client.HTTPConnection, str]:
    parts = urlsplit(url)
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=30), parts.path


def _time(url: str, n: int, method: str = "GET", headers: dict | None = None) -> list[float]:
    """A new TCP connection per call, like the gateway (a new httpx client per call). http.client, not
    httpx: building an httpx client loads an SSL context (~20 ms) that would drown the door's cost."""
    out = []
    body = b'{"messages":[]}' if method == "POST" else None
    for _ in range(n):
        t0 = time.perf_counter()
        c, path = _conn(url)
        c.request(method, path, body=body, headers={"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        r.read()
        c.close()
        out.append((time.perf_counter() - t0) * 1000.0)
        assert r.status == 200, r.status
    return out


def _ttfb(url: str, n: int, headers: dict | None = None) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        c, path = _conn(url)
        c.request("POST", path, body=b"{}", headers={"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        r.read1(1)
        out.append((time.perf_counter() - t0) * 1000.0)
        r.read()
        c.close()
    return out


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def _line(name: str, direct: list[float], door: list[float]) -> dict:
    row = {"case": name, "n": len(direct)}
    for p in (50, 95, 99):
        row[f"direct_p{p}_ms"] = round(_pct(direct, p), 3)
        row[f"door_p{p}_ms"] = round(_pct(door, p), 3)
        row[f"added_p{p}_ms"] = round(_pct(door, p) - _pct(direct, p), 3)
    row["added_mean_ms"] = round(statistics.mean(door) - statistics.mean(direct), 3)
    print(f"{name:28} direct p50 {row['direct_p50_ms']:8.3f}  door p50 {row['door_p50_ms']:8.3f}  "
          f"added p50 {row['added_p50_ms']:7.3f}  p95 {row['added_p95_ms']:7.3f}  p99 {row['added_p99_ms']:7.3f} ms")
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--engine", default=None, help="host:port of a real engine (GET /props only)")
    ap.add_argument("--budget-ms", type=float, default=2.0)
    args = ap.parse_args(argv)

    if args.engine:
        up_host, up_port = args.engine.rsplit(":", 1)
        stand_in = None
    else:
        stand_in = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
        threading.Thread(target=stand_in.serve_forever, daemon=True).start()
        up_host, up_port = "127.0.0.1", str(stand_in.server_address[1])
    door_port = _free_port()
    key = "measure-" + "k" * 56
    logs = tempfile.mkdtemp(prefix="lane-door-measure-")
    env = {**os.environ, "LANE_DOOR_LANE": "measure", "LANE_DOOR_UPSTREAM": f"{up_host}:{up_port}",
           "LANE_DOOR_MODE": "shadow", "LANE_DOOR_KEY": key, "LANE_DOOR_LISTEN_HOST": "127.0.0.1",
           "LANE_DOOR_LISTEN_PORT": str(door_port), "LANE_DOOR_LOG_DIR": logs}
    proc = subprocess.Popen([sys.executable, "-m", "app.main"], cwd=SERVICE, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait(door_port)
        direct, door = f"http://{up_host}:{up_port}", f"http://127.0.0.1:{door_port}"
        keyed = {"X-Orion-Lane-Key": key, "X-Orion-Lease-Id": "measure-lease"}
        rows = []
        _time(direct + "/props", 20), _time(door + "/props", 20)   # warm up
        rows.append(_line("GET /props", _time(direct + "/props", args.n), _time(door + "/props", args.n)))
        if stand_in is not None:
            rows.append(_line("POST chat (keyed, logged)", _time(direct + "/v1/chat/completions", args.n, "POST"),
                              _time(door + "/v1/chat/completions", args.n, "POST", keyed)))
            k = max(20, args.n // 10)
            rows.append(_line("stream first byte", _ttfb(direct + "/v1/stream", k),
                              _ttfb(door + "/v1/stream", k, keyed)))
        print(json.dumps({"engine": args.engine or "local stand-in", "rows": rows}, indent=1))
        if stand_in is not None and rows[1]["added_p50_ms"] > args.budget_ms:
            print(f"FAIL: the door adds {rows[1]['added_p50_ms']} ms at p50 (budget {args.budget_ms} ms)")
            return 1
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        if stand_in is not None:
            stand_in.shutdown()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(None)
    sys.exit(main())
