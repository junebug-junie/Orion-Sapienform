"""The lane door against a real engine stand-in on a local socket: what it forwards, what it strips, what
it logs, what it never refuses, and that a stream's tokens and a hang-up both pass straight through."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from app.door import EXEMPT_GET_PATHS, EXEMPT_POST_PATHS, Door, RequestLog, is_exempt, key_state
from app.settings import Settings, key_hash

KEY = "k" * 64


class Engine:
    """A llama.cpp stand-in: records what reached it; JSON, SSE stream, slow stream, or 500."""

    def __init__(self):
        self.seen: list[dict] = []
        self.closed_early = asyncio.Event()
        self.server = None

    async def handle(self, reader, writer):
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.decode().split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {k.lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
            if "chunked" in headers.get("transfer-encoding", ""):
                body = b""
                while True:
                    size = int((await reader.readuntil(b"\r\n")).strip(), 16)
                    body += (await reader.readexactly(size + 2))[:-2]
                    if size == 0:   # "0\r\n\r\n": the readexactly above took the final CRLF
                        break
            else:
                body = await reader.readexactly(int(headers.get("content-length", "0")))
            self.seen.append({"method": method, "path": target, "headers": headers, "body": body})
            if target.startswith("/v1/stream"):
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                await writer.drain()
                delay = 0.2 if target.endswith("slow") else 0.05
                for i in range(20 if target.endswith("slow") else 3):
                    chunk = f"data: {{\"tok\": {i}}}\n\n".encode()
                    writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    try:
                        await writer.drain()
                    except ConnectionError:
                        self.closed_early.set()
                        return
                    await asyncio.sleep(delay)
                    if reader.at_eof():
                        self.closed_early.set()
                        return
                writer.write(b"0\r\n\r\n")
            else:
                status = b"500 Internal Server Error" if target == "/boom" else b"200 OK"
                data = json.dumps({"path": target, "n": len(body)}).encode()
                # Keep-alive on purpose: the door must end the reply by Content-Length, not by close.
                writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\nConnection: keep-alive\r\n\r\n" + data)
            await writer.drain()
            await asyncio.sleep(0.05)
        except (asyncio.IncompleteReadError, ConnectionError):
            self.closed_early.set()
        finally:
            writer.close()

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]


async def _door(tmp_path: Path, engine_port: int, **kw) -> tuple[Door, asyncio.AbstractServer, int]:
    settings = Settings(lane="chat", upstream_host="127.0.0.1", upstream_port=engine_port, key=KEY,
                        log_dir=str(tmp_path / "logs"), **kw)
    door = Door(settings)
    server = await asyncio.start_server(door.handle, "127.0.0.1", 0)
    return door, server, server.sockets[0].getsockname()[1]


def _rows(tmp_path: Path) -> list[dict]:
    return [json.loads(l) for f in sorted((tmp_path / "logs").glob("chat-*.jsonl")) for l in f.read_text().splitlines()]


def run(coro):
    return asyncio.run(coro)


def test_a_keyed_call_reaches_the_engine_without_the_door_headers_and_is_logged_without_text(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "SECRET-PROMPT"}]},
                             headers={"X-Orion-Lane-Key": KEY, "X-Orion-Lease-Id": "lease-abc123",
                                      "Host": "100.112.254.99:8011"})
        server.close()
        return engine, door, r

    engine, door, r = run(go())
    assert r.status_code == 200 and r.json()["path"] == "/v1/chat/completions"
    (seen,) = engine.seen
    assert "x-orion-lane-key" not in seen["headers"] and "x-orion-lease-id" not in seen["headers"]
    assert seen["headers"]["host"] == "100.112.254.99:8011"          # Strata accepts the caller's Host today
    assert seen["headers"]["connection"] == "close"
    assert b"SECRET-PROMPT" in seen["body"]                          # the body passes byte for byte
    (row,) = _rows(tmp_path)
    assert (row["key"], row["lease_id"], row["status"], row["path"]) == ("ok", "lease-abc123", 200,
                                                                        "/v1/chat/completions")
    assert set(row) == {"ts", "lane", "src", "method", "path", "key", "lease_id", "status", "ms"}
    text = json.dumps(row)
    assert "SECRET-PROMPT" not in text and KEY not in text
    assert door.stats.calls == {"ok": 1, "missing": 0, "bad": 0}


@pytest.mark.parametrize("headers,state", [({}, "missing"), ({"X-Orion-Lane-Key": "wrong"}, "bad")])
def test_shadow_forwards_an_unkeyed_call_and_records_who_sent_it(tmp_path, capsys, headers, state):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            r = await c.post("/v1/chat/completions", json={"messages": []}, headers=headers)
        server.close()
        return engine, door, r

    engine, door, r = run(go())
    assert r.status_code == 200 and len(engine.seen) == 1               # shadow refuses nothing
    (row,) = _rows(tmp_path)
    assert row["key"] == state and row["src"] == "127.0.0.1" and row["lease_id"] is None
    assert door.stats.unkeyed_by_source == {"127.0.0.1": 1}
    assert door.stats.unkeyed_by_path == {"/v1/chat/completions": 1}
    assert "lane_door_unkeyed" in capsys.readouterr().out               # and in the container log


def test_exempt_reads_pass_unkeyed_are_counted_and_never_logged_per_line(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            codes = [(await c.get(p)).status_code for p in sorted(EXEMPT_GET_PATHS)]
            codes += [(await c.post(p, json={"content": "x"})).status_code for p in sorted(EXEMPT_POST_PATHS)]
            codes.append((await c.get("/slots?fail_on_no_slot=1")).status_code)
        server.close()
        return door, codes

    door, codes = run(go())
    assert set(codes) == {200}
    assert door.stats.exempt == len(EXEMPT_GET_PATHS) + len(EXEMPT_POST_PATHS) + 1
    assert door.stats.calls == {"ok": 0, "missing": 0, "bad": 0}
    assert _rows(tmp_path) == []


def test_the_exempt_list_is_pinned():
    """Spec Phase 2: GET /health /props /slots /v1/models /metrics never refused. Plus /models (llama.cpp's
    alias) and the three no-model POST reads the gateway and the probes use. A change here is a change to
    what the door lets through unkeyed: update the spec and this test together."""
    assert EXEMPT_GET_PATHS == {"/health", "/props", "/slots", "/v1/models", "/models", "/metrics", "/orion/server-info"}
    assert EXEMPT_POST_PATHS == {"/tokenize", "/detokenize", "/apply-template"}
    assert not is_exempt("POST", "/slots") and not is_exempt("POST", "/completion")
    assert not is_exempt("GET", "/v1/chat/completions") and is_exempt("HEAD", "/health")


def test_stats_show_the_key_hash_never_the_key(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            await c.post("/v1/chat/completions", json={}, headers={"X-Orion-Lane-Key": KEY})
            r = await c.get("/door/stats")
        server.close()
        return engine, r

    engine, r = run(go())
    stats = r.json()
    assert stats["key_hash"] == key_hash(KEY) and len(stats["key_hash"]) == 16
    assert KEY not in r.text
    assert stats["calls"]["ok"] == 1 and stats["mode"] == "shadow" and stats["lane"] == "chat"
    assert [s["path"] for s in engine.seen] == ["/v1/chat/completions"]   # /door/stats is the door's own


def test_a_stream_is_relayed_as_it_arrives(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        stamps = []
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            t0 = time.monotonic()
            async with c.stream("POST", "/v1/stream", json={}, headers={"X-Orion-Lane-Key": KEY}) as r:
                async for chunk in r.aiter_text():
                    stamps.append((time.monotonic() - t0, chunk))
        server.close()
        return stamps

    stamps = run(go())
    text = "".join(c for _, c in stamps)
    assert text.count("data:") == 3
    assert stamps[0][0] < 0.09   # the first token arrives before the engine has written the rest
    assert _rows(tmp_path)[0]["status"] == 200


def test_a_client_that_hangs_up_mid_stream_closes_the_engine_connection(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/stream-slow HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nX-Orion-Lane-Key: "
                     + KEY.encode() + b"\r\n\r\n{}")
        await writer.drain()
        await reader.readuntil(b"data:")
        writer.close()
        await asyncio.wait_for(engine.closed_early.wait(), 3.0)
        await asyncio.sleep(0.1)
        server.close()
        return door

    run(go())
    (row,) = _rows(tmp_path)
    assert (row["status"], row.get("error")) == (499, "client_gone")


def test_an_engine_that_is_down_is_a_502_and_is_logged(tmp_path):
    async def go():
        door, server, port = await _door(tmp_path, 9)   # nothing listens on port 9
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            r = await c.post("/v1/chat/completions", json={}, headers={"X-Orion-Lane-Key": KEY})
            h = await c.get("/health")
        server.close()
        return door, r, h

    door, r, h = run(go())
    assert r.status_code == 502 and r.json()["error"]["type"] == "lane_door_upstream_unreachable"
    assert h.status_code == 502            # the pool's probe sees a down seat, not silence
    assert _rows(tmp_path)[0]["error"].startswith("upstream_unreachable")
    assert door.stats.upstream_errors == 2


def test_an_engine_error_passes_through_unchanged(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            r = await c.post("/boom", json={}, headers={"X-Orion-Lane-Key": KEY})
        server.close()
        return r

    r = run(go())
    assert r.status_code == 500 and r.json()["path"] == "/boom"
    assert _rows(tmp_path)[0]["status"] == 500


def test_mode_off_is_a_plain_pass_through_that_still_strips_the_key(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start(), mode="off")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            await c.post("/v1/chat/completions", json={}, headers={"X-Orion-Lane-Key": KEY})
            await c.post("/v1/chat/completions", json={})
        server.close()
        return engine, door

    engine, door = run(go())
    assert len(engine.seen) == 2 and all("x-orion-lane-key" not in s["headers"] for s in engine.seen)
    assert _rows(tmp_path) == [] and door.stats.calls == {"ok": 0, "missing": 0, "bad": 0}


def test_a_large_body_and_a_chunked_request_arrive_intact(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        big = json.dumps({"messages": [{"role": "user", "content": "x" * 2_000_000}]}).encode()
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            r = await c.post("/v1/chat/completions", content=big, headers={"X-Orion-Lane-Key": KEY})

            async def gen():
                yield b'{"a":'
                yield b"1}"
            r2 = await c.post("/v1/chat/completions", content=gen(), headers={"X-Orion-Lane-Key": KEY})
        server.close()
        return engine, r, r2, big

    engine, r, r2, big = run(go())
    assert r.json()["n"] == len(big) and engine.seen[0]["body"] == big
    assert r2.status_code == 200 and engine.seen[1]["body"] == b'{"a":1}'
    assert "chunked" in engine.seen[1]["headers"]["transfer-encoding"]


def test_a_malformed_request_is_a_400_and_reaches_nothing(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GARBAGE\r\nno colon here\r\n\r\n")
        await writer.drain()
        data = await reader.read()
        server.close()
        return engine, door, data

    engine, door, data = run(go())
    assert data.startswith(b"HTTP/1.1 400") and engine.seen == [] and door.stats.bad_requests == 1


def test_key_state():
    assert key_state(None, KEY) == "missing"
    assert key_state(KEY.encode(), KEY) == "ok"
    assert key_state(b"nope", KEY) == "bad"
    assert key_state(b"", "") == "bad"         # no door key configured: nothing reads as ok


def test_log_files_roll_daily_and_old_ones_are_removed(tmp_path):
    log = RequestLog(str(tmp_path), "chat", retention_days=30)
    old = tmp_path / "chat-2000-01-01.jsonl"
    old.write_text("{}\n")
    other_lane = tmp_path / "fast-2000-01-01.jsonl"
    other_lane.write_text("{}\n")
    log.write({"ts": "2026-11-14T00:00:00.000+00:00", "x": 1})
    log.write({"ts": "2026-11-15T00:00:00.000+00:00", "x": 2})
    log.close()
    assert not old.exists() and other_lane.exists()   # each door prunes only its own lane
    assert (tmp_path / "chat-2026-11-14.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "chat-2026-11-15.jsonl").exists()


def test_the_host_header_can_be_set_for_an_engine_that_needs_it(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start(), upstream_host_header="127.0.0.1:8080")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            await c.get("/health")
        server.close()
        return engine

    assert run(go()).seen[0]["headers"]["host"] == "127.0.0.1:8080"


class InterimEngine(Engine):
    """Answers every request with an interim 100 Continue head before the real reply (llama.cpp does when
    a caller sends Expect: 100-continue)."""

    async def handle(self, reader, writer):
        raw = await reader.readuntil(b"\r\n\r\n")
        headers = {k.lower(): v.strip() for k, _, v in (l.partition(":") for l in raw.decode().split("\r\n")[1:] if l)}
        self.seen.append({"headers": headers})
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        body = await reader.readexactly(int(headers.get("content-length", "0")))
        data = json.dumps({"n": len(body)}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
        await writer.drain()
        writer.close()


def test_expect_100_continue_gets_the_final_reply_not_just_the_interim_one(tmp_path):
    """Review finding: an interim 1xx ended the exchange; curl sends Expect on large POST bodies."""
    async def go():
        engine = InterimEngine()
        door, server, port = await _door(tmp_path, await engine.start())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nExpect: 100-continue\r\n"
                     b"X-Orion-Lane-Key: " + KEY.encode() + b"\r\n\r\n")
        await writer.drain()
        first = await reader.readuntil(b"\r\n\r\n")
        writer.write(b"{}")
        await writer.drain()
        rest = await reader.read()
        server.close()
        return engine, first, rest

    engine, first, rest = run(go())
    assert first == b"HTTP/1.1 100 Continue\r\n\r\n"           # the door's own
    assert rest.startswith(b"HTTP/1.1 200 OK") and rest.endswith(b'{"n": 2}')
    assert "expect" not in engine.seen[0]["headers"]
    assert _rows(tmp_path)[0]["status"] == 200


def test_a_truncated_request_body_is_the_callers_fault_not_the_engines(tmp_path):
    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{}")
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.3)
        server.close()
        return door

    door = run(go())
    (row,) = _rows(tmp_path)
    assert row["status"] in (400, 499) and door.stats.upstream_errors == 0


def test_a_silent_caller_is_dropped_after_the_head_timeout(tmp_path, monkeypatch):
    import app.door as door_mod

    monkeypatch.setattr(door_mod, "HEAD_TIMEOUT_SEC", 0.2)

    async def go():
        engine = Engine()
        door, server, port = await _door(tmp_path, await engine.start())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/chat")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(), 3.0)
        server.close()
        return door, data

    door, data = run(go())
    assert data.startswith(b"HTTP/1.1 400") and door.stats.bad_requests == 1
