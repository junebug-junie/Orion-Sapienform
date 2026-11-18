"""The lane door: a thin HTTP/1.1 proxy in front of one model server (leased-evals design, Phase 2).

docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md, "Phase 2". The gateway
sends one header on every call it dispatches under a pool grant: ``X-Orion-Lease-Id``, the granted lease.

The door reads it (``leased`` / ``unleased`` / ``invalid``), strips it, forwards the call byte-for-byte to
the engine on circe's loopback, and logs ONE JSON line per model call: time, lane, source address,
method, path, lease state, lease id, status, duration. Never prompt text. In ``shadow`` mode it refuses
nothing. ``off`` = pure pass-through (header still stripped). In ``enforce`` mode (Phase 3, one lane at a
time, Juniper's call after the shadow evidence) a model call without a lease id gets 401
``lane_door_unleased`` and never reaches the engine; exempt reads still pass.

There is no shared secret. The door is there to catch ACCIDENTAL bypass: one of Orion's own evals or
scripts calling a model server directly instead of the gateway. It does not fence a deliberate caller (a
caller can send any lease id; and on athena the Hub holds the docker socket, so a secret would not stop a
local caller either). A made-up lease id is caught offline: scripts/lane_door_report.py joins the logged
lease ids to ``gpu_pool_events``.

Not a model call, never refused in any mode, counted but not logged per line (the pool probes them
every few seconds): ``GET /health /props /slots /v1/models /models /metrics`` and the no-model reads
``POST /tokenize /detokenize /apply-template``. Pinned by tests/test_door.py and kept equal to
orion/evals/leased_llm.py's WORKER_READ_* by tests/test_lane_door_exempt_parity.py (repo root).

The door asks the pool nothing per call: a pool RPC here would sit on Juniper's chat path.

Taking the lane port (``serve``): the door runs on the host network and binds the lane port only once its
engine answers on the loopback upstream. Before that the engine still publishes the lane port itself, and
the door waits (``lane_door_waiting``) instead of stealing the port from it. So a deploy can start the door
before or after the engine moves, in any order, and an engine restart never loses its port to the door.

One request per client connection: the door answers with ``Connection: close``. The gateway opens a
new connection per call anyway (a new httpx client per request), so this costs nothing it pays today.
Request and response bodies are relayed as they arrive (Content-Length, chunked, or until close), so a
stream's tokens are not held back. A client that hangs up mid-call closes the engine connection, so
the engine stops generating -- the same as when the gateway hangs up on the engine directly.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .settings import Settings

logger = logging.getLogger("lane-door")

LEASE_HEADER = b"x-orion-lease-id"
EXEMPT_GET_PATHS = frozenset({"/health", "/props", "/slots", "/v1/models", "/models", "/metrics",
                              "/orion/server-info"})   # the pool's vLLM probe (orion-gpu-pool app/main.py _probe)
EXEMPT_POST_PATHS = frozenset({"/tokenize", "/detokenize", "/apply-template"})
STATS_PATH = "/door/stats"
SPEC = "docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md"
REFUSAL = {"error": {"type": "lane_door_unleased",
                     "message": f"call orion-llm-gateway with a route; see {SPEC}"}}
MAX_HEAD_BYTES = 64 * 1024
CHUNK = 64 * 1024
_HOP = {b"connection", b"keep-alive", b"proxy-connection"}
_LEASE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
MAX_SOURCES = 256   # distinct unleased sources kept in /door/stats


class BadRequest(Exception):
    pass


class ClientFailed(Exception):
    """The caller's side failed while the door read its request (truncated body, hang-up mid-upload)."""


HEAD_TIMEOUT_SEC = 30.0   # a caller that sends no complete request head in this long is dropped


def is_exempt(method: str, path: str) -> bool:
    return ((method in ("GET", "HEAD") and path in EXEMPT_GET_PATHS)
            or (method == "POST" and path in EXEMPT_POST_PATHS))


def lease_state(presented: Optional[bytes]) -> tuple[str, Optional[str]]:
    """(state, lease id to log). ``leased``: a well-formed lease id; ``unleased``: no header (or an empty
    one); ``invalid``: a header that is not a lease id (logged as ``"invalid"``, never echoed)."""
    lease = (presented or b"").decode("latin-1").strip()
    if not lease:
        return "unleased", None
    if _LEASE_RE.match(lease):
        return "leased", lease
    return "invalid", "invalid"


@dataclass
class Head:
    start: bytes                              # request or status line, no CRLF
    headers: list[tuple[bytes, bytes]]        # original names, in order

    def get(self, name: bytes) -> Optional[bytes]:
        for k, v in self.headers:
            if k.lower() == name:
                return v
        return None

    def without(self, *names: bytes) -> list[tuple[bytes, bytes]]:
        drop = {n.lower() for n in names}
        return [(k, v) for k, v in self.headers if k.lower() not in drop]


def parse_head(raw: bytes) -> Head:
    lines = raw.split(b"\r\n")
    if not lines or not lines[0]:
        raise BadRequest("empty head")
    headers: list[tuple[bytes, bytes]] = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(b":")
        if not sep or not name or name != name.strip():
            raise BadRequest("malformed header")
        headers.append((name, value.strip()))
    return Head(lines[0], headers)


def render_head(start: bytes, headers: list[tuple[bytes, bytes]]) -> bytes:
    return start + b"\r\n" + b"".join(k + b": " + v + b"\r\n" for k, v in headers) + b"\r\n"


async def relay_body(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, head: Head, *,
                     until_eof: bool) -> int:
    """Copy one message body from ``reader`` to ``writer`` by its own framing. Returns bytes copied."""
    te = (head.get(b"transfer-encoding") or b"").lower()
    n = 0
    if b"chunked" in te:
        while True:
            size_line = await reader.readuntil(b"\r\n")
            writer.write(size_line)
            try:
                size = int(size_line.split(b";", 1)[0].strip(), 16)
            except ValueError as exc:
                raise BadRequest("bad chunk size") from exc
            if size == 0:
                while True:   # trailers, then the blank line
                    line = await reader.readuntil(b"\r\n")
                    writer.write(line)
                    if line == b"\r\n":
                        break
                await writer.drain()
                return n
            remaining = size + 2   # data + CRLF
            while remaining:
                data = await reader.read(min(CHUNK, remaining))
                if not data:
                    raise asyncio.IncompleteReadError(b"", remaining)
                writer.write(data)
                remaining -= len(data)
                n += len(data)
            await writer.drain()
    length = head.get(b"content-length")
    if length is not None:
        try:
            remaining = int(length)
        except ValueError as exc:
            raise BadRequest("bad content-length") from exc
        while remaining > 0:
            data = await reader.read(min(CHUNK, remaining))
            if not data:
                raise asyncio.IncompleteReadError(b"", remaining)
            writer.write(data)
            await writer.drain()
            remaining -= len(data)
            n += len(data)
        return n
    if until_eof:
        while True:
            data = await reader.read(CHUNK)
            if not data:
                return n
            writer.write(data)
            await writer.drain()
            n += len(data)
    return 0


class RequestLog:
    """One JSON line per model call, to ``<dir>/<lane>-YYYY-MM-DD.jsonl`` (UTC day). Files older than
    the retention are removed at each day's first write. A write error never fails a call."""

    def __init__(self, directory: str, lane: str, retention_days: int):
        self.dir, self.lane, self.retention_days = Path(directory), lane, retention_days
        self._day: Optional[str] = None
        self._fh = None

    def _open(self, day: str) -> None:
        if self._fh is not None:
            self._fh.close()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.dir / f"{self.lane}-{day}.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        self._day = day
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).strftime("%Y-%m-%d")
        for old in self.dir.glob(f"{self.lane}-*.jsonl"):
            stamp = old.stem[len(self.lane) + 1:]
            if len(stamp) == 10 and stamp < cutoff:
                with contextlib.suppress(OSError):
                    old.unlink()

    def write(self, row: dict[str, Any]) -> None:
        try:
            day = row["ts"][:10]
            if day != self._day:
                self._open(day)
            self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._fh.flush()
        except (OSError, ValueError, TypeError):
            logger.warning("lane_door_log_write_failed", exc_info=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class Stats:
    lane: str
    mode: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    calls: dict[str, int] = field(default_factory=lambda: {"leased": 0, "unleased": 0, "invalid": 0})
    exempt: int = 0
    refused: int = 0
    upstream_errors: int = 0
    bad_requests: int = 0
    unleased_by_source: dict[str, int] = field(default_factory=dict)
    unleased_by_path: dict[str, int] = field(default_factory=dict)

    def count(self, state: str, source: str, path: str) -> None:
        self.calls[state] = self.calls.get(state, 0) + 1
        if state != "leased":
            if source in self.unleased_by_source or len(self.unleased_by_source) < MAX_SOURCES:
                self.unleased_by_source[source] = self.unleased_by_source.get(source, 0) + 1
            if path in self.unleased_by_path or len(self.unleased_by_path) < MAX_SOURCES:
                self.unleased_by_path[path] = self.unleased_by_path.get(path, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {"lane": self.lane, "mode": self.mode, "started_at": self.started_at,
                "calls": dict(self.calls), "exempt": self.exempt, "refused": self.refused,
                "upstream_errors": self.upstream_errors, "bad_requests": self.bad_requests,
                "unleased_by_source": dict(self.unleased_by_source),
                "unleased_by_path": dict(self.unleased_by_path)}


JOURNAL_EVERY_SEC = 60.0
DRAIN_MAX_BYTES = 1 << 20
DRAIN_SEC = 2.0


async def _drain(reader: asyncio.StreamReader) -> None:
    """Read and drop what a refused caller is still sending, within limits."""
    got, deadline = 0, time.monotonic() + DRAIN_SEC
    while got < DRAIN_MAX_BYTES:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        try:
            data = await asyncio.wait_for(reader.read(CHUNK), left)
        except (asyncio.TimeoutError, OSError):
            return
        if not data:
            return
        got += len(data)


def _json_response(status: int, reason: str, body: dict[str, Any]) -> bytes:
    data = json.dumps(body).encode()
    return render_head(f"HTTP/1.1 {status} {reason}".encode(), [
        (b"Content-Type", b"application/json"), (b"Content-Length", str(len(data)).encode()),
        (b"Connection", b"close")]) + data


def _status_of(start: bytes) -> int:
    try:
        return int(start.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return 0


class Door:
    def __init__(self, settings: Settings, log: Optional[RequestLog] = None):
        self.s = settings
        self.stats = Stats(lane=settings.lane, mode=settings.mode)
        self._journaled: dict[tuple[str, str], float] = {}
        self.active = 0   # requests in flight (serve() drains them on SIGTERM)
        self.log = log if log is not None else RequestLog(settings.log_dir, settings.lane,
                                                          settings.log_retention_days)

    # -- one connection = one request ------------------------------------------------------------
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        started = time.monotonic()
        peer = writer.get_extra_info("peername")
        source = str(peer[0]) if isinstance(peer, tuple) and peer else "unknown"
        if source.startswith("::ffff:"):
            source = source[len("::ffff:"):]   # an IPv4 caller on a dual-stack socket
        self.active += 1
        try:
            await self._handle(reader, writer, started, source)
        except Exception:  # noqa: BLE001 -- one bad call never takes the door down
            logger.warning("lane_door_handler_error lane=%s src=%s", self.s.lane, source, exc_info=True)
        finally:
            self.active -= 1
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, started: float,
                      source: str) -> None:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_SEC)
            head = parse_head(raw[:-4])
            method_b, target_b, version = head.start.split(b" ", 2)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, BadRequest, ValueError):
            self.stats.bad_requests += 1
            with contextlib.suppress(Exception):
                writer.write(_json_response(400, "Bad Request", {"error": {"type": "lane_door_bad_request"}}))
                await writer.drain()
            return
        method = method_b.decode("latin-1").upper()
        path = target_b.decode("latin-1").split("?", 1)[0]

        if method == "GET" and path == STATS_PATH:
            writer.write(_json_response(200, "OK", self.stats.as_dict()))
            await writer.drain()
            return

        exempt = is_exempt(method, path)
        state, lease_id = lease_state(head.get(LEASE_HEADER))

        # Expect: 100-continue is answered by the door itself (below), never forwarded: an interim 1xx from
        # the engine would otherwise have to be relayed as a second head on the same reply.
        expect_continue = b"100-continue" in (head.get(b"expect") or b"").lower()
        headers = head.without(LEASE_HEADER, b"expect", *_HOP)
        if self.s.upstream_host_header:
            headers = [(k, v) for k, v in headers if k.lower() != b"host"]
            headers.insert(0, (b"Host", self.s.upstream_host_header.encode("latin-1")))
        headers.append((b"Connection", b"close"))

        refused = self.s.mode == "enforce" and not exempt and state != "leased"
        if refused:
            # Refused before any engine connection. Then the request body is read and dropped (up to
            # DRAIN_MAX_BYTES / DRAIN_SEC): closing a socket with unread data sends a reset, and a caller still
            # uploading would see a broken pipe instead of the 401 and its explanation.
            self.stats.refused += 1
            with contextlib.suppress(Exception):
                writer.write(_json_response(401, "Unauthorized", REFUSAL))
                await writer.drain()
                if writer.can_write_eof():
                    writer.write_eof()
            status, err = 401, None
        else:
            status, err = await self._forward(reader, writer, Head(head.start, headers), version,
                                              expect_continue=expect_continue)
        if self.s.mode == "off":
            return
        if exempt:
            self.stats.exempt += 1
            return
        self.stats.count(state, source, path)
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "lane": self.s.lane,
               "src": source, "method": method, "path": path, "lease": state, "lease_id": lease_id,
               "status": status, "ms": round((time.monotonic() - started) * 1000.0, 1)}
        if err:
            row["error"] = err
        if refused:
            row["refused"] = True
        self.log.write(row)
        if state != "leased" and self._journal_due(source, path):
            # Rare and the whole point of the shadow phase: also in the container log (journald), at most
            # once a minute per source and path (a caller retrying against an enforcing door would flood it;
            # the JSONL keeps every call).
            print("lane_door_unleased " + json.dumps(row, separators=(",", ":")), flush=True)
        if refused:
            await _drain(reader)   # after the log row: the call is recorded whatever the caller does next

    def _journal_due(self, source: str, path: str) -> bool:
        now, key = time.monotonic(), (source, path)
        last = self._journaled.get(key)
        if last is not None and now - last < JOURNAL_EVERY_SEC:
            return False
        if len(self._journaled) > 4096:
            self._journaled.clear()
        self._journaled[key] = now
        return True

    async def _forward(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, head: Head,
                       version: bytes, expect_continue: bool = False) -> tuple[int, Optional[str]]:
        """Relay the request to the engine and its reply back. Returns (status, error)."""
        try:
            up_r, up_w = await asyncio.wait_for(
                asyncio.open_connection(self.s.upstream_host, self.s.upstream_port, limit=MAX_HEAD_BYTES),
                self.s.connect_timeout_sec)
        except (OSError, asyncio.TimeoutError) as exc:
            self.stats.upstream_errors += 1
            with contextlib.suppress(Exception):
                writer.write(_json_response(502, "Bad Gateway", {"error": {
                    "type": "lane_door_upstream_unreachable", "message": f"engine for lane {self.s.lane} "
                    f"did not accept a connection: {type(exc).__name__}"}}))
                await writer.drain()
            return 502, f"upstream_unreachable:{type(exc).__name__}"

        client_gone = asyncio.Event()

        async def exchange() -> int:
            up_w.write(render_head(head.start, head.headers))
            if expect_continue:
                writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await writer.drain()
            try:
                await relay_body(reader, up_w, head, until_eof=False)
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, BadRequest) as exc:
                raise ClientFailed(type(exc).__name__) from exc
            await up_w.drain()
            # The client has sent its whole request. From here, EOF or an error on its side means it hung
            # up (a reset as much as a clean close); pipelined bytes are ignored (one request per connection).
            watch = asyncio.ensure_future(reader.read(1))
            watch.add_done_callback(lambda f: client_gone.set() if not f.cancelled() and (
                f.exception() is not None or f.result() == b"") else None)
            try:
                while True:
                    raw = await up_r.readuntil(b"\r\n\r\n")
                    rhead = parse_head(raw[:-4])
                    status = _status_of(rhead.start)
                    if not (100 <= status < 200) or status == 101:
                        break   # an interim 1xx from the engine is dropped; the caller gets the final reply
                writer.write(render_head(rhead.start, rhead.without(*_HOP) + [(b"Connection", b"close")]))
                await writer.drain()
                if head.start.split(b" ", 1)[0].upper() != b"HEAD" and status not in (204, 304) \
                        and not 100 <= status < 200:
                    await relay_body(up_r, writer, rhead, until_eof=True)
                return status
            finally:
                watch.cancel()

        task = asyncio.ensure_future(exchange())
        gone = asyncio.ensure_future(client_gone.wait())
        try:
            await asyncio.wait({task, gone}, return_when=asyncio.FIRST_COMPLETED)
            if not task.done():
                task.cancel()   # the client hung up: drop the engine connection so it stops generating
                with contextlib.suppress(BaseException):
                    await task
                return 499, "client_gone"
            try:
                return task.result(), None
            except ClientFailed as exc:
                self.stats.bad_requests += 1
                return 400, f"client_request_incomplete:{exc}"
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, BadRequest, OSError) as exc:
                if client_gone.is_set() or writer.is_closing():
                    return 499, "client_gone"   # a write to a client that left, not an engine fault
                self.stats.upstream_errors += 1
                return 502, f"relay:{type(exc).__name__}"
        finally:
            gone.cancel()
            with contextlib.suppress(Exception):
                up_w.close()


WAIT_LOG_EVERY_SEC = 60.0
STATE_FILE = os.environ.get("LANE_DOOR_STATE_FILE", "/tmp/lane-door.state")   # read by app/health.py


def _set_state(state: str) -> None:
    with contextlib.suppress(OSError):
        Path(STATE_FILE).write_text(state + "\n", encoding="utf-8")


async def _upstream_answers(settings: Settings) -> bool:
    """True when the engine accepts a TCP connection on the loopback upstream (it has moved behind the door)."""
    try:
        _, w = await asyncio.wait_for(asyncio.open_connection(settings.upstream_host, settings.upstream_port),
                                      settings.connect_timeout_sec)
    except (OSError, asyncio.TimeoutError):
        return False
    with contextlib.suppress(Exception):
        w.close()
        await w.wait_closed()
    return True


async def take_port(door: Door, settings: Settings, *, stop: Optional[asyncio.Event] = None
                    ) -> Optional[asyncio.AbstractServer]:
    """Bind the lane port, but only once the engine answers on the upstream. Polls every
    ``bind_poll_sec``; logs ``lane_door_waiting`` at most once a minute. Returns None if ``stop`` is set
    while waiting.

    Why wait for the engine first: until an engine is moved to loopback it publishes the lane port itself.
    A door that grabbed the port the moment it was free (an engine restart, a crash) would leave the
    restarted engine unable to bind its old port -- the lane down until someone acts. Waiting for the
    upstream means the door only takes the port from an engine that no longer wants it."""
    last_log = -WAIT_LOG_EVERY_SEC
    host = settings.listen_host or None   # None = every interface, IPv4 and IPv6 (what docker publishing did)
    while stop is None or not stop.is_set():
        reason = "engine_not_on_upstream"
        if await _upstream_answers(settings):
            try:
                return await asyncio.start_server(door.handle, host, settings.listen_port,
                                                  limit=MAX_HEAD_BYTES, backlog=512)
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                reason = "port_in_use"
        now = time.monotonic()
        if now - last_log >= WAIT_LOG_EVERY_SEC:
            last_log = now
            print(f"lane_door_waiting lane={settings.lane} reason={reason} port={settings.listen_port} "
                  f"upstream={settings.upstream_host}:{settings.upstream_port}", flush=True)
        if stop is None:
            await asyncio.sleep(settings.bind_poll_sec)
        else:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), settings.bind_poll_sec)
    return None


async def drain(door: Door, seconds: float) -> int:
    """Wait for in-flight calls to finish, at most ``seconds``. Returns how many were still running."""
    deadline = time.monotonic() + seconds
    while door.active > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    return door.active


async def serve(settings: Settings, door: Optional[Door] = None) -> None:
    door = door or Door(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)
    print(f"lane_door_start lane={settings.lane} mode={settings.mode} upstream={settings.upstream_host}:"
          f"{settings.upstream_port} listen={settings.listen_host or '*'}:{settings.listen_port}", flush=True)

    async def summaries() -> None:
        while True:
            await asyncio.sleep(settings.summary_interval_sec)
            s = door.stats
            print(f"lane_door_summary lane={s.lane} mode={s.mode} leased={s.calls['leased']} "
                  f"unleased={s.calls['unleased']} invalid={s.calls['invalid']} exempt={s.exempt} "
                  f"refused={s.refused} upstream_errors={s.upstream_errors}", flush=True)

    ticker = asyncio.ensure_future(summaries())
    try:
        _set_state("waiting")
        server = await take_port(door, settings, stop=stop)
        if server is None:
            return
        _set_state("listening")
        print(f"lane_door_listening lane={settings.lane} port={settings.listen_port}", flush=True)
        await stop.wait()
        # A stop (a redeploy of the door): no new calls, then let the ones in flight finish -- a Hub chat
        # turn's stream must not be cut by a door image update.
        server.close()
        left = await drain(door, settings.drain_sec)
        print(f"lane_door_stop lane={settings.lane} cut_in_flight={left}", flush=True)
    finally:
        ticker.cancel()
        door.log.close()


__all__ = ["Door", "EXEMPT_GET_PATHS", "EXEMPT_POST_PATHS", "LEASE_HEADER", "RequestLog", "Stats",
           "drain", "is_exempt", "lease_state", "serve", "take_port"]


if os.environ.get("LANE_DOOR_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
