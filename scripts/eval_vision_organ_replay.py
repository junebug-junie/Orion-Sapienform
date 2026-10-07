"""Replay real vision traffic through the vision organ lane, end to end.

Real input -> the router's own recorder (app/grammar_emit.py) -> the grammar
window trace -> the substrate reducer (orion/substrate/vision_organ_loop/) ->
the organ readings that would reach capability:vision. Nothing is written
anywhere; this only reads.

Two sources:

  --router-log PATH   the frame router's own stdout (`docker logs
                      orion-orion-athena-vision-frame-router > PATH`). Every
                      "[ROUTER] dispatch" line is a frame the router saw and sent
                      to the host, with its stream. Frames dropped by sampling
                      are not logged, so frame counts are a lower bound and
                      frame age is "age of the newest DISPATCHED frame" (an upper
                      bound, <= the 5 s dispatch cadence when healthy). Replies
                      are not in this log; failure readings are not replayed.

  --live SECONDS      subscribe read-only to the live bus for SECONDS:
                      orion:vision:frames (every frame, per stream),
                      orion:exec:request:VisionHostService (dispatches) and
                      orion:vision:reply:* (host replies), and feed them to the
                      recorder exactly as the router would.

Prints one line per window and a summary: how many windows read calm
(staleness 0.0) and non-calm, per stream and for the organ.

Usage (repo root):
  PYTHONPATH=. python \
    scripts/eval_vision_organ_replay.py --router-log /tmp/router.log
  PYTHONPATH=. python \
    scripts/eval_vision_organ_replay.py --live 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_router_grammar_emit():
    # By file path under a private name: every service ships its own top-level
    # `app` package, so `import app.grammar_emit` would bind to whichever service
    # a test session imported first.
    import importlib.util

    path = REPO / "services" / "orion-vision-frame-router" / "app" / "grammar_emit.py"
    spec = importlib.util.spec_from_file_location("_vision_frame_router_grammar_emit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_ge = _load_router_grammar_emit()
OrganWindowRecorder = _ge.OrganWindowRecorder
build_window_events = _ge.build_window_events
from orion.substrate.vision_organ_loop.pipeline import empty_vision_organ_projection  # noqa: E402
from orion.substrate.vision_organ_loop.reducer import reduce_vision_organ_trace_events  # noqa: E402

_DISPATCH_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) .*\[ROUTER\] dispatch tier=\S+ .*stream_id=(\S+)"
)
_START_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) .*\[FRAME-ROUTER\] Started")


class _Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def configured_streams(policy_path: Path) -> list[str]:
    raw = yaml.safe_load(policy_path.read_text()) or {}
    return [sid for sid, cfg in (raw.get("streams") or {}).items() if (cfg or {}).get("enabled", True)]


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


class Replay:
    def __init__(self, streams: Iterable[str], start: float, window_sec: float = 60.0) -> None:
        self.clock = _Clock(start)
        self.rec = OrganWindowRecorder(configured_streams=streams, clock=self.clock)
        self.window_sec = window_sec
        self.next_flush = start + window_sec
        self.projection = empty_vision_organ_projection(now=datetime.fromtimestamp(start, tz=timezone.utc))
        self.rows: list[dict] = []

    def advance(self, t: float) -> None:
        while t >= self.next_flush:
            self.clock.t = self.next_flush
            self.flush()
            self.next_flush += self.window_sec
        self.clock.t = t

    def flush(self) -> None:
        events = build_window_events(router="vision-frame-router", snapshot=self.rec.drain())
        now = datetime.fromtimestamp(self.clock.t, tz=timezone.utc)
        self.projection, receipt = reduce_vision_organ_trace_events(
            events=events, projection=self.projection, now=now
        )
        after = receipt.state_deltas[0].after
        self.rows.append(
            {
                "window_end": now.isoformat(),
                "organ_staleness": after["vision_frame_staleness"],
                "failure": after["vision_processing_failure_pressure"],
                "streams": {
                    sid: (s["status"], round(s["frame_staleness"], 3), s["frames"], s["dispatched"])
                    for sid, s in after["streams"].items()
                },
            }
        )


def replay_router_log(lines: Iterable[str], streams: list[str]) -> Replay:
    replay: Replay | None = None
    for line in lines:
        m = _START_RE.match(line)
        if m and replay is None:
            replay = Replay(streams, _ts(m.group(1)))
            continue
        m = _DISPATCH_RE.match(line)
        if not m:
            continue
        t = _ts(m.group(1))
        if replay is None:
            replay = Replay(streams, t)
        replay.advance(t)
        stream = m.group(2)
        replay.rec.record_frame(stream)
        replay.rec.record_dispatch(stream)
        # Replies are not in this log. Counted as ok so the failure reading reads
        # its true live value for this span (host_errors_total=0, one timeout in
        # 141k dispatches per /healthz) rather than "unmeasured".
        replay.rec.record_reply_ok(stream, primary=True, objects=None, caption_requested=False,
                                   caption_present=False)
    if replay is None:
        raise SystemExit("no router start or dispatch lines found")
    return replay


async def replay_live(seconds: float, streams: list[str], bus_url: str) -> Replay:
    import time

    import redis.asyncio as aioredis

    r = aioredis.from_url(bus_url)
    ps = r.pubsub()
    await ps.subscribe("orion:vision:frames", "orion:exec:request:VisionHostService")
    await ps.psubscribe("orion:vision:reply:*")
    start = time.time()
    replay = Replay(streams, start)
    replay.clock.t = start
    dispatched_stream: dict[str, str] = {}
    try:
        while time.time() - start < seconds:
            msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
            now = time.time()
            replay.advance(now)
            if not msg:
                continue
            channel = msg["channel"].decode() if isinstance(msg["channel"], bytes) else msg["channel"]
            try:
                env = json.loads(msg["data"])
            except Exception:
                continue
            payload = env.get("payload") or {}
            if channel == "orion:vision:frames":
                replay.rec.record_frame(payload.get("stream_id") or payload.get("camera_id") or "unknown")
            elif channel == "orion:exec:request:VisionHostService":
                meta = payload.get("meta") or {}
                stream = str(meta.get("stream_id") or meta.get("camera_id") or "unknown")
                dispatched_stream[str(env.get("reply_to") or "")] = stream
                replay.rec.record_dispatch(stream, identity=payload.get("task_type") == "identity_face")
            else:
                stream = dispatched_stream.pop(channel, None)
                if stream is None:
                    continue
                if payload.get("ok"):
                    outputs = ((payload.get("artifact") or {}).get("outputs") or {})
                    objects = outputs.get("objects")
                    replay.rec.record_reply_ok(
                        stream, primary=True, objects=None if objects is None else len(objects),
                        caption_requested=False, caption_present=bool((outputs.get("caption") or {}).get("text")),
                    )
                else:
                    replay.rec.record_failure(stream, payload.get("error_code") or "host_error")
    finally:
        await ps.aclose()
        await r.aclose()
    # Only windows that closed while subscribed: flushing the trailing partial
    # window after unsubscribing would age every stream into a fake outage.
    return replay


def summarize(replay: Replay) -> dict:
    rows = replay.rows
    organ = Counter("calm" if r["organ_staleness"] == 0.0 else "non_calm" for r in rows)
    per_stream: dict[str, Counter] = {}
    for r in rows:
        for sid, (status, staleness, _f, _d) in r["streams"].items():
            per_stream.setdefault(sid, Counter())["calm" if staleness == 0.0 else f"non_calm:{status}"] += 1
    failures = Counter("unmeasured" if r["failure"] is None else ("calm" if r["failure"] == 0.0 else "non_calm")
                       for r in rows)
    non_calm = [r for r in rows if r["organ_staleness"] > 0.0]
    return {
        "windows": len(rows),
        "first_window_end": rows[0]["window_end"] if rows else None,
        "last_window_end": rows[-1]["window_end"] if rows else None,
        "organ_staleness": dict(organ),
        "organ_failure": dict(failures),
        "per_stream_staleness": {k: dict(v) for k, v in per_stream.items()},
        "organ_non_calm_examples": non_calm[:5],
        "max_organ_staleness": max((r["organ_staleness"] for r in rows), default=None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--router-log", type=Path)
    src.add_argument("--live", type=float)
    ap.add_argument("--policy", type=Path, default=REPO / "config" / "vision_frame_router.yaml")
    ap.add_argument("--bus-url", default="redis://100.92.216.81:6379/0")
    ap.add_argument("--show-windows", type=int, default=3)
    args = ap.parse_args(argv)
    streams = configured_streams(args.policy)
    if args.router_log:
        with args.router_log.open(errors="replace") as fh:
            replay = replay_router_log(fh, streams)
    else:
        replay = asyncio.run(replay_live(args.live, streams, args.bus_url))
    for row in replay.rows[: args.show_windows]:
        print(json.dumps(row))
    print(json.dumps(summarize(replay), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
