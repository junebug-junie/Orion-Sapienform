#!/usr/bin/env python3
"""Wait until no chat turn is in flight, so a rebuild does not kill one.

The post-pull rebuild (scripts/rebuild_services_from_git_diff.sh) recreates
orion-hub and orion-harness-governor. A chat turn in flight dies with either:
Hub holds the browser's connection and the governor runs the `claude` motor.
Before this, a pull during a chat killed the reply with no retry.

Hub already tracks every turn it hands to the governor
(orion/hub/runtime_activity.py, GET /api/runtime-activity). Juniper's chat turns
carry ``source == "chat"``. Durable-run turns (curiosity, consults, world pulse)
carry their own source tag and are not waited for: durable-runs retries them and
resumes their CLI session (orion/harness/session_resume.py).

Never blocks a rebuild for good: it waits at most ``--wait-s`` and always exits 0.
Prints one verdict line on stdout:

  idle                                 nothing in flight at the first look
  drained waited_s=N                   a chat turn finished inside the wait
  timeout in_flight=N waited_s=N       still busy at the deadline; rebuild proceeds
  unknown reason=<why>                 Hub unreachable or tracker off; rebuild proceeds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

CHAT_SOURCE = "chat"
IN_FLIGHT_PHASES = ("running", "queued")
DEFAULT_PORT = "8080"
ROOT = Path(__file__).resolve().parents[1]


def chat_turns_in_flight(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Chat turns (any lane) that are running or queued in a runtime-activity snapshot."""
    lanes = snapshot.get("lanes")
    if not isinstance(lanes, dict):
        return []
    out: list[dict[str, Any]] = []
    for lane in lanes.values():
        if not isinstance(lane, dict):
            continue
        for phase in IN_FLIGHT_PHASES:
            for turn in lane.get(phase) or []:
                if isinstance(turn, dict) and turn.get("source") == CHAT_SOURCE:
                    out.append(turn)
    return out


def hub_url(env: dict[str, str] | None = None, env_file: Path | None = None) -> str:
    """``ORION_HUB_URL``, else 127.0.0.1 on Hub's HUB_PORT (Hub runs network_mode: host)."""
    env = os.environ if env is None else env
    if env.get("ORION_HUB_URL"):
        return env["ORION_HUB_URL"].rstrip("/")
    port = env.get("HUB_PORT") or ""
    path = env_file or ROOT / "services" / "orion-hub" / ".env"
    if not port and path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "HUB_PORT":
                port = value.strip().strip("'\"")
    return f"http://127.0.0.1:{port or DEFAULT_PORT}"


def fetch_snapshot(base: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base}/api/runtime-activity", timeout=timeout) as resp:  # noqa: S310 -- loopback
        return json.loads(resp.read().decode("utf-8"))


def wait_idle(
    fetch: Callable[[], dict[str, Any]],
    *,
    wait_s: float,
    poll_s: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    start = clock()
    deadline = start + max(0.0, wait_s)
    waited = False
    while True:
        try:
            busy = chat_turns_in_flight(fetch())
        except urllib.error.HTTPError as exc:
            return f"unknown reason=http_{exc.code}"
        except Exception as exc:  # noqa: BLE001 -- Hub down: nothing to protect, proceed
            return f"unknown reason={type(exc).__name__}"
        elapsed = int(clock() - start)
        if not busy:
            return f"drained waited_s={elapsed}" if waited else "idle"
        if clock() >= deadline:
            return f"timeout in_flight={len(busy)} waited_s={elapsed}"
        waited = True
        sleep(max(0.0, min(poll_s, deadline - clock())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wait-s", type=float, default=600.0)
    parser.add_argument("--poll-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    base = hub_url()
    print(wait_idle(lambda: fetch_snapshot(base), wait_s=args.wait_s, poll_s=args.poll_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
