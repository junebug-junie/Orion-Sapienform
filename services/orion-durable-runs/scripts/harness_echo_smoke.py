#!/usr/bin/env python3
"""Hand-triggered smoke for the durable-run -> harness step seam (workflow ``harness.echo_smoke``).

Submits one admitted run to orion-durable-runs (``POST /runs``), then polls ``GET /runs/<id>`` until it
ends. The run waits for an agent-lane GPU pool hold, runs ONE harness turn (the same ``claude -p``
motor chat and curiosity use, via Hub), and checkpoints the turn's final text as the step's result.

    python3 services/orion-durable-runs/scripts/harness_echo_smoke.py            # submit + wait
    python3 services/orion-durable-runs/scripts/harness_echo_smoke.py --print    # request JSON only

Exit 0 only when the run completed AND the step text carries the expected marker. Exit 1: the smoke
ran and FAILED. Exit 2: the submit was refused or durable-runs was unreachable. Exit 3: still running
when polling gave up. The turn's
correlation id is printed with the query that finds it in ``cockpit_turn_sighting``.

Stdlib only (no ``orion`` import), so it runs with the host's plain ``python3``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

MARKER = "harness echo smoke ok"
DEFAULT_PROMPT = (
    "Durable-run harness smoke test. Use your Bash tool exactly once to run "
    "`date -u +%Y-%m-%dT%H:%M:%SZ`. Then reply with exactly one line: "
    f"`{MARKER} <that output>`. Do nothing else."
)
TERMINAL = {"completed", "failed", "cancelled"}


def build_request(*, prompt: str = DEFAULT_PROMPT, timeout_sec: float = 600.0, deadline_hours: float = 2.0,
                  now: datetime | None = None) -> dict:
    """The ``DurableRunRequestV1`` payload (plain JSON). Agent lane, background priority, a deadline
    so a smoke that never gets a hold ends instead of queueing forever."""
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"harness-echo-smoke-{stamp}-{uuid.uuid4().hex[:6]}"
    return {
        "run_id": run_id,
        "workflow": "harness.echo_smoke",
        "correlation_id": str(uuid.uuid4()),
        "requested_at": now.isoformat(),
        "brief": {"smoke": "harness.echo_smoke", "prompt": prompt, "timeout_sec": float(timeout_sec)},
        "admission": {
            "resource": "llm.route.agent", "preferred_lane": "agent", "priority": "background",
            "deadline_at": (now + timedelta(hours=deadline_hours)).isoformat(),
        },
    }


def _http(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode() or "{}")


def verdict(status: dict) -> tuple[bool, str]:
    """(ok, one-line reason) for a terminal ``GET /runs/<id>`` body."""
    if status.get("status") != "completed":
        return False, f"run ended {status.get('status')}: {status.get('error')}"
    step = (status.get("harness_steps") or {}).get("echo") or {}
    text = str(step.get("text") or "")
    if not text.strip():
        return False, "completed with no step text (must never happen: empty is a step failure)"
    if MARKER not in text.lower():
        return False, f"step text lacks the marker {MARKER!r}: {text[:200]!r}"
    return True, f"step text: {text.strip()[:200]!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8124", help="orion-durable-runs base URL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout-sec", type=float, default=600.0, help="the step's turn budget")
    parser.add_argument("--wait-sec", type=float, default=7200.0, help="how long to poll before giving up")
    parser.add_argument("--print", dest="print_only", action="store_true", help="print the request and exit")
    args = parser.parse_args(argv)

    request = build_request(prompt=args.prompt, timeout_sec=args.timeout_sec)
    if args.print_only:
        print(json.dumps(request, indent=2))
        return 0
    base = args.url.rstrip("/")
    try:
        receipt = _http("POST", f"{base}/runs", request)
    except urllib.error.HTTPError as exc:
        print(f"submit refused: HTTP {exc.code} {exc.read().decode()[:300]}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError) as exc:
        print(f"submit failed: cannot reach {base}: {exc}", file=sys.stderr)
        return 2
    run_id = request["run_id"]
    print(f"submitted {run_id}: {receipt}")
    deadline = time.monotonic() + args.wait_sec
    last = None
    status: dict = {}
    while time.monotonic() < deadline:
        try:
            status = _http("GET", f"{base}/runs/{run_id}")
        except (urllib.error.URLError, OSError) as exc:
            # durable-runs restarting mid-smoke is not a smoke failure: the run resumes; keep polling.
            print(f"  poll error (will retry): {exc}")
            time.sleep(10)
            continue
        now_status = (status.get("status"), tuple(status.get("next") or ()), status.get("attempt"))
        if now_status != last:
            print(f"  status={now_status[0]} next={list(now_status[1])} attempt={now_status[2]}"
                  + (f" error={status.get('error')}" if status.get("error") else ""))
            last = now_status
        if status.get("status") in TERMINAL:
            break
        time.sleep(10)
    else:
        print(f"gave up polling after {args.wait_sec:.0f}s; the run continues: GET {base}/runs/{run_id}")
        return 3
    ok, reason = verdict(status)
    corr = ((status.get("harness_steps") or {}).get("echo") or {}).get("turn_correlation_id") \
        or (status.get("harness_turn_meta") or {}).get("turn_correlation_id")
    print(("PASS " if ok else "FAIL ") + reason)
    if corr:
        print(f"turn correlation id: {corr}")
        print("find the turn: docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "
              f"\"select seq, stage, status from cockpit_turn_sighting where correlation_id='{corr}' order by seq\"")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
