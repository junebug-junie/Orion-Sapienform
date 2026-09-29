"""Opt-in live eval: does a reading question stay on the question?

Sends the 2026-09-28 incident question through Hub's real unified turn
(no_write), then scores the governor's own per-step tool log. Baseline
(corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3): did not finish, 112+ steps,
6 introspect calls then ~25 non-introspect tool calls chasing circe_gpu.

HUNT_TRIPWIRE is a tripwire for reporting, not a gate: raw counts are always
written so a reviewer judges the numbers, not just the verdict.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

QUESTION = "What have you read about graphics cards lately, and what did you actually learn from it?"
HUNT_TRIPWIRE = 10
INTROSPECT_PREFIX = "mcp__orion-introspect__"
DISCOVERY_TOOLS = frozenset({"ToolSearch"})
HUB_CONTAINER = "orion-athena-hub"
GOVERNOR_CONTAINER = "orion-athena-harness-governor"

_STEP_RE = re.compile(r"harness_grammar_step_published corr=(\S+) .*?step=(\d+) tool=(\S+)")


def parse_tool_steps(log_text: str, correlation_id: str) -> list[str]:
    steps: list[tuple[int, str]] = []
    for match in _STEP_RE.finditer(log_text):
        corr, step, tool = match.groups()
        if corr == correlation_id and tool != "none":
            steps.append((int(step), tool))
    return [tool for _, tool in sorted(steps)]


def score_run(tools: list[str], *, finished: bool, reply_text: str) -> dict[str, object]:
    introspect = sum(1 for t in tools if t.startswith(INTROSPECT_PREFIX))
    discovery = sum(1 for t in tools if t in DISCOVERY_TOOLS)
    other = len(tools) - introspect - discovery
    hunt = other > HUNT_TRIPWIRE
    return {
        "introspect_calls": introspect,
        "discovery_calls": discovery,
        "other_tool_calls": other,
        "hunt": hunt,
        "finished": finished,
        "passed": finished and introspect >= 1 and bool(reply_text.strip()) and not hunt,
    }


def _hub_python(script: str) -> str:
    done = subprocess.run(
        ["docker", "exec", "-i", HUB_CONTAINER, "sh", "-c", "cd /app && python3 -"],
        input=script, capture_output=True, text=True, timeout=60, check=True,
    )
    return done.stdout.strip()


def _corr_for_session(session_id: str) -> str | None:
    script = (
        "import os, sqlalchemy as sa\n"
        "e = sa.create_engine(os.environ['DATABASE_URL'])\n"
        "with e.connect() as c:\n"
        "    r = c.execute(sa.text('SELECT correlation_id FROM thought_decision "
        "WHERE session_id = :s ORDER BY created_at DESC LIMIT 1'), "
        f"{{'s': {json.dumps(session_id)}}}).first()\n"
        "print(r[0] if r else '')\n"
    )
    return _hub_python(script) or None


def _cancel(correlation_id: str) -> None:
    script = (
        "import asyncio, os\n"
        "from orion.core.bus.async_service import OrionBusAsync\n"
        "from scripts.harness_governor_client import HarnessGovernorClient\n"
        "async def main():\n"
        "    bus = OrionBusAsync(os.environ['ORION_BUS_URL'])\n"
        "    await bus.connect()\n"
        f"    await HarnessGovernorClient(bus).cancel(correlation_id={json.dumps(correlation_id)}, reason='user_stop')\n"
        "    await bus.close()\n"
        "asyncio.run(main())\n"
    )
    _hub_python(script)


def _governor_log(since_iso: str) -> str:
    done = subprocess.run(
        ["docker", "logs", "--since", since_iso, GOVERNOR_CONTAINER],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return done.stdout + done.stderr


def run_once(hub: str, turn_timeout: float) -> dict[str, object]:
    session_id = f"stance-scope-eval-{uuid.uuid4()}"
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    finished, corr, reply = False, None, ""
    try:
        resp = httpx.post(
            f"{hub}/api/chat",
            headers={"X-Orion-Session-Id": session_id, "Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": QUESTION}],
                "mode": "orion", "no_write": True, "disable_tts": True,
            },
            timeout=turn_timeout,
        )
        body = resp.json()
        corr = body.get("correlation_id")
        reply = str(body.get("llm_response") or "")
        finished = body.get("type") == "final"
    except httpx.TimeoutException:
        corr = _corr_for_session(session_id)
        if corr:
            _cancel(corr)
    elapsed = round(time.monotonic() - t0, 1)
    tools = parse_tool_steps(_governor_log(started.isoformat()), corr) if corr else []
    return {
        "session_id": session_id, "correlation_id": corr, "elapsed_sec": elapsed,
        "tools": tools, "reply_excerpt": reply[:400],
        **score_run(tools, finished=finished, reply_text=reply),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", default="http://127.0.0.1:8080")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--turn-timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/stance-scope-eval/report.json"))
    args = parser.parse_args()
    runs = [run_once(args.hub, args.turn_timeout) for _ in range(args.runs)]
    report = {"question": QUESTION, "hunt_tripwire": HUNT_TRIPWIRE, "runs": runs,
              "passed": all(r["passed"] for r in runs)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in runs:
        print(f"corr={r['correlation_id']} finished={r['finished']} elapsed={r['elapsed_sec']}s "
              f"introspect={r['introspect_calls']} other={r['other_tool_calls']} hunt={r['hunt']} passed={r['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
