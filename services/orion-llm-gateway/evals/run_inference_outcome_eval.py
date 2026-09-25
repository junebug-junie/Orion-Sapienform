#!/usr/bin/env python3
"""What would inference_failure_pressure have read? Replays gateway logs.

Before LLM_GATEWAY_GRAMMAR_ENABLED is flipped there is no grammar to look at,
but the gateway already logs every reply (gateway_llm_route_selected) and every
backend timeout/failure. This rebuilds per-window, per-node outcome counts from
those lines so the channel's rest state and dynamic range can be checked against
real traffic first (CLAUDE.md metric gate item 4), and compared afterwards with
what the live lane actually published.

    docker logs --timestamps orion-llm-gateway 2>&1 | \\
        python services/orion-llm-gateway/evals/run_inference_outcome_eval.py --window-sec 60

Log-derived counts are an approximation of the emitter, not a copy of it: the
emitter classifies the reply dict itself, the logs only show the failure lines
llm_backend.py happens to write. The eval says so in its output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime

_TS = r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
_SELECTED = re.compile(_TS + r".*gateway_llm_route_selected .*?served_by=(?P<served_by>\S+)")
_TIMEOUT = re.compile(_TS + r".*\[LLM-GW\] \S+ TIMEOUT route=\S+ served_by=(?P<served_by>\S+)")
_FAILED = re.compile(_TS + r".*\[LLM-GW\] (llamacpp|vllm|ollama|openai) (error|failed)")
_OVERLOADED = re.compile(_TS + r".*gateway_overloaded .*?stage=(?P<stage>\S+)")


def _node(served_by: str) -> str:
    return (served_by or "").strip().lower().split("-worker")[0] or "unrouted"


def replay(lines, *, window_sec: int) -> dict:
    windows: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unattributed_failures = 0
    for line in lines:
        m = _SELECTED.match(line) or _TIMEOUT.match(line) or _FAILED.match(line) or _OVERLOADED.match(line)
        if not m:
            continue
        ts = int(datetime.fromisoformat(m.group("ts")).timestamp())
        bucket = ts - ts % window_sec
        if m.re is _SELECTED:
            windows[(bucket, _node(m.group("served_by")))]["replied"] += 1
        elif m.re is _TIMEOUT:
            windows[(bucket, _node(m.group("served_by")))]["upstream_failed"] += 1
        elif m.re is _OVERLOADED:
            windows[(bucket, "unrouted")]["refused"] += 1
        else:
            unattributed_failures += 1  # these lines carry no served_by
    per_node: dict[str, list[float]] = defaultdict(list)
    for (_bucket, node), c in windows.items():
        # every reply (failed or not) logs route_selected, so replied already
        # includes the failures
        attempted = c["replied"]
        if node == "unrouted" or attempted <= 0:
            continue
        per_node[node].append(min(1.0, c["upstream_failed"] / attempted))
    summary = {}
    for node, series in sorted(per_node.items()):
        s = sorted(series)
        summary[node] = {
            "windows_with_traffic": len(s),
            "windows_at_zero": sum(1 for v in s if v == 0.0),
            "max": s[-1],
            "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
            "mean": round(sum(s) / len(s), 6),
        }
    return {
        "window_sec": window_sec,
        "windows": len({b for b, _ in windows}),
        "unattributed_backend_failure_lines": unattributed_failures,
        "inference_failure_pressure_by_node": summary,
        "caveat": "log-derived approximation; the emitter classifies reply dicts, not log lines",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window-sec", type=int, default=60)
    args = ap.parse_args()
    print(json.dumps(replay(sys.stdin, window_sec=args.window_sec), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
