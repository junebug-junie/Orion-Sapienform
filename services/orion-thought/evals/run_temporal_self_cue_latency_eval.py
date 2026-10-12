"""Temporal Self patch 4 latency eval: what the "my day" cue adds to a unified turn.

Acceptance check 5 asks that turn p50 stay within +5%. Before deploy that cannot be measured end to
end, so this eval measures each added cost on the live system and bounds the turn-level effect:

1. ``fetch_render_ms``: N real ``GET /temporal-self/frame`` fetches from orion-durable-runs plus
   ``project_stance_cue`` on each body (the exact work orion-thought adds per turn).
2. ``critical_path_ms``: the fetch runs concurrently with Mind enrichment, so a turn only waits for
   ``max(0, fetch_render - mind_ms)``. ``mind_ms`` comes from orion-thought's own
   ``stance_prepare_overlap`` log lines for recent real turns (``docker logs``).
3. ``prompt_tokens_added``: the cue block's size in the stance prompt (chars / 3.5, the repo's
   estimate), reported, not converted to time: prefill cost depends on the serving card.

Pass: p50 critical-path wait <= 5% of p50 ``mind_ms``. ``mind_ms`` is a LOWER bound on turn time
(the stance LLM call and the harness come after it), so the real ratio is smaller. The end-to-end
check (50 Juniper turns before vs after deploy) stays a post-deploy step: UNVERIFIED until then.

    python services/orion-thought/evals/run_temporal_self_cue_latency_eval.py \\
        [--url http://127.0.0.1:8124] [--n 50] [--container orion-athena-thought]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from jinja2 import Environment  # noqa: E402

from orion.temporal_self.stance_cue import project_stance_cue  # noqa: E402

PROMPT = REPO / "orion" / "cognition" / "prompts" / "stance_react.j2"
MIND_MS_RE = re.compile(r"stance_prepare_overlap corr=\S+ side=orion-thought mind_ms=([0-9.]+)")
STANCE_TOKENS_RE = re.compile(r"verb=stance_react .*?prompt_tokens=(\d+) completion_tokens=(\d+)")
CHARS_PER_TOKEN = 3.5
MAX_RATIO = 0.05


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def measure_fetch_render(url: str, n: int, timeout: float) -> tuple[list[float], str]:
    samples: list[float] = []
    text = ""
    with httpx.Client(base_url=url.rstrip("/"), timeout=timeout) as client:
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                resp = client.get("/temporal-self/frame")
                body = resp.json() if resp.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                body = None
            cue = project_stance_cue(body, datetime.now(timezone.utc))
            samples.append((time.perf_counter() - t0) * 1000.0)
            text = cue.text or text
    return samples, text


def _logs(container: str, since: str) -> str:
    try:
        out = subprocess.run(["docker", "logs", "--since", since, container], capture_output=True,
                             text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout + out.stderr


def recent_mind_ms(container: str, since: str) -> list[float]:
    return [float(m.group(1)) for m in MIND_MS_RE.finditer(_logs(container, since))]


def recent_stance_tokens(containers: list[str], since: str) -> list[tuple[int, int]]:
    """(prompt_tokens, completion_tokens) of real stance_react calls (cortex-exec llm_chat_result)."""
    return [(int(m.group(1)), int(m.group(2))) for c in containers for m in STANCE_TOKENS_RE.finditer(_logs(c, since))]


def prompt_delta(cue_text: str) -> int:
    template = Environment().from_string(PROMPT.read_text())
    base = {"user_message": "hi", "stance_inputs": {"user_message": "hi"}, "association": {},
            "repair_bundle": None, "coalition_projection": None}
    return len(template.render(**base, temporal_self_cue=cue_text)) - len(template.render(**base))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8124")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=0.5)
    ap.add_argument("--container", default="orion-athena-thought")
    ap.add_argument("--since", default="168h")
    ap.add_argument("--exec-containers", default="orion-athena-cortex-exec,orion-athena-cortex-exec-chat")
    args = ap.parse_args()

    fetch, cue_text = measure_fetch_render(args.url, args.n, args.timeout)
    mind = recent_mind_ms(args.container, args.since)
    report: dict = {
        "fetch_render_ms": {"n": len(fetch), "p50": round(_pct(fetch, 0.5), 2), "p95": round(_pct(fetch, 0.95), 2),
                            "max": round(max(fetch), 2)},
        "cue_chars": len(cue_text),
        "cue_rendered": bool(cue_text),
    }
    delta = prompt_delta(cue_text or "x" * 400)
    report["prompt_chars_added"] = delta
    report["prompt_tokens_added"] = round(delta / CHARS_PER_TOKEN)
    tokens = recent_stance_tokens([c for c in args.exec_containers.split(",") if c], args.since)
    if tokens:
        prompt_p50 = statistics.median(t[0] for t in tokens)
        report["stance_tokens"] = {"n": len(tokens), "prompt_p50": prompt_p50,
                                   "completion_p50": statistics.median(t[1] for t in tokens)}
        # Prefill share only; decode (completion) dominates stance time and is unchanged by the cue.
        report["prompt_token_growth_ratio"] = round(report["prompt_tokens_added"] / prompt_p50, 4)
    if not mind:
        report["verdict"] = "UNVERIFIED: no stance_prepare_overlap lines in orion-thought logs to compare against"
        print(json.dumps(report, indent=2))
        return 2
    fetch_p50 = _pct(fetch, 0.5)
    waits = [max(0.0, fetch_p50 - m) for m in mind]
    mind_p50 = statistics.median(mind)
    ratio = statistics.median(waits) / mind_p50 if mind_p50 else 1.0
    report["mind_ms"] = {"n": len(mind), "p50": round(mind_p50, 1), "min": round(min(mind), 1)}
    report["critical_path_ms_p50"] = round(statistics.median(waits), 2)
    report["critical_path_ratio_vs_mind_p50"] = round(ratio, 5)
    # Worst case: Mind disabled or faster than the fetch, the whole fetch is on the path.
    report["worst_case_ratio_if_mind_off"] = round(fetch_p50 / mind_p50, 5)
    report["passed"] = ratio <= MAX_RATIO
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
