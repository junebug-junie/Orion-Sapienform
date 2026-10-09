"""Live eval: does the current-turn probe notice what Juniper shares, and only that?

corr beab81a3: "I'll be pretty busy the next few days with work travel" came back
from the probe as [] -- nothing for Orion to be curious about. The fix changed the
probe's prompt/shape (items with a natural follow-up question, plus a
wants_direct_answer judgement that replaced the policy's regexes). This eval runs
the real prompt builder and the real parser against a real model and reports:

- disclosure recall: share-turns that yield >=1 item with a follow-up question
- false alarms: control turns (tasks, status questions, filler) that yield any item
- direct-answer accuracy on turns with an unambiguous label
- parse failures, latency p50/p95 (compare against CURRENT_TURN_SIGNAL_PROBE_TIMEOUT_SEC)

Messages are split into the set used while designing the prompt ("tuning") and a
held-out set written afterwards with no shared vocabulary ("heldout"); both are
reported separately so a prompt overfit to the tuning set shows up.

Calls the llama.cpp OpenAI endpoint directly with the same body the gateway
sends (thinking off, temperature/max_tokens from settings). Read-only.

Run:
  python services/orion-cortex-exec/evals/run_current_turn_disclosure_live_eval.py \
      --url http://<llm-node>:8013/v1/chat/completions [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

_SERVICE_DIR = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
for _path in (_SERVICE_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("SERVICE_NAME", "cortex-exec")
os.environ.setdefault("SERVICE_VERSION", "0.2.0")
os.environ.setdefault("NODE_NAME", "athena")
os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("ORION_BUS_ENFORCE_CATALOG", "false")

from app.current_turn_llm_signals import (  # noqa: E402
    build_current_turn_llm_prompt,
    parse_current_turn_llm_read,
)
from app.settings import settings  # noqa: E402

# (set, kind, message, expected wants_direct_answer or None when ambiguous)
# kind: "share" -> should yield an item with a question; "control" -> no items;
# "mixed" -> a request AND something shared: items expected, direct True.
_CASES: list[tuple[str, str, str, bool | None]] = [
    ("tuning", "share", "so far so good. \n\nI'll be pretty busy the next few days with work travel, so won't have much time to do dev on you.", False),
    ("tuning", "share", "my sister's coming to stay this weekend", False),
    ("tuning", "share", "finally signed up for that pottery class", False),
    ("tuning", "share", "rough day. my manager and I got into it again", False),
    ("tuning", "share", "just got back from the dentist, face is still numb lol", False),
    ("tuning", "share", "thinking about repainting the living room", False),
    ("tuning", "control", "ok thanks", None),
    ("tuning", "control", "run the reading queue check again", True),
    ("tuning", "control", "what's the status of the gpu pool?", True),
    ("tuning", "control", "heck yeah!", None),
    ("heldout", "share", "I have a job interview on thursday, kind of nervous", False),
    ("heldout", "share", "we adopted a dog yesterday!", False),
    ("heldout", "share", "been reading a lot of octavia butler lately", False),
    ("heldout", "share", "my mom's surgery went fine, she's home now", False),
    ("heldout", "share", "gonna try running a half marathon in the spring", False),
    ("heldout", "share", "didn't sleep much, the neighbors had a party", False),
    ("heldout", "mixed", "heading to a conference in denver next week, can you summarize the open PRs before I go?", True),
    ("heldout", "control", "merge it", True),
    ("heldout", "control", "why is the harness governor timing out?", True),
    ("heldout", "control", "cool cool", None),
    ("heldout", "control", "restart cortex-exec please", True),
    ("heldout", "control", "what do you think about recursion?", True),
]

_RECALL_FLOOR = 0.8
_FALSE_ALARM_CEILING = 0.2
_DIRECT_ACCURACY_FLOOR = 0.8


def _call(url: str, prompt: str) -> tuple[str, float]:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": settings.current_turn_signal_probe_max_tokens,
        "temperature": settings.current_turn_signal_probe_temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    elapsed = time.monotonic() - started
    return (data["choices"][0]["message"].get("content") or "").strip(), elapsed


def run(url: str, repeats: int) -> int:
    print(f"\n=== current-turn disclosure live eval  url={url}  repeats={repeats}")
    print(
        f"    temperature={settings.current_turn_signal_probe_temperature} "
        f"max_tokens={settings.current_turn_signal_probe_max_tokens} "
        f"timeout_budget={settings.current_turn_signal_probe_timeout_sec}s"
    )
    stats: dict[str, dict[str, int]] = {}
    latencies: list[float] = []
    parse_failures = 0
    direct_total = direct_correct = 0
    for _ in range(repeats):
        for case_set, kind, message, expected_direct in _CASES:
            s = stats.setdefault(case_set, {"share": 0, "share_hit": 0, "control": 0, "false_alarm": 0})
            try:
                raw, elapsed = _call(url, build_current_turn_llm_prompt(message))
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {message[:50]!r}: {exc}")
                parse_failures += 1
                continue
            latencies.append(elapsed)
            read = parse_current_turn_llm_read(raw)
            if read is None:
                parse_failures += 1
                print(f"  [PARSE_FAIL] {message[:50]!r} -> {raw[:160]!r}")
                continue
            items = read["signals"]
            with_question = [i for i in items if i.get("natural_question")]
            if kind in ("share", "mixed"):
                s["share"] += 1
                hit = bool(with_question)
                s["share_hit"] += int(hit)
                verdict = "HIT " if hit else "MISS"
            else:
                s["control"] += 1
                alarm = bool(items)
                s["false_alarm"] += int(alarm)
                verdict = "FA  " if alarm else "OK  "
            if expected_direct is not None:
                direct_total += 1
                direct_correct += int(read["wants_direct_answer"] is expected_direct)
            shown = [(i["phrase"], i.get("natural_question")) for i in items]
            print(
                f"  [{verdict}] {case_set:7s} {message[:55]!r:60s} direct={read['wants_direct_answer']!s:5s} "
                f"{elapsed:4.2f}s {shown}"
            )

    failed = False
    print()
    for case_set, s in stats.items():
        recall = s["share_hit"] / s["share"] if s["share"] else 0.0
        fa = s["false_alarm"] / s["control"] if s["control"] else 0.0
        print(f"  {case_set:7s} disclosure recall {s['share_hit']}/{s['share']} = {recall:.2f}   false alarms {s['false_alarm']}/{s['control']} = {fa:.2f}")
        failed |= recall < _RECALL_FLOOR or fa > _FALSE_ALARM_CEILING
    direct_acc = direct_correct / direct_total if direct_total else 0.0
    print(f"  direct-answer accuracy {direct_correct}/{direct_total} = {direct_acc:.2f}")
    print(f"  parse failures {parse_failures}")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        print(f"  latency p50 {statistics.median(ordered):.2f}s  p95 {p95:.2f}s  (budget {settings.current_turn_signal_probe_timeout_sec}s)")
    failed |= direct_acc < _DIRECT_ACCURACY_FLOOR or parse_failures > 0
    print(f"\nRESULT: {'FAIL' if failed else 'PASS'} (recall >= {_RECALL_FLOOR}, false alarms <= {_FALSE_ALARM_CEILING}, direct accuracy >= {_DIRECT_ACCURACY_FLOOR}, 0 parse failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--url", required=True, help="llama.cpp OpenAI chat/completions URL for the probe's lane")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    sys.exit(run(args.url, args.repeats))
