#!/usr/bin/env python3
"""Live smoke for the shared Qwen3.5 GPU3 (fast) worker, including both thinking modes.

Leased (leased-evals design, docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md).
This used to post straight to the worker through an SSH tunnel, so the GPU pool never saw it. Now:

1. Single checks: ONE operator hold on the lane (class ``operator_fast``, background priority, behind
   all of Orion's own work), every model call through orion-llm-gateway under that hold
   (orion/evals/leased_llm.py). /slots is read from the URL the pool granted (it runs no model).
2. Parallel isolation: ten calls at once, each its own gateway lease on ``operator_fast``
   (``leased_llm.gateway_call``). One hold runs its calls one at a time in its single slot, so it
   cannot prove overlap. /slots is sampled meanwhile to see how many slots really ran together.

When the pool takes the seat back, the eval stops and exits 3. No direct-URL mode: the lane door
refuses an unkeyed model call.

    ORION_BUS_URL=redis://100.92.216.81:6379/0 \\
      python3 services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py

Exit: 0 pass, 1 a check failed, 2 no seat, 3 recalled, the pool did not run the parallel calls, or
they never overlapped (inconclusive).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
HOLDER = "operator:qwen35-fast-eval"
CODES = ("AX17", "BY29", "CZ41", "DW53", "GH71", "JK83", "LM95", "NP16", "QR28", "ST40")


def _body(messages: list[dict], *, thinking: bool | None = False, max_tokens: int = 512,
          **extra: object) -> dict:
    body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, **extra}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    return body


def read_slots(worker_url: str) -> list:
    """/slots runs no model; the lane door lets it through without a key."""
    with urllib.request.urlopen(f"{worker_url}/slots", timeout=30) as response:
        return json.load(response)


def run(chat: Callable[[dict], dict], slots: list) -> None:
    assert isinstance(slots, list) and len(slots) == 10, f"expected 10 slots, got {slots!r:.200}"
    assert [slot.get("n_ctx") for slot in slots] == [32768] * 10

    default = chat(_body([{"role": "user", "content": "Reply only DEFAULT."}], thinking=None, max_tokens=64))
    default_message = default["choices"][0]["message"]
    assert str(default_message.get("content") or "").strip() == "DEFAULT", default_message
    assert not default_message.get("reasoning_content"), default_message

    thinking = chat(_body([{"role": "user", "content": "Your assigned code is EF67. Repeat only EF67."}],
                          thinking=True, max_tokens=512))
    thought_message = thinking["choices"][0]["message"]
    assert "EF67" in str(thought_message.get("content") or "")
    assert thought_message.get("reasoning_content")
    assert thinking["choices"][0]["finish_reason"] == "stop"

    structured = chat(_body([{"role": "user", "content": "Return STRICT JSON only, without Markdown fences: "
                                                         '{"status":"ok","count":3}'}],
                            temperature=0.0, response_format={"type": "json_object"}))
    obj = json.loads(structured["choices"][0]["message"]["content"])
    assert obj.get("status") == "ok" and obj.get("count") == 3, obj

    # Topic Foundry parses a labels object from a system/user pair and may
    # request no explicit max_tokens. This shape matters more than generic JSON.
    labels = chat({"messages": [
        {"role": "system", "content": "You are an analyst. Return STRICT JSON only."},
        {"role": "user", "content": "Label Topic 17 from keywords: home, garden, plants. "
         'Return STRICT JSON only: {"labels":{"17":"<short label>"}}. No commentary.'},
    ], "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.2,
        "response_format": {"type": "json_object"}})
    label_obj = json.loads(labels["choices"][0]["message"]["content"])
    assert isinstance(label_obj.get("labels", {}).get("17"), str), label_obj

    probe = chat(_body([{"role": "user", "content": "Answer yes."}], max_tokens=32, logprobs=True, top_logprobs=5))
    assert probe["choices"][0].get("logprobs"), probe

    tool = chat(_body([{"role": "user", "content": "Call add with a=2 and b=3."}],
                      tools=[{"type": "function", "function": {"name": "add",
                              "description": "Add integers", "parameters": {"type": "object",
                              "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                              "required": ["a", "b"]}}}], tool_choice="required"))
    calls = tool["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "add", calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}
    # A single prompt over the old 4k limit must stay on this worker.
    long_prompt = "word " * 24000 + "\nYour code is UV52. Reply only UV52."
    long_result = chat(_body([{"role": "user", "content": long_prompt}], max_tokens=64))
    assert "UV52" in str(long_result["choices"][0]["message"].get("content") or "")
    print("PASS: 10 x 32768 slots, long context, thinking on/off, strict JSON, topic labels, logprobs, tools")


def _busy(slots: Callable[[], list]) -> int | None:
    try:
        return sum(slot.get("is_processing") is True for slot in slots())
    except OSError:
        return None   # a missed sample lowers the peak; it never raises it


def parallel_load(call: Callable[[dict], dict], slots: Callable[[], list], baseline_sec: float = 1.0) -> int:
    """Ten calls at once, each its own lease. Every reply must carry only its own code. Returns how many
    MORE slots were seen working at once than before the calls went out (the fast worker is shared:
    Orion's own calls also show in /slots). Under 2 means the eval's calls never overlapped."""
    baseline, until = 0, time.monotonic() + baseline_sec
    while time.monotonic() < until:
        baseline = max(baseline, _busy(slots) or 0)
        time.sleep(0.05)
    barrier = Barrier(11)

    def one(index: int) -> tuple[float, str, bool]:
        code = CODES[index]
        # Distinct, long prompts prevent prefix-cache reuse and hold each slot
        # long enough for /slots sampling to observe real overlap.
        prompt = f"Context for {code}: " + (f"orchard {code} " * 4000)
        prompt += f"\nYour assigned code is {code}. Reply only your assigned code."
        barrier.wait()
        started = time.monotonic()
        result = call(_body([{"role": "user", "content": prompt}], thinking=None, max_tokens=96, temperature=0.0))
        message = result["choices"][0]["message"]
        return time.monotonic() - started, str(message.get("content") or ""), bool(message.get("reasoning_content"))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(one, index) for index in range(10)]
        barrier.wait()
        peak = 0
        while not all(f.done() for f in futures):
            peak = max(peak, _busy(slots) or 0)
            time.sleep(0.05)
        results = [f.result() for f in futures]
    assert all(code in text and not reasoned for code, (_, text, reasoned) in zip(CODES, results)), results
    durations = sorted(row[0] for row in results)
    print(f"parallel isolation: 10 replies, each its own code; peak_slots={peak} (baseline {baseline}), "
          f"median_s={statistics.median(durations):.1f}, p95_s={durations[9]:.1f}")
    return peak - baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--lane", default="fast", help="operator lane (leased_llm.OPERATOR_LANES)")
    parser.add_argument("--skip-parallel", action="store_true", help="single checks only")
    args = parser.parse_args(argv)

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from orion.evals.leased_llm import (_HOLD_GONE_MARKERS, EXIT_RECALLED, LaneCallError, LaneRecalled,
                                        LaneUnavailable, gateway_call, operator_lane_sync)

    try:
        with operator_lane_sync(args.lane, holder=HOLDER) as lane:
            worker_url = lane.worker_url
            run(lane.chat, lane.worker_read("GET", "/slots").json())
            print(f"lease: {json.dumps(lane.summary())}")
    except LaneRecalled as exc:
        print(f"RECALLED: {exc}", file=sys.stderr)
        return EXIT_RECALLED
    except LaneUnavailable as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (AssertionError, LaneCallError) as exc:
        print(f"FAIL: {exc!r}", file=sys.stderr)
        return 1
    if args.skip_parallel:
        return 0

    class NoSeat(RuntimeError):
        """The pool did not place a one-off call (lane busy) or took it back: not the worker's fault."""

    def call(body: dict) -> dict:
        r = gateway_call(args.lane, body, holder=HOLDER)
        if r.status_code != 200:
            if "gpu_pool_unavailable" in r.text or any(m in r.text for m in _HOLD_GONE_MARKERS):
                raise NoSeat(f"http {r.status_code} {r.text[:200]}")
            raise LaneCallError(f"http {r.status_code} {r.text[:200]}", r.status_code)
        return r.json()

    try:
        peak = parallel_load(call, lambda: read_slots(worker_url))
    except NoSeat as exc:
        print(f"INCONCLUSIVE: the pool did not run the parallel calls: {exc}", file=sys.stderr)
        return 3
    except (AssertionError, LaneCallError) as exc:
        print(f"FAIL: parallel isolation: {exc!r}", file=sys.stderr)
        return 1
    if peak < 2:
        print("INCONCLUSIVE: the ten calls never ran side by side (the pool queued them); "
              "isolation was not tested under overlap", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
