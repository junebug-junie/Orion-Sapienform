#!/usr/bin/env python3
"""Exact-token, leased long-context retrieval/reasoning benchmark (no direct inference).

Run with a local copy of the pinned tokenizer. Each capacity label records the actual
prompt and output reserve separately. A configuration value is never a passing result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid


def build_messages(tokenizer, prompt_tokens: int, seed: int):
    rng = random.Random(seed)
    keys = [f"{rng.getrandbits(48):012x}" for _ in range(3)]
    a, b, c = rng.sample(range(100, 900), 3)
    expected = {"early": keys[0], "middle": keys[1], "late": keys[2], "total": a + b - c}
    facts = [f"Early record: early={keys[0]}. Initial inventory={a}.",
             f"Middle record: middle={keys[1]}. Add {b} items to inventory.",
             f"Late record: late={keys[2]}. Remove {c} items from inventory."]
    # Fresh prefix defeats previous trials' prefix hits. Replay uses identical messages.
    intro = f"Trial {seed}. Read this ledger, retaining the three records.\n"
    question = '\nReturn only JSON with keys early, middle, late and total (final inventory).'
    def make(n):
        parts = [intro]
        for i, fact in enumerate(facts):
            parts += [" cedar" * (n // 4), "\n" + fact + "\n"]
        parts += [" cedar" * (n - 3 * (n // 4)), question]
        return [{"role": "user", "content": "".join(parts)}]
    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=True, return_dict=False))
    n = max(0, prompt_tokens - count(make(0)))
    for _ in range(12):
        messages = make(n)
        actual = count(messages)
        if actual == prompt_tokens:
            return messages, expected
        n += prompt_tokens - actual
        if n < 0:
            break
    raise ValueError(f"could not construct exact {prompt_tokens}-token prompt; got {actual}")


def judge(result, *, expected, model, prompt_tokens, output_budget):
    failures = []
    if result.get("model") != model:
        failures.append("wrong_model")
    usage = result.get("usage") or {}
    if usage.get("prompt_tokens") != prompt_tokens:
        failures.append("prompt_token_mismatch_or_missing")
    if not isinstance(usage.get("completion_tokens"), int) or not 0 < usage["completion_tokens"] <= output_budget:
        failures.append("completion_usage_invalid")
    if result.get("finish_reason") != "stop":
        failures.append("incomplete_generation")
    try:
        text = result.get("content") or ""
        if json.loads(text) != expected:
            failures.append("reasoning_or_retrieval_wrong")
    except (ValueError, TypeError):
        failures.append("empty_or_invalid_final")
    return failures


def stream_chat(lane, body, timeout_sec):
    import httpx
    from orion.evals.leased_llm import LaneCallError
    lane.check()
    started_at = time.time()
    start = time.monotonic()
    first = last = None
    result = {"content": "", "reasoning": "", "usage": None, "finish_reason": None}
    with httpx.Client() as client:
        with client.stream("POST", lane.gateway + "/v1/chat/completions",
                           headers={**lane.headers(), "X-Orion-Correlation-Id": str(uuid.uuid4())},
                           json={**body, "model": lane.route, "stream": True,
                                 "stream_options": {"include_usage": True}}, timeout=timeout_sec) as response:
            if response.status_code != 200:
                response.read()
                lane.check()
                raise LaneCallError(f"HTTP {response.status_code}: {response.text[:400]}", response.status_code)
            for line in response.iter_lines():
                lane.check()
                now = time.monotonic()
                if now - start > timeout_sec:
                    raise TimeoutError("whole-request deadline exceeded")
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise LaneCallError(str(chunk["error"]))
                result["model"] = chunk.get("model", result.get("model"))
                if chunk.get("usage"):
                    result["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if content or reasoning:
                        first = first if first is not None else now
                        last = now
                    result["content"] += content
                    result["reasoning"] += reasoning
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
    lane.calls += 1
    result["started_at_epoch_s"] = started_at
    result["finished_at_epoch_s"] = time.time()
    result["elapsed_s"] = time.monotonic() - start
    result["ttft_s"] = first - start if first is not None else None
    usage = result.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    result["cached_tokens"] = cached
    # Proxy, includes scheduling/transport: never call this pure prefill throughput.
    result["uncached_prompt_per_ttft"] = (
        (usage["prompt_tokens"] - cached) / result["ttft_s"]
        if cached is not None and result["ttft_s"] and "prompt_tokens" in usage else None)
    result["stream_decode_tokens_per_s"] = (
        (usage["completion_tokens"] - 1) / (last - first)
        if first is not None and last > first and usage.get("completion_tokens", 0) > 1 else None)
    return result


def main():
    from transformers import AutoTokenizer
    from orion.evals.leased_llm import operator_lane_sync
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="local pinned tokenizer directory")
    parser.add_argument("--model", required=True, help="exact expected served model, not route")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--output-budget", type=int, choices=(16384, 32768), default=32768)
    parser.add_argument("--prompts", type=int, nargs="+", default=None)
    parser.add_argument("--endurance", action="store_true", help="also fill the entire context with forced output, then test recovery")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--queue-wait", type=float, default=900)
    args = parser.parse_args()
    prompts = args.prompts or [8192, 32768, 65536, 132000, 196608, 262144 - args.output_budget]
    if args.repeats < 1 or any(n < 1024 or n + args.output_budget > 262144 for n in prompts):
        parser.error("invalid repeats or prompt plus output reserve exceeds 262144")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    passed = True
    # Exclusive output file prevents accidentally appending unlike runs.
    with args.out.open("x") as output:
        try:
            with operator_lane_sync("agent-deep", holder="operator:hecate-deep-context",
                                    gateway=args.gateway, queue_wait_sec=args.queue_wait) as lane:
                for n in prompts:
                    for repeat in range(args.repeats):
                        seed = args.seed + n + repeat
                        messages, expected = build_messages(tokenizer, n, seed)
                        for mode in ("fresh_prefix", "exact_replay"):
                            row = {"prompt_tokens_requested": n, "output_budget": args.output_budget,
                                   "total_budget": n + args.output_budget, "mode": mode, "seed": seed,
                                   "repeat": repeat, "expected": expected, "seat": lane.summary(),
                                   "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest()}
                            try:
                                row.update(stream_chat(lane, {"messages": messages, "max_tokens": args.output_budget,
                                    "temperature": 0, "seed": seed, "chat_template_kwargs": {"enable_thinking": True}}, args.timeout))
                                row["failures"] = judge(row, expected=expected, model=args.model,
                                                         prompt_tokens=n, output_budget=args.output_budget)
                            except Exception as exc:
                                row.update(failures=[type(exc).__name__], error=str(exc))
                                output.write(json.dumps(row) + "\n"); output.flush()
                                raise  # includes recall: no automatic retry or resume
                            passed &= not row["failures"]
                            output.write(json.dumps(row) + "\n"); output.flush()
                            print(json.dumps({k: row.get(k) for k in ("prompt_tokens_requested", "mode", "ttft_s", "failures")}), flush=True)
                if args.endurance:
                    n = 262144 - args.output_budget
                    messages, _ = build_messages(tokenizer, n, args.seed + 999999)
                    row = {"mode": "capacity_endurance", "quality_claim": False,
                           "prompt_tokens_requested": n, "output_budget": args.output_budget,
                           "seat": lane.summary()}
                    row.update(stream_chat(lane, {"messages": messages, "max_tokens": args.output_budget,
                        "min_tokens": args.output_budget, "ignore_eos": True, "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": True}}, args.timeout))
                    usage = row.get("usage") or {}
                    row["passed"] = (row.get("model") == args.model and usage.get("prompt_tokens") == n
                        and usage.get("completion_tokens") == args.output_budget
                        and row.get("finish_reason") == "length" and bool(row.get("content") or row.get("reasoning")))
                    passed &= row["passed"]
                    output.write(json.dumps(row) + "\n"); output.flush()
                    messages, expected = build_messages(tokenizer, 8192, args.seed + 999998)
                    recovery = stream_chat(lane, {"messages": messages, "max_tokens": args.output_budget,
                        "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}, args.timeout)
                    recovery.update(mode="post_endurance_recovery", failures=judge(recovery, expected=expected,
                        model=args.model, prompt_tokens=8192, output_budget=args.output_budget))
                    passed &= not recovery["failures"]
                    output.write(json.dumps(recovery) + "\n"); output.flush()
        except Exception as exc:
            output.write(json.dumps({"status": "INCOMPLETE", "error": str(exc)}) + "\n")
            raise
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
