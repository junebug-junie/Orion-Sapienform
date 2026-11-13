#!/usr/bin/env python3
"""Direct smoke for the shared Qwen3.5 GPU3 worker, including both thinking modes.

Run against an isolated or newly deployed worker:
    python3 services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py --base-url http://127.0.0.1:18013
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from urllib.request import Request, urlopen


def get_json(url: str) -> dict | list:
    with urlopen(url, timeout=120) as response:
        return json.load(response)


def chat(base: str, *, messages: list[dict], thinking: bool | None = False,
         max_tokens: int = 512, model: str = "Qwen3.5-9B-UD-Q5_K_XL.gguf", **extra: object) -> dict:
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, **extra}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    request = Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def parallel_load(worker: str, *, gateway: str | None = None) -> None:
    """Prove ten requests overlap in the worker; optionally exercise both gateway classes."""
    barrier = Barrier(11)
    codes = ("AX17", "BY29", "CZ41", "DW53", "GH71", "JK83", "LM95", "NP16", "QR28", "ST40")

    def call(index: int) -> tuple[float, str, bool]:
        code = codes[index]
        # Distinct, long prompts prevent prefix-cache reuse and hold each slot
        # long enough for /slots sampling to observe real overlap.
        prompt = f"Context for {code}: " + (f"orchard {code} " * 4000)
        prompt += f"\nYour assigned code is {code}. Reply only your assigned code."
        route = ("quick" if index % 2 == 0 else "metacog_perception") if gateway else "Qwen3.5-9B-UD-Q5_K_XL.gguf"
        barrier.wait()
        started = time.monotonic()
        result = chat(gateway or worker, model=route, thinking=None,
                      messages=[{"role": "user", "content": prompt}], max_tokens=96, temperature=0.0)
        message = result["choices"][0]["message"]
        return time.monotonic() - started, str(message.get("content") or ""), bool(message.get("reasoning_content"))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(call, index) for index in range(10)]
        barrier.wait()
        peak = 0
        while not all(f.done() for f in futures):
            slots = get_json(f"{worker}/slots")
            peak = max(peak, sum(slot.get("is_processing") is True for slot in slots))
            time.sleep(0.05)
        results = [f.result() for f in futures]
    assert peak == 10, f"worker never showed ten occupied slots (peak={peak})"
    assert all(code in text and not reasoned for code, (_, text, reasoned) in zip(codes, results)), results
    durations = sorted(row[0] for row in results)
    label = "mixed gateway routes" if gateway else "direct worker"
    print(f"PASS: {label}, peak_slots={peak}, median_s={statistics.median(durations):.1f}, "
          f"p95_s={durations[9]:.1f}")


def run(base: str) -> None:
    slots = get_json(f"{base}/slots")
    assert isinstance(slots, list) and len(slots) == 10
    assert [slot.get("n_ctx") for slot in slots] == [32768] * 10

    default = chat(base, messages=[{"role": "user", "content": "Reply only DEFAULT."}],
                   thinking=None, max_tokens=64)
    default_message = default["choices"][0]["message"]
    assert str(default_message.get("content") or "").strip() == "DEFAULT", default_message
    assert not default_message.get("reasoning_content"), default_message

    parallel_load(base)

    thinking = chat(base, messages=[{"role": "user", "content":
                                     "Your assigned code is EF67. Repeat only EF67."}],
                    thinking=True, max_tokens=512)
    thought_message = thinking["choices"][0]["message"]
    assert "EF67" in str(thought_message.get("content") or "")
    assert thought_message.get("reasoning_content")
    assert thinking["choices"][0]["finish_reason"] == "stop"

    structured = chat(base, messages=[{"role": "user", "content":
                                       "Return STRICT JSON only, without Markdown fences: "
                                       '{"status":"ok","count":3}'}],
                      temperature=0.0, response_format={"type": "json_object"})
    obj = json.loads(structured["choices"][0]["message"]["content"])
    assert obj.get("status") == "ok" and obj.get("count") == 3, obj

    # Topic Foundry parses a labels object from a system/user pair and may
    # request no explicit max_tokens. This shape matters more than generic JSON.
    labels = chat(base, messages=[
        {"role": "system", "content": "You are an analyst. Return STRICT JSON only."},
        {"role": "user", "content": "Label Topic 17 from keywords: home, garden, plants. "
         'Return STRICT JSON only: {"labels":{"17":"<short label>"}}. No commentary.'},
    ], temperature=0.2, response_format={"type": "json_object"})
    label_obj = json.loads(labels["choices"][0]["message"]["content"])
    assert isinstance(label_obj.get("labels", {}).get("17"), str), label_obj

    probe = chat(base, messages=[{"role": "user", "content": "Answer yes."}],
                 max_tokens=32, logprobs=True, top_logprobs=5)
    assert probe["choices"][0].get("logprobs"), probe

    tool = chat(base, messages=[{"role": "user", "content": "Call add with a=2 and b=3."}],
                tools=[{"type": "function", "function": {"name": "add",
                        "description": "Add integers", "parameters": {"type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"]}}}], tool_choice="required")
    calls = tool["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "add", calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}
    # A single prompt over the old 4k limit must stay on this worker.
    long_prompt = "word " * 24000 + "\nYour code is UV52. Reply only UV52."
    long_result = chat(base, messages=[{"role": "user", "content": long_prompt}], max_tokens=64)
    assert "UV52" in str(long_result["choices"][0]["message"].get("content") or "")
    print("PASS: 10 x 32768 slots, long context, parallel isolation, thinking on/off, strict JSON, topic labels, logprobs, tools")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--gateway-url", help="also load quick and metacog_perception through LLM Gateway")
    args = parser.parse_args()
    run(args.base_url.rstrip("/"))
    if args.gateway_url:
        parallel_load(args.base_url.rstrip("/"), gateway=args.gateway_url.rstrip("/"))


if __name__ == "__main__":
    main()
