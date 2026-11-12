#!/usr/bin/env python3
"""Direct smoke for the Qwen3.5 fast worker, including both thinking modes.

Run against an isolated or newly deployed worker:
    python3 services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py --base-url http://127.0.0.1:18013
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def get_json(url: str) -> dict | list:
    with urlopen(url, timeout=120) as response:
        return json.load(response)


def chat(base: str, *, messages: list[dict], thinking: bool | None = False,
         max_tokens: int = 512, **extra: object) -> dict:
    body = {"model": "Qwen3.5-9B-UD-Q5_K_XL.gguf", "messages": messages,
            "max_tokens": max_tokens, **extra}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    request = Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def answer(base: str, code: str) -> tuple[str, str, bool]:
    result = chat(base, messages=[{"role": "user", "content":
                                   f"Your assigned code is {code}. Repeat only your assigned code."}],
                  max_tokens=64)
    message = result["choices"][0]["message"]
    return code, str(message.get("content") or "").strip(), bool(message.get("reasoning_content"))


def run(base: str) -> None:
    slots = get_json(f"{base}/slots")
    assert isinstance(slots, list) and len(slots) == 4
    assert [slot.get("n_ctx") for slot in slots] == [4096] * 4

    default = chat(base, messages=[{"role": "user", "content": "Reply only DEFAULT."}],
                   thinking=None, max_tokens=64)
    default_message = default["choices"][0]["message"]
    assert str(default_message.get("content") or "").strip() == "DEFAULT", default_message
    assert not default_message.get("reasoning_content"), default_message

    codes = ("AX17", "BY29", "CZ41", "DW53")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda code: answer(base, code), codes))
    assert results == [(code, code, False) for code in codes], results

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
    print("PASS: 4 x 4096 slots, parallel isolation, thinking on/off, strict JSON, topic labels, logprobs, tools")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    run(args.base_url.rstrip("/"))


if __name__ == "__main__":
    main()
