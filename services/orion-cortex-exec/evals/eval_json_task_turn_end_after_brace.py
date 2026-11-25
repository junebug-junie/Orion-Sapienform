"""How likely is a JSON hop to end its reply right after "{"? One number per task-turn wording.

Why: 2026-11-17, stance_react on the Strata agent lane replied "{" and ended (Strata request monitor:
output_tokens=2, finish=stop). The generic task turn ("Carry out the task in the system message.") made
that ending 5-10x more likely than JSON_TASK_TURN on real failing prompts.

Input: one llm.chat.request envelope JSON (for example pulled read-only from the bus mirror's
bus_events table) from a hop whose user turn is a service task turn, asking for plain JSON
(response_format json_object, or return_json). Not modelled: the gateway's structured_output_* path,
response_format_for_backend's schema rewrites, and top_p/stop/penalties. Prompts hold private memory:
keep them out of git.

For each wording it sends the request with "{\"" banned (logit_bias), so the first token is "{", and
reads the probability of "<|im_end|>" as the second token (top_logprobs=20). A reply that really
ended after "{" prints as "ended" (the end token itself is not listed). It runs each wording
--repeat times back to back; the first reading after a cache miss can differ, so look at the later ones.

Every call goes through orion/evals/leased_llm.py: one GPU pool operator hold on --lane (default
``agent``, the Strata agent seat), each call POSTed to orion-llm-gateway under that hold. The gateway
forwards the body as-is (logprobs and logit_bias included) and puts the hold's lease on the call, so the
lane door counts it as leased. There is no direct-URL mode (#3921). A handful of calls only:
    ORION_BUS_URL=redis://<tailnet-ip>:6379/0 \
    python services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py envelope.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orion-cortex-exec"))

from app.executor import JSON_TASK_TURN, SYSTEM_TASK_TURN  # noqa: E402
from orion.evals.leased_llm import EXIT_RECALLED, EXIT_UNAVAILABLE, OPERATOR_LANES, LaneCallError, \
    LaneRecalled, LaneUnavailable, operator_lane_sync  # noqa: E402

HOLDER = "operator:eval-json-task-turn-end-after-brace"


def _gateway_strata_structured():
    """The gateway's strata_structured module, by file: both services name their package `app`."""
    path = ROOT / "services" / "orion-llm-gateway" / "app" / "strata_structured.py"
    spec = importlib.util.spec_from_file_location("gateway_strata_structured", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look themselves up here
    spec.loader.exec_module(module)
    return module

OPEN_BRACE_QUOTE_ID = "4754"  # '{"' in the qwen3.8-flash-next vocabulary
END_TOKEN = "<|im_end|>"


def strata_payload(envelope: dict) -> dict:
    """The body the gateway sends Strata for this request (its structured-output takeover included)."""
    strata_structured = _gateway_strata_structured()
    p = envelope["payload"]
    o = p.get("options") or {}
    body = {
        "model": "Active-GGUF-Model",
        "messages": [{"role": m["role"], "content": m["content"]} for m in p["messages"]],
        "stream": False,
        "temperature": o.get("temperature", 0.7),
        "max_tokens": 2,
    }
    if isinstance(o.get("response_format"), dict):
        body["response_format"] = o["response_format"]
    elif o.get("return_json"):
        body["response_format"] = {"type": "json_object"}
    if o.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = o["chat_template_kwargs"]
    sent, _ = strata_structured.take_over(body)
    return sent


def reading(lane, body: dict) -> float | str:
    """One reading; a failed call or an unreadable reply is a string in the readings, not a crash.
    A recall (LaneRecalled) propagates so the run stops."""
    try:
        return p_end_after_brace(lane, body)
    except (LaneCallError, RuntimeError) as exc:
        if isinstance(exc, LaneRecalled):
            raise
        return f"error: {exc}"[:160]


def p_end_after_brace(lane, body: dict) -> float | str:
    """``lane`` is an orion.evals.leased_llm.OperatorLane (anything with ``chat(body, timeout_sec)``)."""
    b = copy.deepcopy(body)
    b.update({"max_tokens": 2, "logprobs": True, "top_logprobs": 20, "logit_bias": {OPEN_BRACE_QUOTE_ID: -100}})
    out = lane.chat(b, timeout_sec=300)
    lp = ((out["choices"][0].get("logprobs") or {}).get("content")) or []
    if not lp or lp[0]["token"] != "{":
        raise RuntimeError(f"first token was {lp[0]['token'] if lp else None!r}, not '{{': no reading")
    if len(lp) < 2:
        return "ended"  # the sampled reply itself ended after "{": the failure this measures
    return sum(math.exp(a["logprob"]) for a in lp[1]["top_logprobs"] if a["token"] == END_TOKEN)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("envelope")
    ap.add_argument("--lane", default="agent", choices=sorted(OPERATOR_LANES),
                    help="operator lane; OPEN_BRACE_QUOTE_ID is a qwen3.8-flash-next token id, so the lane "
                         "must serve that vocabulary (the agent seat does)")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    body = strata_payload(json.loads(Path(args.envelope).read_text()))
    users = [i for i, m in enumerate(body["messages"]) if m["role"] == "user"]
    if not users or body["messages"][users[-1]]["content"] not in (SYSTEM_TASK_TURN, JSON_TASK_TURN):
        print("the request's last user turn is not a service task turn; nothing to compare")
        return 2
    try:
        with operator_lane_sync(args.lane, holder=HOLDER) as lane:
            for name, text in (("generic", SYSTEM_TASK_TURN), ("json", JSON_TASK_TURN)):
                variant = copy.deepcopy(body)
                variant["messages"][users[-1]]["content"] = text
                readings = [reading(lane, variant) for _ in range(args.repeat)]
                print(f"{name:8} {text!r}: P(end after '{{') = {readings}")
            print(f"lease {lane.lease_id} ({lane.route}, {lane.role}), {lane.calls} call(s)")
    except LaneRecalled as exc:
        print(f"RECALLED: {exc}", file=sys.stderr)
        return EXIT_RECALLED
    except LaneUnavailable as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
