"""runner._call_verb_text / strict_final_text: a freeform verb's answer is the final text only.

Regression for the review blocker on the Orion's Day letter: extract_cortex_payload_text falls back
to reasoning_content when final_text is empty, which a JSON verb's parser rejects but a freeform
verb would have stored as Orion's note. Error text framed as prose and a completion cut off at
max_tokens are attempts too, never successes."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.runner import DurableRunner, strict_final_text  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseRefV1  # noqa: E402

NOTE = "Yesterday I kept circling the hop written_at prior. " * 20


def _payload(final_text="", *, reasoning=None, finish_reason="stop", ok=True, truncation=None):
    block = {"content": final_text, "raw": {"choices": [{"finish_reason": finish_reason,
                                                          "message": {"content": final_text,
                                                                      "reasoning_content": reasoning}}]}}
    if reasoning:
        block["reasoning_content"] = reasoning
    payload = {"ok": ok, "status": "success" if ok else "fail", "final_text": final_text,
               "steps": [{"order": 0, "result": {"LLMGatewayService": block}}]}
    if truncation is not None:
        payload["metadata"] = {"runtime_response_diagnostics": {"truncation_detected": truncation}}
    return payload


def test_final_text_is_returned():
    assert strict_final_text(_payload(NOTE), "orion_day_note_v1") == NOTE.strip()


def test_reasoning_is_never_accepted_as_the_answer():
    thinking = "We need to write a long note. Let me look at the record first... " * 30
    with pytest.raises(RuntimeError, match="verb_empty_final_text"):
        strict_final_text(_payload("", reasoning=thinking), "orion_day_note_v1")


@pytest.mark.parametrize("text", ["[Error: llamacpp timed out after waiting]",
                                  "Error: upstream 503 from gateway, please retry later."])
def test_error_text_framed_as_prose_is_refused(text):
    with pytest.raises(RuntimeError, match="verb_error_text"):
        strict_final_text(_payload(text), "orion_day_carry_forward_v1")


def test_a_completion_cut_off_at_max_tokens_is_refused():
    with pytest.raises(RuntimeError, match="verb_truncated_at_max_tokens"):
        strict_final_text(_payload(NOTE, finish_reason="length"), "orion_day_note_v1")
    with pytest.raises(RuntimeError, match="verb_truncated_at_max_tokens"):
        strict_final_text(_payload(NOTE, truncation=True), "orion_day_note_v1")


def test_prose_that_merely_mentions_an_error_is_kept():
    text = "The codebase kept throwing errors I could not map, and I sat with that. " * 10
    assert strict_final_text(_payload(text), "orion_day_note_v1") == text.strip()


def _runner(reply):
    runner = DurableRunner.__new__(DurableRunner)
    runner.sent = []

    async def rpc(request_payload, *, timeout_sec, label):
        runner.sent.append((request_payload, timeout_sec, label))
        return reply

    runner._cortex_orch_rpc = rpc
    return runner


def test_call_verb_text_sends_the_verb_route_hold_and_metadata():
    runner = _runner(_payload(NOTE))
    ref = GpuLeaseRefV1(lease_id="hold-1", generation=2, role="agent", holder="durable-runs:r")
    text = asyncio.run(runner._call_verb_text(
        "orion_day_note_v1", {"orion_day_input": {"digest_md": "d"}}, "agent",
        gpu_lease=ref, timeout_sec=870.0, user_text="Write your note about the day."))
    assert text == NOTE.strip()
    ((request, timeout, label),) = runner.sent
    assert request["verb"] == "orion_day_note_v1" and label == "orion_day_note_v1" and timeout == 870.0
    assert request["options"]["llm_route"] == "agent" and request["options"]["policy_dispatch_only"] is True
    assert request["options"]["gpu_lease"]["lease_id"] == "hold-1"
    # Thinking off: live 2026-09-30 the note's hidden reasoning ate all 12000 max_tokens.
    assert request["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["context"]["metadata"] == {"orion_day_input": {"digest_md": "d"}}
    assert request["recall"]["enabled"] is False


def test_call_verb_text_raises_on_a_not_ok_result():
    runner = _runner(_payload(NOTE, ok=False))
    with pytest.raises(RuntimeError, match="verb_not_ok"):
        asyncio.run(runner._call_verb_text("orion_day_note_v1", {}, "agent", timeout_sec=10.0, user_text="x"))
