"""The compactor digest step machine (driven by the compactor.digest durable run)."""
from __future__ import annotations

from orion.cognition.compactor import map_reduce as mr
from orion.cognition.compactor.constants import DIGEST_MAX_TOKENS
from orion.schemas.cortex.contracts import CortexClientRequest

SPEC = mr.SPECS["github"]
LEASE = {"lease_id": "hold-1", "generation": 2, "role": "agent-gpu2", "holder": "durable-runs:r"}


def _digest(refs: list[str], body: str = "b") -> dict:
    return {"card_summary": "c", "journal_title": "t", "journal_body": body, "pr_refs": refs}


def _inputs(n: int) -> list[dict]:
    return [{"repo": "acme/widgets", "items": [{"number": i}], "chunk_index": i, "chunk_count": n} for i in range(n)]


def test_request_payload_is_a_valid_cortex_request_with_the_hold_and_no_workflow_reentry():
    payload = mr.build_digest_request_payload(SPEC, {"items": []}, workflow_id="github_compactor_pass",
                                              correlation_id="c", session_id="s", gpu_lease=LEASE)
    req = CortexClientRequest.model_validate(payload)
    assert req.verb == "github_compactor_digest_v1" and req.mode == "brain"
    assert req.options["gpu_lease"] == LEASE and req.options["llm_route"] == "agent"
    assert req.options["max_tokens"] == DIGEST_MAX_TOKENS and req.recall.enabled is False
    assert "workflow_request" not in req.context.metadata and "durable_run" not in req.context.metadata
    assert req.context.metadata["github_compactor_input"] == {"items": []}


def test_digest_from_payload_error_tokens():
    assert mr.digest_from_payload(SPEC, {"ok": False, "error": {"message": "llm_timeout"}})[1] == \
        "github_compactor_digest_failed:llm_timeout"
    assert mr.digest_from_payload(SPEC, {"ok": True, "metadata": {"structured_output_rejected": True}})[1] == \
        "github_compactor_digest_failed:structured_output_rejected"
    assert mr.digest_from_payload(SPEC, {"ok": True, "final_text": ""})[1] == "github_compactor_digest_failed:empty_completion"
    assert mr.digest_from_payload(SPEC, {"ok": True, "final_text": "nope"})[1].startswith(
        "github_compactor_digest_failed:invalid_json:")
    digest, error = mr.digest_from_payload(SPEC, {"ok": True, "metadata": {"github_compactor_digest": _digest(["#1"])}})
    assert error is None and digest.pr_refs == ["#1"]


def test_order_is_chunks_then_merge_then_done():
    inputs = _inputs(3)
    partials: list[dict] = []
    labels = []
    while (call := mr.next_call(SPEC, inputs, partials, None)) is not None and call["kind"] == "chunk":
        labels.append(call["label"])
        partials.append(_digest([f"#{call['index']}"]))
    assert labels == ["chunk_1_of_3", "chunk_2_of_3", "chunk_3_of_3"]
    assert call["kind"] == "merge" and len(call["input"]["partial_digests"]) == 3
    merge, err = mr.record_merge(SPEC, partials, SPEC.model_cls.model_validate(_digest(["#0", "#1", "#2"], "merged")))
    assert err is None and mr.next_call(SPEC, inputs, partials, merge) is None
    out = mr.assemble(SPEC, inputs, partials, merge, window_label="2026-09-28")
    assert out["merge_mode"] == "llm_merge" and out["digest"]["journal_body"] == "merged"


def test_single_input_needs_no_merge():
    inputs = _inputs(1)
    assert mr.next_call(SPEC, inputs, [], None)["label"] == "single"
    assert mr.next_call(SPEC, inputs, [_digest(["#0"])], None) is None
    assert mr.resolve_merge_without_call(SPEC, inputs, [_digest(["#0"])], None) is None
    assert mr.assemble(SPEC, inputs, [_digest(["#0"])], None, window_label="d")["merge_mode"] == "single"


def test_merge_fallbacks_join_every_chunk():
    inputs = _inputs(2)
    partials = [_digest(["#0"], "zero"), _digest(["#1"], "one")]
    dropped, err = mr.record_merge(SPEC, partials, SPEC.model_cls.model_validate(_digest(["#0"])))
    assert dropped["reason"] == "merge_dropped_refs:1" and err.startswith("refs_missing:")
    out = mr.assemble(SPEC, inputs, partials, dropped, window_label="d")
    assert out["merge_mode"] == "concatenated" and "zero" in out["digest"]["journal_body"] and "one" in out["digest"]["journal_body"]
    gave_up = mr.assemble(SPEC, inputs, partials, mr.merge_gave_up("boom"), window_label="d")
    assert gave_up["merge_skipped_reason"] == "merge_failed:boom"


def test_over_budget_merge_is_decided_without_a_call(monkeypatch):
    monkeypatch.setattr(mr, "DIGEST_INPUT_CHAR_BUDGET", 10)
    inputs = _inputs(2)
    partials = [_digest(["#0"]), _digest(["#1"])]
    decided = mr.resolve_merge_without_call(SPEC, inputs, partials, None)
    assert decided == {"status": "skipped", "digest": None, "reason": "merge_input_over_budget"}
    assert mr.next_call(SPEC, inputs, partials, decided) is None


def test_journal_body_is_never_trimmed_but_card_prose_is():
    body = "x" * 60_000
    out = mr.assemble(SPEC, _inputs(1), [{**_digest(["#0"], body), "card_summary": "y" * 5000}], None, window_label="d")
    assert out["digest"]["journal_body"] == body
    assert len(out["digest"]["card_summary"]) < 5000 and "card_summary" in out["trimmed_fields"]


def test_digest_request_turns_thinking_off_through_the_forwarded_switch():
    """Live 2026-09-30 corr 2af9b6ea: thinking stayed on (48k chars reasoning, cut at 16000
    tokens) because the old ``reasoning.effort=none`` option had no consumer. cortex-exec only
    forwards ``chat_template_kwargs`` to the gateway."""
    for spec in mr.SPECS.values():
        payload = mr.build_digest_request_payload(spec, {"items": []}, workflow_id="w",
                                                  correlation_id="c", session_id="s")
        req = CortexClientRequest.model_validate(payload)
        assert req.options["chat_template_kwargs"] == {"enable_thinking": False}
        assert "reasoning" not in req.options
