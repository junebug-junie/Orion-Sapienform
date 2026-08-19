from __future__ import annotations

import pytest

from orion.cognition.cortex_payload_extract import (
    cortex_exec_failure_detail,
    extract_cortex_payload_text,
)
from orion.harness.finalize import (
    extract_finalize_reflection_payload,
    extract_voice_finalize_text,
    parse_finalize_reflection_payload,
)


def test_extract_cortex_payload_text_from_nested_llm_gateway_service() -> None:
    payload = {
        "status": "success",
        "final_text": "",
        "steps": [
            {
                "step_name": "llm_harness_finalize_reflect",
                "order": 0,
                "result": {
                    "LLMGatewayService": {
                        "content": '{"correlation_id":"c-1","alignment_verdict":"aligned"}',
                    }
                },
            }
        ],
    }
    assert "alignment_verdict" in extract_cortex_payload_text(payload)


def test_extract_finalize_reflection_payload_surfaces_step_error() -> None:
    payload = {
        "status": "fail",
        "final_text": "",
        "steps": [
            {
                "error": "API returned an empty or malformed response (HTTP 200)",
                "result": {"LLMGatewayService": {}},
            }
        ],
    }
    with pytest.raises(ValueError, match="harness_finalize_reflect exec failed"):
        extract_finalize_reflection_payload(payload)


def test_parse_finalize_reflection_payload_from_nested_step_content() -> None:
    raw = extract_finalize_reflection_payload(
        {
            "steps": [
                {
                    "result": {
                        "LLMGatewayService": {
                            "content": """{
  "correlation_id": "c-1",
  "thought_event_id": "t-1",
  "substrate_appraisal_id": "a-1",
  "draft_hash": "d-1",
  "imperative": "explain",
  "tone": "curious",
  "strain_refs": [],
  "alignment_verdict": "aligned",
  "alignment_notes": [],
  "strain_unresolved": false
}"""
                        }
                    }
                }
            ]
        }
    )
    reflection = parse_finalize_reflection_payload(raw)
    assert reflection.alignment_verdict == "aligned"


def test_cortex_exec_failure_detail_reads_structured_rejection_preview() -> None:
    detail = cortex_exec_failure_detail(
        {
            "status": "fail",
            "metadata": {
                "structured_output_rejected": True,
                "structured_rejection_preview": '{"correlation_id":"c-1"',
            },
        }
    )
    assert detail is not None
    assert "structured_output_rejected" in detail


def test_extract_voice_finalize_text_rejects_error_shaped_text() -> None:
    """Confirmed live, 2026-08-19: a real circe-worker outage made this exec
    result's own text field literally this string -- a genuine upstream
    failure reported only in the text, which an emptiness-only check (the
    only gate here before this fix) does not catch. Raising here routes
    into the file's own already-correct failure path
    (emit_finalize_failure_artifacts/HarnessFinalizeFailedError) instead of
    shipping the error text as Orion's real answer."""
    payload = {"final_text": "[Error: llamacpp timed out after waiting]"}
    with pytest.raises(ValueError, match="error-shaped text"):
        extract_voice_finalize_text(payload)


def test_extract_voice_finalize_text_accepts_real_text() -> None:
    """Guards the test above: the happy path must not be over-gated."""
    payload = {"final_text": "I keep circling the same unresolved thing."}
    assert extract_voice_finalize_text(payload) == "I keep circling the same unresolved thing."


def test_extract_finalize_reflection_payload_rejects_error_shaped_text() -> None:
    """Same failure shape as the voice-finalize test above, for the reflect
    step. Before this fix this accidentally degraded gracefully ONLY
    because the error string fails the caller's JSON parse -- fragile, not
    a real gate: text that happened to parse as JSON would sail through
    un-degraded. This asserts the explicit check, not the accident."""
    payload = {"final_text": "[Error: llamacpp timed out after waiting]"}
    with pytest.raises(ValueError, match="error-shaped text"):
        extract_finalize_reflection_payload(payload)
