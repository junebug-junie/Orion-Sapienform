from __future__ import annotations

import pytest

from orion.cognition.cortex_payload_extract import (
    cortex_exec_failure_detail,
    extract_cortex_payload_text,
)
from orion.harness.finalize import extract_finalize_reflection_payload, parse_finalize_reflection_payload


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
