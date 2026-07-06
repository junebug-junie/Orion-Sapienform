from __future__ import annotations

import pytest

from orion.harness.finalize import parse_finalize_reflection_payload


def _minimal_reflection_json() -> str:
    return """{
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


def test_parse_finalize_reflection_payload_accepts_wrapped_json() -> None:
    raw = f"Here is the reflection JSON:\n```json\n{_minimal_reflection_json()}\n```"
    reflection = parse_finalize_reflection_payload(raw)
    assert reflection.correlation_id == "c-1"
    assert reflection.alignment_verdict == "aligned"


def test_parse_finalize_reflection_payload_accepts_prose_prefix() -> None:
    raw = f"Integrative check complete.\n{_minimal_reflection_json()}"
    reflection = parse_finalize_reflection_payload(raw)
    assert reflection.thought_event_id == "t-1"


def test_parse_finalize_reflection_payload_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Could not parse JSON object"):
        parse_finalize_reflection_payload("not json at all")
