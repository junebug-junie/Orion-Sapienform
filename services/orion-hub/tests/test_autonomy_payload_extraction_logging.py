from __future__ import annotations

import logging
from types import SimpleNamespace

from scripts.autonomy_payloads import extract_autonomy_payload, log_autonomy_payload_extraction


def test_log_autonomy_payload_extraction_empty(caplog) -> None:
    caplog.set_level(logging.INFO)
    cortex = SimpleNamespace(metadata={"trace_verb": "chat_general"})
    payload = extract_autonomy_payload(cortex)
    assert payload == {}
    log_autonomy_payload_extraction(
        correlation_id="corr-empty",
        cortex_result=cortex,
        payload=payload,
        source="http",
    )
    assert "hub_autonomy_extract_empty corr=corr-empty source=http" in caplog.text
    assert "trace_verb" in caplog.text


def test_log_autonomy_payload_extraction_present(caplog) -> None:
    caplog.set_level(logging.INFO)
    cortex = SimpleNamespace(
        metadata={
            "autonomy_summary": {"stance_hint": "steady"},
            "autonomy_debug": {"orion": {"availability": "available"}},
            "autonomy_state_preview": {"dominant_drive": "coherence"},
            "autonomy_backend": "graph",
            "autonomy_selected_subject": "relationship",
        }
    )
    payload = extract_autonomy_payload(cortex)
    log_autonomy_payload_extraction(
        correlation_id="corr-present",
        cortex_result=cortex,
        payload=payload,
        source="ws",
    )
    assert "hub_autonomy_extract_present corr=corr-present source=ws" in caplog.text
    assert "selected_subject=relationship" in caplog.text
    assert "backend=graph" in caplog.text
    assert "summary_present=True" in caplog.text
    assert "debug_present=True" in caplog.text
    assert "state_preview_present=True" in caplog.text
