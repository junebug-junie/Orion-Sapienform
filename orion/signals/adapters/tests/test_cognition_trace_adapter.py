import json
from pathlib import Path

import pytest

from orion.signals.adapters.cognition_trace import CognitionTraceAdapter
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

FIXTURE = Path(__file__).parent / "fixtures" / "cognition_trace_chat_general.json"
CORR = "corr-chat-general-fixture"


@pytest.fixture
def adapter() -> CognitionTraceAdapter:
    return CognitionTraceAdapter()


def _payload_with_correlation() -> dict:
    payload = json.loads(FIXTURE.read_text())
    payload["correlation_id"] = CORR
    payload["_envelope_correlation_id"] = CORR
    return payload


def test_cognition_trace_adapter_chat_general(adapter: CognitionTraceAdapter) -> None:
    payload = _payload_with_correlation()
    norm = NormalizationContext()
    out = adapter.adapt("orion:cognition:trace", payload, ORGAN_REGISTRY, {}, norm)
    assert isinstance(out, list)
    assert len(out) == 4
    run = next(s for s in out if s.signal_kind == "cognition_run")
    assert run.organ_id == "cortex_exec"
    assert run.source_event_id == CORR
    steps = [s for s in out if s.signal_kind == "cognition_step"]
    assert {s.organ_id for s in steps} == {"graph_cognition", "chat_stance", "llm_gateway"}
    run_id = make_signal_id("cortex_exec", f"{CORR}:run")
    assert run.signal_id == run_id
    for s in steps:
        assert run.signal_id in s.causal_parents


def test_cognition_trace_adapter_no_pii_in_signal(adapter: CognitionTraceAdapter) -> None:
    payload = _payload_with_correlation()
    out = adapter.adapt("orion:cognition:trace", payload, ORGAN_REGISTRY, {}, NormalizationContext())
    blob = json.dumps([s.model_dump(mode="json") for s in out])
    assert "USER_SECRET" not in blob
    assert "USER_SECRET_SHOULD_NOT_APPEAR_IN_SIGNAL" not in blob
    assert '"final_text":' not in blob
