from __future__ import annotations

import asyncio

from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
from scripts import api_routes
from scripts.api_routes import RecallEvalSuiteRecordRequest, handle_chat_request


class _TurnEffectClient:
    async def chat(self, req, correlation_id=None):
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text="ok",
            metadata={
                "turn_effect": {"turn": {"novelty": 0.17}},
                "turn_effect_evidence": {"phi_before": {"novelty": 0.11}},
                "turn_effect_status": "present",
                "autonomy_summary": {"stance_hint": "maintain stable direct response"},
            },
            correlation_id=correlation_id or "corr-1",
        )
        return CortexChatResult(cortex_result=result, final_text="ok")


class _ReasoningQuickClient:
    async def chat(self, req, correlation_id=None):
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_quick",
            status="success",
            final_text="quick",
            metadata={
                "reasoning_content": "quick reasoning content",
                "thinking_source": "provider_reasoning",
            },
            correlation_id=correlation_id or "corr-quick",
        )
        return CortexChatResult(cortex_result=result, final_text="quick")


class _ReasoningBrainClient:
    async def chat(self, req, correlation_id=None):
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text="brain",
            reasoning_trace={"trace_role": "reasoning", "content": "brain trace"},
            correlation_id=correlation_id or "corr-brain",
        )
        return CortexChatResult(cortex_result=result, final_text="brain")


def test_handle_chat_request_exports_turn_effect_into_result_payload() -> None:
    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_TurnEffectClient(), payload, "sid-1", no_write=True))
    assert out["turn_effect"]["turn"]["novelty"] == 0.17
    assert out["turn_effect_status"] == "present"
    assert out["spark_meta"]["turn_effect"]["turn"]["novelty"] == 0.17


def test_handle_chat_request_preserves_quick_reasoning_content() -> None:
    payload = {"mode": "brain", "verbs": ["chat_quick"], "messages": [{"role": "user", "content": "quick"}]}
    out = asyncio.run(handle_chat_request(_ReasoningQuickClient(), payload, "sid-quick", no_write=True))
    assert out["reasoning_content"] == "quick reasoning content"
    assert out["thinking_source"] == "provider_reasoning"
    assert out["reasoning_trace"]["content"] == "quick reasoning content"


def test_handle_chat_request_preserves_brain_reasoning_trace() -> None:
    payload = {"mode": "brain", "messages": [{"role": "user", "content": "brain"}]}
    out = asyncio.run(handle_chat_request(_ReasoningBrainClient(), payload, "sid-brain", no_write=True))
    assert out["reasoning_trace"]["content"] == "brain trace"
    assert out["reasoning_content"] is None


def test_handle_chat_request_records_first_class_pressure_events(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "record", lambda entry: recorded.append(entry))

    class _PressureClient:
        async def chat(self, req, correlation_id=None):
            result = CortexClientResult(
                ok=True,
                mode="brain",
                verb="chat_general",
                status="partial",
                final_text="answer",
                metadata={
                    "runtime_response_diagnostics": {
                        "provider_finish_reason": "length",
                        "truncation_detected": True,
                        "status": "partial",
                    },
                    "chat_stance_debug": {
                        "source_inputs": {"social_bridge": {"hazards": ["not_addressed"]}},
                    },
                },
                correlation_id=correlation_id or "corr-pressure",
            )
            return CortexChatResult(cortex_result=result, final_text="answer")

    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_PressureClient(), payload, "sid-pressure", no_write=True))
    assert out["correlation_id"]
    assert recorded, "expected producer pressure telemetry to be recorded"
    latest = recorded[-1]
    categories = {event.pressure_category for event in latest.pressure_events}
    assert "response_truncation_or_length_finish" in categories
    assert "runtime_degradation_or_timeout" in categories
    assert "social_addressedness_gap" in categories


def test_handle_chat_request_ingests_recall_debug_pressure_events(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "record", lambda entry: recorded.append(entry))

    class _RecallPressureClient:
        async def chat(self, req, correlation_id=None):
            result = CortexClientResult(
                ok=True,
                mode="brain",
                verb="chat_general",
                status="success",
                final_text="answer",
                recall_debug={
                    "pressure_events": [
                        {
                            "source_service": "recall",
                            "source_event_id": "corr-recall-1",
                            "pressure_category": "missing_exact_anchor",
                            "confidence": 0.82,
                            "evidence_refs": ["recall_decision:abc"],
                            "metadata": {
                                "v1_v2_compare": {"v1_latency_ms": 120, "v2_latency_ms": 90, "selected_count_delta": 2},
                                "anchor_plan": {"time_window_days": 1},
                                "selected_evidence_cards": [{"id": "page-1"}],
                            },
                        }
                    ],
                    "debug": {
                        "compare_summary": {"v1_latency_ms": 120, "v2_latency_ms": 90, "selected_count_delta": 2},
                        "anchor_plan_summary": {"time_window_days": 1},
                        "selected_evidence_cards": [{"id": "page-1"}],
                    },
                },
                correlation_id=correlation_id or "corr-recall-pressure",
            )
            return CortexChatResult(cortex_result=result, final_text="answer")

    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_RecallPressureClient(), payload, "sid-recall-pressure", no_write=True))
    assert out["correlation_id"]
    assert recorded
    latest = recorded[-1]
    categories = {event.pressure_category for event in latest.pressure_events}
    assert "missing_exact_anchor" in categories
    event = next(item for item in latest.pressure_events if item.pressure_category == "missing_exact_anchor")
    assert event.metadata.get("v1_v2_compare", {}).get("selected_count_delta") == 2


def test_handle_chat_request_ingests_nested_executor_recall_pressure_events(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "record", lambda entry: recorded.append(entry))

    class _NestedRecallClient:
        async def chat(self, req, correlation_id=None):
            result = CortexClientResult(
                ok=True,
                mode="brain",
                verb="chat_general",
                status="success",
                final_text="answer",
                recall_debug={
                    "count": 0,
                    "decision": {
                        "recall_debug": {
                            "pressure_events": [
                                {
                                    "source_service": "orion-recall",
                                    "source_event_id": "corr-nested-1",
                                    "pressure_category": "stale_memory_selected",
                                    "confidence": 0.7,
                                    "evidence_refs": ["recall_decision:x"],
                                    "metadata": {},
                                }
                            ],
                            "compare_summary": {"v1_latency_ms": 10, "v2_latency_ms": 12, "selected_count_delta": 1},
                            "anchor_plan_summary": {"time_window_days": 14},
                            "selected_evidence_cards": [{"id": "n1"}],
                        }
                    },
                },
                correlation_id=correlation_id or "corr-nested",
            )
            return CortexChatResult(cortex_result=result, final_text="answer")

    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_NestedRecallClient(), payload, "sid-nested", no_write=True))
    assert out["correlation_id"]
    assert recorded
    latest = recorded[-1]
    categories = {event.pressure_category for event in latest.pressure_events}
    assert "stale_memory_selected" in categories
    evt = next(e for e in latest.pressure_events if e.pressure_category == "stale_memory_selected")
    assert evt.metadata.get("v1_v2_compare", {}).get("selected_count_delta") == 1


def test_record_chat_turn_pressure_telemetry_ws_source_event_prefix(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "record", lambda e: recorded.append(e))
    api_routes.record_chat_turn_pressure_telemetry(
        correlation_id="c-ws-parity",
        route_debug={},
        autonomy_payload={},
        recall_debug={
            "pressure_events": [
                {
                    "source_service": "orion-recall",
                    "source_event_id": "e-ws",
                    "pressure_category": "stale_memory_selected",
                    "confidence": 0.66,
                    "evidence_refs": ["ref:1"],
                    "metadata": {},
                }
            ]
        },
        source_event_id="chat_result_ws:c-ws-parity",
    )
    assert recorded
    assert recorded[-1].selection_reason.endswith("chat_result_ws:c-ws-parity")
    assert any(e.pressure_category == "stale_memory_selected" for e in recorded[-1].pressure_events)


def test_recall_eval_suite_record_writes_pressure_events(monkeypatch) -> None:
    import os

    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "op-tok-eval-test")
    monkeypatch.setattr(api_routes.settings, "HUB_RECALL_EVAL_RECORDING_ENABLED", True, raising=False)
    recorded = []
    monkeypatch.setattr(api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "record", lambda e: recorded.append(e))
    body = RecallEvalSuiteRecordRequest(
        suite_run_id="suite-test-1",
        rows=[
            {
                "case_id": "c1",
                "query": "q",
                "type": "t",
                "v1": {"selected_count": 0, "latency_ms": 10, "precision_proxy": 0.1, "irrelevant_cousin_rate": 0.0},
                "v2": {"selected_count": 2, "latency_ms": 8, "precision_proxy": 0.5, "irrelevant_cousin_rate": 0.0},
            }
        ],
    )
    out = api_routes.api_substrate_mutation_runtime_recall_eval_suite_record(
        body,
        x_orion_operator_token=os.environ["SUBSTRATE_MUTATION_OPERATOR_TOKEN"],
    )
    assert out["data"]["recorded"] == 1
    assert recorded
    assert recorded[-1].invocation_surface == "operator_review"
    assert any(
        str((e.metadata or {}).get("recall_evidence_kind")) == "eval_suite" for e in recorded[-1].pressure_events
    )


def test_tripwire_normal_chat_path_does_not_invoke_mutation_cycle_or_review_execute_once(monkeypatch) -> None:
    calls = {"scheduled": 0, "mutation": 0, "review": 0}

    def _scheduled(*args, **kwargs):
        calls["scheduled"] += 1

    def _mutation(*args, **kwargs):
        calls["mutation"] += 1

    def _review(*args, **kwargs):
        calls["review"] += 1

    monkeypatch.setattr(api_routes, "execute_substrate_mutation_scheduled_cycle", _scheduled)
    monkeypatch.setattr(api_routes, "_execute_substrate_mutation_cycle", _mutation)
    monkeypatch.setattr(api_routes, "_execute_substrate_review_cycle", _review)
    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_TurnEffectClient(), payload, "sid-tripwire", no_write=True))
    assert out["correlation_id"]
    assert calls == {"scheduled": 0, "mutation": 0, "review": 0}

