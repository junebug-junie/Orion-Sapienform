"""Synthesize Organ Signals correlation chains from cached cognition traces."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

_STEP_NAME_ORGAN = {
    "collect_metacog_context": "graph_cognition",
    "synthesize_chat_stance_brief": "chat_stance",
    "llm_chat_general": "llm_gateway",
}

_SERVICE_PREFIX_ORGAN = [
    ("RecallService", "recall"),
    ("Mind", "mind"),
    ("LLMGatewayService", "llm_gateway"),
    ("MetacogContextService", "graph_cognition"),
]


def _step_services(step: dict[str, Any]) -> List[str]:
    services = step.get("services")
    if isinstance(services, list):
        return [str(s) for s in services if str(s).strip()]
    result = step.get("result") if isinstance(step.get("result"), dict) else {}
    return [str(k) for k in result.keys()]


def _map_step_organ(step_name: str, services: List[str]) -> str:
    if step_name in _STEP_NAME_ORGAN:
        return _STEP_NAME_ORGAN[step_name]
    if services:
        for prefix, organ in _SERVICE_PREFIX_ORGAN:
            if services[0].startswith(prefix) or prefix in services[0]:
                return organ
    return "cortex_exec"


def correlation_chain_from_cognition_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Build a correlation graph chain when signal inspect cache missed the turn."""
    corr = str(trace.get("correlation_id") or "").strip()
    if not corr:
        raise ValueError("missing correlation_id")

    observed_at = datetime.now(timezone.utc).isoformat()
    ts = trace.get("timestamp")
    if isinstance(ts, (int, float)) and ts > 0:
        observed_at = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    ordered_steps = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda row: int(row.get("order") or 0),
    )

    run_id = f"synth:{corr}:run"
    chain: List[Dict[str, Any]] = [
        {
            "organ_id": "cortex_exec",
            "signal_kind": "cognition_run",
            "signal_id": run_id,
            "observed_at": observed_at,
            "dimensions": {
                "success": 1.0 if str(trace.get("status") or "success") == "success" else 0.0,
                "step_count": min(len(ordered_steps), 20) / 20.0,
            },
            "causal_parents": [],
            "is_stub": False,
        }
    ]

    prev_id = run_id
    for step in ordered_steps:
        step_name = str(step.get("step_name") or "step")
        services = _step_services(step)
        organ_id = _map_step_organ(step_name, services)
        order = int(step.get("order") or 0)
        step_id = f"synth:{corr}:step:{order}:{step_name}"
        chain.append(
            {
                "organ_id": organ_id,
                "signal_kind": "cognition_step",
                "signal_id": step_id,
                "observed_at": observed_at,
                "dimensions": {
                    "success": 1.0 if str(step.get("status") or "") == "success" else 0.0,
                },
                "causal_parents": [prev_id],
                "is_stub": False,
            }
        )
        prev_id = step_id

    return {
        "correlation_id": corr,
        "chain": chain,
        "complete": True,
        "gaps": ["synthesized_from_cognition_trace"],
        "hidden_stubs": 0,
        "source": "cognition_trace_fallback",
    }
