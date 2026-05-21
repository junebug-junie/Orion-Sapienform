"""CognitionTracePayload → OrionSignalV1 run + step signals (spec §5.3)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.adapters.result import AdapterResult
from orion.signals.models import OrganClass, OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

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
    ("AgentChainService", "agent_chain"),
    ("PlannerReactService", "planner"),
]


def _step_services(step) -> List[str]:
    if isinstance(step.result, dict) and step.result:
        return list(step.result.keys())
    return []


def _map_step_organ(step_name: str, services: List[str]) -> tuple[str, List[str]]:
    if step_name in _STEP_NAME_ORGAN:
        return _STEP_NAME_ORGAN[step_name], []
    if services:
        for prefix, organ in _SERVICE_PREFIX_ORGAN:
            if services[0].startswith(prefix) or prefix in services[0]:
                return organ, []
    return "cortex_exec", [f"step_organ_fallback:{step_name}"]


class CognitionTraceAdapter(OrionSignalAdapter):
    organ_id = "cortex_exec"

    def can_handle(self, channel: str, payload: dict) -> bool:
        kind = str(channel or "").lower()
        if "cognition:trace" in kind or kind in {"cognition.trace", "cognition.trace.v1"}:
            return True
        return payload.get("verb") is not None and "steps" in payload

    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> AdapterResult:
        try:
            trace = CognitionTracePayload.model_validate(payload)
        except Exception:
            return None

        corr = str(payload.get("_envelope_correlation_id") or trace.correlation_id or "").strip()
        if not corr:
            corr = f"{trace.verb}:{int(trace.timestamp)}"
            synthetic_note = "synthetic_correlation_id"
        else:
            synthetic_note = None

        now = datetime.now(timezone.utc)
        steps = trace.steps or []
        all_ok = trace.metadata.get("status", "success") == "success" and all(
            s.status == "success" for s in steps
        )
        total_ms = sum(int(s.latency_ms or 0) for s in steps)
        meta = trace.metadata or {}
        reasoning = bool(meta.get("reasoning_present")) or bool(trace.recall_debug)

        run_entry = registry.get("cortex_exec") or ORGAN_REGISTRY["cortex_exec"]
        run_parents = [
            prior_signals[p].signal_id
            for p in (run_entry.causal_parent_organs or [])
            if p in prior_signals
        ]
        run_id = make_signal_id("cortex_exec", f"{corr}:run")
        run_notes: List[str] = []
        if synthetic_note:
            run_notes.append(synthetic_note)
        if not steps:
            run_notes.append("no_steps_in_trace")

        run_sig = OrionSignalV1(
            signal_id=run_id,
            organ_id="cortex_exec",
            organ_class=OrganClass.endogenous,
            signal_kind="cognition_run",
            dimensions={
                "success": 1.0 if all_ok else 0.0,
                "step_count": clamp01(len(steps) / 20.0),
                "latency_level": clamp01(min(total_ms, 120_000) / 120_000.0),
                "recall_used": 1.0 if trace.recall_used else 0.0,
                "reasoning_present": 1.0 if reasoning else 0.0,
                "final_text_present": 1.0 if (trace.final_text or "").strip() else 0.0,
            },
            causal_parents=run_parents,
            source_event_id=corr,
            observed_at=now,
            emitted_at=now,
            summary=(
                f"verb={trace.verb} mode={trace.mode} steps={len(steps)} "
                f"recall={int(trace.recall_used)} latency={total_ms}ms"
            ),
            notes=run_notes[:5],
        )

        out: List[OrionSignalV1] = [run_sig]
        for step in sorted(steps, key=lambda s: s.order):
            services = _step_services(step)
            organ_id, extra_notes = _map_step_organ(step.step_name, services)
            step_parents = [run_id]
            reg_entry = registry.get(organ_id) or ORGAN_REGISTRY.get(organ_id)
            if reg_entry:
                for p in reg_entry.causal_parent_organs or []:
                    if p in prior_signals:
                        step_parents.append(prior_signals[p].signal_id)
                        break
            step_id = make_signal_id(organ_id, f"{corr}:step:{step.order}:{step.step_name}")
            out.append(
                OrionSignalV1(
                    signal_id=step_id,
                    organ_id=organ_id,
                    organ_class=OrganClass.endogenous,
                    signal_kind="cognition_step",
                    dimensions={
                        "success": 1.0 if step.status == "success" else 0.0,
                        "latency_level": clamp01(min(int(step.latency_ms or 0), 60_000) / 60_000.0),
                        "error_present": 1.0 if (step.error or "").strip() else 0.0,
                        "service_count": clamp01(min(len(services), 5) / 5.0),
                    },
                    causal_parents=list(dict.fromkeys(step_parents)),
                    source_event_id=corr,
                    observed_at=now,
                    emitted_at=now,
                    summary=f"step={step.step_name} status={step.status} latency={step.latency_ms or 0}ms",
                    notes=extra_notes[:5],
                )
            )
        return out
