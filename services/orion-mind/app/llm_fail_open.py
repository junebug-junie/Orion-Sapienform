"""Fail-open metadata preserved when LLM synthesis falls back to deterministic Mind."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from orion.mind.v1 import MindRunResultV1

from .phase_telemetry import MindPhaseTelemetry, phase_telemetry_machine_keys

_LLM_FAIL_OPEN_ADVISORY_KEYS = (
    "mind.llm_synthesis_enabled",
    "mind.llm_synthesis_attempted",
    "mind.llm_synthesis_failed_phase",
    "mind.llm_synthesis_error_code",
    "mind.llm_synthesis_error",
    "mind.semantic_route",
    "mind.appraisal_route",
    "mind.stance_route",
    "mind.semantic_synthesis_seen",
    "mind.phase_telemetry",
    "mind.fallback_reason",
)


@dataclass(frozen=True)
class MindLLMFailOpenRecord:
    mind_run_id: UUID
    snapshot_hash: str
    error_code: str
    diagnostics: list[str]
    failed_phase: str | None
    semantic_route: str
    appraisal_route: str
    stance_route: str
    phase_telemetry: list[MindPhaseTelemetry] = field(default_factory=list)
    timing_ms_by_phase: dict[str, float] = field(default_factory=dict)
    fallback_reason: list[str] = field(default_factory=list)

    def machine_contract_patch(self) -> dict[str, Any]:
        patch: dict[str, Any] = {
            "mind.llm_synthesis_enabled": True,
            "mind.llm_synthesis_attempted": True,
            "mind.llm_synthesis_error_code": self.error_code,
            "mind.semantic_route": self.semantic_route,
            "mind.appraisal_route": self.appraisal_route,
            "mind.stance_route": self.stance_route,
            "mind.authorized_for_stance_skip": False,
            "mind.authorized_for_stance_use": False,
            "mind.semantic_synthesis_seen": any(
                r.phase_name == "semantic_synthesis" and not r.skipped for r in self.phase_telemetry
            ),
        }
        if self.failed_phase:
            patch["mind.llm_synthesis_failed_phase"] = self.failed_phase
        if self.diagnostics:
            patch["mind.llm_synthesis_error"] = self.diagnostics[0]
        if self.phase_telemetry:
            patch.update(phase_telemetry_machine_keys(self.phase_telemetry))
        reasons = list(self.fallback_reason)[:8]
        if not reasons:
            reasons = [self.error_code, *self.diagnostics[:3]]
        patch["mind.fallback_reason"] = reasons[:8]
        return patch

    def diagnostics_patch(self) -> list[str]:
        out = [f"llm_fail_open:{self.error_code}"]
        if self.failed_phase:
            out.append(f"llm_failed_phase:{self.failed_phase}")
        out.extend(self.diagnostics[:6])
        return out

    def merge_into_deterministic(self, result: MindRunResultV1) -> MindRunResultV1:
        machine = dict(result.brief.machine_contract)
        machine.update(self.machine_contract_patch())
        advisory = list(result.brief.advisory_keys)
        for key in _LLM_FAIL_OPEN_ADVISORY_KEYS:
            if key not in advisory:
                advisory.append(key)
        brief = result.brief.model_copy(
            update={
                "machine_contract": machine,
                "advisory_keys": advisory,
                "mind_authorized_for_stance_skip": False,
            }
        )
        timing = {**self.timing_ms_by_phase, **(result.timing_ms_by_phase or {})}
        diagnostics = list(result.diagnostics or [])
        for item in self.diagnostics_patch():
            if item not in diagnostics:
                diagnostics.append(item)
        return result.model_copy(
            update={
                "brief": brief,
                "diagnostics": diagnostics,
                "timing_ms_by_phase": timing,
            }
        )
