from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence
from uuid import uuid4

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.self_study import (
    SelfStudyConsumerContextV1,
    SelfStudyHarnessResultV1,
    SelfStudyHarnessScenarioResultV1,
    SelfStudyHarnessSoakResultV1,
    SelfStudyHarnessSummaryV1,
    SelfStudyRetrieveRequestV1,
    SelfStudyRetrieveResultV1,
)

from .self_study import (
    INDUCED_TRUST_TIER,
    REFLECTIVE_TRUST_TIER,
    TRUST_TIER,
    run_self_concept_induce,
    run_self_concept_reflect,
    run_self_repo_inspect,
    run_self_retrieve,
)
from .self_study_policy import (
    build_self_study_consumer_context,
    build_self_study_consumer_request,
    resolve_self_study_consumer_policy,
)

RetrieveRunner = Callable[..., Awaitable[SelfStudyRetrieveResultV1]]


@dataclass(frozen=True)
class _ConsumerScenario:
    name: str
    consumer_name: str
    output_mode: str
    config: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_source() -> ServiceRef:
    return ServiceRef(name="orion-cortex-exec", version="0.2.0", node="self-study-harness")


def _counts_payload(result: SelfStudyRetrieveResultV1) -> dict[str, int]:
    return result.counts.model_dump(mode="json")


def _validate_retrieval_boundaries(
    result: SelfStudyRetrieveResultV1,
    *,
    allowed_tiers: Sequence[str],
    allowed_source_kinds: Sequence[str],
    required_tiers: Sequence[str] = (),
) -> list[str]:
    violations: list[str] = []
    allowed_tier_set = set(allowed_tiers)
    allowed_source_set = set(allowed_source_kinds)
    present_tiers: set[str] = set()

    for group in result.groups:
        if group.trust_tier not in allowed_tier_set:
            violations.append(f"unexpected_group_tier:{group.trust_tier}")
        for item in group.items:
            present_tiers.add(item.trust_tier)
            if item.trust_tier not in allowed_tier_set:
                violations.append(f"unexpected_item_tier:{item.stable_id}:{item.trust_tier}")
            if item.source_kind not in allowed_source_set:
                violations.append(f"unexpected_source_kind:{item.stable_id}:{item.source_kind}")
            if result.retrieval_mode == "factual" and item.record_type != "fact":
                violations.append(f"factual_mode_non_fact:{item.stable_id}:{item.record_type}")
            if item.trust_tier == TRUST_TIER and not item.source_path:
                violations.append(f"missing_authoritative_provenance:{item.stable_id}")
            if item.trust_tier == INDUCED_TRUST_TIER:
                if item.record_type != "concept":
                    violations.append(f"induced_non_concept:{item.stable_id}:{item.record_type}")
                if not item.evidence:
                    violations.append(f"induced_missing_evidence:{item.stable_id}")
                if any(ref.trust_tier != TRUST_TIER for ref in item.evidence):
                    violations.append(f"induced_evidence_upcast:{item.stable_id}")
            if item.trust_tier == REFLECTIVE_TRUST_TIER:
                if item.record_type != "reflection":
                    violations.append(f"reflective_non_reflection:{item.stable_id}:{item.record_type}")
                if not item.evidence:
                    violations.append(f"reflective_missing_evidence:{item.stable_id}")
                if not item.concept_refs:
                    violations.append(f"reflective_missing_concept_refs:{item.stable_id}")
                if any(ref.trust_tier != TRUST_TIER for ref in item.evidence):
                    violations.append(f"reflective_evidence_upcast:{item.stable_id}")
                if any(ref.trust_tier != INDUCED_TRUST_TIER for ref in item.concept_refs):
                    violations.append(f"reflective_concept_ref_upcast:{item.stable_id}")

    for tier in required_tiers:
        if tier not in present_tiers:
            violations.append(f"missing_required_tier:{tier}")

    if result.retrieval_mode == "factual" and result.counts.induced:
        violations.append("factual_counts_include_induced")
    if result.retrieval_mode == "factual" and result.counts.reflective:
        violations.append("factual_counts_include_reflective")
    if result.retrieval_mode == "conceptual" and result.counts.reflective:
        violations.append("conceptual_counts_include_reflective")
    if result.retrieval_mode == "conceptual" and result.counts.reflections:
        violations.append("conceptual_counts_include_reflections")
    if result.retrieval_mode == "factual" and result.counts.concepts:
        violations.append("factual_counts_include_concepts")
    if result.retrieval_mode == "factual" and result.counts.reflections:
        violations.append("factual_counts_include_reflections")

    return violations


async def _run_consumer_scenario(
    *,
    scenario: _ConsumerScenario,
    source: ServiceRef,
    retrieve_runner: RetrieveRunner,
) -> tuple[SelfStudyHarnessScenarioResultV1, SelfStudyConsumerContextV1]:
    decision = resolve_self_study_consumer_policy(
        consumer_name=scenario.consumer_name,
        output_mode=scenario.output_mode,
        config=scenario.config,
    )
    notes = [f"policy_reason={decision.policy_reason}"]
    degradation_notes: list[str] = []
    result = None

    if decision.enabled:
        request = build_self_study_consumer_request(decision, scenario.config)
        try:
            result = await retrieve_runner(
                request=request,
                bus=None,
                source=source,
                correlation_id=f"harness-{scenario.name}-{uuid4()}",
            )
            notes.append(f"retrieval_mode={request.retrieval_mode}")
        except Exception as exc:
            degradation_notes.append(f"self_study_unavailable:{exc}")
    else:
        degradation_notes.append(f"self_study_disabled:{decision.policy_reason}")

    context = build_self_study_consumer_context(decision, result=result, notes=[*notes, *degradation_notes])
    violations: list[str] = []
    counts: dict[str, int] = {}
    if result is not None:
        counts = _counts_payload(result)
        violations.extend(
            _validate_retrieval_boundaries(
                result,
                allowed_tiers=decision.allowed_trust_tiers,
                allowed_source_kinds=("self_repo_inspect", "self_concept_induce", "self_concept_reflect"),
                required_tiers=[decision.allowed_trust_tiers[-1]] if decision.allowed_trust_tiers else [],
            )
        )
        if context.retrieval_mode != result.retrieval_mode:
            violations.append(
                f"consumer_context_mode_mismatch:{context.retrieval_mode}:{result.retrieval_mode}"
            )

    scenario_result = SelfStudyHarnessScenarioResultV1(
        name=scenario.name,
        success=decision.enabled and result is not None and not violations,
        retrieval_mode=context.retrieval_mode,
        consumer_name=context.consumer_name,
        consumer_kind=context.consumer_kind,
        summary_counts=counts,
        boundary_violations=violations,
        degradation_notes=degradation_notes,
        notes=list(context.notes),
    )
    return scenario_result, context


async def run_self_study_harness(
    *,
    source: ServiceRef | None = None,
    retrieve_runner: RetrieveRunner = run_self_retrieve,
    include_degraded: bool = True,
    soak_iterations: int = 1,
) -> SelfStudyHarnessResultV1:
    source_ref = source or _default_source()
    timestamp = _utc_now()
    run_id = f"self-study-harness-{uuid4()}"
    scenarios: list[SelfStudyHarnessScenarioResultV1] = []

    repo_result = await run_self_repo_inspect(bus=None, source=source_ref, correlation_id=f"{run_id}:inspect")
    factual_scan_violations: list[str] = []
    if repo_result.snapshot.trust_tier != TRUST_TIER:
        factual_scan_violations.append(f"snapshot_trust_tier_invalid:{repo_result.snapshot.trust_tier}")
    if any(item.trust_tier != TRUST_TIER for item in repo_result.snapshot.services):
        factual_scan_violations.append("service_snapshot_non_authoritative")
    scenarios.append(
        SelfStudyHarnessScenarioResultV1(
            name="factual_self_study",
            success=not factual_scan_violations,
            summary_counts=repo_result.snapshot.counts.model_dump(mode="json"),
            boundary_violations=factual_scan_violations,
            notes=[repo_result.summary],
        )
    )

    concept_result = await run_self_concept_induce(bus=None, source=source_ref, correlation_id=f"{run_id}:induce")
    concept_violations: list[str] = []
    if not concept_result.concepts:
        concept_violations.append("no_induced_concepts")
    for concept in concept_result.concepts:
        if concept.trust_tier != INDUCED_TRUST_TIER:
            concept_violations.append(f"concept_trust_invalid:{concept.concept_id}:{concept.trust_tier}")
        if not concept.evidence:
            concept_violations.append(f"concept_missing_evidence:{concept.concept_id}")
        if any(ref.trust_tier != TRUST_TIER for ref in concept.evidence):
            concept_violations.append(f"concept_evidence_upcast:{concept.concept_id}")
    scenarios.append(
        SelfStudyHarnessScenarioResultV1(
            name="concept_induction",
            success=not concept_violations,
            summary_counts={"concepts": len(concept_result.concepts)},
            boundary_violations=concept_violations,
            notes=[concept_result.summary],
        )
    )

    reflection_result = await run_self_concept_reflect(bus=None, source=source_ref, correlation_id=f"{run_id}:reflect")
    reflection_violations: list[str] = []
    if not reflection_result.findings:
        reflection_violations.append("no_reflective_findings")
    for finding in reflection_result.findings:
        if finding.trust_tier != REFLECTIVE_TRUST_TIER:
            reflection_violations.append(
                f"reflection_trust_invalid:{finding.reflection_id}:{finding.trust_tier}"
            )
        if any(ref.trust_tier != TRUST_TIER for ref in finding.evidence):
            reflection_violations.append(f"reflection_evidence_upcast:{finding.reflection_id}")
        if any(ref.trust_tier != INDUCED_TRUST_TIER for ref in finding.concept_refs):
            reflection_violations.append(f"reflection_concept_upcast:{finding.reflection_id}")
    scenarios.append(
        SelfStudyHarnessScenarioResultV1(
            name="reflection",
            success=not reflection_violations,
            summary_counts={"findings": len(reflection_result.findings)},
            boundary_violations=reflection_violations,
            notes=[reflection_result.summary, reflection_result.validation_summary],
        )
    )

    retrieval_specs = (
        ("factual_retrieval", "factual", [TRUST_TIER], ["self_repo_inspect"], [TRUST_TIER], 12),
        (
            "conceptual_retrieval",
            "conceptual",
            [TRUST_TIER, INDUCED_TRUST_TIER],
            ["self_repo_inspect", "self_concept_induce"],
            [TRUST_TIER, INDUCED_TRUST_TIER],
            18,
        ),
        (
            "reflective_retrieval",
            "reflective",
            [TRUST_TIER, INDUCED_TRUST_TIER, REFLECTIVE_TRUST_TIER],
            ["self_repo_inspect", "self_concept_induce", "self_concept_reflect"],
            [TRUST_TIER, INDUCED_TRUST_TIER, REFLECTIVE_TRUST_TIER],
            24,
        ),
    )
    for name, mode, allowed_tiers, source_kinds, required_tiers, limit in retrieval_specs:
        request = SelfStudyRetrieveRequestV1.model_validate(
            {"retrieval_mode": mode, "filters": {"limit": limit}}
        )
        result = await retrieve_runner(
            request=request,
            bus=None,
            source=source_ref,
            correlation_id=f"{run_id}:{name}",
        )
        violations = _validate_retrieval_boundaries(
            result,
            allowed_tiers=allowed_tiers,
            allowed_source_kinds=source_kinds,
            required_tiers=required_tiers,
        )
        scenarios.append(
            SelfStudyHarnessScenarioResultV1(
                name=name,
                success=not violations,
                retrieval_mode=result.retrieval_mode,
                summary_counts=_counts_payload(result),
                boundary_violations=violations,
                notes=list(result.notes),
            )
        )

    consumer_specs = (
        _ConsumerScenario(
            name="factual_consumer",
            consumer_name="legacy.plan",
            output_mode="implementation_guide",
            config={"enabled": True, "retrieval_mode": "reflective", "filters": {"limit": 12}},
        ),
        _ConsumerScenario(
            name="conceptual_consumer",
            consumer_name="legacy.plan",
            output_mode="project_planning",
            config={"enabled": True, "retrieval_mode": "conceptual", "filters": {"limit": 18}},
        ),
        _ConsumerScenario(
            name="reflective_consumer",
            consumer_name="actions.respond_to_juniper_collapse_mirror.v1",
            output_mode="reflective_depth",
            config={"enabled": True, "retrieval_mode": "reflective", "filters": {"limit": 24}},
        ),
    )
    for consumer_scenario in consumer_specs:
        scenario_result, _context = await _run_consumer_scenario(
            scenario=consumer_scenario,
            source=source_ref,
            retrieve_runner=retrieve_runner,
        )
        scenarios.append(scenario_result)

    if include_degraded:
        async def _failing_retrieve_runner(*args: Any, **kwargs: Any) -> SelfStudyRetrieveResultV1:
            raise RuntimeError("backend_down")

        degraded_scenario, _context = await _run_consumer_scenario(
            scenario=_ConsumerScenario(
                name="degraded_consumer_backend",
                consumer_name="legacy.plan",
                output_mode="project_planning",
                config={"enabled": True, "retrieval_mode": "conceptual", "filters": {"limit": 6}},
            ),
            source=source_ref,
            retrieve_runner=_failing_retrieve_runner,
        )
        degraded_success = degraded_scenario.success or any(
            note.startswith("self_study_unavailable:") for note in degraded_scenario.degradation_notes
        )
        scenarios.append(
            degraded_scenario.model_copy(
                update={
                    "success": degraded_success,
                    "boundary_violations": [] if degraded_success else degraded_scenario.boundary_violations,
                }
            )
        )

    soak_result: SelfStudyHarnessSoakResultV1 | None = None
    if soak_iterations > 1:
        signatures: list[tuple[tuple[str, ...], tuple[tuple[str, int], ...]]] = []
        duplicate_notes: list[str] = []
        drift_notes: list[str] = []
        for iteration in range(1, soak_iterations + 1):
            result = await retrieve_runner(
                request=SelfStudyRetrieveRequestV1.model_validate(
                    {"retrieval_mode": "reflective", "filters": {"limit": 24}}
                ),
                bus=None,
                source=source_ref,
                correlation_id=f"{run_id}:soak:{iteration}",
            )
            ids = tuple(
                item.stable_id
                for group in result.groups
                for item in group.items
            )
            counts = tuple(sorted(_counts_payload(result).items()))
            signatures.append((ids, counts))
        baseline_ids, baseline_counts = signatures[0]
        for iteration, (ids, counts) in enumerate(signatures[1:], start=2):
            if ids != baseline_ids:
                drift_notes.append(f"stable_id_drift:iteration_{iteration}")
            if counts != baseline_counts:
                duplicate_notes.append(f"count_drift:iteration_{iteration}")
        soak_result = SelfStudyHarnessSoakResultV1(
            enabled=True,
            iterations=soak_iterations,
            warnings=[*duplicate_notes, *drift_notes],
            duplicate_notes=duplicate_notes,
            drift_notes=drift_notes,
        )

    passed = sum(1 for scenario in scenarios if scenario.success)
    failed = len(scenarios) - passed
    warning_count = sum(1 for scenario in scenarios if scenario.degradation_notes) + len(soak_result.warnings if soak_result else [])

    return SelfStudyHarnessResultV1(
        run_id=run_id,
        timestamp=timestamp,
        scenarios_run=scenarios,
        soak=soak_result,
        summary=SelfStudyHarnessSummaryV1(
            passed=passed,
            failed=failed,
            warnings=warning_count,
            duplicate_notes=list(soak_result.duplicate_notes if soak_result else []),
            drift_notes=list(soak_result.drift_notes if soak_result else []),
        ),
    )


def render_self_study_harness(result: SelfStudyHarnessResultV1) -> str:
    lines = [
        f"Self-study harness run_id={result.run_id}",
        f"timestamp={result.timestamp}",
    ]
    for scenario in result.scenarios_run:
        status = "PASS" if scenario.success else "FAIL"
        descriptor: list[str] = [scenario.name, status]
        if scenario.retrieval_mode:
            descriptor.append(f"mode={scenario.retrieval_mode}")
        if scenario.consumer_name:
            descriptor.append(f"consumer={scenario.consumer_name}")
        lines.append(" - " + " ".join(descriptor))
        if scenario.summary_counts:
            rendered_counts = ", ".join(f"{key}={value}" for key, value in sorted(scenario.summary_counts.items()))
            lines.append(f"   counts: {rendered_counts}")
        if scenario.boundary_violations:
            lines.append(f"   boundary_violations: {', '.join(scenario.boundary_violations)}")
        if scenario.degradation_notes:
            lines.append(f"   degradation: {', '.join(scenario.degradation_notes)}")
    if result.soak and result.soak.enabled:
        lines.append(
            f" soak: iterations={result.soak.iterations} warnings={len(result.soak.warnings)}"
        )
        if result.soak.warnings:
            lines.append(f"   soak_notes: {', '.join(result.soak.warnings)}")
    lines.append(
        f"summary: passed={result.summary.passed} failed={result.summary.failed} warnings={result.summary.warnings}"
    )
    return "\n".join(lines)
