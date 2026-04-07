from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.endogenous_runtime import (
    EndogenousRuntimeAdoptionService,
    EndogenousRuntimeConfig,
    apply_calibration_adoption,
    compare_endogenous_runtime_profile_outcomes,
    inspect_calibration_profile_audit,
    inspect_calibration_profile_audit_with_source,
    inspect_calibration_profile_state_with_source,
    inspect_calibration_profiles,
    inspect_endogenous_runtime_records_with_source,
    inspect_endogenous_operator_debug_surface,
    resolve_runtime_calibration_profile,
    rollback_calibration_adoption,
    run_endogenous_offline_evaluation,
)
from app.endogenous_runtime_sql_reader import SqlReadResult
from app.endogenous_runtime_store import InMemoryEndogenousRuntimeRecordStore, JsonlEndogenousRuntimeRecordStore
from orion.core.schemas.calibration_adoption import (
    CalibrationAdoptionRequestV1,
    CalibrationProfileV1,
    CalibrationRollbackRequestV1,
    CalibrationRolloutScopeV1,
)
from orion.core.schemas.endogenous_runtime import EndogenousRuntimeQueryV1
from orion.core.schemas.endogenous_eval import EndogenousEvaluationRequestV1
from orion.core.schemas.mentor import MentorProposalItemV1, MentorResponseV1
from orion.core.schemas.reasoning import ClaimV1, ContradictionV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.reasoning.materializer import ReasoningMaterializer
from orion.reasoning.mentor_gateway import MentorGateway
from orion.reasoning.repository import InMemoryReasoningRepository
from orion.reasoning.calibration_profiles import SqlCalibrationProfileStore


class _Provider:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, request, *, context_packet: list[dict]) -> MentorResponseV1:
        self.calls += 1
        return MentorResponseV1(
            proposal_batch_id=f"batch-{self.calls}",
            mentor_provider=request.mentor_provider,
            mentor_model=request.mentor_model,
            task_type=request.task_type,
            proposals=[
                MentorProposalItemV1(
                    proposal_id=f"proposal-{self.calls}",
                    proposal_type="missing_evidence",
                    confidence=0.7,
                    rationale="bounded",
                )
            ],
        )


class _Bus:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, envelope) -> None:
        if self.fail:
            raise RuntimeError("bus_down")
        self.published.append((channel, envelope.kind))


class _SqlReader:
    def __init__(self, *, runtime_rows: list[dict] | None = None, audit_rows: list[dict] | None = None, source: str = "sql") -> None:
        self.runtime_rows = runtime_rows or []
        self.audit_rows = audit_rows or []
        self.source = source

    def runtime_records(self, **kwargs):
        return SqlReadResult(source=self.source, rows=list(self.runtime_rows), filters=kwargs, error=None)

    def calibration_audit(self, **kwargs):
        return SqlReadResult(source=self.source, rows=list(self.audit_rows), filters=kwargs, error=None)


def _config(**overrides) -> EndogenousRuntimeConfig:
    base = EndogenousRuntimeConfig(
        enabled=True,
        surface_chat_reflective_enabled=True,
        surface_operator_enabled=True,
        allowed_workflow_types=frozenset({"contradiction_review", "concept_refinement", "reflective_journal"}),
        allow_mentor_branch=False,
        sample_rate=1.0,
        max_actions=5,
        store_backend="memory",
        store_path="/tmp/orion_endogenous_runtime_records_test.jsonl",
        store_max_records=500,
    )
    return EndogenousRuntimeConfig(**{**base.__dict__, **overrides})


def _repo() -> InMemoryReasoningRepository:
    repo = InMemoryReasoningRepository()
    claim = ClaimV1(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        status="provisional",
        authority="local_inferred",
        confidence=0.7,
        salience=0.6,
        novelty=0.3,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance={"evidence_refs": ["ev:1"], "source_channel": "orion:test", "source_kind": "unit", "producer": "pytest"},
        claim_text="claim",
        claim_kind="assertion",
    )
    contradiction = ContradictionV1(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        status="proposed",
        authority="local_inferred",
        confidence=0.8,
        salience=0.8,
        novelty=0.2,
        risk_tier="medium",
        observed_at=datetime.now(timezone.utc),
        provenance={"evidence_refs": ["ev:2"], "source_channel": "orion:test", "source_kind": "unit", "producer": "pytest"},
        contradiction_type="evidence_conflict",
        severity="high",
        involved_artifact_ids=[claim.artifact_id, "artifact-x"],
        summary="conflict",
    )
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(source_family="manual", source_kind="unit", source_channel="orion:test", producer="pytest"),
            artifacts=[claim, contradiction],
        )
    )
    return repo


def _payload(**ctx_overrides):
    ctx = {
        "user_message": "please reflect on this contradiction",
        "correlation_id": "corr-1",
        "reasoning_subject_ref": "project:orion_sapienform",
        "reasoning_contradiction_refs": ["c-1"],
        "reasoning_repository": _repo(),
    }
    ctx.update(ctx_overrides)
    reasoning = {
        "request_id": "summary-1",
        "active_subject_refs": ["project:orion_sapienform"],
        "tensions": ["conflict-a", "conflict-b", "conflict-c", "conflict-d"],
        "hazards": ["unresolved_contradictions_present"],
        "fallback_recommended": False,
        "debug": {"suppressed_by_drift": 1},
    }
    reflective = {"tensions": ["pressure"]}
    autonomy = {"summary": {"active_tensions": ["continuity"], "top_drives": ["coherence"]}}
    concept = {"tension": ["fragmented"]}
    return ctx, reasoning, reflective, autonomy, concept


def _audit(result: dict) -> dict:
    return result["audit"]


def test_runtime_signal_builder_uses_reasoning_repository_and_lifecycle_defaults() -> None:
    service = EndogenousRuntimeAdoptionService(config=_config())
    ctx, reasoning, reflective, autonomy, concept = _payload()
    result = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert _audit(result)["status"] == "ok"
    record = result["record"]
    assert record["signal_digest"]["contradiction_count"] >= 1
    assert record["signal_digest"]["signal_sources"]["repository"] == "reasoning_repository"


def test_durable_jsonl_store_persists_across_service_instances(tmp_path) -> None:
    store_path = tmp_path / "runtime_records.jsonl"
    cfg = _config(store_backend="jsonl", store_path=str(store_path), store_max_records=100)

    service = EndogenousRuntimeAdoptionService(config=cfg)
    ctx, reasoning, reflective, autonomy, concept = _payload()
    result = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert _audit(result)["status"] == "ok"

    new_service = EndogenousRuntimeAdoptionService(config=cfg)
    rows = new_service.inspect_recent_records(limit=5)
    assert rows
    assert rows[0]["runtime_record_id"]


def test_persists_suppression_noop_and_failure_records(monkeypatch) -> None:
    store = InMemoryEndogenousRuntimeRecordStore()
    service = EndogenousRuntimeAdoptionService(config=_config(), store=store)
    ctx, reasoning, reflective, autonomy, concept = _payload()

    suppress = service.maybe_invoke(
        ctx=ctx,
        reasoning_summary=reasoning,
        reflective=reflective,
        autonomy=autonomy,
        concept=concept,
    )
    assert _audit(suppress)["status"] == "ok"

    noop_ctx, noop_reasoning, noop_reflective, noop_autonomy, noop_concept = _payload(user_message="reflect")
    noop_reasoning["tensions"] = []
    noop_reflective["tensions"] = []
    noop_autonomy["summary"] = {"active_tensions": [], "top_drives": []}
    noop_concept["tension"] = []
    noop = service.maybe_invoke(
        ctx=noop_ctx,
        reasoning_summary=noop_reasoning,
        reflective=noop_reflective,
        autonomy=noop_autonomy,
        concept=noop_concept,
    )
    assert _audit(noop)["status"] == "ok"

    def _explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.endogenous_runtime.EndogenousWorkflowOrchestrator.orchestrate_runtime", _explode)
    failed = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert _audit(failed)["status"] == "failed"

    rows = service.inspect_recent_records(limit=10)
    assert any(not row["execution_success"] for row in rows)


def test_durable_write_failure_does_not_break_main_path() -> None:
    class _BrokenStore:
        def write(self, record):
            raise RuntimeError("disk_offline")

        def list_recent(self, **kwargs):
            return []

    service = EndogenousRuntimeAdoptionService(config=_config(), store=_BrokenStore())
    ctx, reasoning, reflective, autonomy, concept = _payload()
    result = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert _audit(result)["status"] == "persist_failed"


def test_retrieval_filters_support_surface_workflow_outcome_subject_mentor_and_time() -> None:
    service = EndogenousRuntimeAdoptionService(config=_config())
    ctx, reasoning, reflective, autonomy, concept = _payload()
    service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)

    rows = service.inspect_recent_records(limit=5, invocation_surface="chat_reflective_lane")
    assert rows
    rows2 = service.inspect_recent_records(limit=5, workflow_type=rows[0]["decision"]["workflow_type"])
    assert rows2
    rows3 = service.inspect_recent_records(limit=5, outcome=rows[0]["decision"]["outcome"])
    assert rows3
    rows4 = service.inspect_recent_records(limit=5, subject_ref="project:orion_sapienform")
    assert rows4
    rows5 = service.inspect_recent_records(limit=5, mentor_invoked=False)
    assert rows5
    rows6 = service.inspect_recent_records(limit=5, created_after=datetime.now(timezone.utc) - timedelta(minutes=5))
    assert rows6


def test_store_selection_memory_and_jsonl_have_consistent_shape(tmp_path) -> None:
    ctx, reasoning, reflective, autonomy, concept = _payload()
    mem_service = EndogenousRuntimeAdoptionService(config=_config(store_backend="memory"))
    jsonl_service = EndogenousRuntimeAdoptionService(
        config=_config(store_backend="jsonl", store_path=str(tmp_path / "runtime.jsonl"), store_max_records=100)
    )

    mem = mem_service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    js = jsonl_service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)

    assert set(mem.keys()) == set(js.keys())
    assert set(mem["audit"].keys()) == set(js["audit"].keys())


def test_downstream_consumption_seam_is_read_only_and_bounded() -> None:
    service = EndogenousRuntimeAdoptionService(config=_config())
    ctx, reasoning, reflective, autonomy, concept = _payload()
    service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)

    query = EndogenousRuntimeQueryV1(limit=3, invocation_surface="chat_reflective_lane")
    feed = service.consume_for_reflective_review(query=query)
    assert feed
    assert len(feed) <= 3
    assert feed[0].runtime_record_id


def test_mentor_branch_can_be_disabled_independently() -> None:
    repo = InMemoryReasoningRepository()
    provider = _Provider()
    gateway = MentorGateway(repository=repo, materializer=ReasoningMaterializer(repo), provider=provider)
    service = EndogenousRuntimeAdoptionService(
        config=_config(
            allowed_workflow_types=frozenset({"mentor_critique"}),
            allow_mentor_branch=False,
        )
    )
    ctx, reasoning, reflective, autonomy, concept = _payload(endogenous_runtime_operator_review=True, user_message="operator")
    reasoning["tensions"] = []
    reasoning["fallback_recommended"] = True
    ctx["mentor_gateway"] = gateway

    result = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert _audit(result)["status"] in {"ok", "persist_failed"}
    assert _audit(result)["mentor_invoked"] is False
    assert provider.calls == 0


def test_store_impls_are_usable_directly(tmp_path) -> None:
    mem = InMemoryEndogenousRuntimeRecordStore()
    file_store = JsonlEndogenousRuntimeRecordStore(path=str(tmp_path / "records.jsonl"), max_records=100)
    assert mem.list_recent(limit=5) == []
    assert file_store.list_recent(limit=5) == []


def test_operator_offline_evaluation_output_is_structured(monkeypatch) -> None:
    service = EndogenousRuntimeAdoptionService(config=_config())
    ctx, reasoning, reflective, autonomy, concept = _payload()
    service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)

    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)
    out = run_endogenous_offline_evaluation(request=EndogenousEvaluationRequestV1(limit=50, min_sample_size=1))
    assert "result" in out and "report_markdown" in out and "profile_export" in out
    assert out["result"]["metrics"]["sample_size"] >= 1
    assert out["report_markdown"].startswith("# Endogenous Offline Evaluation")


def test_operator_offline_evaluation_fetches_wider_window_and_forwards_single_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Service:
        def inspect_recent_records(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: _Service())

    request = EndogenousEvaluationRequestV1(
        limit=20,
        min_sample_size=50,
        invocation_surfaces=["operator_review"],
        workflow_types=["reflective_journal"],
        outcomes=["trigger"],
    )
    out = run_endogenous_offline_evaluation(request=request)
    assert out["result"]["metrics"]["sample_size"] == 0
    assert captured["limit"] == 250
    assert captured["invocation_surface"] == "operator_review"
    assert captured["workflow_type"] == "reflective_journal"
    assert captured["outcome"] == "trigger"


def test_calibration_adoption_is_manual_scoped_and_reversible(monkeypatch) -> None:
    bus = _Bus()
    service = EndogenousRuntimeAdoptionService(config=_config(), event_bus=bus)
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)

    profile = CalibrationProfileV1(
        profile_id="phase12-profile-1",
        created_by="operator:qa",
        rationale="narrow canary",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.5"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )

    staged = apply_calibration_adoption(
        request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage", profile=profile)
    )
    assert staged["accepted"] is True
    assert staged["active_profile_id"] is None

    baseline = resolve_runtime_calibration_profile(invocation_surface="operator_review", ctx={"correlation_id": "corr-1"})
    assert baseline["mode"] == "baseline"

    activated = apply_calibration_adoption(
        request=CalibrationAdoptionRequestV1(
            action="activate", operator_id="operator:qa", rationale="activate", profile_id="phase12-profile-1"
        )
    )
    assert activated["accepted"] is True
    assert activated["active_profile_id"] == "phase12-profile-1"

    adopted = resolve_runtime_calibration_profile(
        invocation_surface="operator_review",
        ctx={"correlation_id": "corr-1", "endogenous_runtime_operator_review": True},
    )
    assert adopted["mode"] == "adopted"

    out_of_scope = resolve_runtime_calibration_profile(invocation_surface="chat_reflective_lane", ctx={"correlation_id": "corr-1"})
    assert out_of_scope["mode"] == "baseline"

    rolled = rollback_calibration_adoption(
        request=CalibrationRollbackRequestV1(operator_id="operator:qa", rationale="rollback", rollback_mode="to_baseline")
    )
    assert rolled["accepted"] is True
    assert rolled["active_profile_id"] is None

    profiles = inspect_calibration_profiles(include_rolled_back=True)
    assert any(item["profile_id"] == "phase12-profile-1" for item in profiles)
    audit = inspect_calibration_profile_audit(limit=20)
    assert any(item["event_type"] == "activated" for item in audit)
    assert any(kind == "calibration.profile.audit.v1" for _, kind in bus.published)


def test_bus_sql_backend_publishes_runtime_events_and_falls_back_non_blocking() -> None:
    bus = _Bus()
    service = EndogenousRuntimeAdoptionService(config=_config(store_backend="bus_sql"), event_bus=bus)
    ctx, reasoning, reflective, autonomy, concept = _payload()
    result = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert _audit(result)["status"] == "ok"
    assert any(action.startswith("bus_publish_record:ok") for action in _audit(result)["actions"])
    assert any(kind == "endogenous.runtime.record.v1" for _, kind in bus.published)
    assert any(kind == "endogenous.runtime.audit.v1" for _, kind in bus.published)

    bad_bus = _Bus(fail=True)
    fallback_service = EndogenousRuntimeAdoptionService(config=_config(store_backend="bus_sql"), event_bus=bad_bus)
    fallback = fallback_service.maybe_invoke(
        ctx=ctx,
        reasoning_summary=reasoning,
        reflective=reflective,
        autonomy=autonomy,
        concept=concept,
    )
    assert _audit(fallback)["status"] == "ok"
    rows = fallback_service.inspect_recent_records(limit=5)
    assert rows
    assert any(action.startswith("bus_publish_record:failed") for action in _audit(fallback)["actions"])


def test_sql_read_path_is_preferred_and_exposes_source_metadata(monkeypatch) -> None:
    record = {
        "runtime_record_id": "r1",
        "invocation_surface": "operator_review",
        "decision": {"workflow_type": "reflective_journal", "outcome": "trigger"},
    }
    audit = {"audit_id": "a1", "event_type": "activated", "profile_id": "p1"}
    service = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_SqlReader(runtime_rows=[record], audit_rows=[audit], source="sql"))
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)

    runtime_out = inspect_endogenous_runtime_records_with_source(limit=5, invocation_surface="operator_review")
    assert runtime_out["source"] == "sql"
    assert runtime_out["count"] == 1

    audit_out = inspect_calibration_profile_audit_with_source(limit=5, profile_id="p1")
    assert audit_out["source"] == "sql"
    assert audit_out["count"] == 1


def test_sql_read_fallback_is_explicit_and_non_blocking(monkeypatch) -> None:
    service = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_SqlReader(source="sql_error"))
    ctx, reasoning, reflective, autonomy, concept = _payload()
    service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)

    runtime_out = inspect_endogenous_runtime_records_with_source(limit=5)
    assert runtime_out["fallback_used"] is True
    assert runtime_out["source"].startswith("fallback_local:")
    assert runtime_out["count"] >= 1


def test_before_after_comparison_uses_sql_backed_inputs(monkeypatch) -> None:
    treatment_rows = [{"decision": {"outcome": "trigger"}}, {"decision": {"outcome": "trigger"}}]
    baseline_rows = [{"decision": {"outcome": "noop"}}]

    class _Reader:
        def runtime_records(self, **kwargs):
            profile = kwargs.get("calibration_profile_id")
            rows = treatment_rows if profile == "p1" else baseline_rows
            return SqlReadResult(source="sql", rows=rows, filters=kwargs, error=None)

        def calibration_audit(self, **kwargs):
            return SqlReadResult(source="sql", rows=[], filters=kwargs, error=None)

    service = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_Reader())
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)
    out = compare_endogenous_runtime_profile_outcomes(profile_id="p1", limit=10)
    assert out["source"]["treatment"] == "sql"
    assert out["treatment"]["by_outcome"]["trigger"] == 2
    assert out["baseline"]["by_outcome"]["noop"] == 1


def test_runtime_resolution_uses_durable_sql_profile_state(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path / 'runtime_calibration_state.db'}"
    sql_store = SqlCalibrationProfileStore(database_url=db_url)
    profile = CalibrationProfileV1(
        profile_id="durable-p1",
        created_by="operator:qa",
        rationale="durable",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.5"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )
    sql_store.apply(CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage", profile=profile))
    sql_store.apply(
        CalibrationAdoptionRequestV1(action="activate", operator_id="operator:qa", rationale="activate", profile_id="durable-p1")
    )

    service = EndogenousRuntimeAdoptionService(config=_config(), calibration_profiles=sql_store)
    resolved = service.resolve_calibration_profile(
        invocation_surface="operator_review",
        ctx={"correlation_id": "corr-1", "endogenous_runtime_operator_review": True},
    )
    assert resolved["mode"] == "adopted"
    assert resolved["profile_id"] == "durable-p1"

    restarted = EndogenousRuntimeAdoptionService(config=_config(), calibration_profiles=SqlCalibrationProfileStore(database_url=db_url))
    resolved_again = restarted.resolve_calibration_profile(
        invocation_surface="operator_review",
        ctx={"correlation_id": "corr-1", "endogenous_runtime_operator_review": True},
    )
    assert resolved_again["mode"] == "adopted"
    assert resolved_again["profile_id"] == "durable-p1"


def test_operator_debug_surface_returns_unified_source_annotated_payload(monkeypatch) -> None:
    runtime_rows = [
        {"runtime_record_id": "r1", "decision": {"outcome": "trigger", "workflow_type": "reflective_journal"}, "mentor_invoked": False}
    ]
    audit_rows = [{"audit_id": "a1", "event_type": "activated", "profile_id": "p1"}]
    service = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_SqlReader(runtime_rows=runtime_rows, audit_rows=audit_rows, source="sql"))

    profile = CalibrationProfileV1(
        profile_id="p1",
        created_by="operator:qa",
        rationale="operator",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.6"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage", profile=profile))
    service.apply_calibration_adoption(
        request=CalibrationAdoptionRequestV1(action="activate", operator_id="operator:qa", rationale="activate", profile_id="p1")
    )
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)

    out = inspect_endogenous_operator_debug_surface(runtime_limit=10, audit_limit=10, profile_id="p1")
    assert out["surface"] == "operator_debug"
    assert out["runtime_records"]["source"] == "sql"
    assert out["runtime_records"]["count"] == 1
    assert out["calibration_state"]["active_profile"]["profile_id"] == "p1"
    assert out["calibration_state"]["profile"]["profile_id"] == "p1"
    assert out["calibration_audit"]["source"] == "sql"
    assert "baseline_vs_active" in out["comparisons"]


def test_calibration_profile_state_surface_exposes_active_and_staged(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path / 'phase16_state.db'}"
    sql_store = SqlCalibrationProfileStore(database_url=db_url)
    service = EndogenousRuntimeAdoptionService(config=_config(), calibration_profiles=sql_store)

    p1 = CalibrationProfileV1(
        profile_id="p1",
        created_by="operator:qa",
        rationale="first",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.5"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )
    p2 = p1.model_copy(update={"profile_id": "p2", "rationale": "second"})
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage1", profile=p1))
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="activate", operator_id="operator:qa", rationale="activate1", profile_id="p1"))
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage2", profile=p2))

    state = service.inspect_calibration_profile_state_with_source(profile_id="p2")
    assert state["source"] == "sql"
    assert state["active_profile"]["profile_id"] == "p1"
    assert any(item["profile_id"] == "p2" for item in state["staged_profiles"])
    assert state["profile"]["profile_id"] == "p2"


def test_operator_debug_previous_vs_current_and_insufficient_data(monkeypatch) -> None:
    service = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_SqlReader(runtime_rows=[], audit_rows=[], source="sql"))
    p1 = CalibrationProfileV1(
        profile_id="p1",
        created_by="operator:qa",
        rationale="first",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.5"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )
    p2 = p1.model_copy(update={"profile_id": "p2", "rationale": "second", "overrides": {"trigger.reflective_threshold": "0.65"}})
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage1", profile=p1))
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="activate", operator_id="operator:qa", rationale="activate1", profile_id="p1"))
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage2", profile=p2))
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="activate", operator_id="operator:qa", rationale="activate2", profile_id="p2"))
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)

    out = inspect_endogenous_operator_debug_surface()
    assert out["comparisons"]["previous_vs_current"]["baseline_profile_id"] == "p1"
    assert out["comparisons"]["previous_vs_current"]["profile_id"] == "p2"

    no_active = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_SqlReader(runtime_rows=[], audit_rows=[], source="sql"))
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: no_active)
    degraded = inspect_endogenous_operator_debug_surface()
    assert degraded["comparisons"]["baseline_vs_active"]["status"] == "insufficient_data"
    assert degraded["comparisons"]["previous_vs_current"]["status"] == "insufficient_data"


def test_operator_debug_read_failure_is_non_blocking(monkeypatch) -> None:
    service = EndogenousRuntimeAdoptionService(config=_config())
    ctx, reasoning, reflective, autonomy, concept = _payload()
    runtime = service.maybe_invoke(ctx=ctx, reasoning_summary=reasoning, reflective=reflective, autonomy=autonomy, concept=concept)
    assert runtime["audit"]["enabled"] is True

    class _ExplosiveService:
        def inspect_operator_debug_surface(self, **kwargs):
            raise RuntimeError("operator_read_failed")

    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: _ExplosiveService())
    out = inspect_endogenous_operator_debug_surface()
    assert out["fallback_used"] is True
    assert out["error"] == "operator_read_failed"


def test_inspection_surfaces_are_read_only_no_hidden_mutation(monkeypatch) -> None:
    service = EndogenousRuntimeAdoptionService(config=_config(), sql_reader=_SqlReader(runtime_rows=[], audit_rows=[], source="sql"))
    profile = CalibrationProfileV1(
        profile_id="readonly-p1",
        created_by="operator:qa",
        rationale="readonly",
        source_recommendation_ids=["endogenous-rec-1"],
        source_evaluation_request_id="endogenous-eval-1",
        overrides={"trigger.reflective_threshold": "0.5"},
        scope=CalibrationRolloutScopeV1(invocation_surfaces=["operator_review"], sample_percent=100, operator_only=True),
    )
    service.apply_calibration_adoption(request=CalibrationAdoptionRequestV1(action="stage", operator_id="operator:qa", rationale="stage", profile=profile))
    service.apply_calibration_adoption(
        request=CalibrationAdoptionRequestV1(action="activate", operator_id="operator:qa", rationale="activate", profile_id="readonly-p1")
    )
    before = service.inspect_calibration_profiles(include_rolled_back=True)
    monkeypatch.setattr("app.endogenous_runtime.runtime_service", lambda: service)

    inspect_calibration_profile_state_with_source(profile_id="readonly-p1")
    inspect_endogenous_operator_debug_surface(runtime_limit=5)
    after = service.inspect_calibration_profiles(include_rolled_back=True)
    assert before == after
