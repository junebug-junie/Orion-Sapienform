from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes
from orion.core.schemas.substrate_mutation import MutationDecisionV1, MutationPressureV1
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from scripts.mutation_cognition_context import build_mutation_cognition_context

FIXTURE_CANARY = HUB_ROOT / "tests" / "fixtures" / "recall_canary" / "golden_cases.json"


def _proposal_with_readiness(*, recommendation: str, gates: list[str]) -> tuple[SubstrateMutationStore, str]:
    store = SubstrateMutationStore()
    pressure = MutationPressureV1(
        pressure_id="pressure-stage-1",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_surface="recall_strategy_profile",
        pressure_kind="pressure_event:recall_miss_or_dissatisfaction",
        pressure_score=7.0,
        evidence_refs=["recall_compare:a"],
        source_signal_ids=["sig-a"],
    )
    proposal = ProposalFactory().from_pressure(pressure)
    assert proposal is not None
    patch = dict(proposal.patch.patch)
    patch["recall_strategy_readiness"] = {"recommendation": recommendation, "gates_blocked": list(gates)}
    proposal = proposal.model_copy(update={"patch": proposal.patch.model_copy(update={"patch": patch})})
    store.record_pressure(pressure)
    store.add_proposal(proposal)
    store.record_decision(MutationDecisionV1(proposal_id=proposal.proposal_id, action="require_review", requires_operator_review=True))
    return store, proposal.proposal_id


def test_not_ready_recall_proposal_cannot_stage_without_override(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store, proposal_id = _proposal_with_readiness(recommendation="not_ready", gates=["insufficient_evidence"])
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    with pytest.raises(HTTPException) as exc:
        api_routes.api_substrate_mutation_runtime_recall_strategy_proposal_promote_to_staged_profile(
            proposal_id,
            api_routes.RecallProposalStageRequest(override=False),
            x_orion_operator_token="secret",
        )
    assert exc.value.status_code == 409


def test_override_requires_rationale(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store, proposal_id = _proposal_with_readiness(recommendation="not_ready", gates=["insufficient_evidence"])
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    with pytest.raises(HTTPException) as exc:
        api_routes.api_substrate_mutation_runtime_recall_strategy_proposal_promote_to_staged_profile(
            proposal_id,
            api_routes.RecallProposalStageRequest(override=True, operator_rationale=""),
            x_orion_operator_token="secret",
        )
    assert exc.value.status_code == 422


def test_ready_or_review_candidate_can_stage_and_activate_shadow(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store, proposal_id = _proposal_with_readiness(recommendation="review_candidate", gates=[])
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    stage = api_routes.api_substrate_mutation_runtime_recall_strategy_proposal_promote_to_staged_profile(
        proposal_id,
        api_routes.RecallProposalStageRequest(override=False, created_by="operator:test"),
        x_orion_operator_token="secret",
    )
    profile_id = stage["data"]["profile"]["profile_id"]
    assert stage["data"]["production_recall_mode"] == "v1"
    activated = api_routes.api_substrate_mutation_runtime_recall_strategy_profile_activate_shadow(
        profile_id,
        api_routes.RecallShadowActivateRequest(operator_rationale="shadow ramp"),
        x_orion_operator_token="secret",
    )
    assert activated["data"]["profile"]["status"] == "shadow_active"
    assert activated["data"]["production_recall_mode"] == "v1"


def test_only_one_shadow_active_profile_at_a_time(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    ids: list[str] = []
    for idx in (1, 2):
        staged = store.stage_recall_profile(
            profile=api_routes.RecallStrategyProfileV1(
                source_proposal_id=f"proposal-{idx}",
                source_pressure_ids=[],
                source_evidence_refs=[],
                readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
                strategy_kind="strategy_profile",
                recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
                anchor_policy_snapshot={},
                page_index_policy_snapshot={},
                graph_expansion_policy_snapshot={},
                created_by="operator",
                status="staged",
            )
        )
        ids.append(staged.profile_id)
    api_routes.api_substrate_mutation_runtime_recall_strategy_profile_activate_shadow(
        ids[0], api_routes.RecallShadowActivateRequest(), x_orion_operator_token="secret"
    )
    api_routes.api_substrate_mutation_runtime_recall_strategy_profile_activate_shadow(
        ids[1], api_routes.RecallShadowActivateRequest(), x_orion_operator_token="secret"
    )
    p1 = store.get_recall_strategy_profile(ids[0])
    p2 = store.get_recall_strategy_profile(ids[1])
    assert p1 is not None and p1.status == "staged"
    assert p2 is not None and p2.status == "shadow_active"


def test_shadow_eval_dry_run_does_not_persist_pressure_and_non_dry_run_records(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    profile = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-eval",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="staged",
        )
    )
    store.activate_recall_shadow_profile(profile.profile_id)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)

    recorded: list[str] = []

    def _fake_record(*, events, correlation_id, source_event_id, invocation_surface, ingest_notes):
        recorded.append(f"{source_event_id}:{len(events)}")

    monkeypatch.setattr(api_routes, "_record_pressure_events_as_telemetry", _fake_record)
    rows = [
        {
            "case_id": "c1",
            "query": "what changed in recall",
            "v1": {"selected_count": 1, "latency_ms": 100, "precision_proxy": 0.2},
            "v2": {"selected_count": 2, "latency_ms": 90, "precision_proxy": 0.6},
        }
    ]
    dry = api_routes.api_substrate_mutation_runtime_recall_shadow_profile_evaluate(
        api_routes.RecallShadowEvaluateRequest(dry_run=True, record_pressure_events=True, eval_rows=rows),
        x_orion_operator_token="secret",
    )
    assert dry["data"]["recorded_pressure_events"] == 0
    assert recorded == []
    before_refs = list((store.get_recall_strategy_profile(profile.profile_id) or profile).eval_evidence_refs)

    live = api_routes.api_substrate_mutation_runtime_recall_shadow_profile_evaluate(
        api_routes.RecallShadowEvaluateRequest(dry_run=False, record_pressure_events=True, eval_rows=rows),
        x_orion_operator_token="secret",
    )
    assert live["data"]["recorded_pressure_events"] == 1
    assert recorded
    updated = store.get_recall_strategy_profile(profile.profile_id)
    assert updated is not None
    assert updated.readiness_snapshot.get("recommendation") in {
        "not_ready",
        "review_candidate",
        "ready_for_shadow_expansion",
        "ready_for_operator_promotion",
    }
    assert len(updated.eval_evidence_refs) >= len(before_refs)
    assert live["data"]["production_recall_mode"] == "v1"
    assert store._adoptions == {}
    assert live["data"]["eval_run"]["status"] == "completed"
    listed_runs = api_routes.api_substrate_mutation_runtime_recall_shadow_eval_runs(limit=10, profile_id=profile.profile_id)
    assert listed_runs["data"]["runs"]
    run_id = live["data"]["eval_run"]["run_id"]
    detail = api_routes.api_substrate_mutation_runtime_recall_shadow_eval_run_detail(run_id)
    assert detail["data"]["run"]["run_id"] == run_id


def test_typed_recall_eval_settings_are_honored(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    profile = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-settings",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="staged",
        )
    )
    store.activate_recall_shadow_profile(profile.profile_id)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    monkeypatch.setattr(api_routes.settings, "HUB_RECALL_SHADOW_EVAL_DEFAULT_CORPUS_LIMIT", 2)
    monkeypatch.setattr(api_routes.settings, "HUB_RECALL_SHADOW_EVAL_MAX_ROWS_PER_RUN", 2)
    monkeypatch.setattr(api_routes.settings, "HUB_RECALL_SHADOW_EVAL_TIMEOUT_SEC", 3.5)
    monkeypatch.setattr(api_routes.settings, "HUB_RECALL_SERVICE_URL", "http://recall-shadow-svc:8090")
    monkeypatch.setattr(api_routes.settings, "RECALL_SERVICE_URL", "http://fallback:8090")

    observed: dict[str, object] = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "rows": [
                    {"case_id": "a", "v1": {"selected_count": 1, "latency_ms": 100, "precision_proxy": 0.2}, "v2": {"selected_count": 2, "latency_ms": 90, "precision_proxy": 0.5}},
                    {"case_id": "b", "v1": {"selected_count": 1, "latency_ms": 100, "precision_proxy": 0.2}, "v2": {"selected_count": 2, "latency_ms": 90, "precision_proxy": 0.5}},
                    {"case_id": "c", "v1": {"selected_count": 1, "latency_ms": 100, "precision_proxy": 0.2}, "v2": {"selected_count": 2, "latency_ms": 90, "precision_proxy": 0.5}},
                ]
            }

    def _fake_get(url: str, timeout: float):
        observed["url"] = url
        observed["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(api_routes.requests, "get", _fake_get)
    out = api_routes.api_substrate_mutation_runtime_recall_shadow_profile_evaluate(
        api_routes.RecallShadowEvaluateRequest(dry_run=True, record_pressure_events=False),
        x_orion_operator_token="secret",
    )
    assert observed["url"] == "http://recall-shadow-svc:8090/debug/recall/eval-suite"
    assert observed["timeout"] == 3.5
    assert out["data"]["eval_run"]["eval_row_count"] == 2
    assert out["data"]["eval_run"]["corpus_limit"] == 2


def test_recall_service_unavailable_returns_safe_no_side_effect_result(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    profile = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-down",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "review_candidate", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="staged",
        )
    )
    store.activate_recall_shadow_profile(profile.profile_id)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    monkeypatch.setattr(api_routes.settings, "HUB_RECALL_SERVICE_URL", "http://missing")
    monkeypatch.setattr(api_routes.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))

    out = api_routes.api_substrate_mutation_runtime_recall_shadow_profile_evaluate(
        api_routes.RecallShadowEvaluateRequest(dry_run=False, record_pressure_events=True),
        x_orion_operator_token="secret",
    )
    assert out["data"]["recorded_pressure_events"] == 0
    assert out["data"]["eval_run"]["status"] == "failed"
    assert out["data"]["profile"]["profile_id"] == profile.profile_id
    assert out["data"]["production_recall_mode"] == "v1"
    assert store._adoptions == {}


def test_production_candidate_review_requires_eval_history_unless_override(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    profile = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-review-gate",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "ready_for_shadow_expansion", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="staged",
        )
    )
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    with pytest.raises(HTTPException) as exc:
        api_routes.api_substrate_mutation_runtime_recall_strategy_profile_create_production_candidate_review(
            profile.profile_id,
            api_routes.RecallProductionCandidateReviewCreateRequest(override=False),
            x_orion_operator_token="secret",
        )
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc2:
        api_routes.api_substrate_mutation_runtime_recall_strategy_profile_create_production_candidate_review(
            profile.profile_id,
            api_routes.RecallProductionCandidateReviewCreateRequest(override=True, operator_rationale=""),
            x_orion_operator_token="secret",
        )
    assert exc2.value.status_code == 422


def test_ready_profile_can_create_production_candidate_review(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    profile = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-review-ok",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "ready_for_operator_promotion", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="shadow_active",
        )
    )
    store.record_recall_shadow_eval_run(
        api_routes.RecallShadowEvalRunV1(
            profile_id=profile.profile_id,
            dry_run=False,
            status="completed",
            eval_row_count=3,
            readiness_before={"recommendation": "review_candidate"},
            readiness_after={"recommendation": "ready_for_operator_promotion"},
        )
    )
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    out = api_routes.api_substrate_mutation_runtime_recall_strategy_profile_create_production_candidate_review(
        profile.profile_id,
        api_routes.RecallProductionCandidateReviewCreateRequest(override=False, created_by="operator:test"),
        x_orion_operator_token="secret",
    )
    assert out["data"]["review"]["profile_id"] == profile.profile_id
    assert out["data"]["production_recall_mode"] == "v1"
    assert out["data"]["recall_live_apply_enabled"] is False


def test_recall_profile_posture_appears_in_mutation_cognition_context() -> None:
    store = SubstrateMutationStore()
    staged = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-cx",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={
                "recommendation": "review_candidate",
                "gates_blocked": [],
                "corpus_coverage": 0.55,
                "irrelevant_cousin_rate": 0.1,
                "explainability_completeness": 0.8,
            },
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="staged",
        )
    )
    store.activate_recall_shadow_profile(staged.profile_id)
    store.record_recall_shadow_eval_run(
        api_routes.RecallShadowEvalRunV1(
            profile_id=staged.profile_id,
            dry_run=False,
            status="completed",
            eval_row_count=2,
        )
    )
    store.record_recall_production_candidate_review(
        api_routes.RecallProductionCandidateReviewV1(
            profile_id=staged.profile_id,
            source_eval_run_ids=[],
            readiness_snapshot={"recommendation": "ready_for_shadow_expansion"},
            recommendation="expand_shadow_corpus",
            status="draft",
        )
    )
    ctx = build_mutation_cognition_context(store=store)
    assert ctx["active_shadow_recall_profile_id"] == staged.profile_id
    assert ctx["recall_strategy_readiness_recommendation"] == "review_candidate"
    assert ctx["production_recall_mode"] == "v1"
    assert ctx["last_recall_shadow_eval_run_status"] == "completed"
    assert ctx["latest_production_candidate_review_recommendation"] == "expand_shadow_corpus"
    assert ctx["latest_production_candidate_review_status"] == "draft"
    assert ctx["recall_live_apply_enabled"] is False
    assert "live_surface" in ctx


def test_recall_canary_query_judgment_and_review_artifact_flow(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    staged = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-canary",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "ready_for_shadow_expansion", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="staged",
        )
    )
    store.activate_recall_shadow_profile(staged.profile_id)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    fixture_rows = json.loads(FIXTURE_CANARY.read_text(encoding="utf-8"))
    first = fixture_rows[0]

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "v1": {"bundle": {"items": [{"card_id": "v1-a"}]}},
                "v2": {"bundle": {"items": [{"card_id": "v2-a"}]}},
                "compare": dict(first["compare"]),
            }

    monkeypatch.setattr(api_routes.requests, "post", lambda *args, **kwargs: _Resp())
    query = api_routes.api_substrate_recall_canary_query(
        api_routes.RecallCanaryQueryRequest(query_text=str(first["query_text"])),
        x_orion_operator_token="secret",
    )
    run_id = query["data"]["canary_run_id"]
    assert query["data"]["schema_version"] == "recall_canary_query_result.v1"
    assert query["data"]["safety"]["promotion_performed"] is False
    assert query["data"]["safety"]["apply_performed"] is False
    emitted: list[str] = []

    def _fake_record(*, events, correlation_id, source_event_id, invocation_surface, ingest_notes):
        emitted.extend([event.pressure_event_id for event in events])

    monkeypatch.setattr(api_routes, "_record_pressure_events_as_telemetry", _fake_record)
    judgment = api_routes.api_substrate_recall_canary_run_judgment(
        run_id,
        api_routes.RecallCanaryJudgmentRequest(
            judgment="v2_better",
            failure_modes=["missing_exact_anchor"],
            should_emit_pressure=True,
            should_mark_review_candidate=True,
        ),
        x_orion_operator_token="secret",
    )
    assert judgment["judgment_recorded"] is True
    assert judgment["pressure_emitted"] is True
    assert emitted
    artifact = api_routes.api_substrate_recall_canary_create_review_artifact(
        run_id,
        api_routes.RecallCanaryReviewArtifactRequest(review_type="production_candidate_evidence"),
        x_orion_operator_token="secret",
    )
    assert artifact["review_artifact_created"] is True
    assert artifact["safety"]["production_default_unchanged"] is True
    assert artifact["safety"]["promotion_performed"] is False
    assert artifact["safety"]["apply_performed"] is False
    status = api_routes.api_substrate_recall_canary_status(limit=20)
    assert status["data"]["run_count"] >= 1
    assert status["data"]["review_artifact_count"] >= 1
    assert status["data"]["judgment_counts"]["v2_better"] >= 1


def test_recall_canary_review_artifact_warns_without_active_shadow(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    run = store.record_recall_canary_run(
        api_routes.RecallCanaryRunV1(
            profile_id=None,
            query_text="q",
            comparison_summary={},
            v1_summary={},
            v2_summary={},
        )
    )
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    out = api_routes.api_substrate_recall_canary_create_review_artifact(
        run.canary_run_id,
        api_routes.RecallCanaryReviewArtifactRequest(),
        x_orion_operator_token="secret",
    )
    assert out["review_artifact_created"] is False
    assert "no_active_shadow_profile" in out["warnings"]


def test_tripwire_recall_canary_endpoints_do_not_invoke_mutation_execute_cycle(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret")
    store = SubstrateMutationStore()
    staged = store.stage_recall_profile(
        profile=api_routes.RecallStrategyProfileV1(
            source_proposal_id="proposal-canary-tripwire",
            source_pressure_ids=[],
            source_evidence_refs=[],
            readiness_snapshot={"recommendation": "ready_for_shadow_expansion", "gates_blocked": []},
            strategy_kind="strategy_profile",
            recall_v2_config_snapshot={"profile": "recall.v2.shadow"},
            anchor_policy_snapshot={},
            page_index_policy_snapshot={},
            graph_expansion_policy_snapshot={},
            created_by="operator",
            status="shadow_active",
        )
    )
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    calls = {"mutation": 0, "review": 0}
    monkeypatch.setattr(api_routes, "_execute_substrate_mutation_cycle", lambda *a, **k: calls.__setitem__("mutation", calls["mutation"] + 1))
    monkeypatch.setattr(api_routes, "_execute_substrate_review_cycle", lambda *a, **k: calls.__setitem__("review", calls["review"] + 1))

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"v1": {"bundle": {"items": []}}, "v2": {"bundle": {"items": []}}, "compare": {}}

    monkeypatch.setattr(api_routes.requests, "post", lambda *args, **kwargs: _Resp())
    query = api_routes.api_substrate_recall_canary_query(
        api_routes.RecallCanaryQueryRequest(query_text="tripwire"),
        x_orion_operator_token="secret",
    )
    api_routes.api_substrate_recall_canary_run_judgment(
        query["data"]["canary_run_id"],
        api_routes.RecallCanaryJudgmentRequest(judgment="inconclusive", should_emit_pressure=False),
        x_orion_operator_token="secret",
    )
    api_routes.api_substrate_recall_canary_create_review_artifact(
        query["data"]["canary_run_id"],
        api_routes.RecallCanaryReviewArtifactRequest(),
        x_orion_operator_token="secret",
    )
    assert calls == {"mutation": 0, "review": 0}
    assert api_routes.api_substrate_autonomy_readiness()["recall"]["production_mode"] == "v1"
