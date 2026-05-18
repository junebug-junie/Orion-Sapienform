from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


_VALID_STANCE = {
    "conversation_frame": "playful_relational",
    "task_mode": "playful_exchange",
    "identity_salience": "low",
    "user_intent": "User shares a life moment with Amanda.",
    "self_relevance": "Receive warmly; do not over-interpret.",
    "juniper_relevance": "Honor the shared moment without inventing facts.",
    "answer_strategy": "WarmDirectAnswer",
    "stance_summary": "Respond to a casual relational turn about watching a show.",
    "response_priorities": ["receive warmly", "stay concise"],
    "response_hazards": ["do not invent show details"],
    "active_relationship_facets": ["shared life moment"],
    "reflective_themes": ["evening plans"],
}


def _semantic_payload(*, claim_label: str, evidence_ref: str = "current_turn:0") -> dict:
    return {
        "schema_version": "mind.semantic_synthesis.v1",
        "model_id": "quick",
        "extraction_mode": "llm",
        "claims": [
            {
                "claim_id": "c1",
                "label": claim_label,
                "summary": "User is sharing a casual relational moment.",
                "claim_kind": "relationship_claim",
                "evidence_refs": [evidence_ref],
                "source_kinds": ["current_turn"],
                "anchor": "relationship",
                "confidence": 0.9,
                "salience_hint": 0.8,
                "recommended_effect": "receive_warmly",
            }
        ],
        "suppressed": [],
        "diagnostics": {"evidence_item_count": 1, "llm_ok": True},
    }


def _frontier_payload() -> dict:
    return {
        "schema_version": "mind.active_cognitive_frontier.v1",
        "model_id": "metacog",
        "selected": [
            {
                "matter_id": "m1",
                "source_claim_id": "c1",
                "label": "shared evening moment",
                "summary": "User mentions watching a show with Amanda.",
                "matter_kind": "relationship_opportunity",
                "score": 0.88,
                "features": {"current_turn_relevance": 0.9, "confidence": 0.85},
                "evidence_refs": ["current_turn:0"],
                "reason_selected": "directly grounded in current turn",
                "recommended_effect": "receive_warmly",
            }
        ],
        "deferred": [],
        "suppressed": [],
        "hazards": ["do not invent show details"],
        "response_directives": ["answer the user turn first"],
        "diagnostics": {"selected_count": 1, "llm_ok": True},
    }


@pytest.fixture(autouse=True)
def _prep() -> None:
    _mind_prep()


def test_evidence_pack_includes_projection_autonomy_and_recall() -> None:
    from app.evidence import build_evidence_pack

    pack = build_evidence_pack(
        {
            "user_text": "hello there",
            "messages_tail": [{"role": "user", "content": "hello there"}],
            "facets": {
                "recall_bundle": {"fragments": [{"snippet": "prior bike ride", "source": "journal"}]},
                "cognitive_projection": {
                    "anchors": {
                        "relationship": {
                            "items": [
                                {
                                    "label": "bike outing memory",
                                    "summary": "Earlier mention of a bike ride.",
                                    "salience": 0.7,
                                }
                            ]
                        }
                    }
                },
                "autonomy_compact": {"attention_items": [{"summary": "notice user energy"}]},
                "social_compact": {"social_turn_policy": {"mode": "warm"}},
            },
        }
    )
    kinds = {item.source_kind for item in pack.items}
    assert "current_turn" in kinds
    assert "recall_fragment" in kinds
    assert "cognitive_projection" in kinds
    assert "autonomy_compact" in kinds
    assert "social_compact" in kinds


def test_semantic_parser_rejects_missing_evidence_refs() -> None:
    from app.evidence import build_evidence_pack
    from app.guardrails import filter_semantic_claims
    from orion.mind.synthesis_v1 import SemanticSynthesisV1

    pack = build_evidence_pack({"user_text": "hi"})
    synthesis = SemanticSynthesisV1.model_validate(
        _semantic_payload(claim_label="user greeting", evidence_ref="missing:99")
    )
    filtered = filter_semantic_claims(synthesis, pack)
    assert filtered.claims == []
    assert any(s.reason == "unsupported_or_weak" for s in filtered.suppressed)


def test_source_tag_labels_suppressed() -> None:
    from app.evidence import build_evidence_pack, is_source_tag_label
    from app.guardrails import filter_semantic_claims
    from orion.mind.synthesis_v1 import SemanticSynthesisV1

    assert is_source_tag_label("identity_yaml") is True
    pack = build_evidence_pack({"user_text": "hi"})
    synthesis = SemanticSynthesisV1.model_validate(
        _semantic_payload(claim_label="identity_yaml", evidence_ref="current_turn:0")
    )
    filtered = filter_semantic_claims(synthesis, pack)
    assert filtered.claims == []
    assert any(s.reason == "source_tag_not_semantic" for s in filtered.suppressed)


def test_authorization_rejects_contract_only_handoff() -> None:
    from app.guardrails import evaluate_stance_authorization
    from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1

    synthesis = SemanticSynthesisV1.model_validate(_semantic_payload(claim_label="evening plan"))
    frontier = ActiveCognitiveFrontierV1.model_validate(_frontier_payload())
    payload = dict(_VALID_STANCE)
    payload["stance_summary"] = "contract_only: no meaningful Mind synthesis produced."
    payload["self_relevance"] = "contract_only seed"
    authorized, reasons, quality = evaluate_stance_authorization(
        synthesis=synthesis,
        frontier=frontier,
        stance_payload=payload,
        llm_errors=[],
    )
    assert authorized is False
    assert quality == "invalid_handoff"
    assert "contract_or_boilerplate_dominated" in reasons


def test_authorization_accepts_meaningful_synthesis() -> None:
    from app.guardrails import evaluate_stance_authorization
    from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1

    synthesis = SemanticSynthesisV1.model_validate(
        _semantic_payload(claim_label="shared evening moment")
    )
    frontier = ActiveCognitiveFrontierV1.model_validate(_frontier_payload())
    authorized, reasons, quality = evaluate_stance_authorization(
        synthesis=synthesis,
        frontier=frontier,
        stance_payload=dict(_VALID_STANCE),
        llm_errors=[],
    )
    assert authorized is True
    assert quality == "meaningful_synthesis"
    assert "valid_chat_stance_brief" in reasons


def test_mocked_mind_run_returns_meaningful_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunRequestV1, MindRunPolicyV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    set_llm_client_override(
        FakeMindLLMClient(
            [
                _semantic_payload(claim_label="shared evening moment"),
                _frontier_payload(),
                {"stance_payload": dict(_VALID_STANCE)},
            ]
        )
    )
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "going to watch a show with Amanda"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    assert result.ok is True
    assert result.mind_quality == "meaningful_synthesis"
    assert result.brief.mind_authorized_for_stance_skip is True
    assert result.brief.machine_contract.get("mind.authorized_for_stance_use") is True


def test_llm_failure_falls_back_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunRequestV1, MindRunPolicyV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    set_llm_client_override(FakeMindLLMClient([]))
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "hello"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    assert result.mind_quality == "fallback_contract_only"
    assert result.brief.mind_authorized_for_stance_skip is False
    machine = result.brief.machine_contract
    assert machine.get("mind.llm_synthesis_enabled") is True
    assert machine.get("mind.llm_synthesis_attempted") is True
    assert machine.get("mind.llm_fail_open_to_deterministic") is True
    assert machine.get("mind.llm_synthesis_failed_phase") == "semantic_synthesis"
    assert machine.get("mind.llm_synthesis_error_code") == "semantic_synthesis_failed"
    assert machine.get("mind.llm_synthesis_error") == "fake_exhausted"
    assert machine.get("mind.authorized_for_stance_skip") is False
    assert machine.get("mind.authorized_for_stance_use") is False
    telemetry = machine.get("mind.phase_telemetry") or []
    assert any(t.get("phase_name") == "semantic_synthesis" for t in telemetry)
    assert any(d.startswith("llm_fail_open:") for d in result.diagnostics)


def test_source_tag_semantic_fail_open_preserves_phase_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunRequestV1, MindRunPolicyV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    set_llm_client_override(
        FakeMindLLMClient([_semantic_payload(claim_label="identity_yaml", evidence_ref="current_turn:0")])
    )
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "hello"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    machine = result.brief.machine_contract
    assert machine.get("mind.llm_synthesis_attempted") is True
    assert machine.get("mind.llm_synthesis_failed_phase") == "semantic_synthesis"
    assert machine.get("mind.llm_synthesis_error_code") == "semantic_synthesis_failed"
    telemetry = machine.get("mind.phase_telemetry") or []
    sem = next(t for t in telemetry if t.get("phase_name") == "semantic_synthesis")
    assert sem.get("ok") is True
    assert sem.get("validation_ok") is False
    assert result.mind_quality == "fallback_contract_only"


def test_legacy_semantic_claim_shape_is_normalized() -> None:
    from app.evidence import build_evidence_pack
    from app.synthesis import run_semantic_synthesis, try_normalize_legacy_semantic_raw
    from orion.mind.synthesis_v1 import SemanticSynthesisV1

    legacy_raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [
            {
                "claim": "User shares a casual relational moment.",
                "evidence_refs": ["current_turn:0"],
            }
        ],
        "suppressed": [],
    }
    normalized, did = try_normalize_legacy_semantic_raw(legacy_raw)
    assert did is True
    assert normalized is not None
    synthesis = SemanticSynthesisV1.model_validate(normalized)
    assert synthesis.claims[0].claim_id.startswith("legacy_")
    assert synthesis.claims[0].claim_kind == "situation_claim"
    assert synthesis.diagnostics.notes[-1] == "normalized_from_legacy_claim_shape"

    pack = build_evidence_pack({"user_text": "watching a show with Amanda"})

    class _Client:
        def request_json(self, **kwargs):  # type: ignore[no-untyped-def]
            return legacy_raw, None, {"model_used": "quick"}

    result, err, telemetry = run_semantic_synthesis(
        pack,
        client=_Client(),  # type: ignore[arg-type]
        route="quick",
        model_id="quick",
        max_tokens=512,
    )
    assert err is None
    assert result is not None
    assert result.claims
    assert telemetry.validation_ok is True
    assert telemetry.ok is True
    assert "normalized_from_legacy_claim_shape" in result.diagnostics.notes


def test_unrecoverable_semantic_shape_fail_opens_with_schema_telemetry() -> None:
    from app.evidence import build_evidence_pack
    from app.synthesis import run_semantic_synthesis

    pack = build_evidence_pack({"user_text": "hello"})

    class _Client:
        def request_json(self, **kwargs):  # type: ignore[no-untyped-def]
            return {"schema_version": "mind.semantic_synthesis.v1", "claims": [{"foo": "bar"}]}, None, {}

    result, err, telemetry = run_semantic_synthesis(
        pack,
        client=_Client(),  # type: ignore[arg-type]
        route="quick",
        model_id="quick",
        max_tokens=512,
    )
    assert result is None
    assert err == "semantic_schema_invalid"
    assert telemetry.parse_ok is True
    assert telemetry.validation_ok is False
    assert telemetry.ok is False
    assert telemetry.status == "schema_invalid"
    assert telemetry.error == "semantic_schema_invalid"


def test_pre_llm_budget_fail_open_has_attempted_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.budget import MindRunBudget
    from app.engine import run_mind
    from app.settings import settings
    from orion.mind.v1 import MindRunRequestV1, MindRunPolicyV1

    real_over_budget = MindRunBudget.over_budget
    calls = {"n": 0}

    def _over_budget_first_check_only(self: MindRunBudget) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return True
        return real_over_budget(self)

    monkeypatch.setattr(MindRunBudget, "over_budget", _over_budget_first_check_only)
    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "hello"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    machine = result.brief.machine_contract
    assert machine.get("mind.llm_synthesis_attempted") is True
    assert machine.get("mind.llm_synthesis_failed_phase") == "pre_llm"
    assert machine.get("mind.llm_synthesis_error_code") == "loop_budget_exceeded"


def test_disabled_llm_path_has_no_attempted_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.settings import settings
    from orion.mind.v1 import MindRunRequestV1, MindRunPolicyV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", False)
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "hello"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    machine = result.brief.machine_contract
    assert "mind.llm_synthesis_attempted" not in machine
    assert machine.get("mind.llm_synthesis_enabled") is not True
    assert not any(d.startswith("llm_fail_open:") for d in result.diagnostics)
