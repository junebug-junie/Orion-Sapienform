"""Cortex-governance invariants for Mind LLM synthesis."""

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


def _frontier_payload(*, label: str = "shared evening moment") -> dict:
    return {
        "schema_version": "mind.active_cognitive_frontier.v1",
        "model_id": "metacog",
        "selected": [
            {
                "matter_id": "m1",
                "source_claim_id": "c1",
                "label": label,
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


def test_llm_disabled_never_calls_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    fake = FakeMindLLMClient([_semantic_payload(claim_label="x")])
    set_llm_client_override(fake)
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
    assert fake.calls == []
    assert result.mind_quality == "fallback_contract_only"
    assert result.brief.mind_authorized_for_stance_skip is False


def test_llm_phases_propagate_request_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    corr = str(uuid4())
    session = "sess-gov-1"
    trace = "trace-gov-1"
    fake = FakeMindLLMClient(
        [
            _semantic_payload(claim_label="shared evening moment"),
            _frontier_payload(),
            {"stance_payload": dict(_VALID_STANCE)},
        ]
    )
    set_llm_client_override(fake)
    req = MindRunRequestV1(
        correlation_id=corr,
        session_id=session,
        trace_id=trace,
        snapshot_inputs={"user_text": "going to watch a show with Amanda"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000, router_profile_id="default"),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    assert result.ok
    assert len(fake.calls) == 3
    phases = [c["context"].phase_name for c in fake.calls if c.get("context")]
    assert phases == ["semantic_synthesis", "active_frontier_judge", "stance_handoff"]
    for call in fake.calls:
        ctx = call["context"]
        assert ctx.correlation_id == corr
        assert ctx.session_id == session
        assert ctx.trace_id == trace
    telemetry = result.brief.machine_contract.get("mind.phase_telemetry")
    assert isinstance(telemetry, list) and len(telemetry) == 3


def test_wall_budget_skips_later_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    fake = FakeMindLLMClient(
        [
            _semantic_payload(claim_label="shared evening moment"),
            _frontier_payload(),
            {"stance_payload": dict(_VALID_STANCE)},
        ]
    )
    set_llm_client_override(fake)
    from app.budget import MindRunBudget

    phase_gate = {"n": 0}

    def _can_run_phase(self, *, min_ms: float = 100.0) -> bool:
        phase_gate["n"] += 1
        return phase_gate["n"] == 1

    monkeypatch.setattr(MindRunBudget, "can_run_phase", _can_run_phase)
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "hi"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=50),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["context"].phase_name == "semantic_synthesis"
    assert result.mind_quality != "meaningful_synthesis"
    assert result.brief.mind_authorized_for_stance_skip is False


def test_invalid_route_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    monkeypatch.setattr(settings, "MIND_SEMANTIC_MODEL_ROUTE", "   ")
    fake = FakeMindLLMClient([_semantic_payload(claim_label="x")])
    set_llm_client_override(fake)
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
    assert fake.calls == []
    assert result.mind_quality == "fallback_contract_only"


def test_identity_yaml_cannot_authorize(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    set_llm_client_override(
        FakeMindLLMClient(
            [
                _semantic_payload(claim_label="evening plan"),
                _frontier_payload(label="identity_yaml"),
                {"stance_payload": dict(_VALID_STANCE)},
            ]
        )
    )
    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={"user_text": "hi"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
        mind_settings=settings,
    )
    assert result.brief.mind_authorized_for_stance_skip is False
    assert result.mind_quality == "invalid_handoff"
    top_labels = result.brief.machine_contract.get("mind.active_frontier_top_labels") or []
    assert "identity_yaml" not in top_labels


def test_shadow_synthesis_never_authorizes_skip() -> None:
    from app.engine import run_mind_deterministic
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    req = MindRunRequestV1(
        correlation_id=str(uuid4()),
        snapshot_inputs={
            "user_text": "bike ride with family",
            "facets": {
                "cognitive_projection": {
                    "anchors": {
                        "relationship": {
                            "items": [{"label": "bike outing", "summary": "fun ride", "salience": 0.8}]
                        }
                    }
                }
            },
        },
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=60_000),
    )
    result = run_mind_deterministic(
        req,
        router_profiles_dir=Path(__file__).resolve().parents[1] / "app" / "config",
        snapshot_max_bytes=512_000,
    )
    assert result.mind_quality == "shadow_synthesis"
    assert result.brief.mind_authorized_for_stance_skip is False
    assert result.brief.machine_contract.get("mind.authorized_for_stance_skip") is False


def test_budget_phase_timeout_caps_remaining() -> None:
    from app.budget import MindRunBudget

    budget = MindRunBudget(1000.0, safety_ms=50.0)
    assert budget.phase_timeout_sec(90.0) <= 0.95
    assert budget.can_run_phase(min_ms=100.0) is True
