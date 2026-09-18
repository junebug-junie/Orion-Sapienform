from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from orion.mind.synthesis_v1 import (
    ActiveCognitiveFrontierV1,
    MindEvidencePackV1,
    SemanticSynthesisV1,
)
from orion.schemas.chat_stance import ChatStanceBrief

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy_soft", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


@pytest.fixture(autouse=True)
def _prep() -> None:
    _mind_prep()


_MINIMAL_STANCE = {
    "conversation_frame": "technical",
    "task_mode": "technical_collaboration",
    "identity_salience": "medium",
    "user_intent": "Investigate a cross-service failure.",
    "self_relevance": "This is my own investigation subject.",
    "juniper_relevance": "Juniper asked me to look.",
    "answer_strategy": "DirectAnswer",
    "stance_summary": "Self-authored investigation turn.",
}


def test_chat_stance_brief_soft_fields_roundtrip() -> None:
    brief = ChatStanceBrief(
        conversation_frame="technical",
        user_intent="Investigate a cross-service failure.",
        self_relevance="This is my own investigation subject.",
        juniper_relevance="Juniper asked me to look.",
        answer_strategy="DirectAnswer",
        stance_summary="Self-authored investigation turn.",
        expected_depth="deep",
        cross_cutting="yes",
        foresight_note="Likely multi-service archaeology.",
    )
    dumped = brief.model_dump(mode="json")
    reparsed = ChatStanceBrief.model_validate(dumped)
    assert reparsed.expected_depth == "deep"
    assert reparsed.cross_cutting == "yes"
    assert "multi-service" in (reparsed.foresight_note or "")


def test_soft_keys_survive_coerce_when_present() -> None:
    from app.stance_handoff import try_coerce_stance_payload
    from orion.mind.validation import validate_merged_stance_brief_optional

    payload = dict(_MINIMAL_STANCE)
    payload["conversation_frame"] = "smoketest"
    payload["expected_depth"] = "deep"
    payload["cross_cutting"] = "yes"
    payload["foresight_note"] = "Likely multi-service archaeology."
    coerced, did = try_coerce_stance_payload(payload)
    assert did is True
    valid, err = validate_merged_stance_brief_optional(coerced)
    assert err is None
    assert valid is not None
    assert valid.expected_depth == "deep"
    assert valid.cross_cutting == "yes"
    assert "multi-service" in (valid.foresight_note or "")


def _frontier() -> ActiveCognitiveFrontierV1:
    return ActiveCognitiveFrontierV1(
        selected=[
            {
                "matter_id": "m1",
                "source_claim_id": "c1",
                "label": "cross-service failure",
                "summary": "The claim may span several services.",
                "matter_kind": "curiosity_affordance",
                "score": 0.8,
                "features": {"confidence": 0.72},
            }
        ]
    )


def _synthesis() -> SemanticSynthesisV1:
    return SemanticSynthesisV1(
        claims=[
            {
                "claim_id": "c1",
                "label": "cross-service failure",
                "summary": "The claim may span several services.",
                "claim_kind": "curiosity_affordance_claim",
            }
        ]
    )


def _run_handoff(*, utterance_origin: str | None, payload: dict | None = None) -> dict:
    from app.stance_handoff import run_stance_handoff

    captured: dict = {}

    class _Client:
        def request_json(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["system_prompt"] = kwargs["system_prompt"]
            captured["user_prompt"] = kwargs["user_prompt"]
            return payload or dict(_MINIMAL_STANCE), None, {"model_used": "chat"}

    pack = MindEvidencePackV1(current_user_text="investigate claim X")
    result, err, telemetry = run_stance_handoff(
        _frontier(),
        _synthesis(),
        pack,
        client=_Client(),  # type: ignore[arg-type]
        route="chat",
        model_id="chat",
        max_tokens=256,
        utterance_origin=utterance_origin,
    )
    captured["result"] = result
    captured["err"] = err
    captured["telemetry"] = telemetry
    return captured


def test_orion_origin_instructs_soft_work_shape_fields() -> None:
    captured = _run_handoff(utterance_origin="orion")
    system = captured["system_prompt"]
    assert "expected_depth" in system
    assert "cross_cutting" in system
    assert "foresight_note" in system


def test_juniper_origin_omits_soft_work_shape_instruction() -> None:
    captured = _run_handoff(utterance_origin="juniper")
    system = captured["system_prompt"]
    assert "expected_depth" not in system
    assert "cross_cutting" not in system
    assert "foresight_note" not in system
