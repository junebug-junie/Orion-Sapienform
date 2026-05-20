from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.knowledge_forge.models import (
    ClaimStatusV1,
    ClaimV1,
    ContextPackV1,
    DecisionV1,
    SourceV1,
    SpecV1,
    SpecStatusV1,
)


def test_claim_requires_typed_id_and_statement() -> None:
    claim = ClaimV1.model_validate(
        {
            "type": "claim",
            "id": "claim:orion:recall-routing:0007",
            "statement": "chat_general should be the canonical speech step for thought capture.",
            "status": "accepted",
            "source_refs": ["source:2026-05-20-recall-chat"],
            "confidence": "high",
        }
    )
    assert claim.status == ClaimStatusV1.accepted


def test_claim_rejects_bare_related_links() -> None:
    with pytest.raises(ValidationError):
        ClaimV1.model_validate(
            {
                "type": "claim",
                "id": "claim:orion:bad:0001",
                "statement": "bad",
                "status": "speculative",
                "source_refs": [],
                "related": ["claim:other"],
            }
        )


def test_spec_execution_ready_requires_acceptance_tests() -> None:
    with pytest.raises(ValidationError):
        SpecV1.model_validate(
            {
                "type": "spec",
                "id": "spec:substrate-tier-telemetry-v1",
                "status": "execution_ready",
                "component": "orion-substrate-telemetry",
                "requirements": ["persist tier outcomes"],
                "non_goals": [],
                "acceptance_tests": [],
                "source_claims": [],
            }
        )


def test_context_pack_requires_target_and_task() -> None:
    pack = ContextPackV1.model_validate(
        {
            "type": "context_pack",
            "id": "ctx:substrate-tier-telemetry-v1",
            "target": "cursor",
            "task": "Implement substrate tier telemetry persistence v0",
            "included_specs": ["spec:substrate-tier-telemetry-v1"],
            "allowed_sources": [],
            "excluded_context": ["orion-meta-services v2 graph automation"],
        }
    )
    assert pack.target.value == "cursor"
