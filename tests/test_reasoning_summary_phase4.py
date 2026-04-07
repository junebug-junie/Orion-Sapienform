from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.reasoning import ClaimV1, ContradictionV1, ReasoningSparkStateSnapshotV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_summary import ReasoningSummaryRequestV1
from orion.reasoning.repository import InMemoryReasoningRepository
from orion.reasoning.summary import ReasoningSummaryCompiler


def _prov() -> dict:
    return {
        "evidence_refs": ["ev:1"],
        "source_channel": "orion:test",
        "source_kind": "unit",
        "producer": "pytest",
    }


def _claim(**kwargs) -> ClaimV1:
    return ClaimV1(
        anchor_scope=kwargs.get("anchor_scope", "orion"),
        subject_ref=kwargs.get("subject_ref", "project:orion_sapienform"),
        status=kwargs.get("status", "canonical"),
        authority="local_inferred",
        confidence=kwargs.get("confidence", 0.8),
        salience=kwargs.get("salience", 0.7),
        novelty=0.3,
        risk_tier="low",
        observed_at=kwargs.get("observed_at", datetime.now(timezone.utc)),
        provenance=_prov(),
        claim_text=kwargs.get("claim_text", "stable claim"),
        claim_kind=kwargs.get("claim_kind", "assertion"),
    )


def _repo(*artifacts):
    repo = InMemoryReasoningRepository()
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(
                source_family="manual",
                source_kind="unit",
                source_channel="orion:test",
                producer="pytest",
            ),
            artifacts=list(artifacts),
        )
    )
    return repo


def test_empty_repo_returns_fallback_summary() -> None:
    summary = ReasoningSummaryCompiler(InMemoryReasoningRepository()).compile(ReasoningSummaryRequestV1(anchor_scope="orion"))
    assert summary.fallback_recommended is True
    assert summary.debug.compiler_ran is True
    assert summary.debug.compiler_succeeded is True


def test_promoted_artifacts_compile_and_rejected_deprecated_excluded() -> None:
    canonical = _claim(status="canonical", claim_text="trusted")
    deprecated = _claim(status="deprecated", claim_text="old")
    rejected = _claim(status="rejected", claim_text="bad")
    summary = ReasoningSummaryCompiler(_repo(canonical, deprecated, rejected)).compile(ReasoningSummaryRequestV1(anchor_scope="orion"))
    assert any(c.claim_text == "trusted" for c in summary.active_claims)
    assert all(c.claim_text != "old" for c in summary.active_claims)
    assert all(c.claim_text != "bad" for c in summary.active_claims)


def test_unresolved_contradiction_suppresses_and_resolved_allows() -> None:
    claim = _claim(status="provisional", claim_text="tentative")
    contradiction = ContradictionV1(
        anchor_scope="orion",
        subject_ref=claim.subject_ref,
        status="proposed",
        authority="local_inferred",
        confidence=0.7,
        salience=0.6,
        novelty=0.1,
        risk_tier="medium",
        observed_at=datetime.now(timezone.utc),
        provenance=_prov(),
        contradiction_type="evidence_conflict",
        severity="high",
        resolution_status="open",
        involved_artifact_ids=[claim.artifact_id, "artifact-x"],
        summary="open conflict",
    )
    suppressed = ReasoningSummaryCompiler(_repo(claim, contradiction)).compile(ReasoningSummaryRequestV1(anchor_scope="orion"))
    assert not suppressed.active_claims

    resolved = contradiction.model_copy(update={"artifact_id": "cx", "resolution_status": "resolved"})
    allowed = ReasoningSummaryCompiler(_repo(claim, resolved)).compile(ReasoningSummaryRequestV1(anchor_scope="orion"))
    assert allowed.active_claims


def test_scope_entity_lifecycle_and_drift_handling() -> None:
    stale = _claim(status="canonical", subject_ref="project:stale", observed_at=datetime.now(timezone.utc) - timedelta(days=40), salience=0.1)
    concept_drift = _claim(status="canonical", claim_kind="concept_item", claim_text="drifted concept")
    spark = ReasoningSparkStateSnapshotV1(
        anchor_scope="orion",
        subject_ref="node:atlas",
        status="provisional",
        authority="sensed",
        confidence=0.8,
        salience=0.4,
        novelty=0.6,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance=_prov(),
        dimensions={"coherence": 0.9},
    )
    active = _claim(status="canonical", subject_ref="project:active", claim_text="active domain")
    summary = ReasoningSummaryCompiler(_repo(stale, concept_drift, spark, active)).compile(
        ReasoningSummaryRequestV1(anchor_scope="orion")
    )
    assert "project:active" in summary.active_subject_refs
    assert "project:stale" not in summary.active_subject_refs
    assert all(c.claim_text != "drifted concept" for c in summary.active_claims)
    assert summary.spark.present is True
