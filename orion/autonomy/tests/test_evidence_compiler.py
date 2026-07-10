# orion/autonomy/tests/test_evidence_compiler.py
from __future__ import annotations

from datetime import datetime

from orion.autonomy.evidence_compiler import compile_autonomy_evidence


FIXED = datetime(2026, 7, 10, 15, 0, 0)


def test_empty_reasoning_repo_omits_reasoning_quality() -> None:
    result = compile_autonomy_evidence(
        user_message="hi",
        social={"hazards": []},
        social_bridge={"hazards": []},
        reasoning_summary={"fallback_recommended": True},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "available"}},
        now=FIXED,
    )
    kinds = [e.kind for e in result.evidence]
    assert "reasoning_quality" not in kinds
    assert any(o.get("kind") == "reasoning_quality" for o in result.omitted)
    omit = next(o for o in result.omitted if o["kind"] == "reasoning_quality")
    assert omit["reason"] == "empty_upstream"


def test_reasoning_quality_emits_only_with_upstream_and_fallback() -> None:
    result = compile_autonomy_evidence(
        user_message=None,
        social={},
        social_bridge={},
        reasoning_summary={"fallback_recommended": True},
        reasoning_upstream_nonempty=True,
        autonomy_debug={},
        now=FIXED,
    )
    rq = [e for e in result.evidence if e.kind == "reasoning_quality"]
    assert len(rq) == 1
    assert rq[0].signal_kind == "chat_reasoning_quality"
    assert rq[0].dimension == "fallback"
    assert rq[0].value == 1.0
    assert rq[0].observed_at == FIXED


def test_hazards_from_social_locals_not_ctx() -> None:
    result = compile_autonomy_evidence(
        user_message="x",
        social={"hazards": ["cooldown_active", "context_excluded:memory"]},
        social_bridge={"hazards": ["duplicate_message"]},
        reasoning_summary={"fallback_recommended": False},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "degraded"}},
        now=FIXED,
    )
    rel = [e for e in result.evidence if e.kind == "relational_signal"]
    summaries = {e.summary for e in rel}
    assert "cooldown_active" in summaries
    assert "duplicate_message" in summaries
    assert "context_excluded:memory" in summaries

    mapped = {e.summary: e for e in rel}
    assert mapped["cooldown_active"].signal_kind == "chat_social_hazard"
    assert mapped["cooldown_active"].dimension == "cooldown_active"
    assert mapped["cooldown_active"].value == 1.0
    # Unmapped prefix hazard is audit-only (no pressure fields).
    assert mapped["context_excluded:memory"].signal_kind is None
    assert mapped["context_excluded:memory"].dimension is None

    infra = [e for e in result.evidence if e.kind == "infra_health"]
    assert len(infra) == 1
    assert infra[0].observed_at == FIXED
    assert all(e.observed_at == FIXED for e in result.evidence)


def test_user_turn_and_infra_emitted_without_pressure_fields() -> None:
    result = compile_autonomy_evidence(
        user_message="hello there",
        social={},
        social_bridge={},
        reasoning_summary={},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "available"}},
        now=FIXED,
    )
    user = next(e for e in result.evidence if e.kind == "user_turn")
    assert user.signal_kind is None
    assert user.source == "user_message"
    infra = next(e for e in result.evidence if e.kind == "infra_health")
    assert infra.signal_kind is None


def test_infra_omitted_when_availability_unknown() -> None:
    result = compile_autonomy_evidence(
        user_message=None,
        social={},
        social_bridge={},
        reasoning_summary={},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "weird"}},
        now=FIXED,
    )
    assert not any(e.kind == "infra_health" for e in result.evidence)
    assert any(o.get("reason") == "availability_not_recognized" for o in result.omitted)


def test_compiler_never_raises() -> None:
    result = compile_autonomy_evidence(
        user_message=object(),  # type: ignore[arg-type]
        social="bad",  # type: ignore[arg-type]
        social_bridge=None,
        reasoning_summary=None,
        reasoning_upstream_nonempty=True,
        autonomy_debug=None,
        now=FIXED,
    )
    assert isinstance(result.evidence, list)
