from __future__ import annotations

from orion.cognition.findings_bundle_synth import merge_findings_bundle_dicts, synthesize_findings_bundle


def test_synthesize_insufficient_without_repo_evidence() -> None:
    fb = synthesize_findings_bundle(
        answer_contract={"requires_repo_grounding": True},
        trace=[],
        tools_called=["plan_action"],
    )
    assert fb["grounded_status"] == "insufficient_grounding"
    assert any("repo" in str(x) for x in (fb.get("missing_evidence") or []))


def test_merge_findings_merges_lists() -> None:
    a = {"findings": [{"claim": "a"}], "missing_evidence": ["x"], "unsupported_requests": [], "next_checks": [], "grounded_status": "grounded_partial"}
    b = {"findings": [{"claim": "b"}], "missing_evidence": ["y"], "unsupported_requests": [], "next_checks": [], "grounded_status": "insufficient_grounding"}
    m = merge_findings_bundle_dicts(a, b)
    assert len(m["findings"]) == 2
    assert m["grounded_status"] == "insufficient_grounding"
