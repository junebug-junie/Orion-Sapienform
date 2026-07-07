from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import (
    FinalizeReflectionV1,
    GrammarReceiptV1,
    HarnessDraftMoleculeV1,
    HarnessPostTurnClosureV1,
    HarnessRepairOverlayV1,
    HarnessRunRequestV1,
    HarnessRunV1,
    HarnessTurnOutcomeMoleculeV1,
    HarnessVerdictMoleculeV1,
    SubstrateFinalizeAppraisalV1,
)
from orion.schemas.registry import resolve
from orion.schemas.thought import (
    CoalitionSnapshotV1,
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)


def _stance_slice(**overrides: object) -> StanceHarnessSliceV1:
    base = {
        "task_mode": "direct_response",
        "conversation_frame": "mixed",
        "answer_strategy": "direct",
    }
    base.update(overrides)
    return StanceHarnessSliceV1.model_validate(base)


def _thought(**overrides: object) -> ThoughtEventV1:
    base = {
        "event_id": "t-1",
        "correlation_id": "c-1",
        "session_id": "sess-1",
        "created_at": datetime.now(timezone.utc),
        "imperative": "Check the deploy logs first.",
        "tone": "direct",
        "strain_refs": ["node-a"],
        "evidence_refs": ["node-a"],
        "stance_harness_slice": _stance_slice(),
    }
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def _coalition_snapshot(**overrides: object) -> CoalitionSnapshotV1:
    base = {
        "attended_node_ids": ["node-a"],
        "selected_open_loop_id": None,
        "open_loop_ids": [],
        "generated_at": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return CoalitionSnapshotV1.model_validate(base)


def test_thought_event_evidence_refs_required_subset() -> None:
    thought = _thought()
    assert thought.disposition == "proceed"
    assert thought.evidence_refs == ["node-a"]


def test_thought_event_allows_empty_evidence_refs_for_fail_closed_enforce() -> None:
    """Cortex may emit empty evidence_refs; enforce_thought_stance_quality defers."""
    thought = _thought(evidence_refs=[])
    assert thought.evidence_refs == []


def test_harness_run_finalize_ran_default_false() -> None:
    run = HarnessRunV1(
        correlation_id="c-1",
        final_text=None,
        finalize_ran=False,
        step_count=0,
        compliance_verdict="partial",
        grounding_status="unknown",
    )
    assert run.finalize_ran is False


def test_thought_event_roundtrip() -> None:
    thought = _thought()
    restored = ThoughtEventV1.model_validate(thought.model_dump(mode="json"))
    assert restored.event_id == thought.event_id
    assert restored.stance_harness_slice.answer_strategy == "direct"


def test_stance_react_request_roundtrip() -> None:
    req = StanceReactRequestV1(
        correlation_id="c-1",
        session_id="sess-1",
        user_message="what broke in deploy?",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "what broke in deploy?"},
    )
    restored = StanceReactRequestV1.model_validate(req.model_dump(mode="json"))
    assert restored.schema_version == "stance.react.request.v1"


def test_harness_draft_molecule_roundtrip() -> None:
    thought = _thought()
    molecule = HarnessDraftMoleculeV1(
        correlation_id="c-1",
        thought_event_id=thought.event_id,
        draft_text="Draft answer body.",
        draft_hash="abc123",
        thought_event=thought,
        grammar_receipts=[GrammarReceiptV1(step_index=0, summary="read logs")],
        coalition_snapshot=_coalition_snapshot(),
    )
    restored = HarnessDraftMoleculeV1.model_validate(molecule.model_dump(mode="json"))
    assert restored.thought_event.imperative == thought.imperative
    assert restored.grammar_receipts[0].summary == "read logs"


def test_substrate_finalize_appraisal_roundtrip() -> None:
    appraisal = SubstrateFinalizeAppraisalV1(
        correlation_id="c-1",
        molecule_id="mol-1",
        draft_hash="abc123",
        surprise_level=0.2,
        learning_refs=["learn-1"],
    )
    restored = SubstrateFinalizeAppraisalV1.model_validate(appraisal.model_dump(mode="json"))
    assert restored.tick_source == "substrate_runtime_finalize_appraisal"


def test_substrate_finalize_appraisal_requires_learning_refs() -> None:
    with pytest.raises(ValidationError):
        SubstrateFinalizeAppraisalV1(
            correlation_id="c-1",
            molecule_id="mol-1",
            draft_hash="abc123",
            surprise_level=0.2,
            learning_refs=[],
        )


def test_finalize_reflection_roundtrip() -> None:
    reflection = FinalizeReflectionV1(
        correlation_id="c-1",
        thought_event_id="t-1",
        substrate_appraisal_id="appr-1",
        draft_hash="abc123",
        imperative="Stay concrete.",
        tone="direct",
        strain_refs=["node-a"],
        alignment_verdict="aligned",
        alignment_notes=["matches coalition"],
        strain_unresolved=False,
    )
    restored = FinalizeReflectionV1.model_validate(reflection.model_dump(mode="json"))
    assert restored.reflection_source == "substrate_informed_pass"


def test_harness_verdict_molecule_roundtrip() -> None:
    reflection = FinalizeReflectionV1(
        correlation_id="c-1",
        thought_event_id="t-1",
        substrate_appraisal_id="appr-1",
        draft_hash="abc123",
        imperative="Stay concrete.",
        tone="direct",
        strain_refs=["node-a"],
        alignment_verdict="aligned",
        alignment_notes=[],
        strain_unresolved=False,
    )
    verdict = HarnessVerdictMoleculeV1(
        correlation_id="c-1",
        reflection=reflection,
        cortex_trace_id="trace-1",
    )
    restored = HarnessVerdictMoleculeV1.model_validate(verdict.model_dump(mode="json"))
    assert restored.reflection.alignment_verdict == "aligned"


def test_harness_turn_outcome_molecule_roundtrip() -> None:
    outcome = HarnessTurnOutcomeMoleculeV1(
        correlation_id="c-1",
        thought_event_id="t-1",
        substrate_appraisal_id="appr-1",
        reflection_id="refl-1",
        verdict_molecule_id="verd-1",
        draft_hash="abc123",
        final_hash="def456",
        finalize_changed=False,
        alignment_verdict="aligned",
        surprise_level_at_draft=0.1,
        surprise_resolved=True,
        final_text="Shipped answer.",
    )
    restored = HarnessTurnOutcomeMoleculeV1.model_validate(outcome.model_dump(mode="json"))
    assert restored.final_text == "Shipped answer."


def test_harness_post_turn_closure_roundtrip() -> None:
    closure = HarnessPostTurnClosureV1(
        correlation_id="c-1",
        outcome_molecule_id="out-1",
        verdict_molecule_id="verd-1",
        surprise_unresolved=False,
    )
    restored = HarnessPostTurnClosureV1.model_validate(closure.model_dump(mode="json"))
    assert restored.closure_source == "harness_post_turn_appraisal"


def test_post_turn_closure_carries_referent_excerpts_when_unresolved() -> None:
    closure = HarnessPostTurnClosureV1(
        correlation_id="c-1",
        outcome_molecule_id="out-1",
        verdict_molecule_id="verd-1",
        surprise_unresolved=True,
        user_message_excerpt="What if the deploy slips again?",
        stance_imperative="Name the risk plainly before offering a plan.",
        thought_event_id="te-1",
    )
    restored = HarnessPostTurnClosureV1.model_validate(closure.model_dump(mode="json"))
    assert restored.user_message_excerpt == "What if the deploy slips again?"
    assert restored.stance_imperative.startswith("Name the risk")
    assert restored.thought_event_id == "te-1"


def test_harness_repair_overlay_roundtrip() -> None:
    overlay = HarnessRepairOverlayV1(
        mode="repair_concrete",
        rule_lines=["include file paths"],
        prefix_overlay="Be concrete.",
    )
    restored = HarnessRepairOverlayV1.model_validate(overlay.model_dump(mode="json"))
    assert restored.mode == "repair_concrete"


def test_harness_run_request_roundtrip() -> None:
    req = HarnessRunRequestV1(
        correlation_id="c-1",
        thought_event=_thought(),
        user_message="what broke?",
        permissions=ContextExecPermissionV1(read_repo=True),
        answer_contract=AnswerContract(request_kind="runtime_debug"),
    )
    restored = HarnessRunRequestV1.model_validate(req.model_dump(mode="json"))
    assert restored.answer_contract.request_kind == "runtime_debug"


def test_harness_run_roundtrip() -> None:
    run = HarnessRunV1(
        correlation_id="c-1",
        final_text="Done.",
        draft_text="Draft only.",
        finalize_ran=True,
        step_count=3,
        compliance_verdict="completed",
        grounding_status="grounded_complete",
    )
    restored = HarnessRunV1.model_validate(run.model_dump(mode="json"))
    assert restored.finalize_ran is True
    assert restored.draft_text == "Draft only."


def test_registry_schema_kinds_include_thought_event() -> None:
    from orion.schemas.registry import SCHEMA_REGISTRY

    assert SCHEMA_REGISTRY["ThoughtEventV1"].kind == "thought.event.v1"
    assert SCHEMA_REGISTRY["GrammarReceiptV1"].kind == "grammar.receipt.v1"


@pytest.mark.parametrize(
    "schema_id",
    [
        "CoalitionSnapshotV1",
        "StanceHarnessSliceV1",
        "HubAssociationBundleV1",
        "ThoughtEventV1",
        "StanceReactRequestV1",
        "GrammarReceiptV1",
        "HarnessDraftMoleculeV1",
        "SubstrateFinalizeAppraisalV1",
        "FinalizeReflectionV1",
        "HarnessVerdictMoleculeV1",
        "HarnessTurnOutcomeMoleculeV1",
        "HarnessPostTurnClosureV1",
        "HarnessRepairOverlayV1",
        "HarnessRunRequestV1",
        "HarnessRunStepV1",
        "HarnessRunV1",
    ],
)
def test_registry_resolves_unified_turn_schemas(schema_id: str) -> None:
    assert resolve(schema_id).__name__ == schema_id
