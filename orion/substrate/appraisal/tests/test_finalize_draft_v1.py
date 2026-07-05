from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.harness_finalize import GrammarReceiptV1, HarnessDraftMoleculeV1
from orion.schemas.thought import CoalitionSnapshotV1, StanceHarnessSliceV1, ThoughtEventV1
from orion.substrate.appraisal.finalize_draft_v1 import (
    FinalizeDraftAppraisalError,
    appraise_draft_molecule,
)


def test_draft_molecule_rpc_round_trip_fields() -> None:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id="c-1",
        session_id=None,
        created_at=datetime.now(timezone.utc),
        imperative="x",
        tone="y",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    mol = HarnessDraftMoleculeV1(
        correlation_id="c-1",
        thought_event_id="t-1",
        draft_text="draft answer",
        draft_hash="abc",
        thought_event=thought,
        grammar_receipts=[GrammarReceiptV1(step_index=0, summary="step")],
        coalition_snapshot=CoalitionSnapshotV1(
            attended_node_ids=["n-1"],
            selected_open_loop_id=None,
            open_loop_ids=[],
            generated_at=datetime.now(timezone.utc),
        ),
    )
    appraisal = appraise_draft_molecule(mol)
    assert appraisal.draft_hash == "abc"
    assert len(appraisal.learning_refs) >= 1
    assert 0.0 <= appraisal.surprise_level <= 1.0


def test_empty_learning_refs_raises() -> None:
    thought = ThoughtEventV1(
        event_id="t-empty",
        correlation_id="c-empty",
        session_id=None,
        created_at=datetime.now(timezone.utc),
        imperative="x",
        tone="y",
        strain_refs=[],
        evidence_refs=[],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    mol = HarnessDraftMoleculeV1(
        correlation_id="c-empty",
        thought_event_id="t-empty",
        draft_text="draft",
        draft_hash="empty",
        thought_event=thought,
        grammar_receipts=[],
        coalition_snapshot=CoalitionSnapshotV1(
            attended_node_ids=[],
            selected_open_loop_id=None,
            open_loop_ids=[],
            generated_at=datetime.now(timezone.utc),
        ),
    )
    with pytest.raises(FinalizeDraftAppraisalError, match="learning_refs"):
        appraise_draft_molecule(mol)
