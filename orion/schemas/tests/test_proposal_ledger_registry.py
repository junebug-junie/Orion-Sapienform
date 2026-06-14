"""Schema-level proposal ledger registry tests."""

from __future__ import annotations

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
)
from orion.schemas.proposal_ledger import (
    ProposalExecutionEligibilityV1,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionV1,
    ProposalTriageDecisionV1,
)
from orion.schemas.registry import _REGISTRY, resolve


def test_proposal_ledger_schemas_resolve() -> None:
    for schema_id in (
        "ProposalLedgerRecordV1",
        "ProposalTriageDecisionV1",
        "ProposalReviewDecisionV1",
        "ProposalExecutionEligibilityV1",
        "ProposalEnvelopeV1",
        "PatchProposalV1",
        "MemoryCorrectionProposalV1",
    ):
        assert resolve(schema_id) is _REGISTRY[schema_id]
