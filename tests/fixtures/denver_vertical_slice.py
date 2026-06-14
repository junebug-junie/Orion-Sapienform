"""Shared Denver memory-correction vertical-slice fixture (no Docker)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "services" / "orion-context-exec"
for candidate in (str(ROOT), str(SERVICE_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from orion.schemas.context_exec import ContextExecRequestV1, ProposalEnvelopeV1
from orion.schemas.proposal_ledger import JsonFileProposalLedgerRepository

DENVER_MEMORY_CORRECTION_PROMPT = (
    "Propose a memory correction for the unsupported claim that I am from Denver."
)


async def run_denver_vertical_slice_async(
    store_path: Path,
    *,
    auto_triage: bool = True,
) -> dict[str, Any]:
    """Run context-exec fake engine + ledger intake; return run and ledger record."""
    import app.runner as runner_mod
    from app.rlm_engine import FakeRLMEngine
    from app.runner import ContextExecRunner

    runner_mod.settings.rlm_engine = "fake"
    runner_mod.settings.context_exec_proposal_ledger_enabled = True
    runner_mod.settings.context_exec_proposal_ledger_store_path = str(store_path)
    runner_mod.settings.context_exec_proposal_ledger_auto_triage = auto_triage

    runner = ContextExecRunner(engine=FakeRLMEngine())
    run = await runner.run(
        ContextExecRequestV1(text=DENVER_MEMORY_CORRECTION_PROMPT, mode="memory_correction_proposal"),
    )
    repo = JsonFileProposalLedgerRepository(store_path)
    records = repo.list_all()
    record = records[0] if records else None
    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            sys.modules.pop(mod, None)
    return {"run": run, "record": record, "repo": repo}


def run_denver_vertical_slice(
    store_path: Path,
    *,
    auto_triage: bool = True,
) -> dict[str, Any]:
    return asyncio.run(run_denver_vertical_slice_async(store_path, auto_triage=auto_triage))


def assert_denver_vertical_slice_safety(run: Any, record: Any, envelope: ProposalEnvelopeV1) -> None:
    assert run.artifact_type == "ProposalEnvelopeV1"
    assert envelope.proposal_type == "memory_correction_proposal"
    assert envelope.mutation_allowed is False
    assert envelope.requires_human_approval is True
    assert envelope.review_status in {"draft", "pending_review"}
    assert run.runtime_debug.get("proposal_ledger_persisted") is True
    assert record is not None
    assert record.status == "pending_review"
    assert record.attention_required is True
    reason = (record.attention_reason or "").lower()
    assert any(token in reason for token in ("memory", "identity", "denver", "core"))
