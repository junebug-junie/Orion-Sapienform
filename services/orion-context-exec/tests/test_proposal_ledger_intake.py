"""Proposal ledger intake and JSON ledger hardening tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from orion.schemas.context_exec import (
    ContextExecPermissionV1,
    ContextExecRequestV1,
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
    build_memory_correction_proposal_envelope,
    build_patch_proposal_envelope,
)
from orion.schemas.proposal_ledger import (
    JsonFileProposalLedgerRepository,
    ProposalLedgerRecordV1,
    ProposalLedgerStoreError,
    ProposalReviewDecisionV1,
    ProposalTriageDecisionV1,
)

from app.proposal_ledger_intake import maybe_persist_proposal_envelope
from app.runner import ContextExecRunner
from app.settings import ContextExecSettings

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts" / "orion_proposal_cli.py"
PYTHON = ROOT / "orion_dev" / "bin" / "python"

PATCH_PROMPT = "Propose a patch for weak trace-autopsy root cause synthesis in context-exec."
MEMORY_CORRECTION_PROMPT = (
    "Propose a memory correction for the unsupported claim that I am from Denver."
)


def _patch_envelope() -> ProposalEnvelopeV1:
    patch = PatchProposalV1(
        problem="README typo",
        proposed_change_summary="Fix spelling",
        rollback_plan="Revert",
    )
    return build_patch_proposal_envelope(patch, source_mode="patch_proposal")


def _memory_envelope() -> ProposalEnvelopeV1:
    correction = MemoryCorrectionProposalV1(
        current_belief="User is from Denver",
        rationale="Insufficient evidence",
        rollback_plan="No mutation proposed; no rollback required.",
    )
    return build_memory_correction_proposal_envelope(
        correction,
        source_mode="memory_correction_proposal",
    )


def _ledger_settings(
    *,
    enabled: bool = True,
    store_path: str = "",
    auto_triage: bool = False,
) -> ContextExecSettings:
    return ContextExecSettings(
        CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=enabled,
        CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH=store_path,
        CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE=auto_triage,
    )


def test_proposal_ledger_intake_disabled_does_not_persist(tmp_path: Path) -> None:
    settings = _ledger_settings(enabled=False, store_path=str(tmp_path / "ledger.json"))
    result = maybe_persist_proposal_envelope(_patch_envelope(), settings)
    assert result.persisted is False
    assert result.error is None
    assert not (tmp_path / "ledger.json").exists()


def test_proposal_ledger_intake_enabled_requires_store_path() -> None:
    settings = _ledger_settings(enabled=True, store_path="")
    result = maybe_persist_proposal_envelope(_patch_envelope(), settings)
    assert result.persisted is False
    assert result.error == "store_path_required"


def test_patch_proposal_persists_to_ledger_when_enabled(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    settings = _ledger_settings(enabled=True, store_path=str(store))
    envelope = _patch_envelope()
    result = maybe_persist_proposal_envelope(envelope, settings, source_run_id="ctxrun_test")
    assert result.persisted is True
    assert result.proposal_id == envelope.proposal_id
    assert result.ledger_status == "stored"
    assert result.attention_required is False

    repo = JsonFileProposalLedgerRepository(store)
    record = repo.get(envelope.proposal_id)
    assert record is not None
    assert record.status == "stored"
    assert record.attention_required is False
    assert record.source_run_id == "ctxrun_test"


def test_memory_correction_proposal_persists_to_ledger_when_enabled(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    settings = _ledger_settings(enabled=True, store_path=str(store))
    envelope = _memory_envelope()
    result = maybe_persist_proposal_envelope(envelope, settings)
    assert result.persisted is True
    assert result.proposal_id == envelope.proposal_id
    assert result.ledger_status == "stored"
    assert result.attention_required is False

    repo = JsonFileProposalLedgerRepository(store)
    record = repo.get(envelope.proposal_id)
    assert record is not None
    assert record.envelope.proposal_type == "memory_correction_proposal"


@pytest.mark.asyncio
async def test_investigative_artifacts_do_not_enter_proposal_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "proposals.json"
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    monkeypatch.setattr("app.runner.settings.context_exec_proposal_ledger_enabled", True)
    monkeypatch.setattr(
        "app.runner.settings.context_exec_proposal_ledger_store_path",
        str(store),
    )
    runner = ContextExecRunner()
    for mode, text in [
        ("belief_provenance", "Where did the Denver belief come from?"),
        ("trace_autopsy", "Autopsy the failed trace for Denver claim."),
        ("repo_impact_analysis", "What repo files affect trace autopsy synthesis?"),
    ]:
        run = await runner.run(ContextExecRequestV1(text=text, mode=mode))
        assert run.artifact_type != "ProposalEnvelopeV1"
        assert run.runtime_debug.get("proposal_ledger_persisted") is not True
    assert not store.exists()


@pytest.mark.asyncio
async def test_runner_ledger_enabled_without_store_path_skips_persist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "proposals.json"
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    monkeypatch.setattr("app.runner.settings.context_exec_proposal_ledger_enabled", True)
    monkeypatch.setattr("app.runner.settings.context_exec_proposal_ledger_store_path", "")
    runner = ContextExecRunner()
    run = await runner.run(
        ContextExecRequestV1(text=PATCH_PROMPT, mode="patch_proposal"),
    )
    assert run.runtime_debug["proposal_ledger_enabled"] is True
    assert run.runtime_debug["proposal_ledger_persisted"] is False
    assert run.runtime_debug["proposal_ledger_error"] == "store_path_required"
    assert not store.exists()


@pytest.mark.asyncio
async def test_runner_persists_patch_proposal_when_ledger_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "proposals.json"
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    monkeypatch.setattr("app.runner.settings.context_exec_proposal_ledger_enabled", True)
    monkeypatch.setattr(
        "app.runner.settings.context_exec_proposal_ledger_store_path",
        str(store),
    )
    runner = ContextExecRunner()
    run = await runner.run(
        ContextExecRequestV1(text=PATCH_PROMPT, mode="patch_proposal"),
    )
    assert run.status == "ok"
    assert run.runtime_debug["proposal_ledger_enabled"] is True
    assert run.runtime_debug["proposal_ledger_persisted"] is True
    assert run.runtime_debug["ledger_status"] == "stored"
    assert "Proposal stored in ledger:" in run.final_text

    repo = JsonFileProposalLedgerRepository(store)
    records = repo.list_all()
    assert len(records) == 1


@pytest.mark.asyncio
async def test_runner_ledger_disabled_does_not_persist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "proposals.json"
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    monkeypatch.setattr("app.runner.settings.context_exec_proposal_ledger_enabled", False)
    runner = ContextExecRunner()
    run = await runner.run(
        ContextExecRequestV1(text=PATCH_PROMPT, mode="patch_proposal"),
    )
    assert run.runtime_debug["proposal_ledger_enabled"] is False
    assert run.runtime_debug["proposal_ledger_persisted"] is False
    assert not store.exists()


def test_json_ledger_requires_explicit_store_path() -> None:
    with pytest.raises(ProposalLedgerStoreError, match="store path is required"):
        JsonFileProposalLedgerRepository(Path(""))


def test_json_ledger_malformed_json_fails_without_overwrite(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    corrupt = '{"records": ['
    store.write_text(corrupt, encoding="utf-8")

    with pytest.raises(ProposalLedgerStoreError, match="malformed JSON"):
        JsonFileProposalLedgerRepository(store)

    assert store.read_text(encoding="utf-8") == corrupt


def test_json_ledger_missing_proposal_id_fails_cleanly(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    repo = JsonFileProposalLedgerRepository(store)
    envelope = _patch_envelope()
    repo.store(
        ProposalLedgerRecordV1(
            proposal_id=envelope.proposal_id,
            envelope=envelope,
            status="stored",
        )
    )

    with pytest.raises(KeyError, match="proposal not found"):
        repo.apply_triage(
            ProposalTriageDecisionV1(
                proposal_id="prop_missing",
                action="store_only",
                rationale="test",
            )
        )

    with pytest.raises(KeyError, match="proposal not found"):
        repo.apply_review(
            ProposalReviewDecisionV1(
                decision_id="dec_missing",
                proposal_id="prop_missing",
                decision="approve",
                reviewer_type="human",
                reviewer_id="operator",
                rationale="test",
            )
        )


def test_json_ledger_review_approve_creates_eligibility_not_receipt(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    repo = JsonFileProposalLedgerRepository(store)
    envelope = _patch_envelope()
    record = ProposalLedgerRecordV1(
        proposal_id=envelope.proposal_id,
        envelope=envelope,
        status="stored",
    )
    repo.store(record)
    pending = repo.apply_triage(
        ProposalTriageDecisionV1(
            proposal_id=envelope.proposal_id,
            action="promote_to_review",
            rationale="review",
            attention_required=True,
            attention_reason="review",
        )
    )
    approved = repo.apply_review(
        ProposalReviewDecisionV1(
            decision_id="dec_approve",
            proposal_id=pending.proposal_id,
            decision="approve",
            reviewer_type="human",
            reviewer_id="operator",
            rationale="ok",
        )
    )
    assert approved.status == "approved"
    assert approved.status != "executed"
    assert approved.status != "execution_requested"

    raw = json.loads(store.read_text(encoding="utf-8"))
    assert "execution_receipt" not in raw
    assert "receipt" not in raw


def test_json_ledger_write_preserves_valid_json_after_review_update(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    repo = JsonFileProposalLedgerRepository(store)
    envelope = _patch_envelope()
    repo.store(
        ProposalLedgerRecordV1(
            proposal_id=envelope.proposal_id,
            envelope=envelope,
            status="stored",
        )
    )
    repo.apply_triage(
        ProposalTriageDecisionV1(
            proposal_id=envelope.proposal_id,
            action="promote_to_review",
            rationale="review",
            attention_required=True,
            attention_reason="review",
        )
    )
    repo.apply_review(
        ProposalReviewDecisionV1(
            decision_id="dec_review",
            proposal_id=envelope.proposal_id,
            decision="reject",
            reviewer_type="human",
            reviewer_id="operator",
            rationale="no",
        )
    )

    raw_text = store.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    assert isinstance(parsed, dict)
    assert len(parsed["records"]) == 1
    assert parsed["records"][0]["status"] == "rejected"


def test_cli_can_show_context_exec_persisted_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "proposals.json"
    settings = _ledger_settings(enabled=True, store_path=str(store))
    envelope = _memory_envelope()
    maybe_persist_proposal_envelope(envelope, settings, source_run_id="ctxrun_cli")

    cmd = [
        str(PYTHON if PYTHON.exists() else sys.executable),
        str(CLI),
        "show",
        envelope.proposal_id,
        "--store",
        str(store),
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proposal_id"] == envelope.proposal_id
    assert payload["status"] == "stored"
    assert payload["envelope"]["proposal_type"] == "memory_correction_proposal"
