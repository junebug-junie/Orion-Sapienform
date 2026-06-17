"""Assert context-exec proposal ledger smoke scripts use durable storage defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FRESH_MAIN_SMOKE = ROOT / "scripts" / "repl" / "orion_fresh_main_smoke.sh"
PROPOSAL_API_SMOKE = ROOT / "scripts" / "proposal_review_api_smoke.sh"
PROPOSAL_REVIEW_DOC = ROOT / "docs" / "proposal-review-api.md"

HOST_STORAGE_ROOT = "/mnt/rlm-nvme/context-exec"
HOST_LEDGER_SUFFIX = "ledger/orion-proposals.json"
CONTAINER_LEDGER = "/var/lib/orion/context-exec/ledger/orion-proposals.json"


def _active_doc_text() -> str:
    text = PROPOSAL_REVIEW_DOC.read_text(encoding="utf-8")
    marker = "### Migration from `/tmp`"
    if marker in text:
        return text.split(marker, maxsplit=1)[0]
    return text


def test_fresh_main_smoke_defaults_use_durable_storage() -> None:
    text = FRESH_MAIN_SMOKE.read_text(encoding="utf-8")
    assert "/tmp/orion-proposals.json" not in text
    assert "./.orion-smoke-logs" not in text
    assert "CONTEXT_EXEC_STORAGE_ROOT" in text
    assert HOST_STORAGE_ROOT in text
    assert HOST_LEDGER_SUFFIX in text
    assert "orion-proposals.reject.json" in text
    assert "smoke-logs" in text


def test_proposal_review_api_smoke_defaults_use_durable_storage() -> None:
    text = PROPOSAL_API_SMOKE.read_text(encoding="utf-8")
    assert "/tmp/orion-proposals.json" not in text
    assert "CONTEXT_EXEC_STORAGE_ROOT" in text
    assert HOST_STORAGE_ROOT in text
    assert HOST_LEDGER_SUFFIX in text
    assert "orion-proposals.reject.json" in text


def test_proposal_review_doc_active_examples_use_durable_storage() -> None:
    text = _active_doc_text()
    assert "/tmp/orion-proposals.json" not in text
    assert CONTAINER_LEDGER in text
    assert f"{HOST_STORAGE_ROOT}/{HOST_LEDGER_SUFFIX}" in text
