from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.lint import lint_corpus
from orion.knowledge_forge.store import KnowledgeStore


def test_lint_reports_dangling_claim_ref() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    store = KnowledgeStore(root)
    store.load()
    report = lint_corpus(store)
    codes = {issue.code for issue in report.issues}
    assert "dangling_ref" in codes


def test_lint_passes_minimal_valid_fixture() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    # Remove bad fixture temporarily by linting only accepted claim file subset
    store = KnowledgeStore(root)
    store.load()
    store.by_id.pop("claim:test:bad-ref", None)
    report = lint_corpus(store)
    assert report.ok is True
