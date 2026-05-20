from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.knowledge_forge.sources import (
    build_source_delta_review,
    ingest_source,
    parse_source_markdown,
    slug_from_source_id,
)


SAMPLE_DOC = """# Test Design Doc

## Requirements

- Knowledge Forge should track changed design docs.
- Source ingest should propose claims, not accept them.
- Review artifacts should be written to reviews/pending.

## Non-goals

- No MCP.
- No vector search.

## Acceptance checks

- Dry run writes no files.
- Write mode creates a pending review.
"""


def test_slug_from_source_id() -> None:
    assert slug_from_source_id("source:test-design-doc") == "test-design-doc"


def test_parse_source_markdown_extracts_claims() -> None:
    parsed = parse_source_markdown(SAMPLE_DOC)
    assert parsed.title == "Test Design Doc"
    assert len(parsed.proposed_claims) >= 5
    assert any("MCP" in c.text for c in parsed.proposed_claims)


def test_build_source_delta_review_includes_human_action() -> None:
    parsed = parse_source_markdown(SAMPLE_DOC)
    content = build_source_delta_review(
        source_id="source:test-design-doc",
        source_kind="design_doc",
        source_rel_path="raw/sources/test-design-doc.md",
        parsed=parsed,
        possibly_affected_specs=["spec:test:compile"],
    )
    assert "# Source Delta Review" in content
    assert "## Proposed claims" in content
    assert "- [ ]" in content
    assert "## Human action needed" in content
    assert "accept/reject proposed claims" in content
    assert "spec:test:compile" in content


def test_ingest_source_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "design.md"
    src.write_text(SAMPLE_DOC, encoding="utf-8")
    corpus = tmp_path / "corpus"
    (corpus / "raw" / "sources").mkdir(parents=True)
    (corpus / "reviews" / "pending").mkdir(parents=True)

    result = ingest_source(
        corpus,
        source_path=src,
        source_id="source:test-design-doc",
        kind="design_doc",
        write_review=True,
        dry_run=True,
    )

    assert result.review_path is None
    assert result.source_path is None
    assert result.content
    assert any("dry run" in w for w in result.warnings)
    assert list((corpus / "reviews" / "pending").glob("*.md")) == []
    assert list((corpus / "raw" / "sources").glob("*.md")) == []


def test_ingest_source_write_mode_only_allowed_dirs(tmp_path: Path) -> None:
    src = tmp_path / "design.md"
    src.write_text(SAMPLE_DOC, encoding="utf-8")
    corpus = tmp_path / "corpus"
    for sub in (
        "raw/sources",
        "reviews/pending",
        "claims/accepted",
        "specs/execution_ready",
        "decisions",
    ):
        (corpus / sub).mkdir(parents=True)

    before_claims = set((corpus / "claims" / "accepted").iterdir())
    before_specs = set((corpus / "specs" / "execution_ready").iterdir())
    before_decisions = set((corpus / "decisions").iterdir())

    result = ingest_source(
        corpus,
        source_path=src,
        source_id="source:test-design-doc",
        kind="design_doc",
        write_review=True,
        dry_run=False,
        now=datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert result.source_path == "raw/sources/test-design-doc.md"
    assert result.review_path == "reviews/pending/source-delta-20260520T120000Z-test-design-doc.md"
    assert (corpus / result.source_path).is_file()
    assert (corpus / result.review_path).is_file()
    assert (corpus / "raw" / "sources" / "source-test-design-doc.yaml").is_file()

    registry = (corpus / "raw" / "sources" / "source-test-design-doc.yaml").read_text(encoding="utf-8")
    assert "source:test-design-doc" in registry
    assert "design_doc" in registry

    review = (corpus / result.review_path).read_text(encoding="utf-8")
    assert "## Proposed claims" in review
    assert "## Human action needed" in review

    assert set((corpus / "claims" / "accepted").iterdir()) == before_claims
    assert set((corpus / "specs" / "execution_ready").iterdir()) == before_specs
    assert set((corpus / "decisions").iterdir()) == before_decisions


def test_ingest_source_missing_path_raises(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(FileNotFoundError):
        ingest_source(
            corpus,
            source_path=tmp_path / "missing.md",
            source_id="source:missing",
            kind="design_doc",
        )


def test_ingest_source_invalid_kind_raises(tmp_path: Path) -> None:
    src = tmp_path / "design.md"
    src.write_text(SAMPLE_DOC, encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="unsupported source kind"):
        ingest_source(
            corpus,
            source_path=src,
            source_id="source:bad-kind",
            kind="not_a_kind",
        )
