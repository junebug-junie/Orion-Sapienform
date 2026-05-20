from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.probes import probe_source_coverage
from orion.knowledge_forge.store import KnowledgeStore


def test_probe_flags_uncited_sentences_in_source() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    source = root / "raw" / "sources" / "test-source.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "Alpha requirement must persist telemetry.\nBeta requirement is unrelated.\n",
        encoding="utf-8",
    )
    store = KnowledgeStore(root)
    store.load()
    report = probe_source_coverage(
        store,
        source_path=source,
        source_id="source:test:fixture",
        min_keyword="telemetry",
    )
    assert report.ok is False
    assert "uncited_keyword" in {i.code for i in report.issues}
