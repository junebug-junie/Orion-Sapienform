from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.compile import compile_context_pack_markdown
from orion.knowledge_forge.store import KnowledgeStore


def test_compile_context_pack_includes_only_accepted_claims() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    store = KnowledgeStore(root)
    store.load()
    md = compile_context_pack_markdown(store, spec_id="spec:test:compile", task="Ship health endpoint")
    assert "## Goal" in md
    assert "Expose GET /health" in md
    assert "Fixture claim for store indexing" in md
    assert "## Known traps" in md
    assert "Do not import hub redis" in md
