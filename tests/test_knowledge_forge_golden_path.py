from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.compile import compile_context_pack_markdown
from orion.knowledge_forge.lint import lint_corpus
from orion.knowledge_forge.store import KnowledgeStore


def test_live_corpus_lints_clean() -> None:
    root = Path(__file__).resolve().parents[1] / "orion-knowledge"
    store = KnowledgeStore(root)
    store.load()
    report = lint_corpus(store)
    assert report.ok, [f"{i.code}: {i.message}" for i in report.issues]


def test_compile_substrate_telemetry_context_pack() -> None:
    root = Path(__file__).resolve().parents[1] / "orion-knowledge"
    store = KnowledgeStore(root)
    store.load()
    md = compile_context_pack_markdown(
        store,
        spec_id="spec:substrate-tier-telemetry-v1",
        task="Verify substrate tier telemetry persistence matches spec",
    )
    assert "orion-substrate-telemetry" in md
    assert "MindRunRequestV1" in md
