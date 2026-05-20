from __future__ import annotations

from pathlib import Path

import pytest

from orion.knowledge_forge.paths import resolve_corpus_root
from orion.knowledge_forge.yaml_doc import load_yaml_doc, save_yaml_doc


def test_resolve_corpus_root_defaults_to_repo_orion_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo = tmp_path / "repo"
    corpus = fake_repo / "orion-knowledge"
    corpus.mkdir(parents=True)
    monkeypatch.chdir(fake_repo)
    monkeypatch.delenv("ORION_KNOWLEDGE_ROOT", raising=False)
    assert resolve_corpus_root() == corpus.resolve()


def test_yaml_roundtrip_preserves_keys(tmp_path: Path) -> None:
    path = tmp_path / "claim.yaml"
    doc = {"type": "claim", "id": "claim:test:0001", "statement": "x", "status": "accepted", "source_refs": []}
    save_yaml_doc(path, doc)
    loaded = load_yaml_doc(path)
    assert loaded == doc
