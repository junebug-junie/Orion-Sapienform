from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orion.knowledge_forge.models import (
    ClaimV1,
    ContextPackV1,
    DecisionV1,
    SourceV1,
    SpecV1,
)
from orion.knowledge_forge.yaml_doc import load_yaml_doc

DocModel = SourceV1 | ClaimV1 | DecisionV1 | SpecV1 | ContextPackV1

_MODEL_BY_TYPE: dict[str, type[DocModel]] = {
    "source": SourceV1,
    "claim": ClaimV1,
    "decision": DecisionV1,
    "spec": SpecV1,
    "context_pack": ContextPackV1,
}


@dataclass
class KnowledgeStore:
    root: Path
    by_id: dict[str, DocModel] = field(default_factory=dict)
    paths_by_id: dict[str, Path] = field(default_factory=dict)

    def load(self) -> None:
        self.by_id.clear()
        self.paths_by_id.clear()
        for path in sorted(self.root.rglob("*.yaml")):
            if "reviews/" in path.as_posix():
                continue
            doc = load_yaml_doc(path)
            typed = self._coerce(doc)
            if typed.id in self.by_id:
                raise ValueError(f"duplicate id {typed.id} in {path}")
            self.by_id[typed.id] = typed
            self.paths_by_id[typed.id] = path

    def get(self, doc_id: str) -> DocModel | None:
        return self.by_id.get(doc_id)

    def claims(self) -> list[ClaimV1]:
        return [d for d in self.by_id.values() if isinstance(d, ClaimV1)]

    def specs(self) -> list[SpecV1]:
        return [d for d in self.by_id.values() if isinstance(d, SpecV1)]

    def _coerce(self, doc: dict[str, Any]) -> DocModel:
        doc_type = doc.get("type")
        model = _MODEL_BY_TYPE.get(str(doc_type))
        if model is None:
            raise ValueError(f"unknown document type: {doc_type}")
        return model.model_validate(doc)  # type: ignore[return-value]
