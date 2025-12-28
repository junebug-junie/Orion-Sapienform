# orion/core/bus/router.py
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from pydantic import ValidationError

from .bus_schemas import BaseEnvelope


class OrionRouter:
    """
    Maps `kind` -> concrete envelope model for strict validation.

    Business logic should accept only typed envelope models from this router.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseEnvelope]] = {}

    def register(self, kind: str, model: Type[BaseEnvelope]) -> None:
        self._registry[kind] = model

    def model_for(self, kind: str) -> Optional[Type[BaseEnvelope]]:
        return self._registry.get(kind)

    def validate(self, raw: Dict[str, Any]) -> BaseEnvelope:
        kind = str(raw.get("kind") or "")
        model = self._registry.get(kind) or BaseEnvelope
        return model.model_validate(raw)

    def safe_validate(self, raw: Dict[str, Any]) -> tuple[Optional[BaseEnvelope], Optional[dict]]:
        try:
            return self.validate(raw), None
        except ValidationError as e:
            return None, {"error": "validation_failed", "details": e.errors(), "kind": raw.get("kind")}
