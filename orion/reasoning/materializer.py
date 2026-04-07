from __future__ import annotations

from orion.core.schemas.reasoning_io import ReasoningWriteRequestV1, ReasoningWriteResultV1
from orion.reasoning.repository import InMemoryReasoningRepository, ReasoningRepository


class ReasoningMaterializer:
    """Narrow write/materialization seam for canonical reasoning artifacts."""

    def __init__(self, repository: ReasoningRepository | None = None) -> None:
        self._repository = repository or InMemoryReasoningRepository()

    @property
    def repository(self) -> ReasoningRepository:
        return self._repository

    def materialize(self, request: ReasoningWriteRequestV1) -> ReasoningWriteResultV1:
        return self._repository.write_artifacts(request)
