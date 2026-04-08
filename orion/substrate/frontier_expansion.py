from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from orion.core.schemas.frontier_expansion import (
    FrontierExpansionRequestV1,
    FrontierExpansionResponseV1,
    FrontierGraphDeltaBundleV1,
)
from orion.substrate.frontier_context import FrontierContextPackBuilder, FrontierContextPackV1
from orion.substrate.frontier_mapper import FrontierDeltaMapper
from orion.substrate.store import InMemorySubstrateGraphStore


class FrontierExpansionProvider(Protocol):
    def expand(self, *, request: FrontierExpansionRequestV1, context: FrontierContextPackV1) -> FrontierExpansionResponseV1:
        ...


@dataclass(frozen=True)
class FrontierExpansionResultV1:
    context_pack: FrontierContextPackV1
    response: FrontierExpansionResponseV1
    delta_bundle: FrontierGraphDeltaBundleV1


class FrontierExpansionService:
    """Provider-agnostic bounded frontier expansion seam."""

    def __init__(
        self,
        *,
        store: InMemorySubstrateGraphStore,
        provider: FrontierExpansionProvider,
        context_builder: FrontierContextPackBuilder | None = None,
        mapper: FrontierDeltaMapper | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._context_builder = context_builder or FrontierContextPackBuilder()
        self._mapper = mapper or FrontierDeltaMapper()

    def expand(self, *, request: FrontierExpansionRequestV1) -> FrontierExpansionResultV1:
        state = self._store.snapshot()
        context_pack = self._context_builder.build(state=state, request=request)
        response = self._provider.expand(request=request, context=context_pack)
        self._validate_response(request=request, response=response)
        delta_bundle = self._mapper.map_response(request=request, response=response)
        return FrontierExpansionResultV1(context_pack=context_pack, response=response, delta_bundle=delta_bundle)

    @staticmethod
    def _validate_response(*, request: FrontierExpansionRequestV1, response: FrontierExpansionResponseV1) -> None:
        if response.request_id != request.request_id:
            raise ValueError("frontier response request_id mismatch")
        if response.task_type != request.task_type:
            raise ValueError("frontier response task_type mismatch")
        if response.target_zone != request.target_zone:
            raise ValueError("frontier response target_zone mismatch")
