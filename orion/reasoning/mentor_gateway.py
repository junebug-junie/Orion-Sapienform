from __future__ import annotations

import logging
from typing import Protocol

from orion.core.schemas.mentor import MentorGatewayResultV1, MentorRequestV1, MentorResponseV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.reasoning.materializer import ReasoningMaterializer
from orion.reasoning.mentor_context import build_mentor_context
from orion.reasoning.mentor_mapper import map_mentor_response_to_proposals
from orion.reasoning.repository import InMemoryReasoningRepository

logger = logging.getLogger("orion.reasoning.mentor_gateway")


class MentorProvider(Protocol):
    def run(self, request: MentorRequestV1, *, context_packet: list[dict]) -> MentorResponseV1:
        ...


class StubMentorProvider:
    """Deterministic stub provider for bounded phase-6 integration/testing."""

    def run(self, request: MentorRequestV1, *, context_packet: list[dict]) -> MentorResponseV1:
        return MentorResponseV1(
            proposal_batch_id=f"stub-batch-{request.request_id}",
            mentor_provider=request.mentor_provider,
            mentor_model=request.mentor_model,
            task_type=request.task_type,
            proposals=[],
        )


class MentorGateway:
    def __init__(
        self,
        *,
        repository: InMemoryReasoningRepository,
        materializer: ReasoningMaterializer | None = None,
        provider: MentorProvider | None = None,
    ) -> None:
        self._repository = repository
        self._materializer = materializer or ReasoningMaterializer(repository)
        self._provider = provider or StubMentorProvider()

    def execute(self, request: MentorRequestV1) -> MentorGatewayResultV1:
        context, packet = build_mentor_context(request, self._repository)
        request = request.model_copy(update={"context": context})
        logger.info(
            "mentor_gateway_request request_id=%s task_type=%s provider=%s model=%s artifact_count=%s",
            request.request_id,
            request.task_type,
            request.mentor_provider,
            request.mentor_model,
            len(context.artifact_ids),
        )

        try:
            response = self._provider.run(request, context_packet=packet)
        except Exception as exc:
            logger.warning("mentor_gateway_failed request_id=%s error=%s", request.request_id, exc)
            return MentorGatewayResultV1(
                request_id=request.request_id,
                success=False,
                failure_reason=str(exc),
                audit={
                    "task_type": request.task_type,
                    "provider": request.mentor_provider,
                    "model": request.mentor_model,
                    "context_artifact_count": len(context.artifact_ids),
                },
            )

        proposals = map_mentor_response_to_proposals(request, response)
        write_result = self._materializer.materialize(
            ReasoningWriteRequestV1(
                context=ReasoningWriteContextV1(
                    source_family="other",
                    source_kind="mentor_gateway",
                    source_channel="orion:mentor:gateway",
                    producer="mentor_gateway",
                    correlation_id=request.correlation_id,
                    trace_id=request.request_id,
                ),
                artifacts=proposals,
                idempotency_key=f"mentor:{request.request_id}:{response.proposal_batch_id}",
            )
        )

        logger.info(
            "mentor_gateway_result request_id=%s task_type=%s proposals=%s materialized=%s",
            request.request_id,
            request.task_type,
            len(response.proposals),
            write_result.stored_count,
        )
        return MentorGatewayResultV1(
            request_id=request.request_id,
            success=True,
            response=response,
            materialized_count=write_result.stored_count,
            materialized_artifact_ids=list(write_result.artifact_ids),
            audit={
                "task_type": request.task_type,
                "provider": request.mentor_provider,
                "model": request.mentor_model,
                "context_artifact_count": len(context.artifact_ids),
                "context_evidence_count": len(context.evidence_refs),
                "proposal_count": len(response.proposals),
            },
        )
