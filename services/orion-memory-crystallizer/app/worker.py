"""Bus worker: ingest crystallization proposals published by other services.

Listens on orion:memory:crystallization:proposed, validates payloads, and
persists them as proposals. It never canonizes: approval stays behind the
governor HTTP endpoints (or future policy hooks).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orion.memory.crystallization import governor
from orion.memory.crystallization.repository import CrystallizationRepository
from orion.schemas.memory_crystallization import MemoryCrystallizationV1

logger = logging.getLogger("orion.memory-crystallizer.worker")


class ProposalIngestWorker:
    def __init__(self, *, bus: Any, repo: CrystallizationRepository, channel: str, service_name: str):
        self._bus = bus
        self._repo = repo
        self._channel = channel
        self._service_name = service_name
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._bus is None:
            logger.info("bus disabled; proposal ingest worker not started")
            return
        self._task = asyncio.create_task(self._run(), name="crystallizer-proposal-ingest")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        logger.info("proposal_ingest_started channel=%s", self._channel)
        try:
            async with self._bus.subscribe(self._channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    try:
                        await self._handle_message(msg)
                    except Exception as exc:
                        logger.warning("proposal_ingest_failed reason=%s", exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("proposal_ingest_loop_crashed reason=%s", exc)

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        decoded = self._bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            logger.warning("proposal_decode_failed error=%s", decoded.error)
            return
        envelope = decoded.envelope
        source = getattr(getattr(envelope, "source", None), "name", "")
        if source == self._service_name:
            return  # our own publication echo; already persisted

        payload = envelope.payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        try:
            proposal = MemoryCrystallizationV1.model_validate(payload)
        except Exception as exc:
            logger.warning("proposal_schema_invalid reason=%s", exc)
            return
        if proposal.status != "proposed":
            logger.warning(
                "proposal_ignored cid=%s status=%s (only 'proposed' accepted on this channel)",
                proposal.crystallization_id,
                proposal.status,
            )
            return

        validated, entry = governor.validate(proposal, source or "bus")
        # Guarded transactional write: an existing row is only updated while
        # still 'proposed', so this can never revert governed state.
        written = await asyncio.to_thread(
            self._repo.apply_transition,
            validated,
            entry,
            after=validated.model_dump(mode="json"),
            expected_statuses=["proposed"],
        )
        if not written:
            logger.warning(
                "proposal_conflict cid=%s — row already governed; refusing to overwrite",
                proposal.crystallization_id,
            )
            return
        logger.info(
            "proposal_ingested cid=%s kind=%s validation=%s",
            validated.crystallization_id,
            validated.kind,
            validated.governance.validation_status,
        )
