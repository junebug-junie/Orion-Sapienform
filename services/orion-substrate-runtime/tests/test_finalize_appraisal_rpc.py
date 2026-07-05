from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.harness_finalize import GrammarReceiptV1, HarnessDraftMoleculeV1
from orion.schemas.thought import CoalitionSnapshotV1, StanceHarnessSliceV1, ThoughtEventV1

from app.finalize_appraisal_listener import (
    _handle_bus_message,
    handle_finalize_appraisal_request,
)
from app.settings import Settings


def _sample_molecule(*, correlation_id: str = "c-1") -> HarnessDraftMoleculeV1:
    thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id=correlation_id,
        session_id=None,
        created_at=datetime.now(timezone.utc),
        imperative="x",
        tone="y",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )
    return HarnessDraftMoleculeV1(
        correlation_id=correlation_id,
        thought_event_id="t-1",
        draft_text="draft answer",
        draft_hash="abc",
        thought_event=thought,
        grammar_receipts=[GrammarReceiptV1(step_index=0, summary="step")],
        coalition_snapshot=CoalitionSnapshotV1(
            attended_node_ids=["n-1"],
            selected_open_loop_id=None,
            open_loop_ids=[],
            generated_at=datetime.now(timezone.utc),
        ),
    )


@pytest.mark.asyncio
async def test_handle_finalize_appraisal_request_publishes_substrate_appraisal() -> None:
    mol = _sample_molecule()
    bus = AsyncMock()
    settings = Settings(
        POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion",
        CHANNEL_FINALIZE_APPRAISAL_RESULT_PREFIX="orion:substrate:finalize_appraisal:result:",
    )

    appraisal = await handle_finalize_appraisal_request(
        bus,
        mol,
        reply_to="orion:substrate:finalize_appraisal:result:c-1",
        settings=settings,
    )

    assert appraisal.draft_hash == "abc"
    assert appraisal.correlation_id == "c-1"
    assert appraisal.tick_source == "substrate_runtime_finalize_appraisal"
    assert len(appraisal.learning_refs) >= 1
    bus.publish.assert_awaited_once()
    envelope = bus.publish.await_args.args[1]
    assert envelope.kind == "substrate.finalize.appraisal.v1"
    assert envelope.payload["draft_hash"] == "abc"


@pytest.mark.asyncio
async def test_handle_bus_message_validates_harness_draft_molecule() -> None:
    corr_id = uuid4()
    corr = str(corr_id)
    mol = _sample_molecule(correlation_id=corr)
    settings = Settings(
        POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion",
        CHANNEL_FINALIZE_APPRAISAL_RESULT_PREFIX="orion:substrate:finalize_appraisal:result:",
    )
    reply_to = f"orion:substrate:finalize_appraisal:result:{corr}"
    request_envelope = BaseEnvelope(
        kind="harness.draft.molecule.v1",
        source=ServiceRef(name="orion-harness-governor", version="0.1.0"),
        correlation_id=corr_id,
        reply_to=reply_to,
        payload=mol.model_dump(mode="json"),
    )

    class _Codec:
        @staticmethod
        def decode(data):
            class _Decoded:
                ok = True
                envelope = request_envelope
                error = None

            return _Decoded()

    bus = AsyncMock()
    bus.codec = _Codec()

    await _handle_bus_message(bus, {"data": b"ignored"}, settings)

    bus.publish.assert_awaited_once()
    channel, envelope = bus.publish.await_args.args
    assert channel == reply_to
    assert envelope.kind == "substrate.finalize.appraisal.v1"
    assert envelope.payload["correlation_id"] == corr
    assert envelope.payload["draft_hash"] == "abc"


@pytest.mark.asyncio
async def test_handle_bus_message_invalid_payload_publishes_error() -> None:
    corr_id = uuid4()
    corr = str(corr_id)
    settings = Settings(
        POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion",
        CHANNEL_FINALIZE_APPRAISAL_RESULT_PREFIX="orion:substrate:finalize_appraisal:result:",
    )
    reply_to = f"orion:substrate:finalize_appraisal:result:{corr}"
    request_envelope = BaseEnvelope(
        kind="harness.draft.molecule.v1",
        source=ServiceRef(name="orion-harness-governor", version="0.1.0"),
        correlation_id=corr_id,
        reply_to=reply_to,
        payload={"correlation_id": corr},
    )

    class _Codec:
        @staticmethod
        def decode(data):
            class _Decoded:
                ok = True
                envelope = request_envelope
                error = None

            return _Decoded()

    bus = AsyncMock()
    bus.codec = _Codec()

    await _handle_bus_message(bus, {"data": b"ignored"}, settings)

    bus.publish.assert_awaited_once()
    channel, envelope = bus.publish.await_args.args
    assert channel == reply_to
    assert envelope.kind == "system.error"
    assert envelope.payload["error"] == "finalize_appraisal_failed"
