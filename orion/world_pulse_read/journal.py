"""Replayable journal commands using the existing journal write/index rail."""
from uuid import NAMESPACE_URL, uuid5

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.journaler.schemas import JournalEntryWriteV1
from orion.world_pulse_read.queue import request_for_seed


def journal_entry(handoff, result=None, *, round_trips=None):
    if round_trips is None:
        round_trips = result.round_trips if result else 0
    seed = handoff.seed_ref
    request = request_for_seed(seed)
    trace_id = result.trace_id if result else handoff.trace_id
    source_ref = f"world_pulse_read_stage2:{trace_id}" if result else f"world_pulse_read:{trace_id}"
    body = (result.summary if result else handoff.what_i_learned).strip()
    body += f"\n\n{seed.url}\ntrace_id={trace_id}\nreading_request={request.model_dump(mode='json')}"
    if result:
        body += f"\nstage1_trace_id={handoff.trace_id}\nstage2_trace_id={trace_id}\nround_trips={round_trips}"
    return JournalEntryWriteV1(
        entry_id=str(uuid5(NAMESPACE_URL, source_ref)),
        created_at=result.created_at if result else handoff.created_at,
        author="orion", mode="manual", title=seed.title or ("Reading follow-up" if result else "Reading"),
        body=body, source_kind="world_pulse" if request.requested_by == "world_pulse" else "self_study",
        source_ref=source_ref, correlation_id=trace_id,
    )


async def publish_journal(bus, source, handoff, result=None, *, round_trips=None):
    if bus is None or getattr(bus, "enabled", True) is False:
        raise RuntimeError("journal_bus_unavailable")
    entry = journal_entry(handoff, result, round_trips=round_trips)
    await bus.publish("orion:journal:write", BaseEnvelope(
        kind="journal.entry.write.v1", source=source, payload=entry.model_dump(mode="json"),
    ))
