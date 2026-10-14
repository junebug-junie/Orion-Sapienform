"""Sleep only admits due windows; durable-runs owns all LLM work and publication."""
import asyncio
import logging
from uuid import NAMESPACE_URL, uuid5

from orion.schemas.curiosity_compaction import CURIOSITY_COMPACTION_WORKFLOW, CuriosityCompactionBriefV1
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1

logger = logging.getLogger(__name__)


def request_for(row):
    return DurableRunRequestV1(
        run_id=row["run_id"], workflow=CURIOSITY_COMPACTION_WORKFLOW,
        correlation_id=str(uuid5(NAMESPACE_URL, row["run_id"])),
        brief=CuriosityCompactionBriefV1(compaction_id=row["compaction_id"]),
        admission=ResourceRequirementV1(resource="llm.route.agent", preferred_lane="agent", priority="background"),
    )


async def _db_call(fn, *args):
    # to_thread cancellation does not stop the worker. Wait for it before letting
    # the caller close/reuse its connection, even during shutdown.
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


async def submit_due(store, submit, publish, *, now, cycle_id):
    for cadence in ("weekly", "monthly"):
        session = None
        try:
            acquisition = asyncio.create_task(asyncio.to_thread(store.acquire, cadence))
            try:
                session = await asyncio.shield(acquisition)
            except asyncio.CancelledError:
                session = await acquisition  # finally owns a late-acquired session too
                raise
            if session is None:
                continue
            row = await _db_call(session.prepare, now, cycle_id)
            if row:
                reason = await submit(request_for(row))
                if reason:
                    raise RuntimeError(reason)
                logger.info("curiosity_compaction_admitted window=%s run=%s cycle=%s", row["compaction_id"], row["run_id"], cycle_id)
            # Repair delivery, never regenerate a finished artifact. SQL existence is the ack.
            from orion.curiosity.compaction import journal_entry
            for pending in await _db_call(session.unjournaled):
                await publish(journal_entry(pending))
        except Exception:
            logger.exception("curiosity_compaction_admission_failed cadence=%s cycle=%s", cadence, cycle_id)
        finally:
            if session is not None:
                await _db_call(session.close)
