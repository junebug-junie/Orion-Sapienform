"""Recover runs that a process restart stranded mid-flight.

Runs execute **in this process**, as FastAPI ``BackgroundTasks``
(``enqueue_enrichment``, ``enqueue_training``). ``services/orion-topic-foundry/
docker-compose.yml`` declares no ``deploy.replicas``, so there is exactly one.
Together those give a real invariant, not a staleness heuristic:

    at process start, no run can legitimately be `running` or `queued`

-- there is no worker anywhere that could still be advancing it. Any such row
is the residue of a container restart, and it will stay that way forever,
because nothing else ever writes a terminal status for it.

Why this exists (confirmed live 2026-08-29): six runs sat in
``running/enriching``, the oldest for 21 hours, and **zero** runs were
``complete`` for the Orion model. ``fetch_latest_completed_run`` filters on
``status='complete'``, so the concept-atlas ingest returned
``{"available": false, "reason": "topic_foundry_no_completed_run"}`` and the
whole graph had no source run at all. Two of the six were stranded by ordinary
redeploys of this service; one had been stuck since the previous morning.

Two distinct defects produced them, both fixed alongside this module:

1. ``_run_enrichment`` had no ``try/finally``. It wrote ``status="running"``
   up front and restored the previous status only on the success path, so any
   raise -- or any container restart -- left the run ``running`` permanently.
2. It restored the status it had *read at entry*. A second enrichment started
   while a first was in flight read ``"running"`` and wrote ``"running"``
   back as the terminal state, latching the run even on a clean finish. The
   scheduler triggers enrichment every tick, so this was reachable in normal
   operation.

The decision function here is pure and takes counts rather than a connection,
so the policy is testable without a database. See ``tests/test_run_recovery.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("topic-foundry.run-recovery")

# Statuses that cannot survive a process restart, because only this process
# ever advances them.
NON_TERMINAL_STATUSES = frozenset({"running", "queued"})

TERMINAL_STATUSES = frozenset({"complete", "failed"})

INTERRUPTED_ERROR = (
    "Run was interrupted by a service restart and produced no segments; "
    "marked failed at startup by run_recovery (see app/services/run_recovery.py)."
)


@dataclass(frozen=True)
class RunRecovery:
    """The terminal state a stranded run should be moved to."""

    run_id: str
    status: str
    stage: str
    error: Optional[str]


def recovery_decision(
    run: Dict[str, Any], *, segment_count: int, enriched_count: int
) -> Optional[RunRecovery]:
    """Decide how to close out one possibly-stranded run.

    Returns ``None`` when the run needs no action -- it is already terminal.

    A run that produced segments did real work, so it is closed as
    ``complete``: its topics are usable, and marking it ``failed`` would
    withhold a perfectly good run from ``fetch_latest_completed_run`` for no
    reason. Its stage records how far it actually got, from the segments
    themselves rather than from the stage string it was interrupted at --
    that string is exactly what cannot be trusted after a crash.

    A run that produced nothing is ``failed`` with an explicit error, because
    there is nothing to serve and silence would leave an operator guessing.
    """
    status = str(run.get("status") or "").strip().lower()
    if status not in NON_TERMINAL_STATUSES:
        return None
    run_id = str(run.get("run_id"))
    if segment_count > 0:
        return RunRecovery(
            run_id=run_id,
            status="complete",
            stage="enriched" if enriched_count > 0 else "trained",
            # Deliberately no error: this run is usable, and writing an error
            # onto a run that will be served as `complete` would make every
            # future reader wonder what is wrong with it.
            error=None,
        )
    return RunRecovery(run_id=run_id, status="failed", stage="failed", error=INTERRUPTED_ERROR)


def terminal_status_for_enrichment(run: Dict[str, Any]) -> str:
    """The status ``_run_enrichment`` should restore when it finishes.

    Never echoes back a non-terminal status. Enrichment is a post-pass over a
    run whose segments already exist, so "finished enriching" means the run is
    complete; the only way to end ``running`` here was the latch described in
    this module's docstring.

    ``failed`` is preserved: enrichment refuses to touch a failed run at all,
    and if that guard is ever relaxed, a failed run must not be silently
    promoted to complete by a successful enrichment pass.
    """
    status = str(run.get("status") or "").strip().lower()
    if status == "failed":
        return "failed"
    return "complete"


def recover_stranded_runs() -> int:
    """Close out every non-terminal run found at process start.

    Called once from ``app/main.py``'s lifespan, before anything can read a
    run. Best-effort: a failure here must never stop the service from
    starting, since the service is what would let an operator diagnose it.

    Returns the number of runs moved to a terminal status.
    """
    from app.storage.repository import (  # imported here to keep the policy above import-light
        count_segments,
        list_non_terminal_runs,
        update_run,
        utc_now,
    )
    from uuid import UUID

    from app.models import RunRecord, RunSpecSnapshot

    try:
        stranded = list_non_terminal_runs()
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_recovery_scan_failed error=%s", exc)
        return 0

    if not stranded:
        logger.info("run_recovery_scan_clean stranded=0")
        return 0

    recovered = 0
    for run in stranded:
        run_id = run.get("run_id")
        try:
            segment_count = count_segments(UUID(str(run_id)))
            enriched_count = count_segments(UUID(str(run_id)), has_enrichment=True)
            decision = recovery_decision(
                run, segment_count=segment_count, enriched_count=enriched_count
            )
            if decision is None:
                continue
            record = RunRecord(
                run_id=UUID(decision.run_id),
                model_id=UUID(str(run["model_id"])),
                dataset_id=UUID(str(run["dataset_id"])),
                specs=RunSpecSnapshot(**run["specs"]),
                spec_hash=run.get("spec_hash"),
                status=decision.status,
                stage=decision.stage,
                stats=run.get("stats") or {},
                artifact_paths=run.get("artifact_paths") or {},
                created_at=run["created_at"],
                started_at=run.get("started_at"),
                completed_at=run.get("completed_at") or utc_now(),
                error=decision.error or run.get("error"),
            )
            update_run(record)
            recovered += 1
            logger.warning(
                "run_recovery_closed_stranded_run run_id=%s status=%s stage=%s segments=%s enriched=%s "
                "-- was '%s/%s' with no worker able to advance it",
                decision.run_id,
                decision.status,
                decision.stage,
                segment_count,
                enriched_count,
                run.get("status"),
                run.get("stage"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_recovery_failed run_id=%s error=%s", run_id, exc)

    logger.warning(
        "run_recovery_complete stranded=%s recovered=%s", len(stranded), recovered
    )
    return recovered
