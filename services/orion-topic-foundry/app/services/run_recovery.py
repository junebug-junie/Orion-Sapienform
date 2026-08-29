"""Recover runs that a process restart stranded mid-flight.

Runs execute **in this process**, as FastAPI ``BackgroundTasks``
(``enqueue_enrichment``, ``enqueue_training``). ``app/storage/repository.py``
is the only writer of ``topic_foundry_runs`` anywhere in the repo, the
Dockerfile's CMD is a bare ``uvicorn app.main:app`` with no ``--workers``, and
the compose file declares no ``deploy.replicas``. Together those give a real
invariant, not a staleness heuristic:

    at process start, no run can legitimately be `running` or `queued`

-- there is no worker anywhere that could still be advancing it. Any such row
is the residue of a container restart, and it will stay that way forever,
because nothing else ever writes a terminal status for it.

The precondition is precisely **one service instance per database**, which is
stronger than "one replica". ``container_name: orion-${NODE_NAME}-topic-foundry``
says this service is meant to be node-scoped and ``TOPIC_FOUNDRY_PG_DSN`` is a
plain DSN, so if a second node were ever pointed at the same database, node B's
startup would reap node A's live run. Do not relax that without replacing this
invariant with a real lease or heartbeat.

Why this exists (confirmed live 2026-08-29): six runs sat in
``running/enriching``, the oldest for 21 hours, and **zero** runs were
``complete`` for the Orion model. ``fetch_latest_completed_run`` filters on
``status='complete'``, so the concept-atlas ingest returned
``{"available": false, "reason": "topic_foundry_no_completed_run"}`` and the
whole graph had no source run at all. Two of the six were stranded by ordinary
redeploys of this service; one had been stuck since the previous morning.

**The incident had one cause**, and a second latent defect was found while
fixing it. Both are fixed alongside this module; do not conflate them.

1. **The cause.** ``_run_enrichment`` had no ``try/finally``. It wrote
   ``status="running"`` up front and restored the previous status only on the
   success path, so any raise -- or any container restart -- left the run
   ``running`` permanently. All six stranded rows carried a pre-existing
   ``completed_at`` and real ``topics_summary`` artifacts, i.e. they entered
   enrichment already ``complete``; defect 2 would have restored ``complete``
   for every one of them, harmlessly. Defect 1 alone explains all six.
2. **Latent, not the cause.** It restored the status it had *read at entry*,
   so a second pass starting while a first was in flight read ``"running"``
   and wrote it back as the terminal state. Reachable via ``_run_training``'s
   inline ``run_enrichment_sync`` (which genuinely presents
   ``status="running"`` at entry) and via a concurrent manual or smoke-script
   ``POST /runs/{id}/enrich`` -- but **not** via the Hub scheduler, contrary
   to what an earlier version of this docstring claimed. The scheduler
   resolves its target through ``GET /runs?status=complete&limit=1``, so the
   moment pass 1 writes ``status="running"`` the run drops out of that result
   set and cannot be handed to a second pass.

The decision function here is pure and takes counts rather than a connection,
so the policy is testable without a database. See ``tests/test_run_recovery.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("topic-foundry.run-recovery")

# Statuses that cannot survive a process restart, because only this process
# ever advances them. ``repository.list_non_terminal_runs`` builds its SQL
# predicate from this exact set rather than hardcoding the literals, so the
# policy here and the rows the reaper is handed cannot desynchronize -- adding
# a status to one without the other would silently leave every run in it
# un-reaped forever.
NON_TERMINAL_STATUSES = frozenset({"running", "queued"})

# Every status a run is allowed to come to rest in. ``recovery_decision`` and
# ``terminal_status_for_enrichment`` may only ever return one of these; that is
# the property the whole module exists to guarantee.
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


def enrich_refusal_reason(run: Dict[str, Any]) -> Optional[str]:
    """Why ``POST /runs/{id}/enrich`` must refuse this run, or ``None``.

    Enrichment is a post-pass over a run whose segments already exist, and it
    ends by writing a terminal status (see ``terminal_status_for_enrichment``).
    Without this precondition, enriching a ``queued`` or ``running`` run
    promotes it to ``complete`` -- and the Hub resolves "latest completed run"
    by ``created_at DESC``, so a brand-new zero-segment run would win and the
    concept atlas would ingest a run with no segments and no topics.

    Pure, and separate from the route, so the rule is testable without
    importing ``app.routers.runs`` (which pulls in the sklearn/joblib
    training stack).
    """
    status = str(run.get("status") or "").strip().lower()
    if status == "complete":
        return None
    return f"Run is {status or 'unknown'}, not complete; enrichment runs on completed runs only"
