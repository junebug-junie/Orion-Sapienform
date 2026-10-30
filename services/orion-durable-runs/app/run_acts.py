"""Act-ledger producer for durable runs Orion starts for itself (body economy Patch A5, producer 1).

docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Patch A results" -> producer
split item 1. Every run in a WANT family below writes an efference copy to ``grammar_acts``
(orion/grammar/act_ledger.py, via orion-sql-writer): ``act_opened`` when the run starts real work,
``act_closed`` when it ends. The economy grades and prices these families from those rows.

When an act opens and closes:

* **Open** on the call that records the run's FIRST pool grant (``run.resource_granted``): the run
  starts real work there. The GPU queue wait before it is the scheduler's, not the act's (the same
  rule the reading act uses, services/orion-hub/scripts/act_emit.py READ_BASIS). A run refused or
  cancelled before any grant never did anything and opens no act.
* **Close** in ``AdmissionRuntime._terminal`` with the run's actual terminal status: ``completed``
  -> 1, anything else (failed, cancelled; the registry has no other terminal) -> 0. A run that never
  reaches a terminal (driver bug, crash after the projection) is closed at 0 by the ledger's own
  expiry at ``horizon_at`` + EXPIRY_GRACE_SEC: every act here carries an ``artifact_produced``
  expectation for exactly that reason.

Idempotency: ``act_id = stable_act_id(actor, verb, run_id)``. The open is published only by the
call whose ``record_event`` actually INSERTED the run's first grant row, so a restart that re-reads
the same grant (ON CONFLICT DO NOTHING) or a later re-grant publishes nothing. Should an open still
arrive twice, the writer ignores it (``ON CONFLICT ... WHERE opened_at IS NULL``); the first close
wins. The one gap: a crash between that insert and the publish loses the open; the close then
creates the row itself (act_ledger close-before-open path) and the act is still graded.

Correlation: the act carries the run's ``correlation_id``, which is the ``turn_correlation_id`` on
the run's pool HOLD lease (``PoolHolds.acquire(..., correlation_id=state["correlation_id"])``). The
ledger's GPU cost fill (PR #2864) joins on it. Child calls inside a curiosity turn carry the turn's
own fenced id (app/graph.py ``turn_correlation_id``) and are not joined; live 7 d to 2026-10-23
they are about 4 % of these families' slot time, and the ones on the hold's own lane share its slot.

Cost: the producer sends only ActCostV1 ``seconds`` = wall time from the first grant to the
terminal. Never an ActCostV2-only field: the ledger fills GPU slot-seconds itself.

Gated by ORION_DURABLE_ACT_EMIT_ENABLED. Best-effort: never raises into the run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from orion.grammar.acts import act_closed_event, act_opened_event, build_act
from orion.grammar.publish import GRAMMAR_EVENT_CHANNEL
from orion.schemas.grammar import ActCostV1, ActV1, ActVerb, GrammarEventV1
from orion.schemas.journal_compose_run import JOURNAL_COMPOSE_WORKFLOW

logger = logging.getLogger(__name__)

GRAMMAR_EVENT_KIND = "grammar.event.v1"
SOURCE_SERVICE = "orion-durable-runs"
GRANT_EVENT = "run.resource_granted"
# Same strings as app.admission_runtime DEFAULT_WORKFLOW / SELF_SENSE_WORKFLOW (that module imports
# this one; tests/test_run_acts.py pins the equality).
CURIOSITY_WORKFLOW = "curiosity.investigate"
SELF_SENSE_WORKFLOW = "self_sense_eval"
# Margin past a run's own admission deadline: the driver fails a run past its deadline on its next
# drive (one admission tick), so its producer close arrives inside this.
DEADLINE_HORIZON_MARGIN_SEC = 600.0


@dataclass(frozen=True)
class RunActFamily:
    family: str
    actor: str
    verb: ActVerb
    horizon_sec: float
    basis: str


# Horizons: from the run's first pool grant to its terminal, measured 2026-10-23 over 30 days of
# durable_admission_runs + durable_resource_events (non-urgent runs). The horizon only backstops a
# run that never reaches a terminal: a late producer close loses to the expiry (first close wins),
# so each horizon covers the slowest completed run seen, not the median.
_CURIOSITY_BASIS = (
    "artifact_produced (the durable run ends completed) by 86400s after its first GPU pool grant: "
    "first-grant->terminal over 30d for completed {line} runs p99 {p99}s, max {max}s (n={n}, measured "
    "2026-10-23); the run closes its own act at its terminal, the horizon only expires a run that never ends"
)
FAMILIES: dict[str, RunActFamily] = {
    "curiosity_investigate": RunActFamily(
        "curiosity_investigate", "durable.curiosity_investigate", "infer", 86400.0,
        _CURIOSITY_BASIS.format(line="investigate", p99=55648, max=74107, n=147)),
    "self_inquiry": RunActFamily(
        "self_inquiry", "durable.self_inquiry", "ask", 86400.0,
        _CURIOSITY_BASIS.format(line="self_inquiry", p99=49700, max=52130, n=157)),
    "self_sense_eval": RunActFamily(
        "self_sense_eval", "durable.self_sense_eval", "sense", 43200.0,
        "artifact_produced (the durable run ends completed) by 43200s after its first GPU pool grant: "
        "first-grant->terminal over 30d for completed self_sense_eval runs p99 14843s, max 25905s (n=181, "
        "measured 2026-10-23); the run closes its own act at its terminal, the horizon only expires a run "
        "that never ends"),
    "journal_compose": RunActFamily(
        "journal_compose", "durable.journal_compose", "make", 3600.0,
        "artifact_produced (the durable run ends completed) by 3600s after its first GPU pool grant, or by "
        "the run's own admission deadline + 600s when that is later: first-grant->terminal over 30d for "
        "completed journal.compose runs p99 26s, max 27s (n=26, measured 2026-10-23)"),
}


def run_act_family(state: dict[str, Any]) -> RunActFamily | None:
    """The want family of a run, or None for a run that is not one of Orion's own wants.

    Included: curiosity.investigate (line investigate -> infer, self_inquiry -> ask),
    self_sense_eval (sense), journal.compose (make). Excluded on purpose:

    * urgent curiosity runs (``brief.urgent`` set): an incident report a sensor alarm forced
      (cabinet AC, GPU heat), a reflex rather than a want;
    * every other workflow: reading.turn and reverie.visual already have acts in Hub and thought,
      dream.carry belongs to the dream producer, and memory.episode_distill, curiosity.compaction,
      compactor.digest and orion_day.letter are upkeep, not wants (design Organ 3b
      Tier 1). temporal_self / physiology / personality / situation are timer threads, not durable runs.
    """
    workflow = state.get("workflow") or CURIOSITY_WORKFLOW
    brief = state.get("brief") if isinstance(state.get("brief"), dict) else {}
    if workflow == CURIOSITY_WORKFLOW:
        if brief.get("urgent"):
            return None
        line = brief.get("line") or "investigate"
        if line == "self_inquiry":
            return FAMILIES["self_inquiry"]
        if line == "investigate":
            return FAMILIES["curiosity_investigate"]
        return None
    if workflow == SELF_SENSE_WORKFLOW:
        return FAMILIES["self_sense_eval"]
    if workflow == JOURNAL_COMPOSE_WORKFLOW:
        return FAMILIES["journal_compose"]
    return None


def _horizon_sec(family: RunActFamily, state: dict[str, Any], opened_at: datetime) -> float:
    deadline = (state.get("admission") or {}).get("deadline_at")
    if not deadline:
        return family.horizon_sec
    try:
        remaining = (datetime.fromisoformat(str(deadline).replace("Z", "+00:00")) - opened_at).total_seconds()
    except (TypeError, ValueError):
        return family.horizon_sec
    return max(family.horizon_sec, remaining + DEADLINE_HORIZON_MARGIN_SEC)


def build_run_act(family: RunActFamily, state: dict[str, Any], *, opened_at: datetime, grant_entry_id: str) -> ActV1:
    run_id = str(state["run_id"])
    act = build_act(
        actor=family.actor,
        verb=family.verb,
        key=run_id,
        signal_id="artifact_produced",
        horizon_sec=_horizon_sec(family, state, opened_at),
        basis=family.basis,
        object_ref=f"durable_run:{run_id}",
        test_before=f"gpu_pool_grant:{grant_entry_id}",
    )
    return act


def _session(state: dict[str, Any]) -> str | None:
    brief = state.get("brief") if isinstance(state.get("brief"), dict) else {}
    value = brief.get("session_id")
    return str(value) if value else None


def opened_event(family: RunActFamily, state: dict[str, Any], *, opened_at: datetime, grant_entry_id: str) -> GrammarEventV1:
    act = build_run_act(family, state, opened_at=opened_at, grant_entry_id=grant_entry_id)
    return act_opened_event(act, source_service=SOURCE_SERVICE, source_component=f"durable_run.{family.family}.act",
                            correlation_id=str(state.get("correlation_id") or "") or None,
                            session_id=_session(state), at=opened_at)


def closed_event(family: RunActFamily, state: dict[str, Any], *, status: str, opened_at: datetime,
                 grant_entry_id: str, closed_at: datetime) -> GrammarEventV1:
    act = build_run_act(family, state, opened_at=opened_at, grant_entry_id=grant_entry_id)
    wall = (closed_at - opened_at).total_seconds()
    note = status
    error = state.get("last_error")
    if status != "completed" and error:
        note = f"{status}: {error}"
    return act_closed_event(
        act, produced=status == "completed", source_service=SOURCE_SERVICE,
        source_component=f"durable_run.{family.family}.act", note=note,
        cost=ActCostV1(seconds=round(wall, 3)) if wall >= 0 else None,
        correlation_id=str(state.get("correlation_id") or "") or None,
        session_id=_session(state), at=closed_at)


Publish = Callable[[str, str, Any, Any], Awaitable[bool]]


class RunActEmitter:
    """Opens and closes the act of each want-family durable run. ``publish`` is the runner's
    ``_publish(channel, kind, model, corr) -> bool`` (reconnecting, never raises)."""

    def __init__(self, *, enabled: bool, store: Any, publish: Publish, corr_for: Callable[[Any], Any], now):
        self.enabled = bool(enabled)
        self.store, self.publish, self.corr_for, self.now = store, publish, corr_for, now
        self._opened: set[str] = set()   # runs whose open this process already published

    async def _send(self, event: GrammarEventV1, state: dict[str, Any]) -> bool:
        return await self.publish(GRAMMAR_EVENT_CHANNEL, GRAMMAR_EVENT_KIND, event,
                                  self.corr_for(state.get("correlation_id")))

    async def on_grant(self, state: dict[str, Any], granted: Any) -> bool:
        """After ``store.record_event(run.resource_granted)`` returned ``granted`` (the event payload):
        open the act when THIS call inserted the run's first grant. A later grant (re-queue after a
        lost hold) or a re-read of the same grant after a restart (ON CONFLICT DO NOTHING: the stored
        row keeps its original time) publishes nothing, so a resume never re-opens. True when an
        open was published."""
        if not self.enabled or not isinstance(granted, dict):
            return False
        run_id = str(state.get("run_id") or "")
        try:
            family = run_act_family(state)
            if family is None or not run_id or run_id in self._opened:
                return False
            first = await self.store.first_event(run_id, GRANT_EVENT)
            at = granted.get("generated_at")
            at = datetime.fromisoformat(at) if isinstance(at, str) else at
            if first is None or first[0] != granted.get("entry_id") or first[1] != at:
                return False
            sent = await self._send(opened_event(family, state, opened_at=first[1], grant_entry_id=first[0]), state)
            if sent:
                self._opened.add(run_id)
            return sent
        except Exception:  # noqa: BLE001 -- the act is a trace of the run; never block the run
            logger.warning("durable_act_open_failed run=%s", run_id, exc_info=True)
            return False

    async def on_terminal(self, state: dict[str, Any], status: str | None) -> bool:
        """After ``finish_projection`` returned the run's actual terminal status: close its act.
        A run with no grant never opened one and closes nothing. True when a close was published."""
        run_id = str(state.get("run_id") or "")
        self._opened.discard(run_id)
        if not self.enabled or status not in ("completed", "failed", "cancelled"):
            return False
        try:
            family = run_act_family(state)
            if family is None or not run_id:
                return False
            first = await self.store.first_event(run_id, GRANT_EVENT)
            if first is None:
                return False
            event = closed_event(family, state, status=status, opened_at=first[1], grant_entry_id=first[0],
                                 closed_at=self.now())
            return await self._send(event, state)
        except Exception:  # noqa: BLE001
            logger.warning("durable_act_close_failed run=%s status=%s", run_id, status, exc_info=True)
            return False
