"""The dream cycle as one act in the act ledger (body economy Patch A5).

One sleep that starts replay is one ``make`` act, actor ``dream.cycle``:

* opened when the cycle has passed its gates and has material to replay (a cycle id exists);
* closed once, when the cycle is persisted, or 0 if anything raised (or the task was cancelled)
  after the open. A check that is not due, is held, or finds nothing to replay (the ``empty``
  cycle) opens nothing: no work started, so there is nothing to grade.

The act key is the cycle id, so the act id is stable for the cycle (orion.grammar.acts
stable_act_id). A process restart mid-cycle leaves the open act unclosed; orion-sql-writer
expires it at 0 at its horizon. The next sleep has a new cycle id and a new act.

GPU cost join: while the cycle runs, every REM recombination LLM call carries one correlation id
derived from the cycle id (``cycle_correlation_id``), which the gateway passes to the GPU pool as
the lease's ``turn_correlation_id``. The act carries the same id, so the act ledger's close-time
cost fill (orion/grammar/act_ledger.py fill_close_cost) finds the cycle's leases. Before this, each
call minted its own random id (live 10-16..23: 60 dream leases, 60 distinct ids), so no join.
Known side effect: orion-llm-gateway keys its assistant-reply embedding doc_id on the envelope
correlation id, so a cycle's recombination replies now upsert one embedding doc per cycle (the
last reply) instead of one per call. Hub chat turns already share one id across their calls the
same way. Second side effect: orion-bus-mirror reads each gateway reply -> next request under the
shared id as a cross-organ causal hop (latency = the cycle's think time between calls). At ~4 calls
per cycle, ~2 cycles/day and cycles of ~10 s (max 190 s), that is ~6 extra hop observations a day;
accepted for the cost join. The structural fix (gateway forwards an explicit turn id to the pool,
envelope id fresh per call) is the follow-up named in PR #2890. Known gap (UNVERIFIED until live): the ledger fills GPU cost once, when it processes
the close; a lease released after that query misses the fill.

The event is built here, not with orion.grammar.acts.act_opened_event: that helper only knows
the hub and thought producer identities. ``build_act`` and ``stable_act_id`` are reused as is.
Gated by DREAM_ACT_EMIT_ENABLED. Every publish is best-effort: losing the trace never fails a sleep.
"""

from __future__ import annotations

import contextvars
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.schemas.grammar import ActCostV1, ActOutcomeV1, ActV1, GrammarEventV1, GrammarProvenanceV1

logger = logging.getLogger("orion-dream.act_emit")

ACTOR = "dream.cycle"
VERB = "make"
SOURCE_SERVICE = "orion-dream"
SOURCE_COMPONENT = "dream_cycle.act"

# REM compaction runs inside the sleep with no LLM call; this is its slack on top of the
# recombination calls. Live 30 d (n=81 non-empty cycles, measured 2026-10-23): wall time
# p50 5.5 s, p90 9.7 s, max 190 s, so the horizon below is never the binding constraint
# for a healthy cycle; it is "the deadline the cycle already gives itself".
_REM_AND_PERSIST_SLACK_SEC = 120.0

# The id the cycle's GPU leases carry. Set by the cycle around its recombination calls,
# read by app.llm.complete. Unset (None) = each call mints its own id, as before.
_LEASE_CORRELATION: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "dream_cycle_lease_correlation", default=None
)


def current_lease_correlation() -> Optional[str]:
    return _LEASE_CORRELATION.get()


def set_lease_correlation(value: Optional[str]) -> contextvars.Token:
    return _LEASE_CORRELATION.set(value)


def reset_lease_correlation(token: contextvars.Token) -> None:
    _LEASE_CORRELATION.reset(token)


def cycle_correlation_id(cycle_id: str) -> str:
    """A UUID (bus envelopes need one) derived only from the cycle id."""
    return str(uuid5(NAMESPACE_URL, f"orion:dream.cycle:{cycle_id}"))


def horizon_sec(*, hypotheses_per_cycle: int, control_per_cycle: int, llm_timeout_sec: float) -> float:
    """Recombination is sequential, one LLM call per pair, each bounded by DREAM_LLM_TIMEOUT_SEC."""
    pairs = max(1, int(hypotheses_per_cycle) + int(control_per_cycle))
    return float(pairs) * float(llm_timeout_sec) + _REM_AND_PERSIST_SLACK_SEC


def _basis(horizon: float) -> str:
    return (
        f"artifact_produced (a persisted, completed sleep with >= 1 hypothesis) by {horizon:.0f}s: "
        "upper bound from the cycle's configured pairs, one sequential LLM call per pair x "
        f"DREAM_LLM_TIMEOUT_SEC, + {_REM_AND_PERSIST_SLACK_SEC:.0f}s slack for REM compaction, the "
        "arc/walkway reads and persistence (those DB reads have no deadline of their own); live 30d "
        "wall time p90 9.7s, max 190s (n=81, measured 2026-10-23)"
    )


def cycle_act(cycle_id: str, *, horizon: float) -> ActV1:
    from orion.grammar.acts import build_act

    return build_act(
        actor=ACTOR,
        verb=VERB,
        key=str(cycle_id),
        signal_id="artifact_produced",
        horizon_sec=horizon,
        basis=_basis(horizon),
        object_ref=f"dream_cycle:{cycle_id}",
        test_before=f"dream_cycle:{cycle_id}",
    )


def _event(kind: str, act: ActV1, *, correlation_id: str, at: datetime) -> GrammarEventV1:
    # Same shape as orion.grammar.acts._event; the provenance literal is what
    # tests/test_grammar_event_producer_catalog.py reads to keep channels.yaml honest.
    return GrammarEventV1(
        event_id=f"{kind}:{act.act_id}:{uuid4().hex[:8]}",
        event_kind=kind,  # type: ignore[arg-type]
        trace_id=f"act:{act.act_id}",
        correlation_id=correlation_id,
        emitted_at=datetime.now(timezone.utc),
        observed_at=at,
        layer="act",
        dimensions=["act", act.verb, act.level],
        act=act,
        provenance=GrammarProvenanceV1(source_service="orion-dream", source_component=SOURCE_COMPONENT),
    )


def opened_event(cycle_id: str, *, horizon: float, at: datetime) -> GrammarEventV1:
    return _event("act_opened", cycle_act(cycle_id, horizon=horizon),
                  correlation_id=cycle_correlation_id(cycle_id), at=at)


def closed_event(cycle_id: str, *, horizon: float, produced: bool, note: str, seconds: float,
                 at: datetime) -> GrammarEventV1:
    act = cycle_act(cycle_id, horizon=horizon).model_copy(update={
        "outcome": ActOutcomeV1(value=1.0 if produced else 0.0, observed_at=at, source="producer",
                                note=note[:500]),
        "cost": ActCostV1(seconds=round(max(0.0, seconds), 3)),
    })
    return _event("act_closed", act, correlation_id=cycle_correlation_id(cycle_id), at=at)


def cycle_outcome(cycle: Any, *, persisted: bool) -> tuple[bool, str]:
    """(produced, note). Produced = the sleep is saved, completed, and made at least one hypothesis
    (live 30 d: 3 of 81 non-empty cycles completed with 0 hypotheses, every pair no-link)."""
    status = str(getattr(cycle, "status", "") or "")
    n_hyp = len(getattr(cycle, "hypotheses", None) or [])
    produced = bool(persisted) and status == "completed" and n_hyp > 0
    note = (
        f"status={status} persisted={bool(persisted)} hypotheses={n_hyp} "
        f"llm_failures={getattr(cycle, 'llm_failures', 0)} no_link={getattr(cycle, 'no_link_count', 0)}"
    )
    return produced, note


async def publish_grammar(bus: Any, event: GrammarEventV1) -> None:
    from orion.grammar.publish import publish_grammar_event

    await publish_grammar_event(bus, event, source_name=SOURCE_SERVICE)
