"""A durable run's GPU: one GPU pool *hold* for the whole run (stage 4.5).

Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md, Decision 1.
The pool is the only scheduler; durable-runs no longer grants anything itself. This module is the
thin seam between the run's graph and the pool's lease verbs (orion.gpu_pool.client):

* ``acquire`` a hold (kind=hold, retryable, idempotent on ``<run_id>:<seq>``) -- answers at once
  (granted / queued / backlogged / unavailable), it never waits in line;
* ``status`` to wake a queued run (and to resume after a restart), ``heartbeat`` while the run
  works, ``release`` when it ends.

The hold's ref (``GpuLeaseRefV1``: lease_id, generation, role, holder) is what every LLM call the
run makes carries (``CuriosityTurnRequestV1.gpu_lease``, bus ``options.gpu_lease``); the gateway
then *attaches* the call to the hold instead of queueing a second lease behind it.

A hold's ``role`` (e.g. ``agent-gpu2``) is where the pool put the run. It is NOT a route: it is
never written into ``assigned_lane``, an FCC model label or an ``llm_route`` (spec, "Corrections
from building 4.4" item 2). Held calls name the ``agent`` route (``GPU_LEASE_ROUTE``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from orion.gpu_pool.client import (
    DURABLE_RUN_HOLDER_PREFIX, acquire_hold, durable_run_holder, heartbeat_lease, lease_status, release_lease,
)
from orion.gpu_pool.config import PoolConfig, load_pool_config
from orion.schemas.gpu_pool import GpuLeaseRefV1, GpuLeaseReplyV1

# What a checkpointed ``state["lease"]`` holds once the pool granted the run: the GpuLeaseRefV1
# fields. A pre-cutover checkpoint may still carry a legacy ResourceLeaseV1 dict (lane,
# backend_key, demand_id): ``is_hold_ref`` tells the two apart.
HOLD_REF_KEYS = ("lease_id", "generation", "role", "holder")
# Pool reply statuses that mean "still yours, keep working" for a granted hold. ``recall`` = the
# pool wants the seat back: finish the current node, then release (hold_clawback_grace_sec).
HELD = frozenset({"granted", "recall"})
# Still in line (or re-queued after a lost heartbeat): the same lease_id will be granted later.
WAITING = frozenset({"queued", "backlogged"})


# --- Which pool refusals end a run, and which only mean "the pool cannot answer right now" ------
# Incident 2026-09-26: durable-runs 4.5 came up ~20 s before the 4.5 pool. The old pool's
# GpuLeaseRequestV1 did not know the new ``hold_lease_id``/``hold_generation`` fields (its model is
# extra="forbid"), so it answered every acquire ``unavailable reason=invalid:...extra_forbidden``
# with no lease_id -- and durable-runs failed all 11 resumed runs terminally. A refusal like that
# is a fact about the POOL (its version, its config, its health), not about the run: a rolled pool
# answers the very same request. Such refusals keep the run waiting and retry with backoff.
#
# Reply reasons that say the pool could not even process the request (it never looked at the run):
#   ``invalid:<validation error>`` -- the pool's request model rejected ours: contract/version skew.
#   ``unknown_verb:<verb>``        -- the pool predates the verb: version skew.
POOL_TROUBLE_PREFIXES = ("invalid:", "unknown_verb:")
# Reasons that are properties of the run's own request, whatever pool version answers:
#   ``deadline``                -- the hold's deadline_at IS the run's admission deadline_at: it passed.
#   ``min_ctx_exceeds_class``   -- the run asks for more context than any role of its class serves
#                                  (the pool remembers each role's last live ctx, so a role that is
#                                  briefly restarting does not trigger it).
#   ``backlog_max_age``         -- the pool already waited backlog_max_age_sec with nothing able to
#                                  serve the class: the configured give-up for a backlogged hold.
#   ``replay_payload_too_large``-- the request itself is over the pool's size cap.
RUN_TERMINAL_PREFIXES = ("deadline", "min_ctx_exceeds_class", "backlog_max_age", "replay_payload_too_large")


def is_pool_trouble(reply: GpuLeaseReplyV1) -> bool:
    """A reply to status/heartbeat/release that says nothing about the lease it names: the pool
    could not parse or does not know the verb (version skew). Treat it like an unanswered RPC."""
    return (reply.status == "unavailable" and not reply.lease_id
            and str(reply.reason or "").startswith(POOL_TROUBLE_PREFIXES))


def refusal_is_terminal(cfg: PoolConfig, reason: str, work_class: str) -> bool:
    """Does a pool refusal (``unavailable``/``unknown_lease`` reason) mean this run can never be
    placed? Only when the reason is a property of the run's request, or when durable-runs' OWN
    copy of gpu_pool.yaml -- the config the run was admitted against -- agrees with the refusal.

    * ``unknown_class:<c>`` is terminal only if ``c`` is not a class in our loaded config either.
      If we know the class and the pool does not, the pool is running a different gpu_pool.yaml
      (a config roll in progress): transient.
    * ``operator_only_class`` is terminal only if our config also puts an operator-only role in the
      class (durable-runs is never an operator); otherwise the same config-skew argument applies.
    * Everything else -- ``invalid:*``, ``unknown_verb:*``, an ``unknown_lease`` answer to an
      acquire, an empty reason, any reason this code does not recognise -- is transient. Failing a
      run on a reason nobody classified is exactly the incident this guards against; the run stays
      visible (run.waiting_resource with the reason) and cancellable while it waits.
    """
    reason = str(reason or "")
    if reason.startswith(POOL_TROUBLE_PREFIXES):
        return False
    if reason.startswith(RUN_TERMINAL_PREFIXES):
        return True
    if reason.startswith("unknown_class"):
        named = reason.partition(":")[2] or work_class
        return named not in cfg.classes
    if reason.startswith("operator_only_class"):
        spec = cfg.classes.get(work_class)
        return spec is not None and any(cfg.roles[r].operator_only for r in spec.roles if r in cfg.roles)
    return False


class UnknownRoute(ValueError):
    """The run names a route the pool does not know: it can never be placed, so the run fails
    with that reason instead of waiting forever (the gateway refuses such a route the same way)."""


def is_hold_ref(value: Any) -> bool:
    return (isinstance(value, dict) and bool(value.get("lease_id"))
            and isinstance(value.get("generation"), int)
            and str(value.get("holder") or "").startswith(DURABLE_RUN_HOLDER_PREFIX))


def ref_model(value: dict) -> GpuLeaseRefV1:
    return GpuLeaseRefV1.model_validate({key: value[key] for key in HOLD_REF_KEYS})


def ref_dict(reply: GpuLeaseReplyV1, holder: str) -> dict[str, Any]:
    grant = reply.grant
    if grant is None:
        raise ValueError(f"no grant in a {reply.status} reply")
    return {"lease_id": grant.lease_id, "generation": grant.generation, "role": grant.role, "holder": holder}


def hold_placement(cfg: PoolConfig, admission: dict[str, Any]) -> tuple[str, str, int]:
    """(work_class, priority, min_ctx_tokens) for a run's hold. The class comes from the run's
    route in gpu_pool.yaml ``routes``; ResourceRequirementV1.priority overrides the route's
    (spec Decision 1 rule 1) -- it is always ``background`` today."""
    route = str(admission.get("preferred_lane") or "agent")
    spec = cfg.routes.get(route)
    if spec is None:
        raise UnknownRoute(f"unknown_route:{route}")
    priority = str(admission.get("priority") or spec.priority)
    requirements = admission.get("requirements") or {}
    min_ctx = requirements.get("minimum_context_tokens") or requirements.get("min_ctx_tokens") or 0
    try:
        min_ctx = int(min_ctx)
    except (TypeError, ValueError):
        min_ctx = 0
    return spec.work_class, priority, max(0, min_ctx)


def _deadline(value: Any) -> datetime | None:
    if not value:
        return None
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


class PoolHolds:
    """The four pool verbs durable-runs uses, over the bus. ``bus`` is the service's rpc bus."""

    def __init__(self, bus: Any, *, source: str, cfg: PoolConfig | None = None):
        self.bus, self.source = bus, source
        self.cfg = cfg or load_pool_config()

    @property
    def hold_ttl_sec(self) -> float:
        return float(self.cfg.defaults.hold_lease_ttl_sec)

    def placement(self, admission: dict[str, Any]) -> tuple[str, str, int]:
        return hold_placement(self.cfg, admission)

    async def acquire(self, run_id: str, request_id: str, admission: dict[str, Any], *,
                      correlation_id: str | None) -> GpuLeaseReplyV1:
        work_class, priority, min_ctx = self.placement(admission)
        return await acquire_hold(
            self.bus, holder=durable_run_holder(run_id), work_class=work_class, request_id=request_id,
            priority=priority, deadline_at=_deadline(admission.get("deadline_at")), min_ctx_tokens=min_ctx,
            turn_correlation_id=correlation_id, source=self.source)

    async def status(self, lease_id: str) -> GpuLeaseReplyV1:
        return await lease_status(self.bus, lease_id, source=self.source)

    async def heartbeat(self, lease_id: str) -> GpuLeaseReplyV1:
        return await heartbeat_lease(self.bus, lease_id, source=self.source)

    async def release(self, lease_id: str, *, outcome: str = "ok", detail: str | None = None) -> GpuLeaseReplyV1:
        # outcome must be "ok" or "cancelled": any other value on a granted lease is the pool's
        # ``release_failed``, which sends a retryable hold back to the queue (re-granted to nobody).
        return await release_lease(self.bus, lease_id, source=self.source,
                                   outcome="cancelled" if outcome == "cancelled" else "ok", detail=detail)
