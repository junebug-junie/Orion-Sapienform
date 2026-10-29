"""cortex-orch's shadow bid at dispatch (body economy Patch C3a, SHADOW: logged, never obeyed).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Execution seam":
"cortex-orch is the single bid site for self-initiated verbs ... when cortex-orch is about to
dispatch a verb that is not part of Juniper's turn, it calls ``physiology.client.bid()``, using the
verb's ``physiology`` block". "Interactive turns ... never bid."

When a verb is about to be dispatched, ``shadow_bid`` decides whether it may bid at all
(``bid_skip_reason``), and if so POSTs a ``WantV1`` built from the verb's ``physiology`` block
(``orion.cognition.plan_loader.load_verb_physiology``) through ``orion.physiology.client.abid``
(0.5 s timeout, ``None`` on any failure). The dispatch that follows is byte-for-byte unchanged
whatever comes back: the decision id is only logged on the dispatch trace (``orch_physiology_shadow_bid
corr=...``). When the dispatch returns, ``release`` ends the want's reservation (the act closed) --
by want id, also when the bid timed out client-side (the market may still have reserved for it).

Never bids (each a named skip reason, logged):

* ``turn_ingress``: the request came through cortex-gateway, Juniper's chat ingress;
* ``interactive_lane``: the verb is a chat verb, or the caller pinned the interactive ``chat`` lane;
* ``interactive_priority``: the caller marked it ``interactive`` / ``urgent``;
* ``inside_durable_run``: orion-durable-runs is the caller -- the run already bid at its admission;
* ``nested_exec_call``: cortex-exec is the caller (its capability bridge inside a turn, or a nested
  self-study call inside a verb that already bid) -- the parent turn or dispatch owns the act;
* ``tier_0`` / ``no_physiology`` / ``physiology_invalid``: the verb never bids, or declares nothing.

Gated by ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED. Never raises into dispatch.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.physiology_want import WantDecisionV1, WantV1, WantWindowV1

from .execution_lanes import resolve_execution_lane

logger = logging.getLogger("orion.cortex.orch.physiology_bid")

PRODUCER = "orion-cortex-orch.dispatch"
TURN_INGRESS_SOURCES = frozenset({"cortex-gateway"})
DURABLE_RUNS_SOURCES = frozenset({"orion-durable-runs"})
NESTED_EXEC_SOURCES = frozenset({"cortex-exec"})
INTERACTIVE_PRIORITIES = frozenset({"interactive", "urgent"})
INTERACTIVE_LANE_REASONS = frozenset({"verb_chat"})


def bid_skip_reason(req: CortexClientRequest, source_name: Optional[str]) -> Optional[str]:
    """Why this request must not bid (a turn, or already bid), or None when it may."""
    source = (source_name or "").strip().lower()
    if source in TURN_INGRESS_SOURCES:
        return "turn_ingress"
    if source in DURABLE_RUNS_SOURCES:
        return "inside_durable_run"
    options = req.options if isinstance(req.options, dict) else {}
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    if source in NESTED_EXEC_SOURCES or options.get("policy_dispatch_only") or metadata.get("capability_bridge"):
        return "nested_exec_call"
    if str(options.get("priority") or "").strip().lower() in INTERACTIVE_PRIORITIES:
        return "interactive_priority"
    lane = resolve_execution_lane(req)
    if lane.reason in INTERACTIVE_LANE_REASONS or (lane.explicit and lane.lane == "chat"):
        return "interactive_lane"
    return None


def want_for_dispatch(verb: str, block: Any, correlation_id: str, now: datetime) -> WantV1:
    """The want a tier 1/2 verb dispatch becomes. ``want_id`` is unique per dispatch; the
    correlation id is the request's (the id any act this dispatch opens carries)."""
    return WantV1(
        want_id=f"orch:{correlation_id}:{verb}"[:200], family=block.family, producer=PRODUCER, verb=verb[:120],
        object_ref=f"cortex_verb:{verb}"[:500], tier=block.tier, opened_at=now, window=WantWindowV1(),
        lanes=list(block.lanes), correlation_id=str(correlation_id)[:200])


@functools.lru_cache(maxsize=512)
def _physiology(verb: str) -> tuple[Any, Optional[str]]:
    """The verb's block, or the skip reason. Cached: the verb YAML is baked into the image."""
    from orion.cognition.plan_loader import load_verb_physiology

    try:
        block = load_verb_physiology(verb)
    except FileNotFoundError:
        return None, "no_physiology"     # not a YAML verb at all
    except Exception:  # noqa: BLE001 -- an invalid block (CI keeps them valid) is a skip, never a crash
        return None, "physiology_invalid"
    if block is None:
        return None, "no_physiology"
    if block.tier not in (1, 2):
        return None, f"tier_{block.tier}"
    return block, None


@dataclass(frozen=True)
class ShadowBid:
    """A bid that was sent: its want id (to release by) and the decision, or None (no answer)."""
    want_id: str
    decision: Optional[WantDecisionV1]


async def shadow_bid(req: CortexClientRequest, *, source_name: Optional[str], correlation_id: str,
                     enabled: bool, now: Optional[datetime] = None) -> Optional[ShadowBid]:
    """The bid this dispatch sent, or None when it sent none (skipped or disabled). Logs one line
    either way. Never raises."""
    verb = str(req.verb or "").strip()
    try:
        if not enabled or not verb:
            return None
        skip = bid_skip_reason(req, source_name)
        block = None
        if skip is None:
            block, skip = _physiology(verb)
        if skip is not None:
            logger.info("orch_physiology_shadow_bid_skipped corr=%s verb=%s reason=%s", correlation_id, verb, skip)
            return None
        from orion.physiology.client import abid

        want = want_for_dispatch(verb, block, correlation_id, now or datetime.now(timezone.utc))
    except Exception:  # noqa: BLE001 -- the dispatch never depends on the economy
        logger.warning("orch_physiology_shadow_bid_failed corr=%s verb=%s", correlation_id, verb, exc_info=True)
        return None
    try:
        decision = await abid(want)
    except Exception:  # noqa: BLE001
        decision = None
    if decision is None:
        logger.info("orch_physiology_shadow_bid corr=%s verb=%s family=%s decision_id=none reason=no_answer "
                    "shadow=true obeyed=false", correlation_id, verb, want.family)
        return ShadowBid(want.want_id, None)
    logger.info("orch_physiology_shadow_bid corr=%s verb=%s family=%s decision_id=%s outcome=%s reason=%s "
                "binding=%s shadow=true obeyed=false", correlation_id, verb, want.family, decision.decision_id,
                decision.outcome, decision.reason, decision.binding_constraint)
    return ShadowBid(want.want_id, decision)


async def release(sent: Optional[ShadowBid]) -> None:
    """The dispatch returned (the act closed): end the want's reservation. A funded decision holds
    one; so may a bid with no answer (the client gave up, the market may not have). A deferral
    holds none. Never raises."""
    if sent is None:
        return
    d = sent.decision
    if d is not None and (d.outcome != "funded" or not d.reserved):
        return
    try:
        from orion.physiology.client import arelease

        await arelease(sent.want_id)
    except Exception:  # noqa: BLE001
        logger.info("orch_physiology_shadow_release_failed want=%s", sent.want_id, exc_info=True)


__all__ = ["PRODUCER", "ShadowBid", "bid_skip_reason", "release", "shadow_bid", "want_for_dispatch"]
