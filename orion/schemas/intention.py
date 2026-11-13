"""Intentions Orion takes on (Step -1 of docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md).

One primitive with one adapter per source. Prime holds a concern; with the ``take_on`` move it decides to carry the
concern through. Prime opens a ``make`` act (actor ``prime``, ``object_ref intention:<concern_id>``) and orders ONE
durable run from Hub. Hub calls the source's adapter (orion/prime/intention.py ``ADAPTERS``), builds the brief and
submits the run (workflow ``intention.carry``, family ``durable.intention``, tier 1 by Juniper's decision 2026-11-13).
The run is one unified turn (session ``orion_intention``) whose prompt starts with ``prime_concern:<id>``, so the
turn hand-back settles the concern. When the run is terminal, Prime asks Hub to deliver the reply on the adapter's
route and closes the act: 1 = the reply was stored on its route.

    Prime  --IntentionOrderV1  (POST INTENTION_ORDER_PATH)-->   Hub  --DurableRunRequestV1-->  orion-durable-runs
    Prime  <--IntentionOrderResultV1--                          Hub
    Prime  --GET /runs/<run_id>-->                                                             orion-durable-runs
    Prime  --IntentionDeliverV1 (POST INTENTION_DELIVER_PATH)--> Hub: chat thread history + one outreach attempt
    Prime  <--IntentionDeliverResultV1--                         Hub

HTTP only: no new bus channel, no new table.

ROLLOUT: ``intention.carry`` is an ADDITIVE ``DurableWorkflowV1`` value on ``Literal``/``extra="forbid"`` models.
Deploy the state consumers (orion-sql-writer, orion-hub, orion-actions) and orion-cortex-orch, then
orion-durable-runs, and orion-mind (Prime, the only producer of orders) last.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

INTENTION_CARRY_WORKFLOW = "intention.carry"
# The want family (config/physiology/families.yaml, tier 1) and act-ledger actor of the run's own act.
INTENTION_FAMILY = "durable.intention"
# The carry turn's Hub session. Prime's turn hand-back reads it as "intention" (orion/prime/handback.py); the
# loop guard keys on it: a promise made inside this session opens no concern.
INTENTION_SESSION_ID = "orion_intention"
INTENTION_CARRY_TAG = "intention_carry"
INTENTION_CARRY_STEP_KEY = "carry"
INTENTION_CARRY_NODES: tuple[str, ...] = ("harness_step", "finish")
INTENTION_ORDER_PATH = "/api/intention/run"
INTENTION_DELIVER_PATH = "/api/intention/deliver"
# The carry turn's budget per attempt (the attempt timer and Hub's turn limit derive from it).
INTENTION_TURN_TIMEOUT_SEC = 1800.0
CONCERN_ID_PATTERN = r"^[0-9a-f]{16}$"

# One value per adapter. ``letter_thread`` is declared now so the second adapter is one file plus one map entry
# (acceptance check S-1.9), with no schema change.
IntentionSource = Literal["turn_promise", "letter_thread"]
IntentionRoute = Literal["chat_thread", "next_letter", "notes_only"]


def intention_run_id(concern_id: str) -> str:
    """One durable run per take-on (no retries in v1), so one per concern."""
    return f"intention-{concern_id}"


def intention_correlation_id(concern_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"orion:intention:{concern_id}"))


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentionSourceV1(_Model):
    """What an adapter returns for one concern: the turn's task, its deadline and where the result goes."""

    source: IntentionSource
    # The concern's own deadline (the run's admission deadline: WantWindowV1.latest).
    horizon_at: AwareDatetime
    prompt: str = Field(min_length=1, max_length=20000)
    route: IntentionRoute
    # chat_thread: the chat session_id of the originating exchange.
    route_target: Optional[str] = Field(default=None, max_length=200)
    # harness_turn_trace:<corr>#promises_made[i] (the intention ledger's reference form).
    quote_ref: Optional[str] = Field(default=None, max_length=300)
    # What recall searches for during the turn.
    retrieval_query: Optional[str] = Field(default=None, max_length=1000)


class IntentionOrderV1(_Model):
    """``POST {HUB}/api/intention/run``: Prime took a concern on and orders its one carry run."""

    schema_version: Literal["intention.order.v1"] = "intention.order.v1"
    concern_id: str = Field(pattern=CONCERN_ID_PATTERN)
    # The concern's event_ref: the adapter is chosen by its prefix.
    event_ref: str = Field(min_length=1, max_length=500)
    run_id: str = Field(min_length=6, max_length=80)
    correlation_id: str = Field(min_length=1, max_length=200)
    prime_act_id: str = Field(min_length=1, max_length=200)
    why: str = Field(min_length=1, max_length=600)
    timeout_sec: float = Field(default=INTENTION_TURN_TIMEOUT_SEC, ge=60.0, le=3600.0)


IntentionOrderStatus = Literal["submitted", "exists", "refused", "unavailable"]


class IntentionOrderResultV1(_Model):
    """Hub's answer. ``submitted``/``exists``: follow ``run_id``. ``refused``: no adapter, the adapter refused
    (no promise record, a carry turn's own promise, past the horizon) or durable-runs answered 4xx; ``detail`` says
    which. ``unavailable``: transient (no Postgres pool, durable-runs down or 5xx): Prime sends it again."""

    schema_version: Literal["intention.order_result.v1"] = "intention.order_result.v1"
    concern_id: str
    run_id: Optional[str] = None
    status: IntentionOrderStatus
    detail: Optional[str] = Field(default=None, max_length=500)
    source: Optional[IntentionSource] = None
    horizon_at: Optional[AwareDatetime] = None
    route: Optional[IntentionRoute] = None
    route_target: Optional[str] = Field(default=None, max_length=200)


class IntentionCarryBriefV1(_Model):
    """``DurableRunRequestV1.brief`` for ``intention.carry``. ``intention`` is a required literal so the brief is
    never confused with another member of the brief union (they all forbid unknown keys)."""

    intention: Literal["intention.carry"]
    concern_id: str = Field(pattern=CONCERN_ID_PATTERN)
    source: IntentionSource
    prime_act_id: str = Field(min_length=1, max_length=200)
    # Starts with ``prime_concern:<concern_id>`` (the hand-back settles the concern when the turn completes).
    prompt: str = Field(min_length=1, max_length=20000)
    route: IntentionRoute
    route_target: Optional[str] = Field(default=None, max_length=200)
    quote_ref: Optional[str] = Field(default=None, max_length=300)
    horizon_at: AwareDatetime
    retrieval_query: Optional[str] = Field(default=None, max_length=1000)
    timeout_sec: float = Field(default=INTENTION_TURN_TIMEOUT_SEC, ge=60.0, le=3600.0)


class IntentionDeliverV1(_Model):
    """``POST {HUB}/api/intention/deliver``: the carry run is completed; put its reply on its route."""

    schema_version: Literal["intention.deliver.v1"] = "intention.deliver.v1"
    concern_id: str = Field(pattern=CONCERN_ID_PATTERN)
    run_id: str = Field(min_length=6, max_length=80)
    correlation_id: str = Field(min_length=1, max_length=200)
    route: IntentionRoute
    route_target: Optional[str] = Field(default=None, max_length=200)


IntentionDeliverStatus = Literal["published", "already_delivered", "not_ready", "failed", "refused", "unavailable"]


class IntentionDeliverResultV1(_Model):
    """``already_delivered`` (``stored`` true): a ``chat_history_log`` row with this intention's correlation id is
    in the thread. That row is the only proof that closes Prime's act 1. ``published``: the reply was published
    on the bus (sent live, or written to the thread's history only) and its row is not visible yet: ask again.
    ``pinged``: it was sent live through the normal outreach gates (quiet hours, the shared daily cap, the
    economy); ``ping_reason`` is the outreach decision's reason either way. ``not_ready``: the run is not
    completed. ``failed``: the run ended without a reply, or outreach's content guards refused the reply (a PASS,
    an ungrounded internal-signal claim): it is never written into Juniper's thread. ``unavailable``: transient."""

    schema_version: Literal["intention.deliver_result.v1"] = "intention.deliver_result.v1"
    concern_id: str
    status: IntentionDeliverStatus
    stored: bool = False
    pinged: bool = False
    ping_reason: Optional[str] = Field(default=None, max_length=200)
    detail: Optional[str] = Field(default=None, max_length=500)
    reply_chars: int = 0


def aware(value: Optional[datetime]) -> Optional[datetime]:
    """A naive timestamp from Postgres (``timestamp without time zone``, UTC by convention) made aware."""
    from datetime import timezone

    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


__all__ = [
    "CONCERN_ID_PATTERN", "INTENTION_CARRY_NODES", "INTENTION_CARRY_STEP_KEY", "INTENTION_CARRY_TAG",
    "INTENTION_CARRY_WORKFLOW", "INTENTION_DELIVER_PATH", "INTENTION_FAMILY", "INTENTION_ORDER_PATH",
    "INTENTION_SESSION_ID", "INTENTION_TURN_TIMEOUT_SEC", "IntentionCarryBriefV1", "IntentionDeliverResultV1",
    "IntentionDeliverStatus", "IntentionDeliverV1", "IntentionOrderResultV1", "IntentionOrderStatus",
    "IntentionOrderV1", "IntentionRoute", "IntentionSource", "IntentionSourceV1", "aware", "intention_correlation_id",
    "intention_run_id",
]
