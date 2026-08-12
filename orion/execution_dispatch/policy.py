from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DispatchModeConfigV1(BaseModel):
    default_dispatch_mode: str = "dry_run"
    allow_dispatch_read_only: bool = False
    allow_mutating_dispatch: bool = False


# The single non-read-only route scope. Lives here, not in builder.py, because
# BOTH builder and envelopes need it and builder already imports envelopes --
# putting it in builder makes that a cycle. (Learned the same day, from
# orion/field/action_warrant.py's own circular import; see PR #1581.)
MAINTENANCE_SCOPE = "maintenance_bounded"


class CortexRouteTemplateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_kind: str
    cortex_verb: str
    cortex_mode: str = "brain"
    allowed_scope: str
    # 2026-08-12: per-route RPC budget, None = use the runtime's global
    # EXECUTION_DISPATCH_RPC_TIMEOUT_SEC. Exists because the two needs are
    # genuinely opposed: a real build-cache prune can run for minutes, while
    # this consumer is single-threaded, so a global ceiling large enough for
    # the prune would let ONE hung inspect stall all dispatch for that long.
    # Only routes that actually need a longer budget get one.
    rpc_timeout_sec: float | None = None


class DispatchLimitsV1(BaseModel):
    max_dispatch_candidates: int = 5
    max_dispatches_per_tick: int = 1

    # 2026-08-12: both of the fields below exist because of ONE measured
    # defect. Over 3 live hours the top five slots were held by the same five
    # read-only templates on essentially every tick (inspect_bus_channel_
    # catalog / inspect_attended_target / summarize_transport_contract_drift
    # 267 dispatches each, inspect_field_topology_catalog 243,
    # watch_transport_backpressure 209) while prune_build_cache -- the only
    # action in the arena that changes anything about the host -- was
    # dispatched 0 times and blocked 4 times on capacity.
    #
    # That is not the prune being unlucky. A single scalar ranking conflates
    # urgency (a spike) with importance (a persistent level), and a
    # steady-state-high signal like disk fullness loses that comparison
    # forever. Two independent corrections, because they fix different halves:

    # (a) RESERVATION -- capacity held for a route scope so a whole class of
    # action is not required to out-spike inspection to exist. Reservations
    # are NOT a throughput loss: an unclaimed reserved slot is reclaimed by
    # the general fill in the same tick (builder._admit_candidates), so
    # reserving 1 of 5 costs nothing on ticks where nothing claims it.
    reserved_slots_by_scope: dict[str, int] = Field(default_factory=dict)

    # (b) AGING -- the general fix. A candidate that keeps losing the ranking
    # accrues a rising bonus, so starvation is bounded in time rather than
    # permanent. This is the standard cure for scheduler starvation and it
    # applies to every kind, not just the one that prompted it: anything else
    # quietly losing this race gets the same relief without being named.
    # Bonus = min(cap, consecutive_starved_ticks * per_tick), added to
    # priority_score for admission ordering only -- the stored priority_score
    # is never rewritten, so the proposal arena's own scoring stays auditable.
    starvation_aging_bonus_per_tick: float = 0.0
    starvation_aging_bonus_cap: float = 0.0


class ExecutionDispatchPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["execution_dispatch_policy.v1"] = "execution_dispatch_policy.v1"
    policy_id: str = "execution_dispatch_policy.v1"

    mode: DispatchModeConfigV1 = Field(default_factory=DispatchModeConfigV1)
    allowed_policy_decisions: list[str] = Field(default_factory=list)
    blocked_policy_decisions: list[str] = Field(default_factory=list)
    proposal_kind_to_cortex: dict[str, CortexRouteTemplateV1] = Field(default_factory=dict)
    hard_blocks: list[str] = Field(default_factory=list)
    limits: DispatchLimitsV1 = Field(default_factory=DispatchLimitsV1)


def load_execution_dispatch_policy(path: str | Path) -> ExecutionDispatchPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExecutionDispatchPolicyV1.model_validate(data)
