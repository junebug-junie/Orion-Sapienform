from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orion.schemas.grammar import ActSignal, ActVerb


class DispatchModeConfigV1(BaseModel):
    default_dispatch_mode: str = "dry_run"
    allow_dispatch_read_only: bool = False
    allow_mutating_dispatch: bool = False
    # Gates EXPRESS_SCOPE routes. Default False: an action whose product exists
    # outside Orion and costs a physical resource does not become dispatchable
    # by a config file appearing -- it takes a deliberate operator decision,
    # same posture allow_mutating_dispatch has held since 2026-08-12.
    allow_express_dispatch: bool = False
    # 2026-10-01: gates SELF_REVERSIBLE_SCOPE routes (attend-to-act loop A1). Structural, like the two
    # above; the operational switch is the dispatch runtime's ORION_WORLD_ACTIONS_ENABLED plus the
    # template named in ORION_WORLD_ACTIONS_ALLOWED -- all three must be open for a send.
    allow_self_reversible_dispatch: bool = False


# The single non-read-only route scope. Lives here, not in builder.py, because
# BOTH builder and envelopes need it and builder already imports envelopes --
# putting it in builder makes that a cycle. (Learned the same day, from
# orion/field/action_warrant.py's own circular import; see PR #1581.)
MAINTENANCE_SCOPE = "maintenance_bounded"

# 2026-08-30: the second non-read-only scope, for Orion's first OUTWARD action.
# Its own scope and its own gate rather than borrowing maintenance_bounded's,
# for two reasons. It is not maintenance -- overloading that name would make the
# route table lie about what the action does. And it must be switchable
# independently: turning off image generation should not also stop docker
# pruning, and vice versa. A shared flag makes both an all-or-nothing choice.
EXPRESS_SCOPE = "express_bounded"

# 2026-10-01 (attend-to-act loop A1): a bounded, auto-reverting action on Orion's OWN resources
# (today: the GPU pool's orion_self_shed reason). Its own scope so it can be closed without touching
# maintenance or express, and so the route table says what the action is. Not a cortex verb: the
# route's cortex_verb is an executor key the dispatch runtime maps to a GPU-pool shed RPC.
SELF_REVERSIBLE_SCOPE = "self_reversible"
SHED_EXECUTOR_VERB = "orion.gpu_pool.shed.v1"
# Thinking rung 3 follow-up (2026-10-21): not a cortex verb either. The executor key the dispatch
# runtime maps to the reading pull RPC (orion/schemas/world_pulse_read_pull.py), which asks Hub to
# start one extra world-pulse read. Hub runs every gate; the dispatch runtime only admits or refuses.
READ_PULL_EXECUTOR_VERB = "orion.world_pulse_read.pull.v1"
# Orion plays Zelda (2026-10-22, spec section E): the executor key the dispatch runtime maps to one
# retro leisure start RPC (orion/schemas/retro_leisure.py). Hub's leisure runner runs every gate.
RETRO_LEISURE_EXECUTOR_VERB = "orion.retro_leisure.start.v1"

# Keys a route's static `skill_args` may never carry, because a real value
# for them is derived from runtime safety state rather than chosen by an
# operator. See CortexRouteTemplateV1.skill_args.
_CONFIG_FORBIDDEN_SKILL_ARGS = frozenset({"mode", "run_mode", "dry_run"})


class ActRouteRefV1(BaseModel):
    """Rung 3: the rung-2 act (orion/schemas/grammar.py ActV1) this route's dispatch really
    produces. The allocator scores the route's candidates on that act's EffectPosterior
    (orion/autonomy/act_candidates.py). Declared only where the act is joined to the dispatch in
    code (render_scene -> thought.painting: test_before=visual_dispatch:<dispatch_id>)."""

    model_config = ConfigDict(extra="forbid")

    verb: ActVerb
    actor: str = Field(min_length=1)
    signal_id: ActSignal
    # How long one decision of the proven-effect path, treated or held back, keeps the act in flight
    # and (for a "queue" object) when its outcome is read. For a "dispatch" object it is the
    # producer's own ActExpectV1.horizon_sec.
    horizon_sec: float = Field(gt=0.0)
    # Where the act's object comes from (orion/autonomy/act_episodes.py holdback_rule):
    # - "dispatch": the dispatch mints it (render_scene: the painting run id derives from the
    #   dispatch id). No control arm (the outcome cannot occur without the act), measured cost is the
    #   route's settled dispatch latency, and the actor is single-flight (any open act holds it).
    # - "queue": the dispatch picks an existing object another lane also acts on (a pending reading
    #   seed). Control arm drawn and resolved by the object's fate, cost is the act's own measured
    #   open->close time (the dispatch is only an RPC), and the actor's other open acts (the lane's
    #   own concurrent reads) do not hold it.
    object_source: Literal["dispatch", "queue"] = "dispatch"
    # Where the act's measured cost comes from. None keeps the object_source rule above. "act_duration":
    # the act's own measured open->close time (grammar_acts), for a route whose dispatch is only an
    # RPC that starts the work (retro_leisure_session: a session runs minutes after the RPC answers).
    cost_source: Literal["act_duration"] | None = None
    # Charged only while the act has no measured cost yet (never graded). Must be the doer's own
    # enforced ceiling for one act, so a cold act is charged its worst case, never a typical guess:
    # an act-backed route otherwise has no first cost, and no cost means `no_cost_estimate` forever.
    cold_cost_ceiling_sec: float | None = Field(None, gt=0.0)


class CortexRouteTemplateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_kind: str
    cortex_verb: str
    cortex_mode: str = "brain"
    allowed_scope: str
    act: ActRouteRefV1 | None = None
    # 2026-08-12: per-route RPC budget, None = use the runtime's global
    # EXECUTION_DISPATCH_RPC_TIMEOUT_SEC. Exists because the two needs are
    # genuinely opposed: a real build-cache prune can run for minutes, while
    # this consumer is single-threaded, so a global ceiling large enough for
    # the prune would let ONE hung inspect stall all dispatch for that long.
    # Only routes that actually need a longer budget get one.
    rpc_timeout_sec: float | None = None

    # 2026-08-25: static per-route arguments for the target verb, merged into
    # the envelope's `context.skill_args` (orion/execution_dispatch/
    # envelopes.py). Exists because `maintain` routes already needed exactly
    # this and had it hardcoded there as a single `mode` key -- so a second
    # route wanting any argument at all had nowhere to put it.
    #
    # STATIC ONLY, and deliberately so: these are config, chosen by the
    # operator writing the policy file, never derived from candidate state.
    # A route that needs to vary its arguments per tick is describing a
    # decision, and a decision belongs in the proposal arena or inside the
    # skill, not smuggled through a route table.
    #
    # `mode` is REFUSED here outright, not merely overridden. envelopes.py
    # forces the derived `mode` only for MAINTENANCE_SCOPE routes, and
    # `allowed_scope` / `cortex_verb` are independent unvalidated config
    # fields -- so a route declaring a mutating verb under `summarize_only`
    # (which builder.py::scope_allowed admits unconditionally, without
    # mode.allow_mutating_dispatch) plus `skill_args: {mode: execute}` would
    # reach that verb with `mode=execute` regardless of dry_run. Before route
    # skill_args existed there was no such channel at all; this validator is
    # what keeps it that way. Caught in review, 2026-08-25.
    skill_args: dict[str, str] = Field(default_factory=dict)

    @field_validator("skill_args")
    @classmethod
    def _refuse_run_mode(cls, value: dict[str, str]) -> dict[str, str]:
        forbidden = sorted(k for k in value if k.lower() in _CONFIG_FORBIDDEN_SKILL_ARGS)
        if forbidden:
            raise ValueError(
                f"route skill_args may not set {forbidden}: a skill's run mode is derived "
                "from the dispatch runtime's own dry_run state, never from config"
            )
        return value


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

    # 2026-08-13: per-TEMPLATE route override, checked before the per-kind map
    # above. Empty by default, so with no entries here every candidate resolves
    # exactly as it did before this field existed (asserted by test, not
    # assumed).
    #
    # Why this exists: `proposal_kind_to_cortex` is keyed on `proposal_kind`
    # alone, so a whole kind can address exactly ONE verb. `maintain` meant
    # `skills.runtime.builder_prune.v1` and could never mean anything else --
    # Orion's entire mutating repertoire was one action BY CONSTRUCTION, not by
    # arbitration. No amount of scoring, budgeting, or sampling downstream can
    # widen a repertoire that the routing layer caps at one.
    #
    # Deliberately an override rather than a replacement: the kind map stays the
    # default for every template that does not name its own route, so adding a
    # second maintenance action does not require re-declaring the twelve
    # read-only templates that were already working.
    template_to_cortex: dict[str, CortexRouteTemplateV1] = Field(default_factory=dict)
    hard_blocks: list[str] = Field(default_factory=list)
    limits: DispatchLimitsV1 = Field(default_factory=DispatchLimitsV1)

    def act_refs_by_cortex_verb(self) -> dict[str, ActRouteRefV1]:
        """cortex_verb -> the rung-2 act its dispatch produces (rung 3). A dispatch candidate
        carries its cortex_verb, so this is how the allocator finds the act without any
        dispatch-frame schema change."""
        out: dict[str, ActRouteRefV1] = {}
        for route in [*self.proposal_kind_to_cortex.values(), *self.template_to_cortex.values()]:
            if route.act is not None:
                out[route.cortex_verb] = route.act
        return out

    @model_validator(mode="after")
    def _one_act_per_verb(self) -> "ExecutionDispatchPolicyV1":
        seen: dict[str, ActRouteRefV1] = {}
        for route in [*self.proposal_kind_to_cortex.values(), *self.template_to_cortex.values()]:
            if route.act is None:
                continue
            prior = seen.setdefault(route.cortex_verb, route.act)
            if prior != route.act:
                raise ValueError(
                    f"routes sharing cortex_verb {route.cortex_verb!r} declare different acts "
                    f"({prior.model_dump()} vs {route.act.model_dump()}): one dispatch, one act"
                )
        return self


def load_execution_dispatch_policy(path: str | Path) -> ExecutionDispatchPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExecutionDispatchPolicyV1.model_validate(data)
