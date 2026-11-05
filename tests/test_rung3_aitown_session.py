"""Ladder retirement PR 3 (2026-11-05): AI Town left the proposal -> policy -> dispatch ladder.

orion-embodiment starts Orion's town sessions on its own timer (services/orion-embodiment/app/
town_self_start.py), under the session controller's caps. One trigger, never two. Pins that every
piece of the old trigger is gone, so a revert of only part of this PR (a template without a route, a
route without a receiver) fails here:

- no `aitown_session` template in the proposal policy;
- no dispatch route (template or kind) that runs `orion.aitown_session.start.v1` or declares the
  `aitown.session` act;
- no dispatch-side AI Town code (module, mixin, settings keys, executor key);
- the act-duration cost rule that only AI Town used is gone (worker method, policy fields);
- no `orion:aitown_session:start:*` channel and no AITownSessionStart* schema;
- the session ledger contract the controller and the physiology report share is still there.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from orion.execution_dispatch.policy import ActRouteRefV1, load_execution_dispatch_policy  # noqa: E402
from orion.proposals.policy import load_proposal_policy  # noqa: E402


def test_no_aitown_session_template() -> None:
    policy = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
    assert "aitown_session" not in policy.proposal_templates
    assert not [k for k, t in policy.proposal_templates.items() if t.target_id == "world:aitown"]


def test_no_dispatch_route_runs_aitown() -> None:
    policy = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml")
    routes = {**policy.template_to_cortex, **policy.proposal_kind_to_cortex}
    assert "aitown_session" not in policy.template_to_cortex
    assert not [k for k, r in routes.items() if r.cortex_verb == "orion.aitown_session.start.v1"]
    assert not [k for k, r in routes.items() if r.act is not None and r.act.actor == "aitown.session"]


def test_no_dispatch_side_aitown_code() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("app.aitown_session_dispatch")
    import orion.execution_dispatch.policy as dispatch_policy

    assert not hasattr(dispatch_policy, "AITOWN_SESSION_EXECUTOR_VERB")
    from app.settings import Settings
    from app.worker import ExecutionDispatchRuntimeWorker

    assert not [c for c in ExecutionDispatchRuntimeWorker.__mro__ if "AITown" in c.__name__]
    for name in ("_send_aitown_session", "_aitown_session_held_reason", "_hold_aitown_session"):
        assert not hasattr(ExecutionDispatchRuntimeWorker, name), name
    fields = {f.alias or name for name, f in Settings.model_fields.items()}
    assert not [f for f in fields if "AITOWN" in f.upper()]
    env = (SVC / ".env_example").read_text()
    assert "ORION_DISPATCH_AITOWN_SESSION" not in env.replace("ORION_DISPATCH_AITOWN_SESSION_*", "")
    compose = (SVC / "docker-compose.yml").read_text()
    assert "AITOWN" not in compose


def test_the_act_duration_cost_rule_left_with_its_only_route() -> None:
    from app.worker import ExecutionDispatchRuntimeWorker
    from orion.autonomy.act_candidates import ActRef

    assert not hasattr(ExecutionDispatchRuntimeWorker, "_act_duration_or_ceiling_cost")
    assert "cost_source" not in ActRouteRefV1.model_fields
    assert "cold_cost_ceiling_sec" not in ActRouteRefV1.model_fields
    assert "cost_source" not in ActRef.__dataclass_fields__
    # A route that still carries the old keys is refused at load (extra=forbid), never silently ignored.
    with pytest.raises(Exception):
        ActRouteRefV1.model_validate({"verb": "play", "actor": "aitown.session", "signal_id": "artifact_produced",
                                      "horizon_sec": 2400, "cost_source": "act_duration"})
    # The reading pull keeps its own queue-object act-duration cost.
    assert hasattr(ExecutionDispatchRuntimeWorker, "_act_duration_cost")


def test_no_aitown_session_rpc_channel_or_schema() -> None:
    channels = yaml.safe_load((REPO / "orion" / "bus" / "channels.yaml").read_text())
    names = {c["name"] for c in channels["channels"]}
    assert not [n for n in names if n.startswith("orion:aitown_session:start")]
    from orion.schemas import registry

    assert "AITownSessionStartRequestV1" not in registry._REGISTRY
    assert "AITownSessionStartResultV1" not in registry._REGISTRY
    import orion.schemas.aitown_session as contract

    for gone in ("AITownSessionStartRequestV1", "AITownSessionStartResultV1",
                 "AITOWN_SESSION_START_REQUEST_CHANNEL", "AITOWN_SESSION_START_REPLY_PREFIX"):
        assert not hasattr(contract, gone), gone


def test_the_session_ledger_contract_stays() -> None:
    from orion.schemas.aitown_session import (
        AITOWN_SESSION_FAMILY,
        AITOWN_SESSION_LEDGER_REDIS_KEY,
        AITownSessionRowV1,
        aitown_session_object_ref,
    )

    assert AITOWN_SESSION_FAMILY == "aitown.session"
    assert AITOWN_SESSION_LEDGER_REDIS_KEY == "orion:aitown:sessions"
    assert aitown_session_object_ref("aitown_self_start:x") == "aitown_session_dispatch:aitown_self_start:x"
    # A row written by the dispatch era still parses (the ledger is read back for the caps).
    AITownSessionRowV1.model_validate({
        "session_id": "s", "kind": "orion", "started_at": 1.0, "last_seen_at": 2.0, "ended_at": 2.0,
        "start_reason": "want_funded", "stop_reason": "session_cap",
        "dispatch_id": "dispatch:proposal:aitown_session:tick_1:none:execution_dispatch_policy.v1"})
