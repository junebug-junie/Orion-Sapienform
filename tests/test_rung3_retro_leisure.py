"""Ladder retirement PR 1 (2026-11-05): retro leisure left the proposal -> policy -> dispatch ladder.

Hub starts leisure on its own timer (services/orion-hub/scripts/retro_leisure_self_start.py). One
trigger, never two. Pins that every piece of the old trigger is gone, so a revert of only part of
this PR (a template without a route, a route without a receiver) fails here:

- no `retro_leisure_session` template in the proposal policy;
- no dispatch route (template or kind) that runs `orion.retro_leisure.start.v1` or declares the
  `hub.retro_leisure` act;
- no dispatch-side retro code (module, mixin, settings keys);
- no `orion:retro_leisure:start:*` channel and no RetroLeisureStart* schema.

(The act-duration cost the retired route shared with AI Town left with AI Town's route, ladder
retirement PR 3; tests/test_rung3_aitown_session.py pins that.)
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

from orion.execution_dispatch.policy import load_execution_dispatch_policy  # noqa: E402
from orion.proposals.policy import load_proposal_policy  # noqa: E402


def test_no_retro_leisure_template() -> None:
    policy = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
    assert "retro_leisure_session" not in policy.proposal_templates


def test_no_dispatch_route_runs_retro_leisure() -> None:
    policy = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml")
    routes = {**policy.template_to_cortex, **policy.proposal_kind_to_cortex}
    assert "retro_leisure_session" not in policy.template_to_cortex
    assert not [k for k, r in routes.items() if r.cortex_verb == "orion.retro_leisure.start.v1"]
    assert not [k for k, r in routes.items() if r.act is not None and r.act.actor == "hub.retro_leisure"]


def test_no_dispatch_side_retro_code() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("app.retro_leisure_dispatch")
    import orion.execution_dispatch.policy as dispatch_policy

    assert not hasattr(dispatch_policy, "RETRO_LEISURE_EXECUTOR_VERB")
    from app.settings import Settings
    from app.worker import ExecutionDispatchRuntimeWorker

    assert not [c for c in ExecutionDispatchRuntimeWorker.__mro__ if "Retro" in c.__name__]
    fields = {f.alias or name for name, f in Settings.model_fields.items()}
    assert not [f for f in fields if "RETRO" in f.upper()]
    env = (SVC / ".env_example").read_text()
    assert "ORION_DISPATCH_RETRO_LEISURE" not in env.replace("ORION_DISPATCH_RETRO_LEISURE_*", "")


def test_no_retro_leisure_rpc_channel_or_schema() -> None:
    channels = yaml.safe_load((REPO / "orion" / "bus" / "channels.yaml").read_text())
    names = {c["name"] for c in channels["channels"]}
    assert not [n for n in names if n.startswith("orion:retro_leisure:start")]
    from orion.schemas import registry

    assert "RetroLeisureStartRequestV1" not in registry._REGISTRY
    assert "RetroLeisureStartResultV1" not in registry._REGISTRY
    with pytest.raises(ImportError):
        importlib.import_module("orion.schemas.retro_leisure")
