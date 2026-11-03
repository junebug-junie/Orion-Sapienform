from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest
import yaml
from pydantic import ValidationError

from orion.consolidation.policy import load_consolidation_policy
from orion.execution_dispatch.policy import (
    MAX_DISPATCH_RPC_TIMEOUT_SEC,
    CortexRouteTemplateV1,
)

REPO = Path(__file__).resolve().parents[1]
CONSOLIDATION_POLICY = REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml"
EXECUTION_POLICY = REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
EXECUTION_SETTINGS = REPO / "services" / "orion-execution-dispatch-runtime" / "app" / "settings.py"


def _execution_settings_class():
    spec = importlib.util.spec_from_file_location("execution_settings_for_lag_test", EXECUTION_SETTINGS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Settings


def test_visibility_lag_covers_every_configured_dispatch_rpc_timeout() -> None:
    """Dispatch stamps before its concurrent send batch and persists only afterward."""
    consolidation = load_consolidation_policy(CONSOLIDATION_POLICY)
    execution = yaml.safe_load(EXECUTION_POLICY.read_text(encoding="utf-8"))
    route_groups = (
        execution.get("proposal_kind_to_cortex", {}),
        execution.get("template_to_cortex", {}),
    )
    configured = [
        float(route["rpc_timeout_sec"])
        for group in route_groups
        for route in group.values()
        if route.get("rpc_timeout_sec") is not None
    ]

    assert configured
    assert consolidation.window.source_visibility_lag_sec == 900
    assert max(configured) <= MAX_DISPATCH_RPC_TIMEOUT_SEC
    assert consolidation.window.source_visibility_lag_sec >= MAX_DISPATCH_RPC_TIMEOUT_SEC


def test_route_and_runtime_timeouts_cannot_outgrow_visibility_lag() -> None:
    with pytest.raises(ValidationError):
        CortexRouteTemplateV1(
            dispatch_kind="inspect",
            cortex_verb="substrate.inspect",
            allowed_scope="inspect_only",
            rpc_timeout_sec=MAX_DISPATCH_RPC_TIMEOUT_SEC + 1,
        )

    settings = _execution_settings_class()
    common = {"POSTGRES_URI": "postgresql://test/test"}
    with pytest.raises(ValidationError):
        settings(
            **common,
            EXECUTION_DISPATCH_RPC_TIMEOUT_SEC=MAX_DISPATCH_RPC_TIMEOUT_SEC + 1,
        )
    with pytest.raises(ValidationError):
        settings(
            **common,
            ORION_GPU_POOL_SHED_RPC_TIMEOUT_SEC=MAX_DISPATCH_RPC_TIMEOUT_SEC + 1,
        )
