"""The request orion-hardware-watch's disk rule sends reaches the image-prune skill in EXECUTE mode.

Ladder retirement PR 4 (2026-11-06). The disk rule (services/orion-hardware-watch/app/image_prune.py)
builds the payload; this test feeds that exact payload through cortex-exec's own path:
``build_exec_ctx`` -> ``executor._plan_request_from_step_ctx`` (reads ``ctx["skill_args"]``) ->
``verb_adapters._skill_args`` -> ``_resolve_docker_prune_run_mode``. If any of those stops carrying
``skill_args``, the skill would quietly run as a preview (``would_act``) twice a day and prune nothing.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path

from app.exec_ctx import build_exec_ctx
from app.executor import _plan_request_from_step_ctx
from orion.schemas.cortex.schemas import PlanExecutionRequest


def _verb_adapters():
    """The already-loaded verb_adapters module if any (tests/test_skill_verbs.py loads it under its own
    package name; a second import would register every verb twice), else ``app.verb_adapters``."""
    for mod in list(sys.modules.values()):
        if (getattr(mod, "__file__", "") or "").replace("\\", "/").endswith("orion-cortex-exec/app/verb_adapters.py"):
            return mod
    return importlib.import_module("app.verb_adapters")

REPO = Path(__file__).resolve().parents[3]
HW_IMAGE_PRUNE = REPO / "services" / "orion-hardware-watch" / "app" / "image_prune.py"


def _hardware_watch_payload() -> dict:
    # Loaded by path under its own name: hardware-watch's `app` package must not shadow cortex-exec's.
    spec = importlib.util.spec_from_file_location("_hw_image_prune_for_exec_contract", HW_IMAGE_PRUNE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from orion.core.bus.bus_schemas import ServiceRef

    sent: dict = {}

    class _Stop(Exception):
        pass

    class _Bus:
        async def rpc_request(self, channel, env, **kw):
            sent["channel"], sent["env"] = channel, env
            raise _Stop

    ask = mod.build_bus_ask(_Bus(), request_channel="orion:cortex:exec:request:background",
                            result_prefix="orion:exec:result", source=ServiceRef(name="orion-hardware-watch"),
                            timeout_sec=720.0)
    try:
        asyncio.run(ask("image_prune_disk_rule:contract"))
    except _Stop:
        pass
    assert sent["channel"] == "orion:cortex:exec:request:background"
    return sent["env"].payload


def test_the_disk_rules_request_resolves_to_execute_in_the_skill():
    payload = _hardware_watch_payload()
    req = PlanExecutionRequest.model_validate(payload)
    assert [s.verb_name for s in req.plan.steps] == ["skills.runtime.image_prune.v1"]
    assert req.plan.steps[0].services == []   # the local-verb branch, where the skill runs
    ctx = build_exec_ctx(payload_context=payload.get("context"), req=req, trace_id="t", parent_event_id=None,
                         corr_id="c")
    step_req = _plan_request_from_step_ctx(req.plan.steps[0], ctx, "c")
    va = _verb_adapters()
    args = va._skill_args(step_req)
    assert args.get("mode") == "execute"
    mode, resolution = va._resolve_docker_prune_run_mode(args)
    assert mode == "execute" and resolution["source"] == "mode_arg"
