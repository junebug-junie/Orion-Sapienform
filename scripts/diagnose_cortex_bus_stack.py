#!/usr/bin/env python3
"""
Cortex bus stack diagnostic (same logic as Hub GET /api/debug/cortex-bus-stack).

Runs from repo root with Hub + Orion on PYTHONPATH::

  cd services/orion-hub
  export ORION_BUS_URL=redis://...
  ../../.venv/bin/python ../../scripts/diagnose_cortex_bus_stack.py

Or with explicit URL::

  ORION_BUS_URL=redis://host:6379/0 ../../.venv/bin/python ../../scripts/diagnose_cortex_bus_stack.py --no-rpc

Outputs JSON: Redis PING, PUBSUB NUMSUB for gateway + orch request channels, optional
Hub-equivalent RPC to the cortex gateway using ``skills.system.time_now.v1``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _ensure_sys_path_stdlib_safe() -> None:
    here = str(Path(__file__).resolve().parent)
    while sys.path and sys.path[0] in ("", here):
        sys.path.pop(0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    _ensure_sys_path_stdlib_safe()
    root = _repo_root()
    hub_root = root / "services" / "orion-hub"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(hub_root))

    os.chdir(str(hub_root))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("ORION_BUS_URL", "redis://127.0.0.1:6379/0"),
        help="Redis URL (default: ORION_BUS_URL)",
    )
    parser.add_argument("--rpc-timeout", type=float, default=45.0, help="RPC wait seconds")
    parser.add_argument("--no-rpc", action="store_true", help="Skip gateway RPC probe")
    parser.add_argument("--verb", default="skills.system.time_now.v1")
    parser.add_argument("--prompt", default="What time is it right now?")
    args = parser.parse_args()

    from scripts.cortex_bus_stack_diagnostic import run_cortex_bus_stack_diagnostic
    from scripts.settings import settings

    async def _run() -> dict:
        return await run_cortex_bus_stack_diagnostic(
            own_bus=None,
            redis_url=str(args.redis_url),
            gateway_request_channel=str(settings.CORTEX_GATEWAY_REQUEST_CHANNEL),
            gateway_result_prefix=str(settings.CORTEX_GATEWAY_RESULT_PREFIX),
            orch_request_channel=str(settings.CORTEX_ORCH_REQUEST_CHANNEL),
            orch_result_prefix=str(settings.CORTEX_ORCH_RESULT_PREFIX),
            rpc_timeout_sec=float(args.rpc_timeout),
            run_rpc=not bool(args.no_rpc),
            verb=str(args.verb),
            prompt=str(args.prompt),
            service_name="diagnose_cortex_bus_stack_cli",
            enforce_catalog=bool(getattr(settings, "ORION_BUS_ENFORCE_CATALOG", False)),
        )

    out = asyncio.run(_run())
    print(json.dumps(out, indent=2))
    rpc = out.get("rpc") or {}
    if isinstance(rpc, dict) and rpc.get("ok") is False and not rpc.get("skipped"):
        return 2
    if out.get("redis_ping_ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
