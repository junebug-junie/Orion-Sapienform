#!/usr/bin/env python3
"""Ask a GPU pool host actuator two read-only questions over the bus, and print its answers.

Used by docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md before and after each deploy step:

- ``status``: is the controller up, can it parse its own checkout's config/gpu_pool.yaml
  (``config_unloadable:*`` = image not rebuilt after a pull), what does it observe on the card,
  and is anything in flight. A pure read.
- ``digest``: does the controller's checkout agree with THIS checkout's YAML? Sends a ``load`` naming
  a profile that is never in any allow-list. The controller checks the launch digest before the
  profile, and both before the generation fence or any docker call, so the answer is always a
  refusal: ``profile_not_allowed`` = digests agree; ``launch_digest_mismatch`` = the two hosts are on
  different commits (or the controller image predates the YAML). No generation is spent and no
  container is touched (controller tests: test_launch_exec.py
  ``test_profile_outside_allow_list_refused_before_any_docker_call``).

The pool sees these results too and logs them as stale (not its action_ids); its card state does
not change. Run from a checkout at the commit the POOL runs (or pass --config with the pool's YAML):

    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python \
        scripts/gpu_pool_actuator_probe.py [--role agent-gpu2] [--check status digest]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.gpu_pool.config import PoolConfig, launch_digest, load_pool_config  # noqa: E402
from orion.schemas.gpu_pool import (  # noqa: E402
    GPU_ACTUATE_KIND, GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_POOL_ACTUATE_RESULT_CHANNEL, GpuActuateV1,
)

# Never a config/llm_profiles.yaml name, so never in a launch.profiles allow-list.
PROBE_PROFILE = "gpu-pool-actuator-probe-not-a-profile"
SOURCE = "operator:gpu-pool-actuator-probe"


def build(cfg: PoolConfig, role: str, check: str, *, now: datetime | None = None) -> GpuActuateV1:
    """The request for one check. ``status`` carries profile None; ``digest`` is a ``load`` with
    PROBE_PROFILE and THIS checkout's launch digest."""
    spec = cfg.roles[role]
    if spec.launch is None:
        raise SystemExit(f"{role} has no launch block: no actuator to ask")
    now = now or datetime.now(timezone.utc)
    return GpuActuateV1(
        action_id=f"probe-{check}:{role}:{uuid.uuid4().hex[:8]}", generation=1, actuator=spec.launch.actuator,
        role=role, action="status" if check == "status" else "load", cards=list(spec.cards),
        profile=None if check == "status" else PROBE_PROFILE, launch_digest=launch_digest(cfg, role),
        deadline_at=now + timedelta(seconds=120), reason="operator_probe")


def verdict(check: str, results: list[dict]) -> str:
    """One line a human can act on, from the results carrying this probe's action_id."""
    final = [r for r in results if r.get("status") in ("succeeded", "failed", "refused")]
    if not final:
        return "NO ANSWER (controller down, wrong actuator name, or bus unreachable)"
    last = final[-1]
    reason = last.get("reason") or ""
    if check == "digest":
        if reason == "profile_not_allowed":
            return "OK: launch digests agree"
        if reason == "launch_digest_mismatch":
            return "MISMATCH: controller checkout/image is not on this commit"
        return f"UNEXPECTED: {last.get('status')} {reason}"
    if last.get("status") == "succeeded":
        return (f"OK: observed={last.get('observed')} in_flight={last.get('in_flight')} "
                f"last_action_id={last.get('last_action_id')}")
    return f"NOT OK: {last.get('status')} {reason}"


async def probe(role: str, checks: list[str], wait_sec: float, config: str | None = None) -> int:
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    cfg = load_pool_config(config) if config else load_pool_config()
    bus = OrionBusAsync(url=os.environ["ORION_BUS_URL"])
    await bus.connect()
    bad = 0
    try:
        async with bus.subscribe(GPU_POOL_ACTUATE_RESULT_CHANNEL) as pubsub:
            for check in checks:
                msg = build(cfg, role, check)
                await bus.publish(GPU_POOL_ACTUATE_REQUEST_CHANNEL, BaseEnvelope(
                    kind=GPU_ACTUATE_KIND, source=ServiceRef(name=SOURCE), correlation_id=uuid.uuid4(),
                    payload=msg.model_dump(mode="json")))
                got: list[dict] = []
                deadline = asyncio.get_running_loop().time() + wait_sec
                while asyncio.get_running_loop().time() < deadline:
                    m = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if not m:
                        continue
                    try:
                        env = json.loads(m["data"])
                    except (TypeError, ValueError):
                        continue
                    payload = env.get("payload") if isinstance(env, dict) else None
                    if isinstance(payload, dict) and payload.get("action_id") == msg.action_id:
                        got.append(payload)
                        if payload.get("status") in ("succeeded", "failed", "refused"):
                            break
                line = verdict(check, got)
                bad += not line.startswith("OK")
                print(f"{check:7s} {role} digest={msg.launch_digest[:16]} -> {line}")
    finally:
        await bus.close()
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", default="agent-gpu2")
    ap.add_argument("--check", nargs="+", choices=("status", "digest"), default=["status", "digest"])
    ap.add_argument("--wait-sec", type=float, default=90.0,
                    help="per check; status waits on `docker compose ps` (up to ~60 s)")
    ap.add_argument("--config", default=None,
                    help="the pool's config/gpu_pool.yaml when it differs from this checkout's "
                         "(e.g. `git show <pool commit>:config/gpu_pool.yaml > /tmp/pool.yaml`)")
    args = ap.parse_args()
    return asyncio.run(probe(args.role, args.check, args.wait_sec, args.config))


if __name__ == "__main__":
    sys.exit(main())
