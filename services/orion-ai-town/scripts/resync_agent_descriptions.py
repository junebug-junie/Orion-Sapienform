#!/usr/bin/env python3
"""One-shot: push current Descriptions identity/plan into live agentDescriptions.

After deploying a character/plan rewrite, Convex still serves the old baked
agentDescriptions until this input runs (createAgent is the only other writer).

Usage (from repo root, with AITOWN_* in the environment):

  PYTHONPATH=. python3 services/orion-ai-town/scripts/resync_agent_descriptions.py
"""
from __future__ import annotations

import json
import os
import sys
import time

# Prefer shared client; keep script runnable without package install tricks.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from orion.embodiment import aitown_client as c  # noqa: E402


def main() -> int:
    if not c._base_url() or not c._admin_key() or not c._world_id():
        print("AITOWN_CONVEX_URL, AITOWN_ADMIN_KEY, and AITOWN_WORLD_ID are required", file=sys.stderr)
        return 2
    before = c.convex_query("world:gameDescriptions", {"worldId": c._world_id()})
    plans_before = {
        (a.get("agentId"), (a.get("plan") or "")[:80])
        for a in (before.get("agentDescriptions") or [])
    }
    print("plans before:")
    for row in sorted(plans_before):
        print(f"  {row[0]}: {row[1]}")

    # Idle worlds drop inputs; wake the engine first (same pattern as embodiment actuations).
    try:
        hb = c.heartbeat_world()
        print("heartbeatWorld:", json.dumps(hb, default=str)[:200])
    except Exception as exc:
        print(f"heartbeatWorld failed (continuing): {exc}", file=sys.stderr)

    result = c.send_input(name="resyncAgentDescriptions", args={})
    print("sendInput:", json.dumps(result, default=str)[:300])

    # Engine applies inputs asynchronously; poll briefly for plan change.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        time.sleep(1.5)
        after = c.convex_query("world:gameDescriptions", {"worldId": c._world_id()})
        plans_after = {
            (a.get("agentId"), (a.get("plan") or "")[:80])
            for a in (after.get("agentDescriptions") or [])
        }
        if plans_after != plans_before:
            print("plans after:")
            for row in sorted(plans_after):
                print(f"  {row[0]}: {row[1]}")
            return 0
    print("WARNING: plans unchanged after 30s — engine may still be applying, or input rejected", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
