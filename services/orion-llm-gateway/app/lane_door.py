"""The gateway's half of the lane door (leased-evals design, Phase 2).

docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md. Every call the gateway sends
to a model server under a pool grant carries one header, ``X-Orion-Lease-Id``: the granted lease. The door
in front of the server (services/orion-lane-door) logs it, strips it, and in enforce mode refuses a model
call that has none. scripts/lane_door_report.py joins the logged lease ids to ``gpu_pool_events`` offline,
so a made-up id shows up there.

There is no shared secret. The door catches ACCIDENTAL bypass (an eval or script that calls a model
server directly instead of the gateway). It is not a fence against a deliberate local caller: anything
on athena that can reach the docker socket could read any secret anyway.

Both paths send the header: the HTTP passthrough (passthrough_proxy.py) and the bus path (llm_backend's
HTTP client, which reads the lease id the grant's worker thread was started with, upstream_cancel.py).
"""
from __future__ import annotations

from typing import Optional

LEASE_HEADER = "X-Orion-Lease-Id"


def headers(lease_id: Optional[str]) -> dict[str, str]:
    """The door header for one dispatch: the granted lease id, or nothing outside a grant."""
    return {LEASE_HEADER: str(lease_id)} if lease_id else {}
