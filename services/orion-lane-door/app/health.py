"""Container health check (Dockerfile HEALTHCHECK): ``python -m app.health``.

Healthy when the door answers ``/door/stats`` on its lane port, or when it is still waiting for its engine
to move behind it (serve() writes ``waiting`` to the state file; a waiting door is working as designed,
not broken). Anything else -- the state says listening and the stats do not answer -- is unhealthy.
"""
from __future__ import annotations

import os
import sys
import urllib.request

STATE_FILE = os.environ.get("LANE_DOOR_STATE_FILE", "/tmp/lane-door.state")


def main() -> int:
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            state = fh.read().strip()
    except OSError:
        state = ""
    if state == "waiting":
        return 0
    port = os.environ.get("LANE_DOOR_LISTEN_PORT", "8080").strip() or "8080"
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/door/stats", timeout=4)
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
