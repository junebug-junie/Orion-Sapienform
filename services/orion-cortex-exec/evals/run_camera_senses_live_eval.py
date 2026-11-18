#!/usr/bin/env python3
"""Live, read-only eval: does the situation brief tell Orion what their eyes are?

Builds the brief's perception context against the real database (no Redis
writes: identity ask off) and checks the two things the 2026-10-15 patch
sells, on live rows rather than fixtures:

1. a "Your cameras (your own eyes)" line exists, names every configured
   camera, and marks the street camera outside and public;
2. when there is a street line, it opens with what the camera is.

Plus a non-degeneracy floor (CLAUDE.md 0A metric gate step 4): at least one
camera must read live. The statuses are printed so a human can check them
against what is actually plugged in (carbon has been dark since 2026-09-09).

Usage (from repo root):
    SITUATION_PERCEPTION_DSN=postgresql+psycopg2://postgres:postgres@localhost:55432/conjourney \\
        python services/orion-cortex-exec/evals/run_camera_senses_live_eval.py

Exit 0 = pass, 1 = a check failed, 2 = the database could not be read.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from orion.schemas.situation import SituationDiagnosticsV1  # noqa: E402
from orion.situational import context as situation  # noqa: E402
from orion.situational.camera_eyes import senses_line  # noqa: E402


def main() -> int:
    room = os.getenv("ORION_SITUATION_PERCEPTION_STREAM_IDS", "carbon,cam0")
    cfg = situation.settings_from_runtime(SimpleNamespace(orion_situation_perception_stream_ids=room))
    cfg.perception_enabled = True
    diag = SituationDiagnosticsV1()
    ctx = asyncio.run(situation._build_perception_context(cfg, diag, allow_identity_ask=False))
    print("provider_status:", diag.provider_status)
    if diag.provider_status.get("perception_cameras") != "ok":
        print("UNREAD: camera state could not be read")
        return 2
    line = senses_line(ctx.cameras) or ""
    print(line)
    print(ctx.street_summary or "(no street line: nothing to say)")
    failures: list[str] = []
    wanted = list(dict.fromkeys([*cfg.perception_stream_ids, *cfg.street_stream_ids]))
    for sid in wanted:
        if f"({sid}" not in line:
            failures.append(f"camera {sid} missing from the senses line")
    for cam in ctx.cameras:
        if cam.stream_id in cfg.street_stream_ids and not cam.outside:
            failures.append(f"street camera {cam.stream_id} not marked outside")
    if ctx.street_summary and not ctx.street_summary.startswith("Your "):
        failures.append("street line does not open with what the camera is")
    statuses = [c.status for c in ctx.cameras]
    if "live" not in statuses:
        failures.append(f"no camera reads live: {statuses}")
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
