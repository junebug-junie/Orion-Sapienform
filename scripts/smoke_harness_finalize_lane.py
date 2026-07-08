#!/usr/bin/env python3
"""Live smoke: harness finalize cortex RPC must not queue behind legacy chat traffic.

Usage:
  python scripts/smoke_harness_finalize_lane.py
  python scripts/smoke_harness_finalize_lane.py --channel orion:cortex:exec:request:background
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir in sys.path:
    sys.path.remove(_scripts_dir)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from orion.harness.cortex_client import HarnessCortexClient
from orion.harness.finalize import build_voice_finalize_plan_request
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.core.bus.async_service import OrionBusAsync


def _load_bus_url() -> str:
    for path in (
        REPO_ROOT / "services/orion-harness-governor/.env",
        REPO_ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)
    url = os.environ.get("ORION_BUS_URL", "").strip()
    if not url:
        raise SystemExit("ORION_BUS_URL not set (check services/orion-harness-governor/.env)")
    return url


async def _smoke(channel: str, timeout_sec: float) -> None:
    import uuid

    corr = str(uuid.uuid4())
    thought = make_thought(correlation_id=corr)
    appraisal = make_appraisal(correlation_id=corr)
    reflection = make_reflection(correlation_id=corr)
    plan = build_voice_finalize_plan_request(
        correlation_id=corr,
        draft_text="Smoke draft: harness finalize lane must respond while legacy chat is busy.",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=make_repair_overlay(),
        user_message="smoke harness finalize lane",
    )

    bus = OrionBusAsync(url=_load_bus_url())
    await bus.connect()
    client = HarnessCortexClient(
        bus,
        request_channel=channel,
        result_prefix="orion:exec:result",
        source_name="smoke-harness-finalize",
        timeout_sec=timeout_sec,
        voice_finalize_timeout_sec=timeout_sec,
    )
    started = time.monotonic()
    try:
        result = await client(plan)
    finally:
        await bus.close()
    elapsed = time.monotonic() - started
    final_text = (result.get("final_text") or result.get("content") or "").strip()
    if not final_text:
        raise SystemExit(f"FAIL: empty finalize result from channel={channel} elapsed={elapsed:.1f}s")
    print(
        f"PASS channel={channel} corr={corr} elapsed={elapsed:.1f}s final_len={len(final_text)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke harness finalize exec lane")
    parser.add_argument(
        "--channel",
        default=os.environ.get(
            "CHANNEL_CORTEX_EXEC_REQUEST",
            "orion:cortex:exec:request:background",
        ),
        help="Cortex exec intake channel (default: harness background lane)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=float(os.environ.get("VOICE_FINALIZE_TIMEOUT_SEC", "120")),
        help="RPC wait budget",
    )
    args = parser.parse_args()
    asyncio.run(_smoke(args.channel, args.timeout_sec))


if __name__ == "__main__":
    main()
