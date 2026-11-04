"""No behavior change across the loop-library extraction (docs/superpowers/specs/2026-11-04-curiosity-replica-design.md
section 10): recorded tick sequences give the same publishes, bids, model calls, saved state and log lines, and the
live persisted state captured from Redis on 2026-11-04 still loads and saves back byte for byte."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from app.prime_loop import STATE_KEY
from orion.prime.state import PrimeStateV1
from test_prime_loop import Bus, runtime

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "fixtures" / "prime_replay_golden.json"
LIVE_STATE = HERE / "fixtures" / "prime_state_live_2026-11-04.json"


def test_recorded_tick_sequences_replay_identically():
    """The scenarios run in a fresh interpreter (ids come from a counter there, see the script)."""
    proc = subprocess.run([sys.executable, "-W", "ignore", str(HERE / "prime_replay_scenarios.py")],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got, want = json.loads(proc.stdout), json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    for name in want:
        for part in want[name]:
            assert got[name][part] == want[name][part], f"{name}.{part} changed"


def test_the_live_persisted_state_loads_and_saves_back_unchanged():
    raw = LIVE_STATE.read_text().strip()
    bus = Bus()
    bus.redis.kv[STATE_KEY] = raw.encode()
    rt = runtime(answer=None, bus=bus)
    asyncio.run(rt.load())
    assert rt.state.thoughts_total == 200 and len(rt.state.concern_record) == 20
    assert rt.state.open_nominations and rt.state.concerns
    asyncio.run(rt.save())
    assert bus.redis.kv[STATE_KEY] == raw
    assert STATE_KEY == "orion:prime:state"
    assert PrimeStateV1.model_validate_json(raw).model_dump_json() == raw
