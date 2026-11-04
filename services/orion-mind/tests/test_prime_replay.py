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
from orion.prime.state import PRIME_CONCERNS_KEY, PrimeStateV1
from test_prime_loop import Bus, runtime

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "fixtures" / "prime_replay_golden.json"
LIVE_STATE = HERE / "fixtures" / "prime_state_live_2026-11-04.json"


def test_recorded_tick_sequences_replay_identically():
    """The scenarios run in a fresh interpreter (ids come from a counter there, see the script).

    Re-recorded once since the extraction (2026-11-04, curiosity replica patch 3): Prime now publishes its
    concerns view after every state save. The ONLY difference from the pre-extraction golden was one
    ``save orion:prime:concerns`` timeline line after each ``save orion:prime:state`` (59 in all); every act,
    bid, model call, saved state and log line was unchanged (checked by diffing the two recordings).

    Re-recorded again for curiosity replica patch 4 (organs answer Prime): the recording made on the pre-patch
    tree equalled the golden; the new recording differs from it ONLY by two added scenarios recorded last
    (``organ_challenge_in_frame``, ``organ_refusal_settles_question``). Every older scenario is byte-identical:
    with the replica's answering switch absent (as in all of them) Prime reads no view section and shows none."""
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


def test_a_second_instance_beside_prime_keeps_its_own_key_family_and_actor():
    """Two instances on one bus and one Redis (as they will run in orion-mind): neither touches the other."""
    from orion.mindloop.tests.echo_instance import ECHO_KEY, EchoStateV1, echo_runtime
    from test_prime_loop import MOVE

    bus = Bus()
    prime = runtime(answer=MOVE, bus=bus)
    echo = echo_runtime([({"echo": "hi"}, None, {})], bus=bus, turns=prime.turns)
    asyncio.run(prime.tick())
    prime_saved = bus.redis.kv[STATE_KEY]
    asyncio.run(echo.tick())
    # Prime's published concerns view (orion:prime:concerns) is the only other key: a read-only call, not state.
    assert set(bus.redis.kv) == {STATE_KEY, ECHO_KEY, PRIME_CONCERNS_KEY} and bus.redis.kv[STATE_KEY] == prime_saved
    assert EchoStateV1.model_validate_json(bus.redis.kv[ECHO_KEY]).answers == ["hi"]
    assert [w.family for w in prime.bids] == ["prime"] and [w.family for w in echo.bids] == ["test.echo"]
    actors = [a["actor"] for _, a in bus.acts()]
    assert set(actors) == {"prime", "echo"} and actors[-2:] == ["echo", "echo"]
    assert prime.state is not echo.state and prime.gate is not echo.gate
