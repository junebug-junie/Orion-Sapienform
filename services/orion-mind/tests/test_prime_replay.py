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
    Re-recorded again 2026-11-04 (energy money section): the ONLY change is the new "ENERGY MONEY:" section in
    each model call's user prompt (an unavailable gap here, since the fake has no /physiology/report); with that
    section stripped, all 12 changed parts were identical to the previous golden.

    Re-recorded again for curiosity replica patch 4 (organs answer Prime): the recording made on the pre-patch
    tree equalled the golden; the new recording differs from it ONLY by two added scenarios recorded last
    (``organ_challenge_in_frame``, ``organ_refusal_settles_question``). Every older scenario is byte-identical:
    with the replica's answering switch absent (as in all of them) Prime reads no view section and shows none.
    Re-recorded after merging main's energy money section: against main's golden every part of every older
    scenario is identical; the only difference is the two organ scenarios (which show both sections).
    Re-recorded 2026-11-12 (living Prime Part 2, turn hand-back): with the new CONCERN_BASIS text mapped back
    to the old one and ``last_thought_id`` removed from saved states, every part of every scenario equals the
    previous golden (checked by script; 9 basis and 10 last_thought_id occurrences were the whole diff).
    Re-recorded 2026-11-13 (personality rev 4, Prime's choice point): with the system prompt's new step 4 and
    "pick" key mapped back to the old text and the new ``choice=<outcome>`` field removed from prime_thought log
    lines, every part of every scenario equals the previous golden (checked by script). No scenario holds two
    concerns with a pick, so none publishes a choice (11 no_concerns, 4 no_pick).
    Re-recorded 2026-11-13 (promise concerns: quote in the frame, 72 h TTL, how a promise gets kept): with the
    CONCERN_BASIS fade words, the system prompt's fade sentence and new promise line, and the track record's fade
    words mapped back to the old text, every part of every scenario equals the previous golden (checked by script;
    22 parts changed, 0 after mapping). No scenario holds a promise concern.
    Re-recorded 2026-11-16 (integration patch 3, Prime reads Orion's now): with each frame's CURIOSITY section masked
    (the dashboard's runs left the frame; only whether a focus waits is shown), the new ``prime_now_unread
    reason=off`` log lines removed (the fake has no /temporal-self/now: a 404, so no section) and ``now_use=none``
    removed from prime_thought log lines, every part of every scenario equals the previous golden (checked by
    script; 29 parts changed, 0 after mapping). No id moved: the now's shown_id is drawn by durable-runs. Then one
    scenario was added, recorded last (``now_section_and_one_cite``: a now page, one shown cite and one not shown,
    and the use POST); every older scenario is byte-identical to the recording before it.
    Re-recorded 2026-11-16 (integration patch 5, Prime hunger; the quiet floor deleted): with every "quiet: nothing
    has woken me for N hours" change removed from the old golden's frames and close notes, the ``woke:`` prefix
    re-cut at its 160-character cap, and ``need=-`` removed from prime_thought log lines, every part of every
    scenario equals the previous golden (checked by script; 13 parts changed, 0 after mapping). No scenario turns
    hunger on (the flag is absent from the fake settings), and none woke on the floor alone.
    Re-recorded 2026-11-17 (#3609, a concern's event is reached by concern_id): with the system prompt's new
    two-line rule about a concern's own event removed, every part of every scenario equals the previous golden
    (checked by script; 15 parts changed, 0 after mapping).
    Then re-recorded 2026-11-17 (concern evidence clock, #3574): with the track record's and CONCERN_BASIS's fade words
    ("nothing renewed it for ...") mapped back to "not kept for ...", every part of every scenario equals the
    previous golden (checked by script; 8 parts changed, 0 after mapping). No scenario turns the clock on (the flag
    is absent from the fake settings), so a keep still renews in every scenario.
    Re-recorded 2026-11-24 (#3425, Prime's experience cites in the reply template): with the template's new top-level
    "refs" line and the reworded cite paragraph mapped back, every part of every scenario equals the previous golden
    (checked by script; 1 part changed, 0 after mapping: now_section_and_one_cite.llm_calls, the only scenario that
    shows a now).
    Re-recorded again 2026-11-24 (#3425 follow-up: the experience refs line moved before "move", so the template keeps
    its own closing): same check, same result (1 part changed, 0 after mapping).
    Re-recorded 2026-11-24 (reveries that ask, #3814): the now section's example list says "questions your reveries
    left" instead of "reveries"; with that mapped back, every part of every scenario equals the previous golden
    (checked by script; 1 part changed, 0 after mapping: now_section_and_one_cite.llm_calls).
    Re-recorded 2026-11-24 (#3425: a dropped cite's id is named early in the act note): with " | 1 cite(s) not shown:
    gpu_wait:not-shown" removed, every part equals the previous golden (checked by script; 2 parts changed,
    now_section_and_one_cite.logs and .published, 0 after mapping)."""
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
