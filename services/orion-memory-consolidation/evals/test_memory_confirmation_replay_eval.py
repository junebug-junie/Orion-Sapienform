"""The confirmation replay eval, on the committed v3 counts, against a disposable Postgres.

Pins the cap behavior the eval reports: never more than 5 open cards, at most 3 new a day; with no answers every card
expires to "unconfirmed" (never confirmed); with answers the queue drains five at a time.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from collections import Counter
from pathlib import Path

import pytest

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")

_spec = importlib.util.spec_from_file_location(
    "memconf_replay_eval", Path(__file__).resolve().parent / "run_memory_confirmation_replay_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_v3_replay_respects_the_cap_and_never_turns_silence_into_yes():
    rows = ev.synth_v3()
    high = Counter(r["stakes_reason"] for r in rows if r["stakes"] == "high")
    assert sum(high.values()) == 15

    silent = asyncio.run(ev.run_scenario(ADMIN_DSN, rows, answers=False))
    assert silent["cards_opened"] == 15 and silent["max_open_at_once"] == 5 and silent["cap_held"]
    assert silent["cards_by_category"] == dict(high)
    assert silent["high_stakes_end_states"] == {"unconfirmed": 15}
    # At most 3 new cards a local day (review of #2517) and never more than 5 open.
    assert [(t["day"], t["opened"]) for t in silent["ticks_with_activity"]] == [
        (0, 3), (1, 2), (7, 3), (8, 2), (14, 3), (15, 2), (21, 0), (22, 0)]

    answered = asyncio.run(ev.run_scenario(ADMIN_DSN, rows, answers=True))
    assert answered["high_stakes_end_states"] == {"confirmed": 15} and answered["max_open_at_once"] == 3
    assert [t["opened"] for t in answered["ticks_with_activity"]] == [3, 3, 3, 3, 3, 0]
