"""Regression (pr-field-note-3104, 2026-11-04): the field-note turn must leave the harness motor a real
budget AFTER Hub takes its finalize reserve.

Hub's held-turn path gives the motor ``turn_limit - HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC``
(services/orion-hub/scripts/curiosity_investigation.py ``held_turn_fcc_budget_sec``), and the turn limit
durable-runs sends is ``timeout_sec - TURN_REPLY_GRACE_SEC``. With timeout 420 and reserve 330 the motor
got 60 s; a cold claude spawn alone took 58 s to its first event, and the turn died ``fcc_timeout``.
This reads the shipped defaults of BOTH services, so changing either one cannot silently starve it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.harness_step import TURN_REPLY_GRACE_SEC  # noqa: E402
from app.settings import Settings  # noqa: E402

# A note turn's motor work: a cold claude spawn (~60 s to first event live), the packet read (the dig
# builds the live body, ~30-40 s), the write (builds it again) and the model's own decode.
MIN_MOTOR_SEC = 300.0
# Hub also spends time between receipt and the motor start (lock wait, prompt read), and caps the motor at
# the reply deadline. Both only shrink the budget; assume this much of them so the gate stays conservative.
HUB_OVERHEAD_SEC = 60.0


def _example(service: str, key: str) -> float:
    text = (ROOT / "services" / service / ".env_example").read_text()
    return float(re.search(rf"^{key}=(\S+)$", text, re.M).group(1))


def _compose_default(service: str, key: str) -> float:
    text = (ROOT / "services" / service / "docker-compose.yml").read_text()
    return float(re.search(rf"\${{{key}:-([0-9.]+)}}", text).group(1))


def motor_budget(timeout_sec: float, reserve_sec: float) -> float:
    """Hub's arithmetic for a held turn, from the request durable-runs sends."""
    turn = timeout_sec - TURN_REPLY_GRACE_SEC
    budget = turn - reserve_sec - HUB_OVERHEAD_SEC
    return budget if budget > 0 else turn


def test_the_shipped_field_note_turn_leaves_the_motor_a_real_budget():
    reserves = {_example("orion-hub", "HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC"),
                _compose_default("orion-hub", "HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC")}
    timeouts = {_example("orion-durable-runs", "PR_FIELD_NOTES_TURN_TIMEOUT_SEC"),
                _compose_default("orion-durable-runs", "PR_FIELD_NOTES_TURN_TIMEOUT_SEC"),
                Settings.model_fields["field_note_turn_timeout_sec"].default}
    for reserve in reserves:
        for timeout in timeouts:
            assert motor_budget(timeout, reserve) >= MIN_MOTOR_SEC, (timeout, reserve)


def test_the_old_default_is_exactly_the_live_failure():
    # the "fcc turn timed out after 59.99s" of pr-field-note-3104: 390 - 330 = 60 before any overhead
    assert motor_budget(420.0, 330.0) + HUB_OVERHEAD_SEC == 60.0
