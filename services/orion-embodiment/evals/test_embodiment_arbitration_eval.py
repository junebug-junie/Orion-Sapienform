"""Eval: deliberate-vs-involuntary arbitration + non-empty outcome reasons.

Replays a scripted intent sequence through ``EmbodimentWorker.process_intent``
with a controlled clock and asserts the arbitration contract holds end to end:

  * a deliberate intent at t=0 opens a hold window (``deliberate_hold_sec``),
  * an involuntary intent at t=2s (inside the hold) is ``preempted``,
  * an involuntary intent at t=10s (after the hold) resumes and actuates,
  * EVERY outcome carries a non-empty ``reason`` (anti empty-shell §0A).

Distinct from unit tests: this is a behavior gate over a full replay, not a
single-case assertion.

Run: pytest services/orion-embodiment/evals -q
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

# Path bootstrap so `__main__` (standalone) execution resolves `app` and `orion`
# without pytest's conftest. Mirrors evals/conftest.py; idempotent under pytest.
_HERE = Path(__file__).resolve()
_SERVICE_ROOT = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[3]
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from app.worker import EmbodimentWorker  # noqa: E402
from orion.embodiment.arbiter import ArbiterState  # noqa: E402
from orion.schemas.embodiment import EmbodimentIntentV1  # noqa: E402

_T0 = datetime(2026, 7, 7, 0, 0, 0, tzinfo=timezone.utc)
_HOLD_SEC = 8.0
_PLAYERS = [{"id": "orion", "name": "Orion", "position": {"x": 0.0, "y": 0.0}}]

# (offset_sec, source, kind, expected_status)
SCRIPT: list[tuple[float, str, str, str]] = [
    (0.0, "deliberate", "wander", "actuated"),   # opens the hold window
    (2.0, "involuntary", "wander", "preempted"),  # inside hold -> deliberate wins
    (10.0, "involuntary", "wander", "actuated"),  # after hold -> involuntary resumes
]


def _worker() -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._arbiter = ArbiterState()
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._hold_sec = _HOLD_SEC
    w._wander_radius = 3.0
    w._locations = {}
    w._social_cooldown_sec = 120.0
    w._last_conversation_start = None
    return w


def _score() -> tuple[int, int]:
    w = _worker()
    correct = 0
    with patch("app.worker.aitown_client.list_players", return_value=_PLAYERS), \
         patch("app.worker.aitown_client.move_to", return_value={"ok": True}), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}):
        for offset, source, kind, expected in SCRIPT:
            intent = EmbodimentIntentV1(
                kind=kind, source=source, reason=f"{source} {kind}",
                correlation_id=f"c-{offset}", player_id="orion",
            )
            outcome = w.process_intent(intent, now=_T0 + timedelta(seconds=offset))
            status_ok = outcome.status == expected
            reason_ok = bool(outcome.reason and outcome.reason.strip())
            if status_ok and reason_ok:
                correct += 1
    return correct, len(SCRIPT)


def test_arbitration_deliberate_wins_then_involuntary_resumes():
    correct, total = _score()
    accuracy = correct / total
    # Any miss means either the deliberate hold failed to preempt an involuntary
    # intent (arbitration broken) or an outcome shipped with an empty reason
    # (empty-shell §0A failure).
    assert accuracy == 1.0, f"arbitration eval accuracy {accuracy:.2f} ({correct}/{total})"


if __name__ == "__main__":
    c, t = _score()
    print(f"embodiment arbitration eval: {c}/{t} correct ({c / t:.0%})")
