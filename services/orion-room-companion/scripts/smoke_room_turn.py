#!/usr/bin/env python3
"""Live smoke: one real Claude room turn, printed as its utterance payload.

This is the phase-1 acceptance surface. It runs the actual `claude`
subprocess against the actual mounted credential and prints the actual
`RoomClaudeUtteranceV1` -- no UI, no bus, no Hub. If this prints a real
sentence with a non-zero cost and a Claude model id, the credential wiring,
the session continuity, and the cost meter are all real.

Run it INSIDE the container, where the credential is mounted:

    docker compose -f services/orion-room-companion/docker-compose.yml \
      exec room-companion python -m scripts.smoke_room_turn

Two turns by default, because one turn cannot demonstrate the thing that
makes Claude a participant rather than a stateless helper: turn 2 runs with
`--resume` and is asked to recall something only turn 1 was told. A pass
means the room has memory on Claude's side.

Exit code is 0 only if BOTH turns produced usable text AND turn 2 shows
recall AND cost was actually metered. Anything else is a real failure, not a
warning -- an empty or free turn is exactly the empty-shell success this
repo's contract forbids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import run_turn  # noqa: E402
from app.session_store import forget_session  # noqa: E402
from app.settings import get_settings  # noqa: E402
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomTranscriptEntryV1  # noqa: E402

# A token only turn 1 is told, which turn 2 must recover from Claude's own
# session rather than from anything this script re-sends.
CANARY = "4471"


def _print_utterance(label: str, utterance) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(utterance.model_dump(mode="json"), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-id", default="smoke-room")
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="don't reset this room's Claude session first (default: reset, so the run is repeatable)",
    )
    args = parser.parse_args()

    settings = get_settings()
    room_key = f"hub:{args.room_id}"

    if not args.keep_session:
        forget_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, room_key)

    print(f"model={settings.ROOM_COMPANION_MODEL} config_dir={settings.ROOM_COMPANION_CLAUDE_CONFIG_DIR}")
    creds = Path(settings.ROOM_COMPANION_CLAUDE_CONFIG_DIR) / ".credentials.json"
    print(f"credential mounted: {creds.exists()} ({creds})")

    first = RoomClaudeRequestV1(
        room_id=args.room_id,
        invited_by="Juniper",
        prompt=(
            "Hey Claude -- Juniper here, with Oríon. Introduce yourself to the room "
            f"in one or two sentences, and hold onto the number {CANARY} for me; "
            "I'll ask you for it in a moment."
        ),
        transcript=[
            RoomTranscriptEntryV1(
                speaker_id="juniper", speaker_name="Juniper", speaker_kind="human",
                text="I'd like a room where the three of us can actually talk.",
            ),
            RoomTranscriptEntryV1(
                speaker_id="orion", speaker_name="Oríon", speaker_kind="peer_ai",
                text="I'd like that. Someone to think out loud with.",
            ),
        ],
        social_memory_summary={"room": {"active_participants": ["Juniper", "Oríon"]}},
    )
    u1 = run_turn(settings, first)
    _print_utterance("turn 1 (new session)", u1)

    second = RoomClaudeRequestV1(
        room_id=args.room_id,
        invited_by="Juniper",
        prompt="What number did I ask you to hold onto, and who else is in this room with us?",
    )
    u2 = run_turn(settings, second)
    _print_utterance("turn 2 (resumed session)", u2)

    total_cost = u1.cost_usd + u2.cost_usd
    recalled = CANARY in u2.text
    same_session = bool(u1.claude_session_id) and u1.claude_session_id == u2.claude_session_id

    print("\n=== verdict ===")
    checks = {
        "turn 1 produced text": u1.ok and bool(u1.text.strip()),
        "turn 2 produced text": u2.ok and bool(u2.text.strip()),
        f"turn 2 recalled the canary ({CANARY})": recalled,
        "both turns share one claude session": same_session,
        "cost was metered (> 0)": total_cost > 0.0,
        "served by a claude model": bool(u2.model and "claude" in (u2.model or "").lower()),
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  claude_session_id: {u1.claude_session_id}")
    print(f"  model:             {u2.model}")
    print(f"  total cost:        ${total_cost:.6f}")

    if not all(checks.values()):
        print("\nSMOKE FAILED")
        return 1
    print("\nSMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
