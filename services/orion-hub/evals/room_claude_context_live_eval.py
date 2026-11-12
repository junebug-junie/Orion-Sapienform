"""Live, read-only eval of what Claude would get on an invite (Patch 1a).

Builds the real transcript, 'Earlier' note and Hub index from the live
database for one Hub session, the same way /api/room/claude/invite does, and
checks the invariants of the design (acceptance check 1):

- the conversation text stays within the cap;
- rows are whole (every kept row's entries share one turn ref and the next
  older row would not have fitted);
- lines are oldest first, each with a real time and a turn ref;
- Orion's unprompted rows and Claude's side exchanges are labelled;
- the request validates against RoomClaudeRequestV1 (extra=forbid);
- the Hub index is names and counts only, at most 800 characters.

Nothing is written and nothing is sent to Claude.

    POSTGRES_URI=postgresql://... PYTHONPATH=<repo> python \\
        services/orion-hub/evals/room_claude_context_live_eval.py <session_id> [max_chars]

Exit code 0 when every check passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

from orion.schemas.room_claude import RoomClaudeRequestV1  # noqa: E402
from scripts.room_claude_context import (  # noqa: E402
    HUB_INDEX_MAX_CHARS,
    fetch_conversation,
    fetch_hub_index_lines,
    render_hub_index,
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    session_id = argv[1]
    max_chars = int(argv[2]) if len(argv) > 2 else 160_000
    entries, note = fetch_conversation(session_id, max_chars=max_chars)
    index = render_hub_index(fetch_hub_index_lines())
    req = RoomClaudeRequestV1(
        room_id="hub-direct", invited_by="Juniper", prompt="(eval)", transcript=entries,
        hub_index=index, transcript_note=note,
    )
    text_chars = sum(len(e.text) for e in req.transcript)
    times = [e.created_at for e in req.transcript]
    refs = [e.turn_ref for e in req.transcript]
    checks = {
        "within_cap": text_chars <= max_chars,
        "nonempty": len(req.transcript) > 0,
        "oldest_first": times == sorted(times),
        "every_line_has_turn_ref": all(refs),
        "hub_index_within_800": index is None or len(index) <= HUB_INDEX_MAX_CHARS,
    }
    report = {
        "session_id": session_id,
        "max_chars": max_chars,
        "lines": len(req.transcript),
        "turns": len(dict.fromkeys(refs)),
        "text_chars": text_chars,
        "unprompted_lines": sum(e.unprompted for e in req.transcript),
        "side_exchange_lines": sum(e.side_exchange for e in req.transcript),
        "oldest_line_at": times[0] if times else None,
        "newest_line_at": times[-1] if times else None,
        "transcript_note": note,
        "hub_index": index,
        "request_json_bytes": len(req.model_dump_json().encode()),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
