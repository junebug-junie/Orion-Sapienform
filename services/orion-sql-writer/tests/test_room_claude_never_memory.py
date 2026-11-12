"""A side exchange with Claude (client_meta.room_claude) never becomes Orion's memory.

Open question 11 (2026-11-12): Hub now stores Juniper's @claude prompt with the
room_claude row, in client_meta.room_claude.prompt. Orion may read the side
exchange; it must not be appraised as a Juniper turn.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SQL_WRITER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SPEC = importlib.util.spec_from_file_location(
    "sql_writer_worker_room_claude_memory_tests", SQL_WRITER_ROOT / "app" / "worker.py"
)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

ROOM_CLAUDE = {"room_claude": {"request_id": "r1", "prompt": "@claude what do you think?"}}


def test_room_claude_row_without_prompt_is_not_memory():
    assert worker._memory_turn_initiator("", ROOM_CLAUDE) is None


def test_room_claude_row_with_prompt_column_is_still_not_memory():
    assert worker._memory_turn_initiator("what do you think?", ROOM_CLAUDE) is None


def test_room_claude_meta_as_json_string_is_not_memory():
    import json

    assert worker._memory_turn_initiator("x", json.dumps(ROOM_CLAUDE)) is None


def test_ordinary_turns_are_unchanged():
    assert worker._memory_turn_initiator("hello", None) == "juniper"
    assert worker._memory_turn_initiator("", {"unsolicited": True}) == "orion"
    assert worker._memory_turn_initiator("", {}) is None
    # A string client_meta never counted as unsolicited before; it still does not.
    assert worker._memory_turn_initiator("", '{"unsolicited": true}') is None
