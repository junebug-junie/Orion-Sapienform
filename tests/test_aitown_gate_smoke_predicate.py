"""The ai-town gate smoke's FAIL branch must be reachable.

Two earlier versions of that check were wrong in opposite directions and both
read as passing evidence:

  1. `if total and not auto: fail` -- inverted the moment the gate worked. In
     steady state no ai-town window reaches status='proposed', so `auto` is
     legitimately empty and the smoke failed forever.
  2. `resolved_platform is allowlisted but cid not in auto` -- provably always
     empty. The smoke replays live rows through the same pure policy functions
     that produce both sides of that comparison, so with dominant_shift
     hardcoded to STANCE and sensitivity hardcoded to "private" the two
     conditions are literally equivalent. The FAIL branch was unreachable.

The fix was to check the DEPLOYED service's own output instead of the replay's.
This pins that the resulting predicate can actually fire, which is the one
property neither previous version had.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_aitown_crystallization_gate import find_leaked  # noqa: E402


def test_fail_branch_is_reachable():
    """The whole point. An ai-town-stamped row sitting in the queue is a leak."""
    leaked = find_leaked([("crys-1", "aitown")])
    assert leaked == [("crys-1", "aitown")]


def test_direct_rows_are_not_leaks():
    assert find_leaked([("crys-1", None), ("crys-2", "")]) == []


def test_unlisted_platform_is_not_a_leak():
    """`hub` turns up in live data (chat_history_log, 2026-08-14) and is not on
    the allowlist, so queueing one is correct behavior, not a leak."""
    assert find_leaked([("crys-1", "hub")]) == []


def test_empty_queue_is_not_a_leak():
    assert find_leaked([]) == []


def test_honors_a_caller_supplied_allowlist():
    rows = [("crys-1", "discord")]
    assert find_leaked(rows) == []
    assert find_leaked(rows, platforms=frozenset({"discord"})) == rows


def test_mixed_batch_reports_only_the_leaks():
    leaked = find_leaked(
        [("a", None), ("b", "aitown"), ("c", "hub"), ("d", "aitown")]
    )
    assert [r[0] for r in leaked] == ["b", "d"]
