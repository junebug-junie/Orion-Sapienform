"""Gate: Juniper is not deleted from the town on a join-time timer.

HUMAN_IDLE_TOO_LONG (5 minutes) lived in player.tick() and called player.leave()
when lastInput aged out. lastInput was only written at join, so this was a
session fuse, not an idle check. Live 2026-08-29: Juniper was removed mid-Cam
chat ~5 minutes after joining.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-no-human-idle-kick.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_idle_kick_patch_registered_last():
    text = _APPLY.read_text(encoding="utf-8")
    order = [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]
    assert "orion-no-human-idle-kick.patch" in order
    assert order[-1] == "orion-no-human-idle-kick.patch"


def test_idle_kick_patch_removes_human_leave_on_last_input():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "HUMAN_IDLE_TOO_LONG" in patch
    assert "-      this.leave(game, now);" in patch
    assert "+      this.leave(game, now);" not in patch
    assert "lastInput < now - HUMAN_IDLE_TOO_LONG" in patch
    # The comparison must only appear on removed lines.
    for line in patch.splitlines():
        if "lastInput < now - HUMAN_IDLE_TOO_LONG" in line:
            assert line.startswith("-")
