from __future__ import annotations

from orion.cockpit.markers import COCKPIT_WORKING_NOTES_MARKER
from orion.hub.cockpit_emit import emit_motor_hop_from_claude_step


def test_working_notes_marker_becomes_its_own_stage() -> None:
    step = {"_cockpit": COCKPIT_WORKING_NOTES_MARKER, "carried_in": "> - 07:45Z | fcc | ls | ok",
            "enabled": True, "written_bytes": 30, "written": "x", "kept": 1, "dropped": 2, "suspicious": 0,
            "tainted_by": ["WebFetch"], "carried_forward": False, "read_error": None}
    frame = emit_motor_hop_from_claude_step("corr-wn", {"kind": "claude_step", "step": step, "step_index": 4})
    hop = frame["hop"]
    assert hop["stage"] == "working_notes"
    assert hop["visor_line"] == "working_notes · 30 bytes · held back (WebFetch)"
    assert hop["raw"]["carried_in"] == step["carried_in"] and "_cockpit" not in hop["raw"]
    assert hop["summary"]["dropped"] == 2 and hop["summary"]["tainted_by"] == ["WebFetch"]
