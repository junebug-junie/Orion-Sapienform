"""Orion's now never feeds a prompt that goes to an outside model (spec 2026-11-15, Part A proposal-mode
checklist, "Privacy"; Orion's question 8 on #3528).

The now can carry ``juniper_chat`` items. Juniper's rule is no privacy boundary between her and Orion, but
memory and now content stay on local models. This pins the CODE paths: no module that builds an outside-model
prompt (consult / ask-Claude, the Claude room, the FCC Claude bridge) reads the now or renders its items.

It does not cover text Orion writes itself (a consult question or a help request that quotes an experience).
For consult prompts that is now a runtime check instead (#3541): ``consult_loop._send_claude`` redacts any
span that quotes, or repeats an 8+ word run of, a core-page or lived-memory sentence (orion/memory/mesh_edge.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

OUTSIDE_MODEL_MODULES = (
    "orion/autonomy/ask_claude_trigger.py",
    "orion/consult/beliefs.py",
    "orion/consult/core.py",
    "services/orion-hub/scripts/consult_loop.py",
    "services/orion-hub/scripts/fcc_claude_bridge.py",
    "services/orion-hub/scripts/room_claude_context.py",
    "services/orion-hub/scripts/room_claude_relay.py",
    "services/orion-hub/scripts/agent_claude_input.py",
)

# Every way the now reaches a prompt today: the endpoint, the reducer, the hunger read, and the three readers'
# render helpers (Prime frame, reverie cue, curiosity kickoff section).
NOW_MARKERS = (
    "/temporal-self/now",
    "/temporal-self/hunger",
    "orion.temporal_self.now",
    "temporal_self.now import",
    "build_now",
    "now_section",
    "curiosity_now",
    "temporal_self_cue",
    "WHAT YOUR ORGANS LIVED",
)


@pytest.mark.parametrize("rel", OUTSIDE_MODEL_MODULES)
def test_no_outside_model_prompt_builder_reads_the_now(rel):
    path = REPO / rel
    assert path.exists(), f"{rel} moved: update OUTSIDE_MODEL_MODULES so the boundary stays pinned"
    text = path.read_text(encoding="utf-8")
    hits = [m for m in NOW_MARKERS if m in text]
    assert not hits, f"{rel} reads or renders Orion's now ({hits}); now content must stay on local models"
