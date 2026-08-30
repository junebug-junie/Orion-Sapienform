"""Gate tests for the AI Town NPC continuity ingest/summary patch.

Companion to orion-concrete-grounding-prompt.patch: NPC-to-NPC (and
NPC-to-Juniper) speech should read aitown-town continuity once at conversation
start and publish SocialRoomTurnV1 after a successful continue/leave line.
Orion↔anyone stays embodiment-only. Fail-open: ingest and summary fetch never
block speech. Slugs are hardcoded from orion/town_cast.py — never inferred.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-town-continuity-ingest.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_continuity_patch_registered_after_grounding():
    # apply script lists orion-town-continuity-ingest.patch after orion-concrete-grounding-prompt.patch
    text = _APPLY.read_text(encoding="utf-8")
    order = [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]
    assert "orion-town-continuity-ingest.patch" in order
    assert order.index("orion-concrete-grounding-prompt.patch") < order.index(
        "orion-town-continuity-ingest.patch"
    )


def test_continuity_patch_hardcodes_slug_map():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "'Sofia Bell': 'sofia-bell'" in patch
    assert "name.lower()" not in patch
    assert 'replace(" ", "-")' not in patch


def test_continuity_patch_skips_orion_as_other():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "isOrionName" in patch
    assert "AITOWN_ORION_NAME" in patch


def test_continuity_patch_fetches_summary_once_at_start():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "startConversationMessage" in patch
    assert "/summary" in patch
    assert "aitown-town" in patch


def test_continuity_patch_ingests_on_continue_and_leave():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "ingestTownTurn" in patch
    assert "/ingest-turn" in patch
    assert "continueConversationMessage" in patch
    assert "leaveConversationMessage" in patch
