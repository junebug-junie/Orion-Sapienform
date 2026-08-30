"""Gate tests for the AI Town NPC continuity ingest/pair-turn patch.

Companion to orion-concrete-grounding-prompt.patch: NPC-to-NPC openers fetch
aitown-town pair/town turns in startConversationMessage. Human (Juniper) chats
skip that start path (orion-town-chat-turns.patch) and GET /town-continuity on
the first continue instead. Continue/leave still publish SocialRoomTurnV1 when
the other is not Orion. Orion↔anyone stays embodiment-only. Fail-open: ingest
and continuity fetch time out in 3s and never block the speech return path.
Slugs are hardcoded from orion/town_cast.py — never inferred.
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
    assert "/town-continuity" in patch
    assert "/summary" not in patch
    assert "/ingest-turn" in patch
    assert "aitown-town" in patch


def test_continuity_patch_ingests_on_continue_and_leave():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "ingestTownTurn" in patch
    assert "/ingest-turn" in patch
    assert "continueConversationMessage" in patch
    assert "leaveConversationMessage" in patch


def test_continuity_patch_aborts_fetches_in_3s():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "AbortSignal" in patch or "timeout(3000)" in patch


def test_continuity_patch_does_not_await_ingest_on_speech_return():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "await ingestTownTurn" not in patch
    assert "ingestTownTurn(" in patch


def test_continuity_patch_fetches_on_first_continue():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "continueConversationMessage" in patch
    assert "priorMessages.length <= 1" in patch or "previousMessages.length <= 1" in patch
    assert "Earlier with them:" in patch
    assert "From your other conversations:" in patch
    assert "What you remember:" not in patch


def test_continuity_patch_fetch_uses_speaker_and_other():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "fetchTownContinuity(player.name, otherPlayer.name)" in patch
    assert "thread_id" in patch
    assert "speaker_id" in patch


def test_continuity_patch_omits_topic_bag_labels():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "What you remember:" not in patch
    assert "safe_continuity_summary" not in patch


def test_readme_says_juniper_gets_continuity_on_first_continue():
    readme = (_SERVICE / "README.md").read_text(encoding="utf-8")
    assert "first continue" in readme.lower()
    assert "Juniper chats never GET" not in readme
    assert "GET /town-continuity" in readme or "`GET /town-continuity`" in readme
    assert "thread_id" in readme
    assert "speaker_id" in readme
    assert "embodiment" in readme.lower() and "/summary" in readme
