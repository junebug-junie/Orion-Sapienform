"""Identity hypothesis reaching presence.subject + council's window
evidence -- the 2026-08-26 patch (docs/superpowers/specs/2026-08-21-
seeing-juniper-identity-and-situated-observation-design.md sections 4/6.1).

Same direct-call pattern as test_belief_flush_wiring.py: instantiate
WindowService, mock only bus.publish, call _flush_and_publish directly, and
assert on the resulting live payload / presence registry -- no bus
subscribe-loop mocking needed since _identity_by_stream is populated
directly here, exactly like a real _consume_identity() call would.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject

from app.main import WindowService
from app import main as app_main


def _door_artifact() -> VisionArtifactPayload:
    """No 'person' label at all -- for the person-gating regression test."""
    return VisionArtifactPayload(
        artifact_id="art-door",
        correlation_id="c2",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="door", score=0.9, box_xyxy=[0, 0, 1, 1])]
        ),
        timing={},
        model_fingerprints={},
    )


def _person_artifact() -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id="art-person",
        correlation_id="c1",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="person", score=0.9, box_xyxy=[0, 0, 1, 1])]
        ),
        timing={},
        model_fingerprints={},
    )


@pytest.mark.asyncio
async def test_fresh_identity_hint_folds_into_window_evidence() -> None:
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert evidence["identity_hypothesis"] == {
        "subject": "juniper",
        "state": "probable",
        "similarity": 0.61,
    }


@pytest.mark.asyncio
async def test_fresh_identity_hint_not_folded_in_when_current_window_has_no_person() -> None:
    """Review finding, 2026-08-26: a fresh hint alone is not enough --
    council's own hedging rule ("AND person is in hard_labels") is a
    prompt instruction, not a guarantee, so this must be enforced in code.
    A hint can outlive the person it was about by up to
    WINDOW_IDENTITY_MAX_AGE_SEC; a window whose OWN observed labels have no
    person at all must never carry identity_hypothesis, regardless of how
    fresh the cached hint is."""
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _door_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert "identity_hypothesis" not in evidence
    assert "person" not in evidence.get("hard_labels", [])


@pytest.mark.asyncio
async def test_stale_identity_hint_is_not_folded_in() -> None:
    """Older than WINDOW_IDENTITY_MAX_AGE_SEC (default 90s) -- must not
    reach council's evidence as if it were current."""
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now - app_main.settings.WINDOW_IDENTITY_MAX_AGE_SEC - 1.0,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert "identity_hypothesis" not in evidence


@pytest.mark.asyncio
async def test_no_identity_hint_omits_the_evidence_key_entirely() -> None:
    """No hint at all (never enrolled, never dispatched, no face detected)
    -- the field must be ABSENT, not present-and-empty/unsure. Council's
    prompt rule only activates on the field's presence."""
    svc = WindowService()
    now = time.time()

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert "identity_hypothesis" not in evidence


@pytest.mark.asyncio
async def test_window_identity_disabled_ignores_a_fresh_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_main.settings, "WINDOW_IDENTITY_ENABLED", False)
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert "identity_hypothesis" not in evidence


@pytest.mark.asyncio
async def test_window_belief_disabled_also_ignores_a_fresh_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """WINDOW_IDENTITY_ENABLED alone is not sufficient (review finding,
    2026-08-26) -- the identity fetch/fold is gated on BOTH flags, since
    presence and evidence-folding both live inside the WINDOW_BELIEF_ENABLED
    branch. Belief disabled, identity left at its true default: no fetch,
    no fold -- matches start()'s own refusal to even launch the identity
    consumer in this combination."""
    monkeypatch.setattr(app_main.settings, "WINDOW_BELIEF_ENABLED", False)
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    # WINDOW_BELIEF_ENABLED=False means summary["evidence"] is never built
    # at all in this codepath -- the window's own raw summary is what's live.
    evidence = svc._live_by_stream["cam0"].summary.get("evidence", {})
    assert "identity_hypothesis" not in evidence


@pytest.mark.asyncio
async def test_fresh_identity_hint_narrows_presence_subject() -> None:
    """The same flush that folds identity into council's evidence must also
    narrow presence.subject -- one hint, two consumers, per the design
    doc's own §5/§6.1 fusion."""
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        # SceneBeliefTracker requires WINDOW_BELIEF_ENTER_VOTES (live: 3)
        # consecutive observations before a label is "believed" -- a single
        # flush's raw hard_labels are not yet what presence.py reads.
        # believed_labels is what _note_presence is actually called with,
        # same as test_belief_flush_wiring.py's own multi-flush pattern.
        for _ in range(3):
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )

    snapshot = svc._presence_registry.current_snapshot("cam0")
    assert snapshot["subject"] == "juniper"


@pytest.mark.asyncio
async def test_identity_hint_for_a_different_stream_does_not_bleed_over() -> None:
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["carbon"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for _ in range(3):
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert "identity_hypothesis" not in evidence
    snapshot = svc._presence_registry.current_snapshot("cam0")
    assert snapshot["subject"] == "unknown"


# -- identity_uncertain: the "is that you?" signal, 2026-08-26 ---------------
# Juniper's direct ask: confirmed -> carry on and never mention it; genuinely
# uncertain -> a real, but at-most-once, clarifying signal; broken/not
# running -> silence. presence.subject/identity_hypothesis above are the
# "confirmed" half; these are the "uncertain" half.


@pytest.mark.asyncio
async def test_fresh_uncertain_confidence_sets_presence_identity_uncertain() -> None:
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {"hint": None, "confidence": "uncertain", "ts": now}

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for _ in range(3):  # clear WINDOW_BELIEF_ENTER_VOTES, same as the presence tests above
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )

    snapshot = svc._presence_registry.current_snapshot("cam0")
    assert snapshot["identity_uncertain"] is True


@pytest.mark.asyncio
async def test_confirmed_confidence_never_sets_presence_identity_uncertain() -> None:
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": {"subject": "juniper", "state": "probable", "similarity": 0.61},
        "confidence": "confirmed",
        "ts": now,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for _ in range(3):
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )

    snapshot = svc._presence_registry.current_snapshot("cam0")
    assert snapshot["subject"] == "juniper"
    assert snapshot["identity_uncertain"] is False


@pytest.mark.asyncio
async def test_no_identity_signal_at_all_never_sets_presence_identity_uncertain() -> None:
    """The subsystem simply not running/no fresh read -- must stay silent,
    not manufacture a false "I don't recognize you"."""
    svc = WindowService()
    now = time.time()
    # _identity_by_stream deliberately left empty for cam0.

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    snapshot = svc._presence_registry.current_snapshot("cam0")
    assert snapshot["identity_uncertain"] is False


@pytest.mark.asyncio
async def test_stale_uncertain_confidence_does_not_set_presence_identity_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = WindowService()
    now = time.time()
    svc._identity_by_stream["cam0"] = {
        "hint": None,
        "confidence": "uncertain",
        "ts": now - app_main.settings.WINDOW_IDENTITY_MAX_AGE_SEC - 1.0,
    }

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _person_artifact(), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    snapshot = svc._presence_registry.current_snapshot("cam0")
    assert snapshot["identity_uncertain"] is False


# -- _should_ignore_uncertain_reading: pure decision logic --------------------
# Extracted from _consume_identity's loop for direct testability (2026-08-26
# review). "Sticky confirmed" is the behavior under test: a single flickery
# unsure frame must not undo an already-settled, still-fresh confirmed read.


def test_ignores_uncertain_reading_when_a_fresh_confirmed_entry_exists() -> None:
    existing = {"confidence": "confirmed", "ts": 100.0}
    assert app_main._should_ignore_uncertain_reading(
        existing, "uncertain", now=105.0, max_age_sec=90.0
    ) is True


def test_does_not_ignore_uncertain_reading_once_the_confirmed_entry_goes_stale() -> None:
    existing = {"confidence": "confirmed", "ts": 100.0}
    assert app_main._should_ignore_uncertain_reading(
        existing, "uncertain", now=100.0 + 90.1, max_age_sec=90.0
    ) is False


def test_does_not_ignore_uncertain_reading_when_no_existing_entry() -> None:
    assert app_main._should_ignore_uncertain_reading(
        None, "uncertain", now=105.0, max_age_sec=90.0
    ) is False


def test_does_not_ignore_uncertain_reading_when_existing_entry_is_also_uncertain() -> None:
    """Only a CONFIRMED existing entry gets the hold-off -- successive
    unsure readings should keep refreshing the uncertain signal's own
    freshness, not get stuck on the first one."""
    existing = {"confidence": "uncertain", "ts": 100.0}
    assert app_main._should_ignore_uncertain_reading(
        existing, "uncertain", now=105.0, max_age_sec=90.0
    ) is False


def test_a_new_confirmed_reading_is_never_subject_to_the_hold_off() -> None:
    """The hold-off only ever applies when new_confidence == 'uncertain' --
    a new confirmed reading always overwrites immediately, regardless of
    what the existing entry says."""
    existing = {"confidence": "confirmed", "ts": 100.0}
    assert app_main._should_ignore_uncertain_reading(
        existing, "confirmed", now=105.0, max_age_sec=90.0
    ) is False
