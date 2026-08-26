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
