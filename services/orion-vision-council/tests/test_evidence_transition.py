import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionWindowPayload

from app.evidence_transition import (
    EvidenceTransitionTracker,
    snapshot_from_window,
)
from app.main import CouncilService


def _window(
    *,
    window_id: str = "w1",
    stream_id: str = "cam0",
    hard_labels: list[str] | None = None,
    object_counts: dict[str, int] | None = None,
) -> VisionWindowPayload:
    hard_labels = hard_labels or []
    object_counts = object_counts or {}
    return VisionWindowPayload(
        window_id=window_id,
        start_ts=1.0,
        end_ts=2.0,
        stream_id=stream_id,
        summary={
            "object_counts": object_counts,
            "top_labels": [],
            "captions": [],
            "item_count": 1,
            "detection_count": sum(object_counts.values()),
            "evidence": {
                "hard_labels": hard_labels,
                "soft_labels": [],
                "host_person_hits": 1 if "person" in hard_labels else 0,
                "caption_count": 0,
            },
        },
        artifact_ids=["art-1"],
    )


def test_snapshot_ignores_object_count_drift() -> None:
    a = snapshot_from_window(_window(hard_labels=["person"], object_counts={"person": 1}))
    b = snapshot_from_window(_window(hard_labels=["person"], object_counts={"person": 9}))
    assert a == b


def test_tracker_first_window_interprets() -> None:
    tracker = EvidenceTransitionTracker()
    snap = snapshot_from_window(_window(hard_labels=["person"]))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=snap,
        now=100.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is True
    assert decision.reason == "first_window"


def test_tracker_skips_stable_labels_despite_count_drift() -> None:
    tracker = EvidenceTransitionTracker()
    snap = snapshot_from_window(_window(hard_labels=["person"], object_counts={"person": 1}))
    tracker.record_interpretation(stream_key="cam0", snapshot=snap, now=100.0)
    drift = snapshot_from_window(_window(hard_labels=["person"], object_counts={"person": 8}))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=drift,
        now=110.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is False
    assert decision.reason == "stable_scene"


def test_tracker_person_entered() -> None:
    tracker = EvidenceTransitionTracker()
    empty = snapshot_from_window(_window(hard_labels=["door"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=empty, now=100.0)
    entered = snapshot_from_window(_window(hard_labels=["door", "person"]))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=entered,
        now=110.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is True
    assert decision.reason == "person_entered"


def test_tracker_person_exited() -> None:
    tracker = EvidenceTransitionTracker()
    present = snapshot_from_window(_window(hard_labels=["person"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=present, now=100.0)
    gone = snapshot_from_window(_window(hard_labels=["door"]))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=gone,
        now=110.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is True
    assert decision.reason == "person_exited"


def test_tracker_refresh_ttl() -> None:
    tracker = EvidenceTransitionTracker()
    snap = snapshot_from_window(_window(hard_labels=["person"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=snap, now=100.0)
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=snap,
        now=221.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is True
    assert decision.reason == "refresh_ttl"


def test_tracker_labels_changed() -> None:
    tracker = EvidenceTransitionTracker()
    door = snapshot_from_window(_window(hard_labels=["door"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=door, now=100.0)
    chair = snapshot_from_window(_window(hard_labels=["chair"]))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=chair,
        now=110.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is True
    assert decision.reason == "salient_labels_changed"


def test_tracker_skips_when_interpret_in_flight() -> None:
    tracker = EvidenceTransitionTracker()
    snap = snapshot_from_window(_window(hard_labels=["person"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=snap, now=100.0)
    tracker.begin_interpretation(stream_key="cam0")
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=snap,
        now=110.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is False
    assert decision.reason == "interpret_in_flight"


def test_abort_interpretation_clears_first_window_placeholder() -> None:
    tracker = EvidenceTransitionTracker()
    tracker.begin_interpretation(stream_key="cam0")
    tracker.abort_interpretation(stream_key="cam0")
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=snapshot_from_window(_window()),
        now=100.0,
        max_refresh_sec=120.0,
    )
    assert decision.interpret is True
    assert decision.reason == "first_window"


@pytest.mark.asyncio
async def test_generate_interpretation_skips_on_stable_scene() -> None:
    svc = CouncilService()
    snap = snapshot_from_window(_window(hard_labels=["person"], object_counts={"person": 1}))
    now = 1_000_000.0
    svc._evidence_transition.record_interpretation(stream_key="cam0", snapshot=snap, now=now)
    drift_window = _window(hard_labels=["person"], object_counts={"person": 12})

    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock) as mock_llm, patch(
        "app.main.time.time", return_value=now + 10.0
    ):
        interpretation, outcome = await svc._generate_interpretation(drift_window, source_env=None)  # type: ignore[arg-type]

    assert interpretation is None
    assert outcome.parse_mode == "stable_scene"
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_generate_interpretation_calls_llm_on_person_entered() -> None:
    svc = CouncilService()
    empty = snapshot_from_window(_window(hard_labels=[]))
    svc._evidence_transition.record_interpretation(stream_key="cam0", snapshot=empty, now=1.0)
    entered = _window(hard_labels=["person"])

    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock, return_value=None) as mock_llm:
        await svc._generate_interpretation(entered, source_env=None)  # type: ignore[arg-type]

    mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_generate_interpretation_retries_after_parse_failure() -> None:
    svc = CouncilService()
    empty = snapshot_from_window(_window(hard_labels=[]))
    svc._evidence_transition.record_interpretation(stream_key="cam0", snapshot=empty, now=1.0)
    entered = _window(hard_labels=["person"])

    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock, return_value=None) as mock_llm:
        await svc._generate_interpretation(entered, source_env=None)  # type: ignore[arg-type]
        await svc._generate_interpretation(entered, source_env=None)  # type: ignore[arg-type]

    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_generate_interpretation_rpc_bypasses_transition_gate() -> None:
    svc = CouncilService()
    snap = snapshot_from_window(_window(hard_labels=["person"], object_counts={"person": 1}))
    now = 1_000_000.0
    svc._evidence_transition.record_interpretation(stream_key="cam0", snapshot=snap, now=now)
    drift_window = _window(hard_labels=["person"], object_counts={"person": 12})

    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock, return_value=None) as mock_llm, patch(
        "app.main.time.time", return_value=now + 10.0
    ):
        await svc._generate_interpretation(
            drift_window, source_env=None, allow_transition_gate=False  # type: ignore[arg-type]
        )

    mock_llm.assert_called_once()
