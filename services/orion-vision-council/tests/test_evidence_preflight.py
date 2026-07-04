import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionWindowPayload

from app.evidence_preflight import (
    EvidenceSkipTracker,
    evidence_fingerprint,
)
from app.main import CouncilService


def _window(
    *,
    window_id: str = "w1",
    stream_id: str = "cam0",
    hard_labels: list[str] | None = None,
    host_person_hits: int = 0,
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
                "host_person_hits": host_person_hits,
                "caption_count": 0,
            },
        },
        artifact_ids=["art-1"],
    )


def test_evidence_fingerprint_stable_for_same_material() -> None:
    a = _window(hard_labels=["person"], object_counts={"person": 2})
    b = _window(window_id="w2", hard_labels=["person"], object_counts={"person": 2})
    assert evidence_fingerprint(a) == evidence_fingerprint(b)


def test_evidence_fingerprint_changes_when_hard_labels_change() -> None:
    a = _window(hard_labels=["person"])
    b = _window(hard_labels=["door"])
    assert evidence_fingerprint(a) != evidence_fingerprint(b)


def test_skip_tracker_first_window_never_skips() -> None:
    tracker = EvidenceSkipTracker()
    decision = tracker.evaluate(
        stream_key="cam0",
        fingerprint="abc",
        now=100.0,
        max_skip_sec=120.0,
    )
    assert decision.skip is False
    assert decision.reason == "first_window"


def test_skip_tracker_skips_unchanged_evidence() -> None:
    tracker = EvidenceSkipTracker()
    tracker.record_llm(stream_key="cam0", fingerprint="abc", now=100.0)
    decision = tracker.evaluate(
        stream_key="cam0",
        fingerprint="abc",
        now=110.0,
        max_skip_sec=120.0,
    )
    assert decision.skip is True
    assert decision.reason == "evidence_unchanged"


def test_skip_tracker_refreshes_after_max_skip_sec() -> None:
    tracker = EvidenceSkipTracker()
    tracker.record_llm(stream_key="cam0", fingerprint="abc", now=100.0)
    decision = tracker.evaluate(
        stream_key="cam0",
        fingerprint="abc",
        now=221.0,
        max_skip_sec=120.0,
    )
    assert decision.skip is False
    assert decision.reason == "max_skip_sec_elapsed"


@pytest.mark.asyncio
async def test_generate_interpretation_skips_llm_when_evidence_unchanged() -> None:
    svc = CouncilService()
    window = _window()
    fp = evidence_fingerprint(window)
    now = 1_000_000.0
    svc._evidence_skip.record_llm(stream_key="cam0", fingerprint=fp, now=now)

    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock) as mock_llm, patch(
        "app.main.time.time", return_value=now + 10.0
    ):
        interpretation, outcome = await svc._generate_interpretation(window, source_env=None)  # type: ignore[arg-type]

    assert interpretation is None
    assert outcome.parse_mode == "evidence_unchanged"
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_generate_interpretation_calls_llm_when_evidence_changes() -> None:
    svc = CouncilService()
    first = _window(hard_labels=["person"])
    svc._evidence_skip.record_llm(
        stream_key="cam0",
        fingerprint=evidence_fingerprint(first),
        now=1.0,
    )
    changed = _window(hard_labels=["door"])

    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock, return_value=None) as mock_llm:
        await svc._generate_interpretation(changed, source_env=None)  # type: ignore[arg-type]

    mock_llm.assert_called_once()
