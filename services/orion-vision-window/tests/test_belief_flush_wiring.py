from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject

from app.main import WindowService


def _artifact(label: str, score: float = 0.8) -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id=f"art-{label}",
        correlation_id="c1",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label=label, score=score, box_xyxy=[0, 0, 1, 1])]
        ),
        timing={},
        model_fingerprints={},
    )


@pytest.mark.asyncio
async def test_flush_payload_includes_believed_hard_labels() -> None:
    svc = WindowService()
    now = time.time()
    buffered = [{"artifact": _artifact("door"), "ts": now, "env": None}]

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=buffered,
            correlation_id=None,
            causality_chain=[],
        )

    payload = svc._live_by_stream["cam0"]
    evidence = payload.summary["evidence"]
    assert "door" in evidence["hard_labels"]
    assert "believed_hard_labels" in evidence
    assert evidence["belief"]["schema"] == "scene_belief.v1"


@pytest.mark.asyncio
async def test_belief_survives_single_empty_observation() -> None:
    svc = WindowService()
    now = time.time()

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for _ in range(2):
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _artifact("door"), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _artifact("door", score=0.01), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert evidence["hard_labels"] == []
    assert "door" in evidence["believed_hard_labels"]
