from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.worker import (
    _CANDIDATE_SPARK_META,
    _merge_spark_meta_dict,
    _novelty_from_spark_meta,
    handle_spark_meta_patch,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


def test_novelty_from_appraisal_top_level_fallback() -> None:
    meta = {
        "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.78},
        "novelty": 0.78,
    }
    assert _novelty_from_spark_meta(meta) == pytest.approx(0.78)


def test_novelty_missing_when_degraded() -> None:
    meta = {
        "turn_change_appraisal": {"turn_change_status": "degraded"},
        "novelty": 0.0,
    }
    assert _novelty_from_spark_meta(meta) is None


def test_merge_spark_meta_dict_nested_appraisal() -> None:
    base = {"turn_change_appraisal": {"turn_change_status": "degraded"}}
    patch = {
        "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.91},
        "novelty": 0.91,
    }
    merged = _merge_spark_meta_dict(base, patch)
    assert merged["turn_change_appraisal"]["turn_change_status"] == "ok"
    assert merged["novelty"] == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_handle_spark_meta_patch_broadcasts_tissue_update() -> None:
    corr = str(uuid4())
    _CANDIDATE_SPARK_META.pop(corr, None)
    env = BaseEnvelope(
        kind="chat.history.spark_meta.patch.v1",
        correlation_id=corr,
        source=ServiceRef(name="memory-consolidation", node="athena"),
        payload={
            "correlation_id": corr,
            "spark_meta": {
                "turn_change_appraisal": {
                    "turn_change_status": "ok",
                    "novelty_score": 0.83,
                },
                "novelty": 0.83,
            },
        },
    )
    broadcast = AsyncMock()
    with patch("app.worker._broadcast_tissue_update", broadcast):
        await handle_spark_meta_patch(env)
    broadcast.assert_awaited_once()
    kwargs = broadcast.await_args.kwargs
    assert kwargs["correlation_id"] == corr
    assert kwargs["spark_meta"]["novelty"] == pytest.approx(0.83)
    _CANDIDATE_SPARK_META.pop(corr, None)
