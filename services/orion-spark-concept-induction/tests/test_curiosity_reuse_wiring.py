"""Regression: world-pulse curiosity-followup reuse wiring in bus_worker.

Covers the act branch in `ConceptWorker.handle_envelope` that, on a
`world.pulse.run.result.v1` event, calls `select_reusable_followup` +
`outcome_from_followup` and passes the resulting `prefetched_outcome` into
`maybe_execute_substrate_act_after_metabolism`. When a followup matches a gap
signal AND carries articles, the shared world-pulse fetch is reused and the live
fetch backend must NOT be called again.

This drives a real envelope through the worker and does NOT stub
`maybe_execute_substrate_act_after_metabolism`, `select_reusable_followup`, or
`outcome_from_followup`, so the new bus_worker.py lines actually execute. The
only externals stubbed are the heavy I/O seams (store, bus publish).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings

# The followup's action_id must be exactly what flows through to the emitted
# action outcome (proving the reused outcome, not a fresh fetch, was used).
REUSED_ACTION_ID = "fetch-reused-hardware-compute-gpu-9f3a"


def _world_pulse_envelope_with_followup() -> BaseEnvelope:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    return BaseEnvelope(
        id=uuid4(),
        kind="world.pulse.run.result.v1",
        correlation_id=uuid4(),
        created_at=now,
        source=ServiceRef(name="orion-world-pulse", version="0.1.0", node="athena"),
        payload={
            "run": {
                "run_id": "wp-run-reuse",
                "date": "2026-07-06",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "status": "completed",
                "dry_run": False,
            },
            "digest": {
                "run_id": "wp-run-reuse",
                "date": "2026-07-06",
                "generated_at": now.isoformat(),
                "title": "t",
                "executive_summary": "e",
                "sections": {},
                "items": [],
                "orion_analysis_layer": "deterministic",
                "coverage_status": "sparse",
                # This missing-section rollup drives the metabolism gap signal for
                # "hardware_compute_gpu" that the followup below matches.
                "section_rollups": [
                    {
                        "section": "hardware_compute_gpu",
                        "status": "missing",
                        "article_count": 0,
                        "digest_item_count": 0,
                        "confidence": 0.35,
                    }
                ],
                # Shared fetch already performed by world-pulse: findings carried on
                # the run result so the reactive loop can reuse them.
                "curiosity_followups": [
                    {
                        "section": "hardware_compute_gpu",
                        "driving_gap": "missing",
                        "query": "hardware compute gpu recent news coverage",
                        "articles": [
                            {
                                "url": "https://ex/gpu-1",
                                "title": "New GPU",
                                "description": "d",
                                "salience": 0.7,
                            }
                        ],
                        "action_id": REUSED_ACTION_ID,
                        "correlation_id": "wp-run-reuse",
                    }
                ],
                "created_at": now.isoformat(),
            },
        },
    )


def _drive_state_ready() -> dict:
    return {
        "pressures": {"predictive": 0.7, "coherence": 0.5, "continuity": 0.5, "capability": 0.5, "relational": 0.5, "autonomy": 0.5},
        "activations": {"predictive": True},
    }


@pytest.mark.asyncio
async def test_curiosity_followup_reused_skips_live_fetch(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")

    fetch_calls = {"count": 0}

    async def _fetch_backend(query, *, max_articles):
        fetch_calls["count"] += 1
        return {"success": True, "urls": [], "articles": []}

    cfg = ConceptSettings()
    # Keep the journal RPC out of this test: the reuse path we care about ends at
    # the prefetched-outcome pass-through; journal compose is a separate seam.
    cfg.autonomy_episode_journal_enabled = False
    worker = ConceptWorker(cfg, fetch_backend=_fetch_backend)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = _drive_state_ready()
    worker.drive_engine.update = MagicMock(
        return_value=(_drive_state_ready()["pressures"], {"predictive": True})
    )
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker._publish_action_outcome = AsyncMock(return_value=None)
    worker.goal_engine.propose = MagicMock(
        return_value=MagicMock(proposal=None, suppressed_signature=None)
    )

    await worker.handle_envelope(
        _world_pulse_envelope_with_followup(), "orion:world_pulse:run:result"
    )

    # (a) Live fetch backend is never called: the shared world-pulse fetch was reused.
    assert fetch_calls["count"] == 0

    # (b) The reused outcome's action_id (from the followup) flows through to the
    # emitted action outcome.
    worker._publish_action_outcome.assert_awaited_once()
    emitted = worker._publish_action_outcome.await_args.args[0]
    assert emitted.action_id == REUSED_ACTION_ID
    assert emitted.kind == "web.fetch.readonly"
    assert emitted.success is True
