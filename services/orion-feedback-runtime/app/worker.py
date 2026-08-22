from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.feedback.builder import build_feedback_frame
from orion.feedback.outcome_resolution import (
    resolve_action_outcomes,
    summarize_control_observations,
)
from orion.feedback.policy import load_feedback_policy
from orion.schemas.feedback_frame import FeedbackFrameV1

from app.settings import get_settings
from app.store import FeedbackRuntimeStore

logger = logging.getLogger("orion.feedback.runtime")


class FeedbackRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = FeedbackRuntimeStore(
            self._settings.postgres_uri,
            reconcile_interval_sec=self._settings.feedback_reconcile_interval_sec,
        )
        self._policy = load_feedback_policy(Path(self._settings.feedback_policy_path))
        self._stop = asyncio.Event()
        self._bus = OrionBusAsync(
            self._settings.bus_url,
            enabled=self._settings.bus_enabled,
        )

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self._settings.service_name,
            version=self._settings.service_version,
            node=self._settings.node_name,
        )

    async def start(self) -> None:
        await self._bus.connect()
        asyncio.create_task(self._poll_loop(), name="feedback-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()
        await self._bus.close()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = await asyncio.to_thread(self._tick)
                if frame is not None:
                    await self._publish_feedback_frame(frame)
            except Exception:
                logger.exception("feedback_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.feedback_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _publish_feedback_frame(self, frame: FeedbackFrameV1) -> None:
        env = BaseEnvelope(
            kind="feedback.frame.v1",
            source=self._service_ref(),
            payload=frame.model_dump(mode="json"),
        )
        await self._bus.publish(self._settings.feedback_bus_channel, env)
        logger.info(
            "feedback_frame_published frame_id=%s channel=%s",
            frame.frame_id,
            self._settings.feedback_bus_channel,
        )

    def _tick(self) -> Optional[FeedbackFrameV1]:
        if not self._settings.enable_feedback_runtime:
            return None

        # Rate-limited internally (default once per 15 min); safe to call every tick.
        self._store.reconcile_feedback_pending()

        dispatch = self._store.load_latest_dispatch_frame_without_feedback()
        if dispatch is None:
            return None
        if self._store.load_feedback_frame_for_dispatch(dispatch.frame_id) is not None:
            # Marker was stale-true (e.g. a pre-migration row, or a manual write). Clear it, or
            # this guard returns early every tick on the same row and the FIFO never advances.
            self._store.clear_feedback_pending(dispatch.frame_id)
            return None

        policy_frame = self._store.load_policy_frame(dispatch.source_policy_frame_id)
        proposal_frame = self._store.load_proposal_frame(dispatch.source_proposal_frame_id)
        # 2026-07-22 (SelfStateV1 burn): field_before is the exact field
        # tick dispatch was built against; field_after is the next real
        # field tick observed within the policy's window, same "did the
        # world move" comparison self-state used to provide a lossy hop for.
        field_before = self._store.load_field_for_tick(dispatch.source_field_tick_id)
        field_after = self._store.load_latest_field_after(
            dispatch.generated_at,
            window_sec=self._policy.windows.field_after_window_sec,
        )
        cortex_results = self._store.load_cortex_result_evidence(dispatch)

        frame = build_feedback_frame(
            dispatch_frame=dispatch,
            policy_frame=policy_frame,
            proposal_frame=proposal_frame,
            field_before=field_before,
            field_after=field_after,
            cortex_results=cortex_results or None,
            policy=self._policy,
        )
        # Score whatever predictions this tick's dispatched actions made.
        # Never allowed to take the feedback frame down with it: a scoring
        # bug must not stall the pipeline that has been running for months.
        # An empty resolution is a legitimate outcome (a tick can dispatch
        # nothing, or dispatch only actions that declare no signal).
        try:
            # NOT the feedback frame's own field_before/field_after. Those
            # are [t-205s, t+1.1s] and the action has not returned at the
            # closing edge -- see store.load_action_scoring_window for the
            # measurements and why that made every contrast an unbiased
            # estimate of a null quantity.
            score_before, score_after = self._store.load_action_scoring_window(
                dispatch.generated_at,
                settle_sec=self._settings.action_settle_sec,
            )
            resolution = resolve_action_outcomes(
                dispatch_frame=dispatch,
                feedback_frame_id=frame.frame_id,
                field_before=score_before,
                field_after=score_after,
                priors=self._store.load_effect_posteriors(),
                control_priors=self._store.load_control_posteriors(),
                latency_by_dispatch_id=_latencies(cortex_results),
            )
        except Exception:
            logger.exception(
                "action_outcome_resolution_failed dispatch_frame_id=%s", dispatch.frame_id
            )
            resolution = None

        self._store.save_feedback_frame(
            frame,
            outcome_records=resolution.records if resolution else None,
            control_cells=resolution.control_posteriors if resolution else None,
            control_frame_id=dispatch.frame_id,
        )
        logger.info(
            "feedback_frame_saved frame_id=%s dispatch_frame_id=%s outcome_status=%s "
            "observations=%d scored_actions=%d skipped_actions=%d untreated=%s",
            frame.frame_id,
            dispatch.frame_id,
            frame.outcome_status,
            len(frame.observations),
            len(resolution.records) if resolution else 0,
            len(resolution.skipped) if resolution else 0,
            summarize_control_observations(
                resolution.control_observations if resolution else []
            ),
        )
        if resolution and resolution.skipped:
            # Aggregated, not per-dispatch: this fires every tick and the
            # per-reason counts are the part anyone acts on.
            counts: dict[str, int] = {}
            for reason in resolution.skipped.values():
                counts[reason] = counts.get(reason, 0) + 1
            logger.info(
                "action_outcome_skipped dispatch_frame_id=%s reasons=%s",
                dispatch.frame_id,
                sorted(counts.items()),
            )
        return frame


def _latencies(cortex_results: list[dict[str, object]] | None) -> dict[str, float]:
    """Real measured cost per dispatch, when the cortex result reported one.

    Absent keys stay absent -- never coerced to 0.0, which would read as
    "this action was free" and quietly bias any cost-weighted comparison
    built on top of this ledger toward whichever executor happens not to
    report timings.
    """
    out: dict[str, float] = {}
    for raw in cortex_results or []:
        dispatch_id = str(raw.get("dispatch_id") or "")
        if not dispatch_id:
            continue
        for key in ("latency_ms", "duration_ms", "elapsed_ms"):
            value = raw.get(key)
            if value is None:
                continue
            try:
                out[dispatch_id] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            break
    return out
