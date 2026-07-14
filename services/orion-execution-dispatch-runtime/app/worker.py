from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from orion.autonomy.models import ActionOutcomeEmitV1
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.execution_dispatch.builder import (
    build_execution_dispatch_frame,
    build_unevaluable_execution_dispatch_frame,
)
from orion.execution_dispatch.cortex_client import ExecutionDispatchCortexClient
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.execution_dispatch.result_extraction import (
    extract_final_text,
    parse_structured_observation,
)
from orion.notify.client import NotifyClient
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.notify import NotificationRequest

from app.settings import get_settings
from app.store import ExecutionDispatchRuntimeStore

logger = logging.getLogger("orion.execution_dispatch.runtime")

THEATER_TRIPWIRE_WINDOW = 10
THEATER_TRIPWIRE_EMPTY_THRESHOLD = 0.5
# ActionOutcomeEmitV1.summary lands in chat-visible evidence via
# chat_stance.py's _project_recent_dispatch_actions -- bounded here at the
# producer so nothing downstream needs to re-truncate raw model output.
ACTION_OUTCOME_SUMMARY_MAX_CHARS = 280
# Established self-subject convention across orion/autonomy/* (reducer.py,
# substrate_metabolism.py, signal_tension.py all default/use this).
ACTION_OUTCOME_SUBJECT = "orion"


class ExecutionDispatchRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = ExecutionDispatchRuntimeStore(self._settings.postgres_uri)
        self._policy = load_execution_dispatch_policy(
            Path(self._settings.execution_dispatch_policy_path)
        )
        self._notify = NotifyClient(
            base_url=self._settings.notify_url,
            api_token=self._settings.notify_api_token,
            timeout=10,
        )
        self._stop = asyncio.Event()
        # In-memory only, by design: once tripped, stays tripped until this
        # process restarts (mirrors the "re-arm is manual" rule in the
        # parent spec -- a self-clearing tripwire could silently resume
        # sending on a coincidentally-good sample).
        self.theater_tripwire_active = False

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="execution-dispatch-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("execution_dispatch_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.execution_dispatch_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_execution_dispatch_runtime:
            return

        policy_frame = self._store.load_latest_policy_frame_without_dispatch()
        if policy_frame is None:
            return

        proposal = self._store.load_proposal_frame(policy_frame.source_proposal_frame_id)
        if proposal is None:
            # A naive `return` here would retry this exact same policy frame
            # forever -- it's always "the oldest undispatched" until a
            # dispatch frame exists for it, permanently blocking every policy
            # frame queued behind it. Record an honest "could not evaluate"
            # frame instead so the FIFO queue advances.
            logger.warning(
                "execution_dispatch_proposal_unavailable proposal_frame_id=%s",
                policy_frame.source_proposal_frame_id,
            )
            frame = build_unevaluable_execution_dispatch_frame(
                policy_frame=policy_frame,
                policy_id=self._policy.policy_id,
                reason=f"proposal_frame {policy_frame.source_proposal_frame_id} unavailable",
            )
            self._store.save_dispatch_frame(frame)
            logger.info(
                "execution_dispatch_frame_saved_unevaluable frame_id=%s policy_frame_id=%s",
                frame.frame_id,
                policy_frame.frame_id,
            )
            return

        self_state = self._store.load_self_state(policy_frame.source_self_state_id)
        if self_state is None:
            # Same reasoning as the missing-proposal branch above (2026-07-12
            # live incident: a schema change made an old self-state row
            # permanently unloadable, stalling the whole queue for it).
            logger.warning(
                "execution_dispatch_self_state_unavailable self_state_id=%s",
                policy_frame.source_self_state_id,
            )
            frame = build_unevaluable_execution_dispatch_frame(
                policy_frame=policy_frame,
                policy_id=self._policy.policy_id,
                reason=f"self_state {policy_frame.source_self_state_id} unavailable or schema-incompatible",
            )
            self._store.save_dispatch_frame(frame)
            logger.info(
                "execution_dispatch_frame_saved_unevaluable frame_id=%s policy_frame_id=%s",
                frame.frame_id,
                policy_frame.frame_id,
            )
            return

        frame = build_execution_dispatch_frame(
            policy_frame=policy_frame,
            proposal_frame=proposal,
            self_state=self_state,
            policy=self._policy,
            override_dispatch_mode=self._settings.execution_dispatch_mode,
        )

        if (
            self._settings.execution_dispatch_mode == "dispatch_read_only"
            and self._policy.mode.allow_dispatch_read_only
        ):
            # _tick() runs inside asyncio.to_thread -- this thread has no
            # running event loop of its own, so asyncio.run() here is safe
            # and creates one just for this tick's bus RPCs.
            frame = asyncio.run(self._send_prepared_candidates(frame))

        self._store.save_dispatch_frame(frame)
        logger.info(
            "execution_dispatch_frame_saved frame_id=%s policy_frame_id=%s candidates=%d blocked=%d dispatched=%d",
            frame.frame_id,
            policy_frame.frame_id,
            len(frame.candidates),
            frame.blocked_count,
            frame.dispatch_count,
        )

    async def _send_prepared_candidates(
        self, frame: ExecutionDispatchFrameV1
    ) -> ExecutionDispatchFrameV1:
        if self._check_theater_tripwire():
            return frame  # tripped (this tick or a prior one) -- send nothing

        remaining_daily_budget = (
            self._settings.orion_dispatch_max_per_day - self._store.count_dispatches_today()
        )
        if remaining_daily_budget <= 0:
            logger.info(
                "execution_dispatch_daily_cap_reached max_per_day=%d",
                self._settings.orion_dispatch_max_per_day,
            )
            return frame

        budget = max(0, min(remaining_daily_budget, self._policy.limits.max_dispatches_per_tick))
        if budget <= 0:
            return frame

        to_send = [c for c in frame.candidates if c.dispatch_status == "prepared_for_dispatch"][
            :budget
        ]
        if not to_send:
            return frame

        sent_ids = {c.dispatch_id for c in to_send}
        remaining_candidates = [c for c in frame.candidates if c.dispatch_id not in sent_ids]

        bus = OrionBusAsync(
            url=self._settings.orion_bus_url, enabled=self._settings.orion_bus_enabled
        )
        await bus.connect()
        client = ExecutionDispatchCortexClient(
            bus,
            request_channel=self._settings.cortex_exec_channel,
            result_prefix=self._settings.cortex_exec_result_prefix,
            timeout_sec=self._settings.execution_dispatch_rpc_timeout_sec,
        )
        newly_dispatched: list[ExecutionDispatchCandidateV1] = []
        try:
            for candidate in to_send:
                newly_dispatched.append(await self._send_one(client, bus, frame, candidate))
        finally:
            await bus.close()

        dispatched_candidates = list(frame.dispatched_candidates) + newly_dispatched
        updated = frame.model_copy(
            update={
                "candidates": remaining_candidates,
                "dispatched_candidates": dispatched_candidates,
                "dispatch_count": len(dispatched_candidates),
                "dispatch_attempted": True,
            }
        )

        # Re-check after this tick's real sends landed new rows -- lets the
        # tripwire fire the same tick it actually crosses the threshold,
        # not one tick late.
        self._check_theater_tripwire()
        return updated

    def _check_theater_tripwire(self) -> bool:
        recent = self._store.recent_dispatch_result_statuses(THEATER_TRIPWIRE_WINDOW)
        if len(recent) < THEATER_TRIPWIRE_WINDOW:
            return self.theater_tripwire_active
        empty_count = sum(1 for s in recent if s == "empty")
        newly_tripped = (
            not self.theater_tripwire_active
            and empty_count > THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_EMPTY_THRESHOLD
        )
        if newly_tripped:
            self.theater_tripwire_active = True
            logger.warning(
                "execution_dispatch_theater_tripwire_active empty=%d window=%d",
                empty_count,
                len(recent),
            )
            self._notify_tripwire(empty_count, len(recent))
        return self.theater_tripwire_active

    async def _send_one(
        self,
        client: ExecutionDispatchCortexClient,
        bus: OrionBusAsync,
        frame: ExecutionDispatchFrameV1,
        candidate: ExecutionDispatchCandidateV1,
    ) -> ExecutionDispatchCandidateV1:
        now = datetime.now(timezone.utc)
        result_id = f"result:{candidate.dispatch_id}"

        # Idempotency guard: dispatch_id is deterministic (stable_dispatch_id
        # hashes proposal_id+policy_id), so if this process crashed after a
        # prior successful send but before save_dispatch_frame() recorded
        # that this policy frame is now dispatched, the next tick re-selects
        # the same policy frame and rebuilds the identical dispatch_id. A
        # real cortex-exec RPC must never fire twice for the same candidate
        # -- replay the stored result instead of resending.
        existing = self._store.load_dispatch_result_by_dispatch_id(candidate.dispatch_id)
        if existing is not None:
            logger.info(
                "execution_dispatch_result_replayed dispatch_id=%s status=%s",
                candidate.dispatch_id,
                existing["status"],
            )
            # Re-emit on replay too: action_outcomes.action_id is the SQL
            # primary key and sql-writer's route upserts by merge(), so a
            # repeat emit for the same dispatch_id idempotently overwrites
            # the same row -- it does not duplicate. Skipping the emit here
            # would risk permanently losing it instead: if the process died
            # between save_dispatch_result (above, on a prior attempt) and
            # the emit, or the emit itself failed transiently, no later tick
            # would ever retry it, since every later tick also hits this
            # replay branch.
            if existing["status"] == "failed":
                error = existing["result_json"].get("error", "previous attempt failed")
                await self._emit_action_outcome(
                    bus,
                    candidate=candidate,
                    summary=f"attempted {candidate.dispatch_kind} on {candidate.target_id}, send failed",
                    success=False,
                    observed_at=now,
                )
                return candidate.model_copy(
                    update={
                        "dispatch_status": "dispatched",
                        "dispatched_at": now,
                        "dispatch_error": str(error)[:500],
                    }
                )
            existing_observation = existing["result_json"].get("observation") or ""
            await self._emit_action_outcome(
                bus,
                candidate=candidate,
                summary=(
                    existing_observation
                    if existing_observation
                    else f"{candidate.dispatch_kind} on {candidate.target_id} returned no observation"
                ),
                success=bool(existing_observation),
                observed_at=now,
            )
            return candidate.model_copy(
                update={
                    "dispatch_status": "dispatched",
                    "dispatched_at": now,
                    "result_ref": existing["result_id"],
                }
            )

        context = dict(candidate.request_envelope.get("context") or {})

        try:
            payload = await client.dispatch(
                verb=candidate.cortex_verb or "",
                mode=candidate.cortex_mode or "brain",
                context=context,
                dispatch_id=candidate.dispatch_id,
            )
        except Exception as exc:
            logger.warning(
                "execution_dispatch_send_failed dispatch_id=%s error=%s", candidate.dispatch_id, exc
            )
            self._store.save_dispatch_result(
                result_id=result_id,
                dispatch_id=candidate.dispatch_id,
                frame_id=frame.frame_id,
                status="failed",
                result_json={"error": str(exc)[:2000], "evidence_refs": [result_id]},
                raw_len=0,
            )
            await self._emit_action_outcome(
                bus,
                candidate=candidate,
                summary=f"attempted {candidate.dispatch_kind} on {candidate.target_id}, send failed",
                success=False,
                observed_at=now,
            )
            return candidate.model_copy(
                update={
                    "dispatch_status": "dispatched",
                    "dispatched_at": now,
                    "dispatch_error": str(exc)[:500],
                }
            )

        final_text = extract_final_text(payload)
        observation_data = parse_structured_observation(final_text)
        raw_len = len(observation_data["observation"])
        status = "success" if raw_len > 0 else "empty"
        self._store.save_dispatch_result(
            result_id=result_id,
            dispatch_id=candidate.dispatch_id,
            frame_id=frame.frame_id,
            status=status,
            result_json={**observation_data, "evidence_refs": [result_id]},
            raw_len=raw_len,
        )
        logger.info(
            "execution_dispatch_result dispatch_id=%s status=%s raw_len=%d",
            candidate.dispatch_id,
            status,
            raw_len,
        )
        await self._emit_action_outcome(
            bus,
            candidate=candidate,
            summary=(
                observation_data["observation"]
                if raw_len > 0
                else f"{candidate.dispatch_kind} on {candidate.target_id} returned no observation"
            ),
            success=raw_len > 0,
            observed_at=now,
        )
        return candidate.model_copy(
            update={
                "dispatch_status": "dispatched",
                "dispatched_at": now,
                "result_ref": result_id,
            }
        )

    async def _emit_action_outcome(
        self,
        bus: OrionBusAsync,
        *,
        candidate: ExecutionDispatchCandidateV1,
        summary: str,
        success: bool,
        observed_at: datetime,
    ) -> None:
        """Publish onto the same always-on ActionOutcomeEmitV1 route
        orion-spark-concept-induction already uses for curiosity-fetch
        outcomes -- no new pipe, reusing an existing durable sink
        (sql-writer -> action_outcomes) so load_action_outcomes() sees
        Layer 9's real dispatch results the same way it already sees
        curiosity-fetch ones. Never lets a publish failure raise out of
        the tick -- an unreachable bus must not lose a result that's
        already durably recorded via save_dispatch_result above.
        """
        try:
            emit = ActionOutcomeEmitV1(
                subject=ACTION_OUTCOME_SUBJECT,
                action_id=candidate.dispatch_id,
                kind=candidate.dispatch_kind,
                summary=summary[:ACTION_OUTCOME_SUMMARY_MAX_CHARS],
                success=success,
                surprise=0.0,
                observed_at=observed_at,
            )
            env = BaseEnvelope(
                kind="action.outcome.emit.v1",
                source=ServiceRef(name=self._settings.service_name),
                correlation_id=str(uuid4()),
                payload=emit.model_dump(mode="json"),
            )
            await bus.publish(self._settings.action_outcome_channel, env)
        except Exception:
            logger.warning(
                "execution_dispatch_action_outcome_emit_failed dispatch_id=%s",
                candidate.dispatch_id,
                exc_info=True,
            )

    def _notify_tripwire(self, empty_count: int, window: int) -> None:
        try:
            self._notify.send(
                NotificationRequest(
                    source_service="orion-execution-dispatch-runtime",
                    event_kind="execution_dispatch_theater_tripwire",
                    severity="warning",
                    title="Execution dispatch theater tripwire tripped",
                    body_text=(
                        f"{empty_count}/{window} of the last real dispatches returned empty "
                        "observations. Dispatch sending is paused until this worker restarts."
                    ),
                    tags=["execution_dispatch", "tripwire"],
                )
            )
        except Exception:
            logger.exception("execution_dispatch_tripwire_notify_failed")
