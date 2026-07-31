from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.proposals.builder import build_proposal_frame
from orion.proposals.templates import FORBIDDEN_TRANSPORT_PROPOSAL_KEYS, TRANSPORT_PROPOSAL_TEMPLATE_KEYS
from orion.proposals.policy import load_proposal_policy

from app.settings import get_settings
from app.store import ProposalRuntimeStore

logger = logging.getLogger("orion.proposal.runtime")


class ProposalRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = ProposalRuntimeStore(self._settings.postgres_uri)
        self._policy = load_proposal_policy(Path(self._settings.proposal_policy_path))
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="proposal-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("proposal_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.proposal_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        # 2026-07-22 (SelfStateV1 burn): polls FieldStateV1 directly instead of
        # SelfStateV1 -- field was always the real upstream tick, self_state was
        # a lossy pass-through hop for this consumer's purposes.
        if not self._settings.enable_proposal_runtime:
            return

        field = self._store.load_latest_field()
        if field is None:
            return

        if self._store.load_proposal_frame_for_field_tick(field.tick_id) is not None:
            return

        attention = self._store.load_attention_frame_for_field_tick(field.tick_id)

        previous = self._store.load_latest_proposal_frame()

        # Flag-gated non-deterministic producers. Default-off → empty → zero
        # change. Every candidate here carries an operator_review gate, so even
        # when on they cannot auto-dispatch.
        external_candidates: list = []

        # Hop 0 of the stream-of-consciousness chain: a sustained metacog trend
        # competing as a first-class arena candidate. The design doc scopes hop 0
        # as a reducer "registered as a candidate producer the same way reverie
        # is -- not built as a bespoke standalone script"; this is that
        # registration. Stateless by design: the reducer replays the (small)
        # appraisal history each tick rather than checkpointing, because hop 0
        # opens a chain and has nothing to resume. Checkpointed state becomes
        # load-bearing at hop 1, where a real multi-hop chain can be preempted
        # mid-flight -- building it here would be machinery with no chain to
        # carry.
        if getattr(self._settings, "metacog_hop_propose_enabled", False):
            try:
                from orion.metacog.proposal import trend_result_to_candidate
                from orion.metacog.trend_reducer import replay

                readings = self._store.load_repair_pressure_readings()
                if readings:
                    # replay() returns every tick's result; the newest one is
                    # this tick's classification.
                    results = replay(readings)
                    hop = trend_result_to_candidate(
                        results[-1] if results else None,
                        series_name="repair_pressure",
                        observed_at=readings[-1].at,
                        fallback_target_id=field.tick_id,
                    )
                    if hop is not None:
                        external_candidates.append(hop)
            except Exception:
                # A metacog read must never break proposal generation -- same
                # fail-open discipline as the reverie branch below.
                logger.exception("metacog_hop_candidate_failed")

        reverie_candidates = None
        if getattr(self._settings, "reverie_propose_enabled", False):
            thought = self._store.load_recent_reverie_thought()
            if thought is not None:
                from orion.reverie.proposal import spontaneous_thought_to_candidate

                candidate = spontaneous_thought_to_candidate(
                    thought,
                    fallback_target_id=field.tick_id,
                    autoaction_enabled=getattr(
                        self._settings, "reverie_autoaction_enabled", False
                    ),
                )
                if candidate is not None:
                    reverie_candidates = [candidate]
        if reverie_candidates:
            external_candidates.extend(reverie_candidates)

        frame = build_proposal_frame(
            field=field,
            attention=attention,
            policy=self._policy,
            previous_frame=previous,
            external_candidates=external_candidates or None,
        )
        if not self._settings.enable_transport_proposals:
            filtered = [
                c
                for c in frame.candidates
                if not any(key in c.proposal_id for key in TRANSPORT_PROPOSAL_TEMPLATE_KEYS)
            ]
            frame = frame.model_copy(update={"candidates": filtered})
        elif self._settings.transport_proposal_mode == "read_only":
            filtered = [
                c
                for c in frame.candidates
                if not any(key in c.proposal_id for key in FORBIDDEN_TRANSPORT_PROPOSAL_KEYS)
            ]
            frame = frame.model_copy(update={"candidates": filtered})
        self._store.save_proposal_frame(frame)
        logger.info(
            "proposal_frame_saved frame_id=%s field_tick_id=%s candidates=%d",
            frame.frame_id,
            field.tick_id,
            len(frame.candidates),
        )
