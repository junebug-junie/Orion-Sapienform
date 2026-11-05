from __future__ import annotations

import asyncio
import logging
import json
from datetime import datetime, timezone
from urllib.request import urlopen
from orion.reverie.baseline import load_baseline_policy
from orion.schemas.reverie_visual import VisualActivityV1
from pathlib import Path


from orion.regulation.arousal_read import read_arousal_sync
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

    def _baseline_arousal_level(self) -> str | None:
        """R3 reader 4: Orion's arousal at the moment a baseline image would be issued.

        Called by ``orion.reverie.baseline.schedule`` only after every other baseline check
        passed (about 16 times a day, plus retries), so this is one Redis GET per due image,
        not per tick. None (unknown, flag off, Redis down) changes nothing."""
        view = read_arousal_sync(
            self._settings.orion_bus_url,
            enabled=bool(self._settings.visual_baseline_arousal_enabled),
            reader="visual_baseline",
            max_age_sec=float(self._settings.regulation_reader_max_age_sec),
        )
        if view.level == "strained":
            logger.info("visual_baseline_deferred reason=arousal_strained since=%s reasons=%s",
                        view.since, ",".join(view.reasons))
        return view.level if view.known else None

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

        # Flag-gated non-deterministic producers. Every candidate here carries
        # an operator_review gate, so even when on they cannot auto-dispatch.
        # (The metacog "hop 0" producer was removed 2026-10-16: 36 candidates in
        # its whole life, none after 2026-10-08.)
        external_candidates: list = []

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

        eligibility = None
        baseline_reason = None
        baseline_policy = load_baseline_policy()
        if baseline_policy.enabled:
            try:
                with urlopen(baseline_policy.thought_url.rstrip("/") + "/visual-chain/activity", timeout=baseline_policy.timeout_sec) as response:
                    activity = VisualActivityV1.model_validate(json.load(response))
                eligibility, baseline_reason = self._store.baseline_eligibility(activity,
                    now=datetime.now(timezone.utc), policy=baseline_policy,
                    arousal_level=self._baseline_arousal_level)
            except Exception:
                baseline_reason = "visual_activity_unavailable"
                logger.warning("visual_baseline_activity_unavailable", exc_info=True)

        frame = build_proposal_frame(
            field=field,
            attention=attention,
            policy=self._policy,
            previous_frame=previous,
            external_candidates=external_candidates or None,
            baseline_eligibility=eligibility,
        )
        if baseline_reason:
            frame.warnings.append(baseline_reason)
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
