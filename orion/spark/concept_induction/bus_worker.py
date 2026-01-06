from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.schemas.vector.schemas import VectorWriteRequest

from .inducer import ConceptInducer, WindowEvent
from .settings import ConceptSettings
from .store import LocalProfileStore
from .summarizer import Summarizer

logger = logging.getLogger("orion.spark.concept.worker")


def _service_ref(cfg: ConceptSettings) -> ServiceRef:
    return ServiceRef(
        name=cfg.service_name,
        version=cfg.service_version,
        node=cfg.node_name,
    )


class ConceptWorker:
    """Manages windowed intake and triggers induction."""

    def __init__(self, cfg: ConceptSettings) -> None:
        self.cfg = cfg
        self.bus = OrionBusAsync(cfg.orion_bus_url, enabled=cfg.orion_bus_enabled)
        self.store = LocalProfileStore(cfg.store_path)
        self.last_run: Dict[str, datetime] = {}
        service_ref = _service_ref(cfg)
        self.summarizer = Summarizer(
            use_cortex=cfg.use_cortex_orch,
            verb_name=cfg.cortex_orch_verb,
            bus=self.bus,
            request_channel=cfg.cortex_request_channel,
            result_prefix=cfg.cortex_result_prefix,
            timeout_sec=cfg.cortex_timeout_sec,
            service_ref=service_ref,
        )
        self.inducer = ConceptInducer(
            cfg,
            summarizer=self.summarizer if cfg.use_cortex_orch else None,
            store_loader=self.store.load,
            store_saver=self.store.save,
            service_ref=service_ref,
        )
        self.window: Dict[str, List[WindowEvent]] = {s: [] for s in cfg.subjects}

    def _prune_window(self, subject: str) -> None:
        window = self.window.get(subject, [])
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.cfg.window_max_minutes)
        self.window[subject] = [
            w for w in window if w.timestamp >= cutoff
        ][-self.cfg.window_max_events :]

    def _detect_subject(self, env: BaseEnvelope) -> str:
        payload = env.payload if isinstance(env.payload, dict) else {}
        user = payload.get("user") or payload.get("speaker")
        role = payload.get("role")
        if isinstance(user, str) and user.lower().startswith("juniper"):
            return "juniper"
        if role == "assistant" or (env.source and "orion" in (env.source.name or "")):
            return "orion"
        return "relationship"

    def _extract_text(self, env: BaseEnvelope) -> Optional[str]:
        payload = env.payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        if not isinstance(payload, dict):
            if isinstance(payload, str):
                return payload
            return None
        for key in ("content", "text", "message", "summary", "prompt"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return None

    async def _publish_profile(self, profile: ConceptProfile, corr_id) -> None:
        env = BaseEnvelope(
            kind="memory.concepts.profile.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=profile.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.profile_channel, env)

    async def _publish_delta(self, delta: ConceptProfileDelta, corr_id) -> None:
        env = BaseEnvelope(
            kind="memory.concepts.delta.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=delta.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.delta_channel, env)

    async def _forward_vector(self, profile: ConceptProfile, corr_id) -> None:
        for concept in profile.concepts:
            req = VectorWriteRequest(
                id=concept.concept_id,
                kind="memory.concept",
                content=concept.label,
                metadata={
                    "subject": profile.subject,
                    "revision": profile.revision,
                },
                vector=None,
            )
            env = BaseEnvelope(
                kind="vector.write",
                source=_service_ref(self.cfg),
                correlation_id=corr_id,
                payload=req.model_dump(mode="json"),
            )
            await self.bus.publish(self.cfg.forward_vector_channel, env)

    async def handle_envelope(self, env: BaseEnvelope) -> None:
        text = self._extract_text(env)
        if not text:
            return
        subject = self._detect_subject(env)
        event = WindowEvent(text=text, timestamp=env.created_at, envelope=env)
        self.window.setdefault(subject, []).append(event)
        self._prune_window(subject)
        now = datetime.now(timezone.utc)
        last = self.last_run.get(subject)
        if (
            last is None
            or len(self.window[subject]) >= self.cfg.window_max_events
            or (last and (now - last) > timedelta(minutes=self.cfg.window_max_minutes))
        ):
            await self.run_for_subject(subject, corr_id=env.correlation_id)

    async def run_for_subject(self, subject: str, corr_id=None) -> None:
        corr_id = corr_id or uuid4()
        window = list(self.window.get(subject, []))
        if not window:
            return
        result = await self.inducer.run(subject=subject, window=window)
        await self._publish_profile(result.profile, corr_id)
        if result.delta:
            await self._publish_delta(result.delta, corr_id)
        await self._forward_vector(result.profile, corr_id)
        self.last_run[subject] = datetime.now(timezone.utc)

    async def start(self) -> None:
        await self.bus.connect()
        patterns = self.cfg.intake_channels
        hunter = Hunter(
            ChassisConfig(
                service_name=self.cfg.service_name,
                service_version=self.cfg.service_version,
                node_name=self.cfg.node_name,
                bus_url=self.cfg.orion_bus_url,
                bus_enabled=self.cfg.orion_bus_enabled,
                heartbeat_interval_sec=self.cfg.heartbeat_interval_sec,
            ),
            patterns=patterns,
            handler=self.handle_envelope,
        )
        await hunter.start_background()
        logger.info("Concept worker started listening on %s", patterns)
        # Keep the loop alive
        while True:
            await asyncio.sleep(1.0)
