from __future__ import annotations

import time
from typing import Optional
from uuid import uuid4

from loguru import logger

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.pad import PadEventV1, PadLinks

from ..observability.stats import PadStatsTracker
from ..reducers.registry import ReducerRegistry
from ..scoring.default import DefaultScorer


class NormalizationPipeline:
    def __init__(self, *, app_name: str, min_salience: float, pulse_salience: float, stats: PadStatsTracker):
        self.app_name = app_name
        self.min_salience = min_salience
        self.pulse_salience = pulse_salience
        self.stats = stats
        self.registry = ReducerRegistry()
        self.scorer = DefaultScorer()

    async def reduce_and_score(self, env: BaseEnvelope, channel: str) -> Optional[PadEventV1]:
        reducer = self.registry.get(env.kind)
        event = await reducer(env, channel=channel)
        if event is None:
            self.stats.increment_dropped(reason="reducer_none")
            return None

        scored = self.scorer.score(event)
        if scored.salience < self.min_salience:
            self.stats.increment_dropped(reason="below_min_salience")
            return None

        enriched = event.model_copy(update={"salience": scored.salience, "novelty": scored.novelty, "confidence": scored.confidence})
        self.stats.record_salient(enriched.salience)
        return enriched
