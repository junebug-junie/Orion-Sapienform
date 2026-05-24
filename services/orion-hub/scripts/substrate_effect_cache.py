"""In-memory LRU cache of per-turn substrate effect snapshots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.signals.models import OrionSignalV1
from orion.substrate.appraisal.models import (
    RepairEvidenceV1,
    RepairPressureAppraisalV1,
)


@dataclass
class SubstrateEffectSnapshot:
    turn_id: str
    message_id: str | None
    user_text: str
    appraisal: RepairPressureAppraisalV1 | None
    signal: OrionSignalV1 | None
    evidence: list[RepairEvidenceV1] = field(default_factory=list)
    contract_before: dict[str, Any] = field(default_factory=dict)
    contract_after: dict[str, Any] = field(default_factory=dict)
    causal_molecule_ids: list[str] = field(default_factory=list)
    stored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SubstrateEffectCache:
    def __init__(self, *, max_entries: int = 256) -> None:
        self._max = max(1, int(max_entries))
        self._entries: "OrderedDict[str, SubstrateEffectSnapshot]" = OrderedDict()

    def store(self, snapshot: SubstrateEffectSnapshot) -> None:
        if snapshot.turn_id in self._entries:
            self._entries.move_to_end(snapshot.turn_id)
        self._entries[snapshot.turn_id] = snapshot
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def get(self, turn_id: str) -> SubstrateEffectSnapshot | None:
        return self._entries.get(turn_id)

    def recent(self, *, limit: int = 25) -> list[SubstrateEffectSnapshot]:
        items = list(self._entries.values())
        items.reverse()  # newest first
        return items[: max(0, int(limit))]


# Module-level singleton used by the Hub.
substrate_effect_cache = SubstrateEffectCache(max_entries=256)
