from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from orion.schemas.collapse_mirror import (
    CollapseMirrorCausalDensity,
    CollapseMirrorEntryV2,
    normalize_collapse_entry,
)

logger = logging.getLogger("orion.collapse.service")


class CollapseMirrorStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self._entries: Dict[str, CollapseMirrorEntryV2] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception as exc:
            logger.warning("Failed to load collapse mirror store: %s", exc)
            return
        if not isinstance(raw, dict):
            return
        for key, payload in raw.items():
            try:
                entry = CollapseMirrorEntryV2.model_validate(payload)
                self._entries[key] = entry
            except Exception:
                continue

    def _persist(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: v.model_dump(mode="json") for k, v in self._entries.items()}
            self.path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("Failed to persist collapse mirror store: %s", exc)

    def save(self, entry: CollapseMirrorEntryV2) -> CollapseMirrorEntryV2:
        with self._lock:
            self._load_if_needed()
            self._entries[entry.event_id] = entry
            self._persist()
        return entry

    def get(self, event_id: str) -> CollapseMirrorEntryV2:
        with self._lock:
            self._load_if_needed()
            if event_id not in self._entries:
                raise KeyError(event_id)
            return self._entries[event_id]


DEFAULT_STORE_PATH = "/mnt/storage/collapse-mirrors/collapse_mirror_store.json"
_store: CollapseMirrorStore | None = None


def _get_store() -> CollapseMirrorStore:
    global _store
    if _store is None:
        path = os.getenv("COLLAPSE_MIRROR_STORE_PATH", DEFAULT_STORE_PATH)
        _store = CollapseMirrorStore(path)
    return _store


def create_entry_from_v2(payload: Any, *, source_service: str | None = None, source_node: str | None = None) -> CollapseMirrorEntryV2:
    entry = normalize_collapse_entry(payload)
    if source_service:
        entry.source_service = source_service
    if source_node:
        entry.source_node = source_node
    return _get_store().save(entry)


def enrich_entry(event_id: str, *, updates: Optional[Dict[str, Any]] = None) -> CollapseMirrorEntryV2:
    store = _get_store()
    entry = store.get(event_id)

    updates = updates or {}
    entry.what_changed_summary = updates.get("what_changed_summary") or entry.what_changed_summary or entry.summary
    entry.pattern_candidate = updates.get("pattern_candidate") or entry.pattern_candidate or entry.trigger
    entry.resonance_signature = updates.get("resonance_signature") or entry.resonance_signature or entry.field_resonance
    entry.change_type = updates.get("change_type") or entry.change_type or entry.type

    if not entry.change_type_scores:
        if entry.change_type:
            entry.change_type_scores = {entry.change_type: 0.6}

    if updates.get("tags") and isinstance(updates.get("tags"), list):
        entry.tags = list({*entry.tags, *updates.get("tags")})

    if updates.get("tag_scores") and isinstance(updates.get("tag_scores"), dict):
        entry.tag_scores.update({str(k): float(v) for k, v in updates.get("tag_scores").items() if v is not None})

    store.save(entry)
    return entry


def _score_from_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))


def _label_for_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.6:
        return "dense"
    if score >= 0.25:
        return "salient"
    return "ambient"


def score_causal_density(event_id: str) -> CollapseMirrorEntryV2:
    store = _get_store()
    entry = store.get(event_id)

    numeric = entry.numeric_sisters
    signals: list[float] = []

    for value in (numeric.valence, numeric.arousal, numeric.clarity, numeric.overload, numeric.risk_score):
        if value is None:
            continue
        try:
            signals.append(abs(float(value)))
        except (TypeError, ValueError):
            continue

    if numeric.constraints.severity_score is not None:
        try:
            signals.append(float(numeric.constraints.severity_score))
        except (TypeError, ValueError):
            pass

    if entry.tag_scores:
        tag_values = [float(v) for v in entry.tag_scores.values() if v is not None]
        if tag_values:
            signals.append(max(tag_values))

    if entry.change_type_scores:
        change_values = [float(v) for v in entry.change_type_scores.values() if v is not None]
        if change_values:
            signals.append(max(change_values))

    score = _score_from_values(signals)
    label = _label_for_score(score)

    entry.causal_density = CollapseMirrorCausalDensity(
        label=label,
        score=score,
        rationale=entry.causal_density.rationale or "computed_from_numeric_sisters",
    )

    entry.is_causally_dense = score >= 0.6
    if entry.snapshot_kind == "baseline" and entry.is_causally_dense:
        entry.snapshot_kind = "confirmed_dense"

    store.save(entry)
    return entry
