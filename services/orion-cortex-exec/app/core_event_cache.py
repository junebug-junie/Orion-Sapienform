from __future__ import annotations

from collections import deque
from typing import Any, Dict, List


def _is_turn_effect_alert(event: Dict[str, Any]) -> bool:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    event_type = payload.get("event_type") or payload.get("type") or payload.get("kind")
    return event.get("event") == "turn_effect_alert" or event_type == "turn_effect_alert"


def _normalize_turn_effect_alert(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    rule = metadata.get("rule") or payload.get("rule")
    value = metadata.get("value") if metadata.get("value") is not None else payload.get("value")
    threshold = metadata.get("threshold") if metadata.get("threshold") is not None else payload.get("threshold")
    return {
        "event": event.get("event"),
        "event_type": payload.get("event_type") or payload.get("type") or payload.get("kind"),
        "title": payload.get("title"),
        "summary": payload.get("body") or payload.get("summary"),
        "severity": payload.get("severity"),
        "rule": rule,
        "value": value,
        "threshold": threshold,
        "direction": payload.get("direction"),
        "corr_id": (
            metadata.get("corr_id")
            or payload.get("corr_id")
            or payload.get("correlation_id")
            or payload.get("corr_id")
        ),
        "trace_id": metadata.get("trace_id") or payload.get("trace_id"),
    }


def format_recent_turn_effect_alerts(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return "Recent Alerts: 0"
    last = alerts[0]
    rule = last.get("rule") or "unknown"
    value = last.get("value")
    severity = last.get("severity") or "unknown"
    corr_id = last.get("corr_id") or "?"
    trace_id = last.get("trace_id") or "?"
    if isinstance(value, (int, float)):
        value_text = f"{float(value):.3f}"
    else:
        value_text = "?"
    return f"Recent Alerts: {len(alerts)} (last: {rule} {value_text} {severity} corr={corr_id} trace={trace_id})"


class CoreEventCache:
    _instance = None

    def __init__(self, maxlen: int = 20):
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=maxlen)

    @classmethod
    def get_instance(cls) -> CoreEventCache:
        if cls._instance is None:
            cls._instance = CoreEventCache()
        return cls._instance

    def append(self, event: Dict[str, Any]) -> None:
        if not _is_turn_effect_alert(event):
            return
        self.buffer.append(_normalize_turn_effect_alert(event))

    def get_recent_turn_effect_alerts(self, k: int = 5) -> List[Dict[str, Any]]:
        return list(reversed(self.buffer))[:k]


def get_core_event_cache() -> CoreEventCache:
    return CoreEventCache.get_instance()
