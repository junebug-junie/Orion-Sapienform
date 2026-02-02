from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

from sqlalchemy import func

from orion.core.sql_router.db import SessionLocal

from .db_models import NotificationAttemptDB, NotificationRequestDB


@dataclass
class DigestSummary:
    window_start: datetime
    window_end: datetime
    severity_counts: Dict[str, int]
    top_event_kinds: List[Tuple[str, int]]
    top_source_services: List[Tuple[str, int]]
    critical_events: List[NotificationRequestDB]
    warning_events: List[NotificationRequestDB]
    failed_attempts: List[NotificationAttemptDB]
    throttled_count: int
    deduped_count: int


@dataclass
class TopicItem:
    label: str
    value: Optional[float] = None


@dataclass
class DriftItem:
    label: str
    score: float


@dataclass
class TopicsSnapshot:
    window_minutes: int
    summary_items: List[TopicItem]
    drift_items: List[DriftItem]
    summary_error: Optional[str] = None
    drift_error: Optional[str] = None


def window_bounds(window_end: datetime, window_hours: int) -> Tuple[datetime, datetime]:
    window_start = window_end - timedelta(hours=window_hours)
    return window_start, window_end


def summarize_window(window_start: datetime, window_end: datetime) -> DigestSummary:
    with SessionLocal() as db:
        severity_counts = dict(
            db.query(NotificationRequestDB.severity, func.count(NotificationRequestDB.notification_id))
            .filter(NotificationRequestDB.created_at >= window_start, NotificationRequestDB.created_at <= window_end)
            .group_by(NotificationRequestDB.severity)
            .all()
        )

        event_kind_counts = (
            db.query(NotificationRequestDB.event_kind, func.count(NotificationRequestDB.notification_id))
            .filter(NotificationRequestDB.created_at >= window_start, NotificationRequestDB.created_at <= window_end)
            .group_by(NotificationRequestDB.event_kind)
            .order_by(func.count(NotificationRequestDB.notification_id).desc())
            .limit(10)
            .all()
        )

        source_service_counts = (
            db.query(NotificationRequestDB.source_service, func.count(NotificationRequestDB.notification_id))
            .filter(NotificationRequestDB.created_at >= window_start, NotificationRequestDB.created_at <= window_end)
            .group_by(NotificationRequestDB.source_service)
            .order_by(func.count(NotificationRequestDB.notification_id).desc())
            .limit(10)
            .all()
        )

        critical_events = (
            db.query(NotificationRequestDB)
            .filter(
                NotificationRequestDB.created_at >= window_start,
                NotificationRequestDB.created_at <= window_end,
                NotificationRequestDB.severity.in_(["critical", "error"]),
            )
            .order_by(NotificationRequestDB.created_at.desc())
            .limit(10)
            .all()
        )

        warning_events = (
            db.query(NotificationRequestDB)
            .filter(
                NotificationRequestDB.created_at >= window_start,
                NotificationRequestDB.created_at <= window_end,
                NotificationRequestDB.severity == "warning",
            )
            .order_by(NotificationRequestDB.created_at.desc())
            .limit(10)
            .all()
        )

        failed_attempts = (
            db.query(NotificationAttemptDB)
            .filter(
                NotificationAttemptDB.attempted_at >= window_start,
                NotificationAttemptDB.attempted_at <= window_end,
                NotificationAttemptDB.status == "failed",
            )
            .order_by(NotificationAttemptDB.attempted_at.desc())
            .limit(10)
            .all()
        )

        throttled_count = (
            db.query(func.count(NotificationRequestDB.notification_id))
            .filter(
                NotificationRequestDB.created_at >= window_start,
                NotificationRequestDB.created_at <= window_end,
                NotificationRequestDB.status == "throttled",
            )
            .scalar()
            or 0
        )

        deduped_count = (
            db.query(func.count(NotificationRequestDB.notification_id))
            .filter(
                NotificationRequestDB.created_at >= window_start,
                NotificationRequestDB.created_at <= window_end,
                NotificationRequestDB.status == "deduped",
            )
            .scalar()
            or 0
        )

        return DigestSummary(
            window_start=window_start,
            window_end=window_end,
            severity_counts=severity_counts,
            top_event_kinds=event_kind_counts,
            top_source_services=source_service_counts,
            critical_events=critical_events,
            warning_events=warning_events,
            failed_attempts=failed_attempts,
            throttled_count=throttled_count,
            deduped_count=deduped_count,
        )


def build_digest_content(
    summary: DigestSummary,
    topics_snapshot: Optional[TopicsSnapshot] = None,
    drift_max_items: int = 5,
) -> tuple[str, str]:
    lines = []
    lines.append("Orion Daily Digest")
    lines.append(f"Window: {summary.window_start.isoformat()} → {summary.window_end.isoformat()}")
    lines.append("")

    lines.append("Counts by severity:")
    for severity, count in sorted(summary.severity_counts.items()):
        lines.append(f"- {severity}: {count}")
    if not summary.severity_counts:
        lines.append("- (no events)")

    lines.append("")
    lines.append("Top event kinds:")
    for event_kind, count in summary.top_event_kinds:
        lines.append(f"- {event_kind}: {count}")
    if not summary.top_event_kinds:
        lines.append("- (no events)")

    lines.append("")
    lines.append("Top source services:")
    for source_service, count in summary.top_source_services:
        lines.append(f"- {source_service}: {count}")
    if not summary.top_source_services:
        lines.append("- (no events)")

    lines.append("")
    lines.append("Topics:")
    lines.extend(_format_topics_text(topics_snapshot, drift_max_items))

    lines.append("")
    lines.append("Critical/Error events:")
    lines.extend(_format_events(summary.critical_events))

    lines.append("")
    lines.append("Warnings:")
    lines.extend(_format_events(summary.warning_events))

    lines.append("")
    lines.append("Delivery failures:")
    if summary.failed_attempts:
        for attempt in summary.failed_attempts:
            error = attempt.error or "(no error)"
            lines.append(
                f"- {attempt.attempted_at.isoformat()} | {attempt.channel} | {attempt.notification_id} | {error}"
            )
    else:
        lines.append("- (no failures)")

    lines.append("")
    lines.append(f"Throttled count: {summary.throttled_count}")
    lines.append(f"Deduped count: {summary.deduped_count}")

    body_text = "\n".join(lines)

    body_md = "\n".join(
        [
            "# Orion Daily Digest",
            f"**Window:** {summary.window_start.isoformat()} → {summary.window_end.isoformat()}",
            "",
            "## Counts by severity",
            *_format_md_list(summary.severity_counts),
            "",
            "## Top event kinds",
            *_format_md_pairs(summary.top_event_kinds),
            "",
            "## Top source services",
            *_format_md_pairs(summary.top_source_services),
            "",
            "## Topics",
            *_format_topics_md(topics_snapshot, drift_max_items),
            "",
            "## Critical/Error events",
            *_format_md_events(summary.critical_events),
            "",
            "## Warnings",
            *_format_md_events(summary.warning_events),
            "",
            "## Delivery failures",
            *_format_md_failures(summary.failed_attempts),
            "",
            f"**Throttled count:** {summary.throttled_count}",
            f"**Deduped count:** {summary.deduped_count}",
        ]
    )

    return body_text, body_md


def _format_events(events: List[NotificationRequestDB]) -> List[str]:
    if not events:
        return ["- (no events)"]
    return [
        f"- {event.created_at.isoformat()} | {event.severity} | {event.event_kind} | {event.source_service} | {event.title}"
        for event in events
    ]


def _format_md_list(items: Dict[str, int]) -> List[str]:
    if not items:
        return ["- (no events)"]
    return [f"- **{key}**: {value}" for key, value in items.items()]


def _format_md_pairs(items: List[Tuple[str, int]]) -> List[str]:
    if not items:
        return ["- (no events)"]
    return [f"- **{key}**: {value}" for key, value in items]


def _format_md_events(events: List[NotificationRequestDB]) -> List[str]:
    if not events:
        return ["- (no events)"]
    return [
        f"- `{event.created_at.isoformat()}` **{event.severity}** `{event.event_kind}` `{event.source_service}` — {event.title}"
        for event in events
    ]


def _format_md_failures(attempts: List[NotificationAttemptDB]) -> List[str]:
    if not attempts:
        return ["- (no failures)"]
    return [
        f"- `{attempt.attempted_at.isoformat()}` **{attempt.channel}** `{attempt.notification_id}` — {attempt.error or '(no error)'}"
        for attempt in attempts
    ]


def fetch_topics_snapshot(
    landing_pad_url: Optional[str],
    window_minutes: int,
    max_topics: int,
    drift_min_turns: int,
    drift_max_sessions: int,
    timeout_seconds: int = 5,
) -> TopicsSnapshot:
    summary_items: List[TopicItem] = []
    drift_items: List[DriftItem] = []
    summary_error = None
    drift_error = None

    if not landing_pad_url:
        return TopicsSnapshot(
            window_minutes=window_minutes,
            summary_items=[],
            drift_items=[],
            summary_error="LANDING_PAD_URL not set",
            drift_error="LANDING_PAD_URL not set",
        )

    base_url = landing_pad_url.rstrip("/")

    try:
        summary_resp = requests.get(
            f"{base_url}/api/topics/summary",
            params={"window_minutes": window_minutes, "max_topics": max_topics},
            timeout=timeout_seconds,
        )
        summary_resp.raise_for_status()
        summary_items = _parse_topic_summary(summary_resp.json())
    except Exception as exc:
        summary_error = str(exc)

    try:
        drift_resp = requests.get(
            f"{base_url}/api/topics/drift",
            params={
                "window_minutes": window_minutes,
                "min_turns": drift_min_turns,
                "max_sessions": drift_max_sessions,
            },
            timeout=timeout_seconds,
        )
        drift_resp.raise_for_status()
        drift_items = _parse_topic_drift(drift_resp.json())
    except Exception as exc:
        drift_error = str(exc)

    return TopicsSnapshot(
        window_minutes=window_minutes,
        summary_items=summary_items,
        drift_items=drift_items,
        summary_error=summary_error,
        drift_error=drift_error,
    )


def _parse_topic_summary(payload: Any) -> List[TopicItem]:
    items = _extract_topic_items(payload)
    results: List[TopicItem] = []
    for item in items:
        label = _extract_topic_label(item)
        value = _extract_topic_value(item)
        results.append(TopicItem(label=label, value=value))
    return results


def _parse_topic_drift(payload: Any) -> List[DriftItem]:
    items = _extract_topic_items(payload)
    results: List[DriftItem] = []
    for item in items:
        label = _extract_topic_label(item)
        score = _extract_drift_score(item)
        if score is None:
            continue
        results.append(DriftItem(label=label, score=score))
    results.sort(key=lambda entry: entry.score, reverse=True)
    return results


def _extract_topic_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("topics", "items", "results", "data"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _extract_topic_label(item: Dict[str, Any]) -> str:
    for key in ("topic", "label", "name", "title", "id"):
        value = item.get(key)
        if value:
            return str(value)
    return "unknown"


def _extract_topic_value(item: Dict[str, Any]) -> Optional[float]:
    for key in ("count", "weight", "score", "value"):
        value = item.get(key)
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None


def _extract_drift_score(item: Dict[str, Any]) -> Optional[float]:
    for key in ("drift_score", "score", "delta", "drift", "magnitude", "change"):
        value = item.get(key)
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_topics_text(topics_snapshot: Optional[TopicsSnapshot], drift_max_items: int) -> List[str]:
    lines: List[str] = ["Top Topics:"]
    if topics_snapshot is None:
        lines.append("- Topics unavailable (no snapshot)")
        lines.append("Drift Highlights:")
        lines.append("- Drift unavailable (no snapshot)")
        return lines
    if topics_snapshot.summary_error and not topics_snapshot.summary_items:
        lines.append(f"- Top topics unavailable: {topics_snapshot.summary_error}")
    else:
        if topics_snapshot.summary_items:
            for idx, item in enumerate(topics_snapshot.summary_items, start=1):
                value_label = f" ({item.value:.2f})" if item.value is not None else ""
                lines.append(f"{idx}. {item.label}{value_label}")
        else:
            lines.append("- (no topics)")
    lines.append("Drift Highlights:")
    if topics_snapshot.drift_error and not topics_snapshot.drift_items:
        lines.append(f"- Drift unavailable: {topics_snapshot.drift_error}")
    else:
        drift_items = topics_snapshot.drift_items[:drift_max_items]
        if drift_items:
            for idx, item in enumerate(drift_items, start=1):
                lines.append(f"{idx}. {item.label} (score {item.score:.2f})")
        else:
            lines.append("- (no drift)")
    return lines


def _format_topics_md(topics_snapshot: Optional[TopicsSnapshot], drift_max_items: int) -> List[str]:
    lines: List[str] = ["**Top Topics**"]
    if topics_snapshot is None:
        lines.append("- Topics unavailable (no snapshot)")
        lines.append("")
        lines.append("**Drift Highlights**")
        lines.append("- Drift unavailable (no snapshot)")
        return lines
    if topics_snapshot.summary_error and not topics_snapshot.summary_items:
        lines.append(f"- Top topics unavailable: {topics_snapshot.summary_error}")
    else:
        if topics_snapshot.summary_items:
            for item in topics_snapshot.summary_items:
                value_label = f" ({item.value:.2f})" if item.value is not None else ""
                lines.append(f"- {item.label}{value_label}")
        else:
            lines.append("- (no topics)")
    lines.append("")
    lines.append("**Drift Highlights**")
    if topics_snapshot.drift_error and not topics_snapshot.drift_items:
        lines.append(f"- Drift unavailable: {topics_snapshot.drift_error}")
    else:
        drift_items = topics_snapshot.drift_items[:drift_max_items]
        if drift_items:
            for item in drift_items:
                lines.append(f"- {item.label} (score {item.score:.2f})")
        else:
            lines.append("- (no drift)")
    return lines
