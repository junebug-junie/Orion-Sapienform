from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_summary_rows(
    *,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    counts: Iterable[Tuple[int, int]],
    total_docs: int,
    outlier_count: int,
    outlier_pct: float,
    topic_labeler,
    topic_keywords_fn,
    max_topics: int,
    min_docs: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if total_docs <= 0:
        return rows

    sorted_counts = sorted(counts, key=lambda item: item[1], reverse=True)
    for topic_id, doc_count in sorted_counts[: max_topics or len(sorted_counts)]:
        if doc_count < min_docs:
            continue
        keywords = topic_keywords_fn(topic_id)
        label = topic_labeler(topic_id, keywords)
        pct = float(doc_count) / float(total_docs)
        rows.append(
            {
                "model_version": model_version,
                "window_start": window_start,
                "window_end": window_end,
                "topic_id": topic_id,
                "topic_label": label,
                "topic_keywords": keywords,
                "doc_count": doc_count,
                "pct_of_window": pct,
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
            }
        )
    return rows


def compute_drift_rows(
    *,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    rows: Iterable[Dict[str, Any]],
    min_turns: int,
) -> List[Dict[str, Any]]:
    sessions: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
    for row in rows:
        session_id = row.get("session_id")
        topic_id = row.get("topic_id")
        created_at = row.get("created_at") or _utc_now()
        if session_id is None:
            continue
        sessions[str(session_id)].append((created_at, int(topic_id)))

    output: List[Dict[str, Any]] = []
    for session_id, items in sessions.items():
        items.sort(key=lambda item: item[0])
        topics = [topic_id for _, topic_id in items]
        turns = len(topics)
        if turns < min_turns:
            continue
        counts = Counter(topics)
        unique_topics = len(counts)
        entropy = _entropy(counts, turns)
        switch_rate = _switch_rate(topics)
        dominant_topic_id, dominant_pct = _dominant(counts, turns)
        output.append(
            {
                "model_version": model_version,
                "window_start": window_start,
                "window_end": window_end,
                "session_id": session_id,
                "turns": turns,
                "unique_topics": unique_topics,
                "entropy": entropy,
                "switch_rate": switch_rate,
                "dominant_topic_id": dominant_topic_id,
                "dominant_pct": dominant_pct,
            }
        )
    return output


def _entropy(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p <= 0:
            continue
        entropy -= p * math.log2(p)
    return entropy


def _switch_rate(topics: List[int]) -> float:
    if not topics:
        return 0.0
    switches = 0
    for prev, nxt in zip(topics, topics[1:]):
        if prev != nxt:
            switches += 1
    return switches / max(1, len(topics))


def _dominant(counts: Counter, total: int) -> Tuple[int | None, float]:
    if not counts or total <= 0:
        return None, 0.0
    topic_id, count = counts.most_common(1)[0]
    return int(topic_id), float(count) / float(total)
