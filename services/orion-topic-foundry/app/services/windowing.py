from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID, uuid4

from app.models import WindowingSpec
from app.services.types import BoundaryContext, RowBlock


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.conversation_overrides import Conversation


def _row_text(row: Dict[str, Any], text_columns: Sequence[str]) -> str:
    parts: List[str] = []
    for col in text_columns:
        value = row.get(col)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def build_blocks_for_conversation(
    convo_rows: Sequence[Dict[str, Any]],
    *,
    spec: WindowingSpec,
    text_columns: Sequence[str],
    time_column: str,
    id_column: str,
) -> List[RowBlock]:
    mode = spec.windowing_mode
    max_chars = int(spec.max_chars)
    blocks: List[RowBlock] = []

    if mode == "document":
        for idx in range(0, len(convo_rows), 2):
            chunk = convo_rows[idx : idx + 2]
            if not chunk:
                continue
            text = _truncate("\n".join(_row_text(row, text_columns) for row in chunk).strip(), max_chars)
            if not text:
                continue
            blocks.append(
                RowBlock(
                    row_ids=[str(row[id_column]) for row in chunk],
                    timestamps=[
                        row[time_column].isoformat() if hasattr(row[time_column], "isoformat") else str(row[time_column])
                        for row in chunk
                    ],
                    doc_id=str(uuid4()),
                    text=text,
                )
            )
        return blocks

    if mode == "conversation_bound":
        text = _truncate("\n".join(_row_text(row, text_columns) for row in convo_rows).strip(), max_chars)
        if text:
            blocks.append(
                RowBlock(
                    row_ids=[str(row[id_column]) for row in convo_rows],
                    timestamps=[
                        row[time_column].isoformat() if hasattr(row[time_column], "isoformat") else str(row[time_column])
                        for row in convo_rows
                    ],
                    doc_id=str(uuid4()),
                    text=text,
                )
            )
        return blocks

    # time_gap
    row_ids: List[str] = []
    timestamps: List[str] = []
    text_parts: List[str] = []
    last_ts: Optional[datetime] = None
    gap_seconds = int(spec.time_gap_minutes) * 60
    for row in convo_rows:
        text = _row_text(row, text_columns)
        if not text:
            continue
        ts = row[time_column]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if last_ts is not None and (ts - last_ts).total_seconds() > gap_seconds and text_parts:
            blocks.append(RowBlock(row_ids=row_ids, timestamps=timestamps, doc_id=str(uuid4()), text=_truncate("\n".join(text_parts).strip(), max_chars)))
            row_ids, timestamps, text_parts = [], [], []
        candidate = "\n".join([*text_parts, text]).strip()
        if text_parts and len(candidate) > max_chars:
            blocks.append(RowBlock(row_ids=row_ids, timestamps=timestamps, doc_id=str(uuid4()), text=_truncate("\n".join(text_parts).strip(), max_chars)))
            row_ids, timestamps, text_parts = [], [], []
        row_ids.append(str(row[id_column]))
        timestamps.append(ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
        text_parts.append(text)
        last_ts = ts
    if text_parts:
        blocks.append(RowBlock(row_ids=row_ids, timestamps=timestamps, doc_id=str(uuid4()), text=_truncate("\n".join(text_parts).strip(), max_chars)))
    return blocks


def _build_segments_internal(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
    run_id: Optional[UUID] = None,
) -> tuple[List[RowBlock], int]:
    segments: List[RowBlock] = []
    blocks_generated = 0
    for convo in conversations:
        blocks = convo.blocks
        blocks_generated += len(blocks)
        segments.extend(blocks)
    return segments, blocks_generated


def build_segments_from_conversations(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
    run_id: Optional[UUID] = None,
) -> tuple[List[RowBlock], int]:
    return _build_segments_internal(conversations, spec=spec, embedding_url=embedding_url, boundary_context=boundary_context, run_id=run_id)


def build_segments_with_stats(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
    run_id: Optional[UUID] = None,
) -> tuple[List[RowBlock], int]:
    return _build_segments_internal(conversations, spec=spec, embedding_url=embedding_url, boundary_context=boundary_context, run_id=run_id)
