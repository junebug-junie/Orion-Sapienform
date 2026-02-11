from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
from uuid import UUID, uuid4

from app.models import WindowingSpec
from app.services.types import BoundaryContext, RowBlock

if TYPE_CHECKING:
    from app.services.conversation_overrides import Conversation


logger = logging.getLogger("topic-foundry.windowing")


def build_blocks_for_conversation(
    convo_rows: Sequence[Dict[str, Any]],
    *,
    spec: WindowingSpec,
    text_columns: Sequence[str],
    time_column: str,
    id_column: str,
) -> List[RowBlock]:
    blocks: List[RowBlock] = []
    mode = spec.windowing_mode
    if mode not in {"document", "time_gap", "conversation_bound"}:
        mode = "document"

    if mode == "time_gap":
        gap_seconds = max(int(spec.time_gap_minutes), 1) * 60
        row_ids: List[str] = []
        timestamps: List[str] = []
        text_parts: List[str] = []
        last_ts: Optional[datetime] = None
        for row in convo_rows:
            text = _row_text(row, text_columns)
            if not text:
                continue
            ts = row[time_column]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if last_ts is not None and (ts - last_ts).total_seconds() > gap_seconds and text_parts:
                blocks.append(
                    RowBlock(
                        row_ids=row_ids,
                        timestamps=timestamps,
                        doc_id=str(uuid4()),
                        text=_truncate("\n".join(text_parts).strip(), spec.max_chars),
                    )
                )
                row_ids = []
                timestamps = []
                text_parts = []
            candidate_text = "\n".join([*text_parts, text]).strip()
            if text_parts and len(candidate_text) > spec.max_chars:
                blocks.append(
                    RowBlock(
                        row_ids=row_ids,
                        timestamps=timestamps,
                        doc_id=str(uuid4()),
                        text=_truncate("\n".join(text_parts).strip(), spec.max_chars),
                    )
                )
                row_ids = []
                timestamps = []
                text_parts = []
            row_ids.append(str(row[id_column]))
            timestamps.append(ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
            text_parts.append(text)
            last_ts = ts
        if text_parts:
            blocks.append(
                RowBlock(
                    row_ids=row_ids,
                    timestamps=timestamps,
                    doc_id=str(uuid4()),
                    text=_truncate("\n".join(text_parts).strip(), spec.max_chars),
                )
            )
    else:
        text = _make_block_text(convo_rows, text_columns, spec)
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


def _row_text(row: Dict[str, Any], text_columns: Sequence[str]) -> str:
    parts: List[str] = []
    for col in text_columns:
        val = row.get(col)
        if val is None:
            continue
        val_str = str(val).strip()
        if val_str:
            parts.append(val_str)
    return "\n".join(parts).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _make_block_text(rows: Sequence[Dict[str, Any]], text_columns: Sequence[str], spec: WindowingSpec) -> str:
    text = "\n".join(_row_text(row, text_columns) for row in rows).strip()
    return _truncate(text, spec.max_chars)


def _build_segments_internal(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
    run_id: Optional[UUID] = None,
) -> tuple[List[RowBlock], int]:
    del embedding_url, boundary_context, run_id
    segments: List[RowBlock] = []
    blocks_generated = 0
    for convo in conversations:
        blocks_generated += len(convo.blocks)
        segments.extend(convo.blocks)
    return segments, blocks_generated


def build_segments_from_conversations(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
    run_id: Optional[UUID] = None,
) -> tuple[List[RowBlock], int]:
    return _build_segments_internal(
        conversations,
        spec=spec,
        embedding_url=embedding_url,
        boundary_context=boundary_context,
        run_id=run_id,
    )


def build_segments_with_stats(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
    run_id: Optional[UUID] = None,
) -> tuple[List[RowBlock], int]:
    return _build_segments_internal(
        conversations,
        spec=spec,
        embedding_url=embedding_url,
        boundary_context=boundary_context,
        run_id=run_id,
    )
