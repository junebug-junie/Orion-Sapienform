from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4, UUID

import numpy as np

from app.models import WindowingSpec
from app.services.boundary_judge import judge_boundaries
from app.services.llm_client import get_llm_client
from app.settings import settings
from app.storage.repository import insert_window_filters
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.conversation_overrides import Conversation
from app.services.embedding_client import VectorHostEmbeddingProvider
from app.services.semantic_segmentation import SemanticConfig, split_blocks
from app.services.types import BoundaryContext, RowBlock


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
    if mode == "turn_pairs" and spec.block_mode != "turn_pairs":
        if spec.block_mode == "rows":
            mode = "fixed_k_rows"
            spec.fixed_k_rows = 1
        elif spec.block_mode == "triads":
            mode = "fixed_k_rows"
            spec.fixed_k_rows = 3
        elif spec.block_mode == "group_by_column":
            mode = "conversation_bound_then_time_gap"
    if mode == "fixed_k_rows":
        step = spec.fixed_k_rows_step or spec.fixed_k_rows
        for idx in range(0, len(convo_rows), step):
            chunk = convo_rows[idx : idx + spec.fixed_k_rows]
            if not chunk:
                continue
            text = _make_block_text(chunk, text_columns, spec)
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
    elif mode == "time_gap" or mode == "conversation_bound_then_time_gap":
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
            if last_ts is not None and (ts - last_ts).total_seconds() > spec.time_gap_seconds and text_parts:
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
    elif mode == "turn_pairs":
        idx = 0
        while idx < len(convo_rows) - 1:
            first = convo_rows[idx]
            second = convo_rows[idx + 1]
            role_first = _role_of(first)
            role_second = _role_of(second)
            if spec.include_roles and role_first and role_second:
                if role_first not in spec.include_roles or role_second not in spec.include_roles:
                    idx += 1
                    continue
            text = _make_block_text([first, second], text_columns, spec)
            if text:
                blocks.append(
                    RowBlock(
                        row_ids=[str(first[id_column]), str(second[id_column])],
                        timestamps=[
                            first[time_column].isoformat() if hasattr(first[time_column], "isoformat") else str(first[time_column]),
                            second[time_column].isoformat() if hasattr(second[time_column], "isoformat") else str(second[time_column]),
                        ],
                        doc_id=str(uuid4()),
                        text=text,
                    )
                )
            idx += 2
    elif mode == "conversation_bound":
        step = spec.fixed_k_rows_step or spec.fixed_k_rows
        for idx in range(0, len(convo_rows), step):
            chunk = convo_rows[idx : idx + spec.fixed_k_rows]
            if not chunk:
                continue
            text = _make_block_text(chunk, text_columns, spec)
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
    elif spec.block_mode == "rows":
        for row in convo_rows:
            text = _row_text(row, text_columns)
            if not text:
                continue
            blocks.append(
                RowBlock(
                    row_ids=[str(row[id_column])],
                    timestamps=[row[time_column].isoformat() if hasattr(row[time_column], "isoformat") else str(row[time_column])],
                    doc_id=str(uuid4()),
                    text=_truncate(text, spec.max_chars),
                )
            )
    elif spec.block_mode == "triads":
        for idx in range(0, len(convo_rows), 3):
            chunk = convo_rows[idx : idx + 3]
            if len(chunk) < 3:
                break
            text = _make_block_text(chunk, text_columns, spec)
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
    elif spec.block_mode == "group_by_column":
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
            if last_ts is not None and (ts - last_ts).total_seconds() > spec.time_gap_seconds and text_parts:
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
        for row in convo_rows:
            text = _row_text(row, text_columns)
            if not text:
                continue
            blocks.append(
                RowBlock(
                    row_ids=[str(row[id_column])],
                    timestamps=[row[time_column].isoformat() if hasattr(row[time_column], "isoformat") else str(row[time_column])],
                    doc_id=str(uuid4()),
                    text=_truncate(text, spec.max_chars),
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


def _role_of(row: Dict[str, Any]) -> Optional[str]:
    role = row.get("role") or row.get("speaker")
    if role is None:
        return None
    return str(role).lower().strip()


def _make_block_text(rows: Sequence[Dict[str, Any]], text_columns: Sequence[str], spec: WindowingSpec) -> str:
    if spec.block_mode == "turn_pairs":
        if len(rows) == 2:
            user_row, assistant_row = rows
            user_text = _row_text(user_row, text_columns)
            assistant_text = _row_text(assistant_row, text_columns)
            text = f"User: {user_text}\nAssistant: {assistant_text}".strip()
        else:
            text = "\n".join(_row_text(row, text_columns) for row in rows).strip()
    else:
        text = "\n".join(_row_text(row, text_columns) for row in rows).strip()
    return _truncate(text, spec.max_chars)


def _chunk_blocks(blocks: List[RowBlock], spec: WindowingSpec) -> List[RowBlock]:
    if spec.min_blocks_per_segment <= 1:
        return blocks
    segments: List[RowBlock] = []
    for idx in range(0, len(blocks), spec.min_blocks_per_segment):
        chunk = blocks[idx : idx + spec.min_blocks_per_segment]
        if len(chunk) < spec.min_blocks_per_segment:
            break
        row_ids: List[str] = []
        timestamps: List[str] = []
        text_parts: List[str] = []
        for block in chunk:
            row_ids.extend(block.row_ids)
            timestamps.extend(block.timestamps)
            text_parts.append(block.text)
        segments.append(
            RowBlock(
                row_ids=row_ids,
                timestamps=timestamps,
                doc_id=str(uuid4()),
                text=_truncate("\n".join(text_parts).strip(), spec.max_chars),
                conversation_id=blocks[0].conversation_id if blocks else None,
                block_index=blocks[0].block_index if blocks else None,
            )
        )
    return segments


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
        if spec.segmentation_mode == "time_gap":
            segments.extend(_chunk_blocks(blocks, spec))
            continue

        embeddings = None
        if embedding_url and spec.segmentation_mode in {"semantic", "hybrid", "llm_judge", "hybrid_llm"}:
            embedder = VectorHostEmbeddingProvider(embedding_url)
            embeddings = np.array(embedder.embed_texts([block.text for block in blocks]), dtype=np.float32)

        if spec.segmentation_mode in {"semantic", "hybrid"} and embeddings is not None:
            cfg = SemanticConfig(
                threshold=spec.semantic_split_threshold,
                confirm_edges_k=spec.confirm_edges_k,
                smoothing_window=max(1, spec.smoothing_window),
                min_blocks_per_segment=spec.min_blocks_per_segment,
                max_window_seconds=spec.max_window_seconds,
                max_chars=spec.max_chars,
            )
            segments.extend(split_blocks(blocks, embeddings, cfg))
            continue

        if spec.segmentation_mode in {"llm_judge", "hybrid_llm"} and boundary_context is not None:
            splits = _llm_segmentation(blocks, embeddings, spec, boundary_context)
            segments.extend(_segments_from_splits(blocks, splits, spec))
            continue

        segments.extend(_chunk_blocks(blocks, spec))
    if spec.llm_filter_enabled:
        segments, filters = _apply_llm_filter(segments, spec)
        if run_id and filters:
            insert_window_filters(run_id, filters)
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


def _apply_llm_filter(
    segments: List[RowBlock],
    spec: WindowingSpec,
) -> tuple[List[RowBlock], List[Dict[str, Any]]]:
    if not settings.topic_foundry_llm_enable:
        return segments, []
    client = get_llm_client()
    kept: List[RowBlock] = []
    decisions: List[Dict[str, Any]] = []
    max_windows = max(0, int(spec.llm_filter_max_windows))
    for idx, segment in enumerate(segments):
        if max_windows and idx >= max_windows:
            kept.append(segment)
            continue
        prompt = spec.llm_filter_prompt_template.format(window_text=segment.text[:2000])
        response = client.request_json(system_prompt="You are a filtering assistant.", user_prompt=prompt, temperature=0.0)
        decision = response or {}
        keep = bool(decision.get("keep", True))
        if spec.llm_filter_policy == "reject":
            keep = not keep
        if spec.llm_filter_policy == "score":
            keep = float(decision.get("score") or 0) >= 0.5
        decisions.append(
            {
                "segment_id": segment.doc_id,
                "policy": spec.llm_filter_policy,
                "decision": decision,
            }
        )
        if keep:
            kept.append(segment)
    return kept, decisions


def _llm_segmentation(
    blocks: List[RowBlock],
    embeddings: Optional[np.ndarray],
    spec: WindowingSpec,
    boundary_context: BoundaryContext,
) -> List[int]:
    candidates = list(range(len(blocks) - 1))
    if spec.llm_candidate_strategy == "semantic_low_sim" and embeddings is not None:
        sims = _similarities(embeddings)
        threshold = spec.llm_candidate_threshold or spec.semantic_split_threshold
        candidates = [idx for idx, sim in enumerate(sims) if sim < threshold]
        if spec.llm_candidate_top_k:
            candidates = candidates[: spec.llm_candidate_top_k]
    elif spec.llm_candidate_strategy == "all_edges":
        if spec.llm_candidate_top_k:
            candidates = candidates[: spec.llm_candidate_top_k]

    decisions = judge_boundaries(blocks=blocks, candidate_indices=candidates, spec=spec, context=boundary_context)
    split_indices: List[int] = []
    sims = _similarities(embeddings) if embeddings is not None else []
    for idx in candidates:
        decision = decisions.get(idx)
        if decision is None:
            if spec.segmentation_mode == "hybrid_llm" and embeddings is not None:
                threshold = spec.llm_candidate_threshold or spec.semantic_split_threshold
                if idx < len(sims) and sims[idx] < threshold:
                    split_indices.append(idx)
            continue
        if decision.get("split") is True:
            split_indices.append(idx)
    return split_indices


def _segments_from_splits(blocks: List[RowBlock], split_indices: List[int], spec: WindowingSpec) -> List[RowBlock]:
    split_set = set(split_indices)
    segments: List[List[RowBlock]] = []
    current: List[RowBlock] = []
    for idx, block in enumerate(blocks):
        current.append(block)
        if idx in split_set:
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    merged: List[RowBlock] = []
    buffer: List[RowBlock] = []
    for segment in segments:
        buffer.extend(segment)
        if len(buffer) >= spec.min_blocks_per_segment:
            merged.append(_merge_blocks(buffer, spec.max_chars))
            buffer = []
    if buffer:
        if merged:
            merged[-1] = _merge_blocks([merged[-1], *buffer], spec.max_chars)
        else:
            merged.append(_merge_blocks(buffer, spec.max_chars))
    return merged


def _merge_blocks(blocks: List[RowBlock], max_chars: int) -> RowBlock:
    row_ids: List[str] = []
    timestamps: List[str] = []
    text_parts: List[str] = []
    for block in blocks:
        row_ids.extend(block.row_ids)
        timestamps.extend(block.timestamps)
        text_parts.append(block.text)
    text = "\n".join(text_parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return RowBlock(
        row_ids=row_ids,
        timestamps=timestamps,
        doc_id=str(uuid4()),
        text=text,
        conversation_id=blocks[0].conversation_id if blocks else None,
        block_index=blocks[0].block_index if blocks else None,
    )


def _similarities(embeddings: Optional[np.ndarray]) -> List[float]:
    if embeddings is None or len(embeddings) < 2:
        return []
    sims: List[float] = []
    for idx in range(len(embeddings) - 1):
        a = embeddings[idx]
        b = embeddings[idx + 1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(float(np.dot(a, b) / denom) if denom else 0.0)
    return sims
