from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import List, Optional
from uuid import uuid4

import numpy as np

from app.services.types import RowBlock


logger = logging.getLogger("topic-foundry.semantic")


@dataclass
class SemanticConfig:
    threshold: float
    confirm_edges_k: int
    smoothing_window: int
    min_blocks_per_segment: int
    max_window_seconds: int
    max_chars: int


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def split_blocks(blocks: List[RowBlock], embeddings: np.ndarray, cfg: SemanticConfig) -> List[RowBlock]:
    if len(blocks) <= 1:
        return blocks

    sims: List[float] = []
    for idx in range(len(blocks) - 1):
        sims.append(_cosine_similarity(embeddings[idx], embeddings[idx + 1]))

    smoothed: List[float] = []
    window = max(1, cfg.smoothing_window)
    for idx, sim in enumerate(sims):
        start = max(0, idx - window + 1)
        smoothed.append(median(sims[start : idx + 1]))

    segments: List[List[RowBlock]] = []
    current: List[RowBlock] = [blocks[0]]
    low_count = 0

    for edge_idx, sim in enumerate(smoothed):
        low = sim < cfg.threshold
        if low:
            low_count += 1
        else:
            low_count = 0

        next_block = blocks[edge_idx + 1]
        current.append(next_block)

        should_split = low_count >= cfg.confirm_edges_k
        if should_split:
            segments.append(current)
            current = []
            low_count = 0

    if current:
        segments.append(current)

    limited: List[List[RowBlock]] = []
    for segment in segments:
        limited.extend(_split_by_limits_blocks(segment, cfg))

    merged: List[RowBlock] = []
    buffer: List[RowBlock] = []

    for segment in limited:
        buffer.extend(segment)
        if len(buffer) >= cfg.min_blocks_per_segment:
            merged.append(_merge_blocks(buffer, cfg.max_chars))
            buffer = []

    if buffer:
        if merged:
            merged[-1] = _merge_blocks([merged[-1], *buffer], cfg.max_chars)
        else:
            merged.append(_merge_blocks(buffer, cfg.max_chars))

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


def _split_by_limits_blocks(segment: List[RowBlock], cfg: SemanticConfig) -> List[List[RowBlock]]:
    if not segment:
        return []
    output: List[List[RowBlock]] = []
    current: List[RowBlock] = []
    current_chars = 0
    start_ts: Optional[datetime] = None
    for block in segment:
        if block.timestamps:
            block_start = _parse_ts(block.timestamps[0])
            block_end = _parse_ts(block.timestamps[-1])
        else:
            block_start = None
            block_end = None
        if start_ts is None and block_start:
            start_ts = block_start
        exceeds_time = False
        if start_ts and block_end:
            exceeds_time = (block_end - start_ts).total_seconds() > cfg.max_window_seconds
        exceeds_chars = (current_chars + len(block.text)) > cfg.max_chars
        if current and (exceeds_time or exceeds_chars):
            output.append(current)
            current = []
            current_chars = 0
            start_ts = block_start
        current.append(block)
        current_chars += len(block.text)
    if current:
        output.append(current)
    return output
