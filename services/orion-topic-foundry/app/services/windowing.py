from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from uuid import uuid4, UUID

import numpy as np

from app.models import WindowingSpec
from app.services.boundary_judge import judge_boundaries
from app.services.embedding_client import VectorHostEmbeddingProvider
from app.services.semantic_segmentation import SemanticConfig, split_blocks
from app.services.types import BoundaryContext, RowBlock
from app.services.windowing_provenance import dedup_extend, dedup_row_provenance

if TYPE_CHECKING:
    from app.services.conversation_overrides import Conversation


logger = logging.getLogger("topic-foundry.windowing")


@dataclass(frozen=True)
class _Unit:
    """One indivisible piece of source text plus who said it.

    With ``split_text_columns`` on this is a single utterance (one text
    column of one row); with it off it is a whole row's columns fused, and
    ``speaker`` is None because the fused text has more than one author.
    Every block mode below composes blocks out of these, so the speaker
    survives windowing instead of being destroyed by it.
    """

    row: Dict[str, Any]
    text: str
    speaker: Optional[str]


def _expand_units(
    convo_rows: Sequence[Dict[str, Any]],
    *,
    spec: WindowingSpec,
    text_columns: Sequence[str],
) -> List[_Unit]:
    units: List[_Unit] = []
    for row in convo_rows:
        if spec.split_text_columns:
            for col in text_columns:
                value = row.get(col)
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                units.append(_Unit(row=row, text=text, speaker=spec.column_speakers.get(col)))
        else:
            text = _row_text(row, text_columns)
            if text:
                # Fused: the columns of one row become one unit. A source that
                # genuinely carries a per-row role/speaker column still gets a
                # real speaker here -- without this, _role_of had no caller at
                # all and include_roles filtering that used to work for such a
                # source would have gone permanently inert (review finding,
                # 2026-08-28).
                units.append(_Unit(row=row, text=text, speaker=_role_of(row)))
    return units


def _unit_speakers(units: Sequence[_Unit]) -> List[str]:
    """Ordered, de-duplicated speakers across ``units`` (unknowns dropped)."""
    return dedup_extend([], (unit.speaker for unit in units if unit.speaker))


def _block_from_units(
    units: Sequence[_Unit],
    *,
    spec: WindowingSpec,
    time_column: str,
    id_column: str,
) -> Optional[RowBlock]:
    text = _make_block_text(units, spec)
    if not text:
        return None
    # One row can contribute several units (prompt + response), and a block's
    # row_ids are its provenance -- de-duplicate so a 2-unit block from one row
    # does not claim to cover two rows.
    row_ids: List[str] = []
    timestamps: List[str] = []
    for unit in units:
        row_id = str(unit.row[id_column])
        if row_id in row_ids:
            continue
        row_ids.append(row_id)
        raw_ts = unit.row[time_column]
        timestamps.append(raw_ts.isoformat() if hasattr(raw_ts, "isoformat") else str(raw_ts))
    return RowBlock(
        row_ids=row_ids,
        timestamps=timestamps,
        doc_id=str(uuid4()),
        text=text,
        speakers=_unit_speakers(units),
    )


def build_blocks_for_conversation(
    convo_rows: Sequence[Dict[str, Any]],
    *,
    spec: WindowingSpec,
    text_columns: Sequence[str],
    time_column: str,
    id_column: str,
) -> List[RowBlock]:
    units = _expand_units(convo_rows, spec=spec, text_columns=text_columns)
    blocks: List[RowBlock] = []

    def emit(chunk: Sequence[_Unit]) -> None:
        block = _block_from_units(chunk, spec=spec, time_column=time_column, id_column=id_column)
        if block is not None:
            blocks.append(block)

    if spec.block_mode == "rows":
        for unit in units:
            emit([unit])
    elif spec.block_mode == "triads":
        for idx in range(0, len(units), 3):
            chunk = units[idx : idx + 3]
            if len(chunk) < 3:
                break
            emit(chunk)
    else:
        idx = 0
        while idx < len(units) - 1:
            first = units[idx]
            second = units[idx + 1]
            # When split, a "turn pair" is one row's own prompt+response. Pair
            # only within a row: a row with a NULL column contributes a single
            # unit, and a flat idx += 2 walk would silently start pairing one
            # row's prompt with the NEXT row's prompt and stay misaligned for
            # the rest of the conversation (review finding, 2026-08-28). The
            # fused path keeps pairing consecutive rows, which is its point.
            if spec.split_text_columns and first.row is not second.row:
                idx += 1
                continue
            # With split_text_columns on, a unit's speaker is real and this
            # filter finally does something. With it off both speakers are
            # None, the guard short-circuits, and include_roles is inert --
            # which is exactly what it silently was for every run before
            # 2026-08-28, because _role_of read `role`/`speaker` columns that
            # do not exist on the configured source table.
            if spec.include_roles and first.speaker and second.speaker:
                if first.speaker not in spec.include_roles or second.speaker not in spec.include_roles:
                    idx += 1
                    continue
            emit([first, second])
            idx += 2
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
    """Speaker recorded ON THE ROW, for sources that carry one.

    Retained for source tables that really do have a per-row `role`/`speaker`
    column. It is NOT how speakers are resolved for column-split sources --
    see WindowingSpec.column_speakers and _expand_units. Returns None when
    the columns are absent, which is the case for `chat_history_log`.
    """
    role = row.get("role") or row.get("speaker")
    if role is None:
        return None
    return str(role).lower().strip()


def _make_block_text(units: Sequence[_Unit], spec: WindowingSpec) -> str:
    """Join units into one document. The speaker is deliberately NOT in here.

    Before 2026-08-28 the turn_pairs branch hardcoded "User:"/"Assistant:"
    prefixes onto two arbitrary consecutive rows. On a source whose every row
    holds a full exchange both labels were false -- but the deeper problem was
    never their truthfulness, it was that a *role label was inside the text
    that gets vectorized*. Code review 2026-08-28 caught the first fix
    reintroducing exactly that, just with true names: under the new default
    `rows` mode every document would have begun "juniper: " or "orion: ",
    which on a ~508-document corpus split roughly in half is a near-perfect
    high-IDF discriminator -- HDBSCAN could cluster on speaker instead of
    topic, and TfidfVectorizer could hand "juniper"/"orion" a top-keyword slot
    and label topics after the speakers.

    The speaker is carried structurally instead, on RowBlock.speakers ->
    segment provenance, where every consumer that needs it can read it and no
    embedding ever sees it.
    """
    return _truncate("\n".join(unit.text for unit in units).strip(), spec.max_chars)


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
        speakers: List[str] = []
        for block in chunk:
            row_ids.extend(block.row_ids)
            timestamps.extend(block.timestamps)
            text_parts.append(block.text)
            dedup_extend(speakers, block.speakers)
        row_ids, timestamps = dedup_row_provenance(row_ids, timestamps)
        segments.append(
            RowBlock(
                row_ids=row_ids,
                timestamps=timestamps,
                doc_id=str(uuid4()),
                text=_truncate("\n".join(text_parts).strip(), spec.max_chars),
                conversation_id=blocks[0].conversation_id if blocks else None,
                block_index=blocks[0].block_index if blocks else None,
                speakers=speakers,
            )
        )
    return segments


def _build_segments_internal(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
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
    return segments, blocks_generated


def build_segments_from_conversations(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
) -> tuple[List[RowBlock], int]:
    return _build_segments_internal(
        conversations,
        spec=spec,
        embedding_url=embedding_url,
        boundary_context=boundary_context,
    )


def build_segments_with_stats(
    conversations: List[Conversation],
    *,
    spec: WindowingSpec,
    embedding_url: Optional[str] = None,
    boundary_context: Optional[BoundaryContext] = None,
) -> tuple[List[RowBlock], int]:
    return _build_segments_internal(
        conversations,
        spec=spec,
        embedding_url=embedding_url,
        boundary_context=boundary_context,
    )


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
    speakers: List[str] = []
    for block in blocks:
        row_ids.extend(block.row_ids)
        timestamps.extend(block.timestamps)
        text_parts.append(block.text)
        dedup_extend(speakers, block.speakers)
    row_ids, timestamps = dedup_row_provenance(row_ids, timestamps)
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
        speakers=speakers,
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


def _heuristic_gate_score(text: str) -> float:
    """Bounded lightweight score used by tests and fallback heuristics."""
    if not text:
        return 0.0
    tokens = [tok for tok in str(text).lower().split() if tok]
    if not tokens:
        return 0.0
    unique_ratio = min(1.0, len(set(tokens)) / len(tokens))
    length_ratio = min(1.0, len(tokens) / 40.0)
    return round((0.6 * unique_ratio) + (0.4 * length_ratio), 4)
