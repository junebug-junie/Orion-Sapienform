from __future__ import annotations

import logging
from statistics import mean
from typing import List, Optional
from uuid import UUID

from app.models import DatasetPreviewDoc, DatasetPreviewRequest, DatasetPreviewResponse
from app.services.conversation_overrides import OverrideRecord, apply_overrides, build_conversations
from app.services.data_access import fetch_dataset_rows
from app.services.windowing import build_segments_with_stats
from app.settings import settings
from app.storage.repository import list_conversation_overrides


logger = logging.getLogger("topic-foundry.preview")


def preview_dataset(payload: DatasetPreviewRequest) -> DatasetPreviewResponse:
    rows = fetch_dataset_rows(
        dataset=payload.dataset,
        start_at=payload.start_at,
        end_at=payload.end_at,
        limit=payload.limit,
    )
    conversations = build_conversations(
        rows,
        dataset_id=payload.dataset.dataset_id,
        spec=payload.windowing,
        text_columns=payload.dataset.text_columns,
        time_column=payload.dataset.time_column,
        id_column=payload.dataset.id_column,
        boundary_column=payload.dataset.boundary_column,
    )
    overrides = [
        OverrideRecord(
            override_id=UUID(row["override_id"]),
            kind=row["kind"],
            payload=row["payload"],
            created_at=row["created_at"],
        )
        for row in list_conversation_overrides(payload.dataset.dataset_id)
    ]
    if overrides:
        conversations = apply_overrides(conversations, overrides)
    segments, blocks_generated = build_segments_with_stats(
        conversations,
        spec=payload.windowing,
        embedding_url=settings.topic_foundry_embedding_url,
        run_id=None,
    )
    observed_start_at: Optional[str] = None
    observed_end_at: Optional[str] = None
    if rows:
        observed_start_at = rows[0][payload.dataset.time_column]
        observed_end_at = rows[-1][payload.dataset.time_column]
    if not segments:
        return DatasetPreviewResponse(
            rows_scanned=len(rows),
            row_count=len(rows),
            blocks_generated=blocks_generated,
            segments_generated=0,
            segment_count=0,
            docs_generated=0,
            doc_count=0,
            avg_chars=0.0,
            p95_chars=0,
            max_chars=0,
            observed_start_at=observed_start_at,
            observed_end_at=observed_end_at,
            samples=[],
        )

    lengths = [len(seg.text) for seg in segments]
    sorted_lengths = sorted(lengths)
    p95_index = max(0, int(len(sorted_lengths) * 0.95) - 1)
    p95_chars = sorted_lengths[p95_index]
    samples = [
        DatasetPreviewDoc(
            doc_id=seg.doc_id,
            segment_id=seg.doc_id,
            row_ids_count=len(seg.row_ids),
            chars=len(seg.text),
            snippet=seg.text[:200],
        )
        for seg in segments[:5]
    ]
    return DatasetPreviewResponse(
        rows_scanned=len(rows),
        row_count=len(rows),
        blocks_generated=blocks_generated,
        segments_generated=len(segments),
        segment_count=len(segments),
        docs_generated=len(segments),
        doc_count=len(segments),
        avg_chars=float(mean(lengths)),
        p95_chars=p95_chars,
        max_chars=max(lengths),
        observed_start_at=observed_start_at,
        observed_end_at=observed_end_at,
        samples=samples,
    )
