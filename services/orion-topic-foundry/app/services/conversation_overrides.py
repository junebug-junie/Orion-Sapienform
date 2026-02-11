from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, NAMESPACE_DNS, uuid5

from app.models import WindowingSpec
from app.services.types import RowBlock


@dataclass
class Conversation:
    conversation_id: UUID
    dataset_id: UUID
    blocks: List[RowBlock]
    observed_start_at: datetime
    observed_end_at: datetime


@dataclass
class OverrideRecord:
    override_id: UUID
    kind: str
    payload: Dict[str, Any]
    created_at: datetime


def build_conversations(
    rows: List[Dict[str, Any]],
    *,
    dataset_id: UUID,
    spec: WindowingSpec,
    text_columns: List[str],
    time_column: str,
    id_column: str,
    boundary_column: Optional[str] = None,
) -> List[Conversation]:
    from app.services.windowing import build_blocks_for_conversation

    sorted_rows = sorted(rows, key=lambda r: r[time_column])
    conversations: List[List[Dict[str, Any]]] = []
    mode = spec.windowing_mode
    effective_boundary = boundary_column or spec.conversation_bound or spec.boundary_column
    if mode == "conversation_bound":
        if not effective_boundary:
            raise ValueError(f"boundary_column is required for {mode} dataset_id={dataset_id}")
        current: List[Dict[str, Any]] = []
        current_key: Optional[str] = None
        last_ts: Optional[datetime] = None
        conv_start: Optional[datetime] = None
        for row in sorted_rows:
            ts = row[time_column]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            key_val = row.get(effective_boundary)
            key = str(key_val) if key_val is not None else "__none__"
            if current_key is None:
                current_key = key
                current.append(row)
                last_ts = ts
                conv_start = ts
                continue
            if key == current_key:
                current.append(row)
            else:
                if current:
                    conversations.append(current)
                current = [row]
                current_key = key
                conv_start = ts
            last_ts = ts
        if current:
            conversations.append(current)
    elif mode == "time_gap":
        current = []
        last_ts = None
        conv_start: Optional[datetime] = None
        for row in sorted_rows:
            ts = row[time_column]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if last_ts is None:
                current.append(row)
                last_ts = ts
                conv_start = ts
                continue
            gap_ok = (ts - last_ts).total_seconds() <= max(int(spec.time_gap_minutes), 1) * 60
            if gap_ok:
                current.append(row)
            else:
                if current:
                    conversations.append(current)
                current = [row]
                conv_start = ts
            last_ts = ts
        if current:
            conversations.append(current)
    else:
        if sorted_rows:
            conversations = [sorted_rows]

    out: List[Conversation] = []
    for convo_rows in conversations:
        blocks = build_blocks_for_conversation(
            convo_rows,
            spec=spec,
            text_columns=text_columns,
            time_column=time_column,
            id_column=id_column,
        )
        if not blocks:
            continue
        observed_start = _parse_ts(blocks[0].timestamps[0])
        observed_end = _parse_ts(blocks[-1].timestamps[-1])
        convo_id = _stable_conversation_id(blocks)
        for idx, block in enumerate(blocks):
            block.conversation_id = convo_id
            block.block_index = idx
        out.append(
            Conversation(
                conversation_id=convo_id,
                dataset_id=dataset_id,
                blocks=blocks,
                observed_start_at=observed_start,
                observed_end_at=observed_end,
            )
        )
    return out


def apply_overrides(conversations: List[Conversation], overrides: List[OverrideRecord]) -> List[Conversation]:
    convo_map = {convo.conversation_id: convo for convo in conversations}

    for override in sorted((o for o in overrides if o.kind == "merge"), key=lambda o: o.created_at):
        ids = [UUID(cid) for cid in override.payload.get("conversation_ids", [])]
        if not ids or any(cid not in convo_map for cid in ids):
            continue
        merged_blocks: List[RowBlock] = []
        for cid in ids:
            merged_blocks.extend(convo_map[cid].blocks)
        merged_blocks.sort(key=lambda b: b.timestamps[0])
        new_id = UUID(override.payload.get("new_conversation_id")) if override.payload.get("new_conversation_id") else _stable_merge_id(ids)
        for idx, block in enumerate(merged_blocks):
            block.conversation_id = new_id
            block.block_index = idx
        observed_start = _parse_ts(merged_blocks[0].timestamps[0])
        observed_end = _parse_ts(merged_blocks[-1].timestamps[-1])
        for cid in ids:
            convo_map.pop(cid, None)
        convo_map[new_id] = Conversation(
            conversation_id=new_id,
            dataset_id=conversations[0].dataset_id,
            blocks=merged_blocks,
            observed_start_at=observed_start,
            observed_end_at=observed_end,
        )

    for override in sorted((o for o in overrides if o.kind == "split"), key=lambda o: o.created_at):
        payload = override.payload
        source_id = UUID(payload.get("conversation_id")) if payload.get("conversation_id") else None
        if source_id is None or source_id not in convo_map:
            continue
        split_at = int(payload.get("split_at_block_index", 0))
        blocks = convo_map[source_id].blocks
        if split_at <= 0 or split_at >= len(blocks):
            continue
        left_blocks = blocks[:split_at]
        right_blocks = blocks[split_at:]
        left_id, right_id = _split_ids(payload, source_id, left_blocks, right_blocks)
        for idx, block in enumerate(left_blocks):
            block.conversation_id = left_id
            block.block_index = idx
        for idx, block in enumerate(right_blocks):
            block.conversation_id = right_id
            block.block_index = idx
        left_convo = Conversation(
            conversation_id=left_id,
            dataset_id=convo_map[source_id].dataset_id,
            blocks=left_blocks,
            observed_start_at=_parse_ts(left_blocks[0].timestamps[0]),
            observed_end_at=_parse_ts(left_blocks[-1].timestamps[-1]),
        )
        right_convo = Conversation(
            conversation_id=right_id,
            dataset_id=convo_map[source_id].dataset_id,
            blocks=right_blocks,
            observed_start_at=_parse_ts(right_blocks[0].timestamps[0]),
            observed_end_at=_parse_ts(right_blocks[-1].timestamps[-1]),
        )
        convo_map.pop(source_id, None)
        convo_map[left_id] = left_convo
        convo_map[right_id] = right_convo

    return list(convo_map.values())


def _stable_conversation_id(blocks: List[RowBlock]) -> UUID:
    raw = "|".join(
        [
            ",".join(block.row_ids) + ":" + ",".join(block.timestamps)
            for block in blocks
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return uuid5(NAMESPACE_DNS, digest)


def _stable_merge_id(ids: List[UUID]) -> UUID:
    raw = "|".join(sorted(str(cid) for cid in ids))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return uuid5(NAMESPACE_DNS, digest)


def _split_ids(payload: Dict[str, Any], source_id: UUID, left_blocks: List[RowBlock], right_blocks: List[RowBlock]) -> tuple[UUID, UUID]:
    provided = payload.get("new_conversation_ids")
    if isinstance(provided, list) and len(provided) == 2:
        return UUID(provided[0]), UUID(provided[1])
    left_id = uuid5(NAMESPACE_DNS, str(source_id) + ":left")
    right_id = uuid5(NAMESPACE_DNS, str(source_id) + ":right")
    return left_id, right_id


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)
