from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from orion.schemas.social_chat import TownContinuityReadV1, TownContinuityTurnV1
from orion.town_cast import TOWN_PARTICIPANT_SLUGS

_DISPLAY_BY_SLUG = {slug: name for name, slug in TOWN_PARTICIPANT_SLUGS.items()}
_TEXT_CAP = 160
_PAIR_CAP = 8
_TOWN_CAP = 4


def _parse_json(value: Any, empty: dict | list) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return empty
    if isinstance(value, (dict, list)):
        return value
    return empty


def _thread_parts(thread_id: str) -> tuple[str, str] | None:
    parts = str(thread_id or "").split("--")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def speaker_on_thread(thread_id: str, speaker_id: str) -> bool:
    parts = _thread_parts(thread_id)
    return parts is not None and speaker_id in parts


def _display_name(slug: str) -> str:
    return _DISPLAY_BY_SLUG.get(slug, slug)


def _created_at_str(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _room_fields(row: dict) -> tuple[str, str, str] | None:
    client_meta = _parse_json(row.get("client_meta"), {})
    if not isinstance(client_meta, dict):
        return None
    room = client_meta.get("external_room")
    if not isinstance(room, dict):
        return None
    platform = room.get("platform")
    room_id = room.get("room_id")
    thread_id = room.get("thread_id")
    if not platform or not room_id or not thread_id:
        return None
    return str(platform), str(room_id), str(thread_id)


def _row_speaker_id(row: dict) -> str | None:
    client_meta = _parse_json(row.get("client_meta"), {})
    if not isinstance(client_meta, dict):
        return None
    participant = client_meta.get("external_participant")
    if not isinstance(participant, dict):
        return None
    speaker_id = participant.get("participant_id")
    if not speaker_id:
        return None
    return str(speaker_id)


def _row_is_recall_safe(row: dict) -> bool:
    redaction = _parse_json(row.get("redaction"), {})
    if not isinstance(redaction, dict) or "recall_safe" not in redaction:
        return False
    return redaction.get("recall_safe") is not False


def _has_aitown_tag(row: dict) -> bool:
    tags = _parse_json(row.get("tags"), [])
    if not isinstance(tags, list):
        return False
    return "aitown" in tags


def utterances_from_row(
    row: dict,
    *,
    platform: str | None = None,
    room_id: str | None = None,
) -> list[TownContinuityTurnV1]:
    room = _room_fields(row)
    if room is None:
        return []
    row_platform, row_room_id, thread_id = room
    if platform is not None and row_platform != platform:
        return []
    if room_id is not None and row_room_id != room_id:
        return []
    if not _has_aitown_tag(row) or not _row_is_recall_safe(row):
        return []
    parts = _thread_parts(thread_id)
    speaker_id = _row_speaker_id(row)
    if parts is None or speaker_id is None or speaker_id not in parts:
        return []
    other_slug = parts[0] if parts[1] == speaker_id else parts[1]
    created_at = _created_at_str(row.get("created_at"))
    if not created_at:
        return []
    speaker_name = _display_name(speaker_id)
    other_name = _display_name(other_slug)
    turns: list[TownContinuityTurnV1] = []
    for raw_text, turn_speaker, turn_other in (
        (row.get("prompt"), other_name, speaker_name),
        (row.get("response"), speaker_name, other_name),
    ):
        text = str(raw_text or "").strip()[:_TEXT_CAP]
        if not text:
            continue
        turns.append(
            TownContinuityTurnV1(
                speaker=turn_speaker,
                other=turn_other,
                text=text,
                thread_id=thread_id,
                created_at=created_at,
            )
        )
    return turns


def select_town_continuity(
    *,
    platform: str,
    room_id: str,
    thread_id: str,
    speaker_id: str,
    rows: list[dict],
) -> TownContinuityReadV1:
    pair_turns: list[TownContinuityTurnV1] = []
    town_turns: list[TownContinuityTurnV1] = []
    for row in rows:
        room = _room_fields(row)
        if room is None:
            continue
        _row_platform, _row_room_id, row_thread_id = room
        turns = utterances_from_row(row, platform=platform, room_id=room_id)
        if not turns:
            continue
        if row_thread_id == thread_id:
            pair_turns.extend(turns)
        elif speaker_on_thread(row_thread_id, speaker_id):
            town_turns.extend(turns)
    pair_turns.sort(key=lambda turn: (turn.created_at, turn.speaker))
    town_turns.sort(key=lambda turn: (turn.created_at, turn.speaker))
    return TownContinuityReadV1(
        thread_id=thread_id,
        speaker_id=speaker_id,
        pair_turns=pair_turns[:_PAIR_CAP],
        town_turns=town_turns[:_TOWN_CAP],
    )
