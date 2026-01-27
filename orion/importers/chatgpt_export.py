from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from uuid import UUID, uuid5

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.chat_history import CHAT_HISTORY_MESSAGE_KIND, CHAT_HISTORY_TURN_KIND

CHATGPT_IMPORTER_VERSION = "0.1.0"
CHATGPT_IMPORTER_NAME = "chatgpt-import"
CHATGPT_NAMESPACE = UUID("7c8a8f4a-3ef5-4c8c-8c9a-7781f960a1c3")

_TEMP_DIRS: List[tempfile.TemporaryDirectory[str]] = []


@dataclass(frozen=True)
class ChatMessage:
    node_id: str
    role: str
    content: str
    create_time: float
    timestamp_iso: str
    model: Optional[str]
    provider: Optional[str]


@dataclass(frozen=True)
class ChatTurn:
    turn_index: int
    user_msg: ChatMessage
    assistant_msg: ChatMessage


def _to_iso_utc(epoch_seconds: float | None) -> str:
    if epoch_seconds is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _slugify(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif cleaned and cleaned[-1] != "-":
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "untitled"


def _extract_text(message: Dict[str, Any]) -> Optional[str]:
    content = message.get("content") or {}
    if content.get("content_type") != "text":
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts: List[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
    text = "".join(text_parts).strip()
    return text or None


def _service_ref(node_name: Optional[str] = None) -> ServiceRef:
    node = node_name or os.getenv("NODE_NAME") or socket.gethostname()
    return ServiceRef(name=CHATGPT_IMPORTER_NAME, version=CHATGPT_IMPORTER_VERSION, node=node)


def resolve_export_path(export_path: str | Path) -> Path:
    path = Path(export_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "conversations.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"conversations.json not found in directory: {path}")
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.endswith("conversations.json")]
            if not members:
                raise FileNotFoundError("conversations.json not found inside zip export")
            temp_dir = tempfile.TemporaryDirectory(prefix="chatgpt_export_")
            _TEMP_DIRS.append(temp_dir)
            zf.extract(members[0], path=temp_dir.name)
            return Path(temp_dir.name) / members[0]
    if path.is_file() and path.name == "conversations.json":
        return path
    raise FileNotFoundError(f"Unsupported export path: {path}")


def load_conversations(conversations_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(conversations_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("conversations.json did not contain a list")
    return data


def iter_messages(conv: Dict[str, Any], include_branches: bool) -> Iterator[ChatMessage]:
    mapping = conv.get("mapping") or {}
    if not isinstance(mapping, dict):
        return iter(())

    def build_message(node_id: str, node: Dict[str, Any]) -> Optional[ChatMessage]:
        message = node.get("message")
        if not isinstance(message, dict):
            return None
        content = _extract_text(message)
        if content is None:
            return None
        author = message.get("author") or {}
        role = author.get("role") or "user"
        create_time = message.get("create_time")
        if create_time is None:
            create_time = node.get("create_time")
        if create_time is None:
            create_time = conv.get("create_time")
        if create_time is None:
            create_time = conv.get("update_time")
        if create_time is None:
            create_time = 0.0
        metadata = message.get("metadata") or {}
        model = metadata.get("model_slug") or metadata.get("model")
        provider = metadata.get("provider")
        return ChatMessage(
            node_id=node_id,
            role=role,
            content=content,
            create_time=float(create_time),
            timestamp_iso=_to_iso_utc(float(create_time)),
            model=model,
            provider=provider,
        )

    messages: List[ChatMessage] = []
    if include_branches:
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            msg = build_message(node_id, node)
            if msg:
                messages.append(msg)
        messages.sort(key=lambda m: (m.create_time, m.node_id))
        return iter(messages)

    current = conv.get("current_node")
    seen: set[str] = set()
    chain: List[ChatMessage] = []
    while current and current not in seen:
        seen.add(current)
        node = mapping.get(current)
        if not isinstance(node, dict):
            break
        msg = build_message(current, node)
        if msg:
            chain.append(msg)
        parent = node.get("parent")
        current = parent if isinstance(parent, str) else None
    chain.reverse()
    return iter(chain)


def pair_turns(messages: Iterable[ChatMessage]) -> List[ChatTurn]:
    turns: List[ChatTurn] = []
    pending_user: Optional[ChatMessage] = None
    for msg in messages:
        if msg.role == "user":
            pending_user = msg
            continue
        if msg.role == "assistant" and pending_user:
            turns.append(ChatTurn(turn_index=len(turns), user_msg=pending_user, assistant_msg=msg))
            pending_user = None
    return turns


def build_envelopes_for_turn(
    *,
    conversation_id: str,
    conversation_title: str,
    conversation_update_time: float | None,
    conversation_create_time: float | None,
    user_id: str,
    user_speaker: str,
    assistant_speaker: str,
    turn: ChatTurn,
    node_name: Optional[str] = None,
    branch_mode: str = "current",
) -> tuple[BaseEnvelope, BaseEnvelope, BaseEnvelope]:
    session_id = f"chatgpt:{conversation_id}"
    correlation_uuid = uuid5(
        CHATGPT_NAMESPACE,
        f"{conversation_id}:{turn.user_msg.node_id}:{turn.assistant_msg.node_id}:turn",
    )
    user_env_id = uuid5(CHATGPT_NAMESPACE, f"{conversation_id}:{turn.user_msg.node_id}:msg")
    assistant_env_id = uuid5(CHATGPT_NAMESPACE, f"{conversation_id}:{turn.assistant_msg.node_id}:msg")

    tags = [
        "source:chatgpt_export",
        f"conv:{conversation_id}",
        f"title:{_slugify(conversation_title)}",
    ]

    user_payload = {
        "message_id": str(user_env_id),
        "session_id": session_id,
        "role": "user",
        "speaker": user_speaker,
        "content": turn.user_msg.content,
        "timestamp": turn.user_msg.timestamp_iso,
        "model": turn.user_msg.model,
        "provider": turn.user_msg.provider,
        "tags": tags,
    }
    assistant_payload = {
        "message_id": str(assistant_env_id),
        "session_id": session_id,
        "role": "assistant",
        "speaker": assistant_speaker,
        "content": turn.assistant_msg.content,
        "timestamp": turn.assistant_msg.timestamp_iso,
        "model": turn.assistant_msg.model,
        "provider": turn.assistant_msg.provider,
        "tags": tags,
    }

    spark_meta = {
        "conversation_title": conversation_title,
        "source": "chatgpt_export",
        "source_conversation_id": conversation_id,
        "source_update_time": conversation_update_time,
        "source_create_time": conversation_create_time,
        "branch_mode": branch_mode,
        "importer": {
            "name": CHATGPT_IMPORTER_NAME,
            "version": CHATGPT_IMPORTER_VERSION,
            "node": node_name or os.getenv("NODE_NAME") or socket.gethostname(),
        },
    }

    turn_timestamp = turn.assistant_msg.timestamp_iso or turn.user_msg.timestamp_iso
    turn_payload = {
        "id": str(correlation_uuid),
        "correlation_id": str(correlation_uuid),
        "source": "chatgpt_import",
        "prompt": turn.user_msg.content,
        "response": turn.assistant_msg.content,
        "user_id": user_id,
        "session_id": session_id,
        "spark_meta": spark_meta,
        "timestamp": turn_timestamp,
        "created_at": turn_timestamp,
    }

    source = _service_ref(node_name)
    user_env = BaseEnvelope(
        id=user_env_id,
        correlation_id=correlation_uuid,
        kind=CHAT_HISTORY_MESSAGE_KIND,
        source=source,
        payload=user_payload,
    )
    assistant_env = BaseEnvelope(
        id=assistant_env_id,
        correlation_id=correlation_uuid,
        kind=CHAT_HISTORY_MESSAGE_KIND,
        source=source,
        payload=assistant_payload,
    )
    turn_env = BaseEnvelope(
        id=correlation_uuid,
        correlation_id=correlation_uuid,
        kind=CHAT_HISTORY_TURN_KIND,
        source=source,
        payload=turn_payload,
    )
    return user_env, assistant_env, turn_env


def build_message_envelope(
    *,
    conversation_id: str,
    conversation_title: str,
    message: ChatMessage,
    speaker: str,
    node_name: Optional[str] = None,
) -> BaseEnvelope:
    env_id = uuid5(CHATGPT_NAMESPACE, f"{conversation_id}:{message.node_id}:msg")
    tags = [
        "source:chatgpt_export",
        f"conv:{conversation_id}",
        f"title:{_slugify(conversation_title)}",
    ]
    payload = {
        "message_id": str(env_id),
        "session_id": f"chatgpt:{conversation_id}",
        "role": message.role,
        "speaker": speaker,
        "content": message.content,
        "timestamp": message.timestamp_iso,
        "model": message.model,
        "provider": message.provider,
        "tags": tags,
    }
    return BaseEnvelope(
        id=env_id,
        correlation_id=env_id,
        kind=CHAT_HISTORY_MESSAGE_KIND,
        source=_service_ref(node_name),
        payload=payload,
    )


async def publish_import(
    bus: OrionBusAsync,
    envelopes: Iterable[tuple[str, BaseEnvelope]],
    rate_limit: float,
    dry_run: bool,
    counters: Dict[str, int],
) -> None:
    sleep_sec = 1.0 / rate_limit if rate_limit > 0 else 0.0
    for channel, env in envelopes:
        counters["envelopes_published"] = counters.get("envelopes_published", 0) + 1
        if not dry_run:
            await bus.publish(channel, env)
            if sleep_sec:
                await asyncio.sleep(sleep_sec)
