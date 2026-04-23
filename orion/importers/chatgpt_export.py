from __future__ import annotations

import asyncio
import hashlib
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
from orion.schemas.chat_gpt_log import (
    CHAT_GPT_CONVERSATION_KIND,
    CHAT_GPT_EXAMPLE_KIND,
    CHAT_GPT_IMPORT_RUN_KIND,
    CHAT_GPT_LOG_TURN_KIND,
    CHAT_GPT_MESSAGE_KIND,
)

CHATGPT_IMPORTER_VERSION = "0.1.0"
CHATGPT_IMPORTER_NAME = "chatgpt-import"
CHATGPT_NAMESPACE = UUID("7c8a8f4a-3ef5-4c8c-8c9a-7781f960a1c3")

_TEMP_DIRS: List[tempfile.TemporaryDirectory[str]] = []


@dataclass(frozen=True)
class ChatMessage:
    node_id: str
    source_message_id: Optional[str]
    parent_message_id: Optional[str]
    child_message_ids: list[str]
    role: str
    content: str
    content_type: Optional[str]
    content_blocks: list[Dict[str, Any]]
    attachments: list[Dict[str, Any]]
    metadata: Dict[str, Any]
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
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts: List[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    text = "".join(text_parts).strip()
    return text or None


def _fallback_text_from_blocks(blocks: list[Dict[str, Any]]) -> Optional[str]:
    snippets: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for key in ("text", "caption", "title", "alt", "name"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                snippets.append(value.strip())
                break
    if snippets:
        return "\n".join(snippets)
    if blocks:
        return "[non_text_content]"
    return None


def _extract_content_blocks(message: Dict[str, Any]) -> list[Dict[str, Any]]:
    content = message.get("content") or {}
    content_type = content.get("content_type")
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    blocks: list[Dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if isinstance(part, str):
            blocks.append({"index": idx, "type": "text", "text": part})
            continue
        if isinstance(part, dict):
            block = {"index": idx, **part}
            if "type" not in block and isinstance(content_type, str):
                block["type"] = content_type
            blocks.append(block)
    return blocks


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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_import_run_id(
    *,
    artifact_sha256: str,
    include_branches: bool,
    include_system: bool,
    force_full: bool,
    limit: int | None,
) -> str:
    return str(
        uuid5(
            CHATGPT_NAMESPACE,
            f"run:{artifact_sha256}:{include_branches}:{include_system}:{force_full}:{limit}",
        )
    )


def iter_messages(conv: Dict[str, Any], include_branches: bool) -> Iterator[ChatMessage]:
    mapping = conv.get("mapping") or {}
    if not isinstance(mapping, dict):
        return iter(())

    def build_message(node_id: str, node: Dict[str, Any]) -> Optional[ChatMessage]:
        message = node.get("message")
        if not isinstance(message, dict):
            return None
        content_blocks = _extract_content_blocks(message)
        content = _extract_text(message) or _fallback_text_from_blocks(content_blocks)
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
        attachments = message.get("attachments")
        if not isinstance(attachments, list):
            attachments = []
        child_nodes = node.get("children")
        if not isinstance(child_nodes, list):
            child_nodes = []
        child_message_ids = [str(item) for item in child_nodes if isinstance(item, str)]
        content_raw = message.get("content") if isinstance(message.get("content"), dict) else {}
        return ChatMessage(
            node_id=node_id,
            source_message_id=str(message.get("id")) if message.get("id") else None,
            parent_message_id=node.get("parent") if isinstance(node.get("parent"), str) else None,
            child_message_ids=child_message_ids,
            role=role,
            content=content,
            content_type=content_raw.get("content_type") if isinstance(content_raw.get("content_type"), str) else None,
            content_blocks=content_blocks,
            attachments=[item for item in attachments if isinstance(item, dict)],
            metadata=metadata if isinstance(metadata, dict) else {},
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
        "source_message_id": turn.user_msg.source_message_id,
        "parent_message_id": turn.user_msg.parent_message_id,
        "child_message_ids": turn.user_msg.child_message_ids,
        "role": "user",
        "speaker": user_speaker,
        "content": turn.user_msg.content,
        "content_type": turn.user_msg.content_type,
        "content_blocks": turn.user_msg.content_blocks,
        "attachments": turn.user_msg.attachments,
        "timestamp": turn.user_msg.timestamp_iso,
        "model": turn.user_msg.model,
        "provider": turn.user_msg.provider,
        "shared_conversation_id": conversation_id,
        "metadata": {
            "source": "chatgpt_export",
            "node_id": turn.user_msg.node_id,
            "message_metadata": turn.user_msg.metadata,
        },
        "tags": tags,
    }
    assistant_payload = {
        "message_id": str(assistant_env_id),
        "session_id": session_id,
        "source_message_id": turn.assistant_msg.source_message_id,
        "parent_message_id": turn.assistant_msg.parent_message_id,
        "child_message_ids": turn.assistant_msg.child_message_ids,
        "role": "assistant",
        "speaker": assistant_speaker,
        "content": turn.assistant_msg.content,
        "content_type": turn.assistant_msg.content_type,
        "content_blocks": turn.assistant_msg.content_blocks,
        "attachments": turn.assistant_msg.attachments,
        "timestamp": turn.assistant_msg.timestamp_iso,
        "model": turn.assistant_msg.model,
        "provider": turn.assistant_msg.provider,
        "shared_conversation_id": conversation_id,
        "metadata": {
            "source": "chatgpt_export",
            "node_id": turn.assistant_msg.node_id,
            "message_metadata": turn.assistant_msg.metadata,
        },
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
        kind=CHAT_GPT_MESSAGE_KIND,
        source=source,
        payload=user_payload,
    )
    assistant_env = BaseEnvelope(
        id=assistant_env_id,
        correlation_id=correlation_uuid,
        kind=CHAT_GPT_MESSAGE_KIND,
        source=source,
        payload=assistant_payload,
    )
    turn_env = BaseEnvelope(
        id=correlation_uuid,
        correlation_id=correlation_uuid,
        kind=CHAT_GPT_LOG_TURN_KIND,
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
        "source_message_id": message.source_message_id,
        "parent_message_id": message.parent_message_id,
        "child_message_ids": message.child_message_ids,
        "role": message.role,
        "speaker": speaker,
        "content": message.content,
        "content_type": message.content_type,
        "content_blocks": message.content_blocks,
        "attachments": message.attachments,
        "timestamp": message.timestamp_iso,
        "model": message.model,
        "provider": message.provider,
        "shared_conversation_id": conversation_id,
        "metadata": {
            "source": "chatgpt_export",
            "node_id": message.node_id,
            "message_metadata": message.metadata,
        },
        "tags": tags,
    }
    return BaseEnvelope(
        id=env_id,
        correlation_id=env_id,
        kind=CHAT_GPT_MESSAGE_KIND,
        source=_service_ref(node_name),
        payload=payload,
    )


def build_import_run_envelope(
    *,
    import_run_id: str,
    source_artifact_path: str,
    source_artifact_sha256: str,
    source_artifact_bytes: int,
    source_artifact_mtime: str | None,
    include_branches: bool,
    include_system: bool,
    force_full: bool,
    dry_run: bool,
    state_file: str | None,
    counters: Dict[str, int],
    started_at: str,
    completed_at: str,
    metadata: Dict[str, Any] | None = None,
    node_name: Optional[str] = None,
) -> BaseEnvelope:
    payload = {
        "import_run_id": import_run_id,
        "source_artifact_path": source_artifact_path,
        "source_artifact_sha256": source_artifact_sha256,
        "source_artifact_bytes": source_artifact_bytes,
        "source_artifact_mtime": source_artifact_mtime,
        "importer_name": CHATGPT_IMPORTER_NAME,
        "importer_version": CHATGPT_IMPORTER_VERSION,
        "import_mode": "full" if force_full else "incremental",
        "include_branches": include_branches,
        "include_system": include_system,
        "force_full": force_full,
        "dry_run": dry_run,
        "state_file": state_file,
        "conversation_count": counters.get("conversations_processed", 0),
        "message_count": counters.get("messages_published", 0),
        "turn_count": counters.get("turns_published", 0),
        "example_count": counters.get("examples_published", 0),
        "started_at": started_at,
        "completed_at": completed_at,
        "metadata": metadata or {},
    }
    return BaseEnvelope(
        id=uuid5(CHATGPT_NAMESPACE, f"import_run:{import_run_id}"),
        correlation_id=uuid5(CHATGPT_NAMESPACE, f"import_run:{import_run_id}"),
        kind=CHAT_GPT_IMPORT_RUN_KIND,
        source=_service_ref(node_name),
        payload=payload,
    )


def build_conversation_envelope(
    *,
    import_run_id: str,
    conversation: Dict[str, Any],
    message_count: int,
    turn_count: int,
    include_branches: bool,
    node_name: Optional[str] = None,
) -> BaseEnvelope:
    conversation_id = str(conversation.get("id") or "")
    mapping = conversation.get("mapping") if isinstance(conversation.get("mapping"), dict) else {}
    branch_count = len(mapping)
    payload = {
        "conversation_id": conversation_id,
        "import_run_id": import_run_id,
        "session_id": f"chatgpt:{conversation_id}" if conversation_id else None,
        "title": conversation.get("title"),
        "create_time": conversation.get("create_time"),
        "update_time": conversation.get("update_time"),
        "current_node_id": conversation.get("current_node"),
        "message_count": message_count,
        "turn_count": turn_count,
        "branch_count": branch_count if include_branches else max(branch_count, 1 if branch_count else 0),
        "metadata": {
            "source": "chatgpt_export",
            "conversation_ids": conversation.get("conversation_ids"),
            "safe_urls": conversation.get("safe_urls"),
        },
    }
    env_id = uuid5(CHATGPT_NAMESPACE, f"conversation:{import_run_id}:{conversation_id}")
    return BaseEnvelope(
        id=env_id,
        correlation_id=env_id,
        kind=CHAT_GPT_CONVERSATION_KIND,
        source=_service_ref(node_name),
        payload=payload,
    )


def build_example_envelope(
    *,
    import_run_id: str,
    conversation_id: str,
    conversation_title: str,
    turn: ChatTurn,
    node_name: Optional[str] = None,
) -> BaseEnvelope:
    example_id = str(
        uuid5(
            CHATGPT_NAMESPACE,
            f"example:{import_run_id}:{conversation_id}:{turn.user_msg.node_id}:{turn.assistant_msg.node_id}",
        )
    )
    payload = {
        "example_id": example_id,
        "import_run_id": import_run_id,
        "conversation_id": conversation_id,
        "session_id": f"chatgpt:{conversation_id}",
        "user_message_id": turn.user_msg.source_message_id or turn.user_msg.node_id,
        "assistant_message_id": turn.assistant_msg.source_message_id or turn.assistant_msg.node_id,
        "turn_id": str(
            uuid5(
                CHATGPT_NAMESPACE,
                f"{conversation_id}:{turn.user_msg.node_id}:{turn.assistant_msg.node_id}:turn",
            )
        ),
        "prompt": turn.user_msg.content,
        "response": turn.assistant_msg.content,
        "prompt_timestamp": turn.user_msg.timestamp_iso,
        "response_timestamp": turn.assistant_msg.timestamp_iso,
        "model": turn.assistant_msg.model,
        "provider": turn.assistant_msg.provider,
        "tags": [
            "source:chatgpt_export",
            f"conv:{conversation_id}",
            f"title:{_slugify(conversation_title or 'untitled')}",
        ],
        "metadata": {
            "source": "chatgpt_export",
            "user_node_id": turn.user_msg.node_id,
            "assistant_node_id": turn.assistant_msg.node_id,
            "assistant_metadata": turn.assistant_msg.metadata,
        },
    }
    env_id = uuid5(CHATGPT_NAMESPACE, f"example_envelope:{example_id}")
    return BaseEnvelope(
        id=env_id,
        correlation_id=env_id,
        kind=CHAT_GPT_EXAMPLE_KIND,
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
