from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Determine repo root
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_env_file(SCRIPTS_DIR / ".env")
_load_env_file(SCRIPTS_DIR / ".env_example")

if str(SCRIPTS_DIR) in sys.path:
    sys.path.remove(str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vector.schemas import EmbeddingGenerateV1
from orion.importers.chatgpt_export import (
    build_conversation_envelope,
    build_example_envelope,
    build_envelopes_for_turn,
    build_import_run_id,
    build_import_run_envelope,
    build_message_envelope,
    file_sha256,
    iter_messages,
    load_conversations,
    pair_turns,
    publish_import,
    resolve_export_path,
    CHATGPT_IMPORTER_NAME,
)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import ChatGPT export via Orion bus fanout.")
    parser.add_argument("--export", required=True, help="Path to chatgpt export (zip, dir, conversations.json)")
    parser.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--channel-log", default="orion:chat:gpt:message:log")
    parser.add_argument("--channel-turn", default="orion:chat:gpt:turn")
    parser.add_argument("--channel-import-run", default="orion:chat:gpt:import:run")
    parser.add_argument("--channel-conversation", default="orion:chat:gpt:conversation")
    parser.add_argument("--channel-example", default="orion:chat:gpt:example")
    parser.add_argument("--user-id", default="Juniper")
    parser.add_argument("--user-speaker", default="~Juniper")
    parser.add_argument("--assistant-speaker", default="ChatGPT")
    parser.add_argument("--include-branches", action="store_true")
    parser.add_argument("--include-system", action="store_true")
    parser.add_argument("--only-messages", action="store_true", help="Publish message events only.")
    parser.add_argument("--only-turns", action="store_true", help="Publish turn events only.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rate-limit", type=float, default=50.0)
    parser.add_argument("--state-file", default="scripts/.state/chatgpt_import.json")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--emit-embeddings", action="store_true")
    parser.add_argument("--embedding-channel", default="orion:embedding:generate")
    parser.add_argument("--message-collection", default="orion_chat_gpt")
    parser.add_argument("--turn-collection", default="orion_chat_gpt_turns")
    return parser.parse_args(argv)




def _validate_channel_schemas(
    channel_log: str,
    channel_turn: str,
    channel_import_run: str,
    channel_conversation: str,
    channel_example: str,
    *,
    embedding_channel: str | None = None,
) -> None:
    channels_file = ROOT / "orion" / "bus" / "channels.yaml"
    if not channels_file.exists():
        raise ValueError(f"Missing channels catalog: {channels_file}")

    schema_by_name: Dict[str, str] = {}
    current_name: str | None = None
    for raw_line in channels_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith('- name:') or line.startswith('name:'):
            current_name = line.split(':', 1)[1].strip().strip('"')
            continue
        if current_name and line.startswith('schema_id:'):
            schema_by_name[current_name] = line.split(':', 1)[1].strip().strip('"')
            current_name = None

    expected = {
        channel_log: "ChatGptMessageV1",
        channel_turn: "ChatGptLogTurnV1",
        channel_import_run: "ChatGptImportRunV1",
        channel_conversation: "ChatGptConversationV1",
        channel_example: "ChatGptDerivedExampleV1",
    }
    if embedding_channel:
        expected[embedding_channel] = "EmbeddingGenerateV1"
    for channel, expected_schema in expected.items():
        actual = schema_by_name.get(channel)
        if actual != expected_schema:
            raise ValueError(
                f"Channel schema mismatch for {channel}: expected {expected_schema}, found {actual or 'missing'}."
            )


def _embedding_envelope(
    *,
    doc_id: str,
    text: str,
    collection: str,
    correlation_id: Any,
    metadata: Dict[str, Any],
) -> BaseEnvelope:
    payload = EmbeddingGenerateV1(
        doc_id=doc_id,
        text=text,
        collection=collection,
        include_latent=False,
    ).model_dump(mode="json")
    payload["metadata"] = metadata
    return BaseEnvelope(
        kind="embedding.generate.v1",
        source=ServiceRef(
            name=CHATGPT_IMPORTER_NAME,
            version="0.1.0",
            node=os.getenv("NODE_NAME", "importer"),
        ),
        correlation_id=correlation_id,
        payload=payload,
    )


def _build_embedding_envelopes(
    envelopes: Iterable[Tuple[str, BaseEnvelope]],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, BaseEnvelope]], Dict[str, int]]:
    embed_envelopes: List[Tuple[str, BaseEnvelope]] = []
    counters = {"embedding_requests_published": 0}

    for channel, env in envelopes:
        payload = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
        if not isinstance(payload, dict):
            continue

        if channel == args.channel_log and payload.get("message_id") and payload.get("content"):
            doc_id = str(payload.get("message_id"))
            metadata = {
                "role": payload.get("role"),
                "session_id": payload.get("session_id"),
                "timestamp": payload.get("timestamp"),
                "tags": payload.get("tags") or [],
                "original_channel": args.channel_log,
                "source_service": CHATGPT_IMPORTER_NAME,
            }
            embed_env = _embedding_envelope(
                doc_id=doc_id,
                text=str(payload.get("content")),
                collection=args.message_collection,
                correlation_id=env.correlation_id,
                metadata=metadata,
            )
            embed_envelopes.append((args.embedding_channel, embed_env))
            counters["embedding_requests_published"] += 1

        if channel == args.channel_turn and (payload.get("prompt") or payload.get("response")):
            doc_id = str(payload.get("correlation_id") or payload.get("id"))
            spark_meta = payload.get("spark_meta") if isinstance(payload.get("spark_meta"), dict) else {}
            metadata = {
                "session_id": payload.get("session_id"),
                "timestamp": payload.get("timestamp") or payload.get("created_at"),
                "source_conversation_id": spark_meta.get("source_conversation_id"),
                "title": spark_meta.get("conversation_title"),
                "original_channel": args.channel_turn,
                "source_service": CHATGPT_IMPORTER_NAME,
            }
            text = f"{payload.get('prompt') or ''}\n\n{payload.get('response') or ''}".strip()
            embed_env = _embedding_envelope(
                doc_id=doc_id,
                text=text,
                collection=args.turn_collection,
                correlation_id=env.correlation_id,
                metadata=metadata,
            )
            embed_envelopes.append((args.embedding_channel, embed_env))
            counters["embedding_requests_published"] += 1

    return embed_envelopes, counters

def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "last_run_utc": None,
            "per_conversation_update_time": {},
            "per_conversation_max_message_time": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if "per_conversation_update_time" not in state:
        state["per_conversation_update_time"] = {}
    if "per_conversation_max_message_time" not in state:
        state["per_conversation_max_message_time"] = {}
    if "max_conversation_update_time" in state:
        legacy = state.get("max_conversation_update_time")
        state.setdefault("per_conversation_update_time", {})
        state["per_conversation_update_time"].setdefault("__legacy_global__", legacy)
    return state


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _message_allowed(role: str, include_system: bool) -> bool:
    if role in ("user", "assistant"):
        return True
    if role == "system" and include_system:
        return True
    return False


def _build_system_envelope(
    conversation_id: str,
    message: Any,
    conversation_title: str,
    include_system: bool,
) -> BaseEnvelope | None:
    if not include_system or message.role != "system":
        return None
    return build_message_envelope(
        conversation_id=conversation_id,
        conversation_title=conversation_title,
        message=message,
        speaker="System",
    )


def _build_publish_envelopes(
    *,
    conversation: Dict[str, Any],
    messages: Iterable[Any],
    turns: Iterable[Any],
    publish_messages: bool,
    publish_turns: bool,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, BaseEnvelope]], Dict[str, int], List[BaseEnvelope]]:
    conversation_id = conversation.get("id")
    title = conversation.get("title") or "Untitled"
    update_time = conversation.get("update_time")
    create_time = conversation.get("create_time")
    branch_mode = "all" if args.include_branches else "current"

    envelopes: List[Tuple[str, BaseEnvelope]] = []
    samples: List[BaseEnvelope] = []
    counters: Dict[str, int] = {
        "messages_published": 0,
        "turns_published": 0,
    }

    use_turn_messages = publish_messages and not args.include_branches
    if publish_turns or use_turn_messages:
        for turn in turns:
            user_env, assistant_env, turn_env = build_envelopes_for_turn(
                conversation_id=conversation_id,
                conversation_title=title,
                conversation_update_time=update_time,
                conversation_create_time=create_time,
                user_id=args.user_id,
                user_speaker=args.user_speaker,
                assistant_speaker=args.assistant_speaker,
                turn=turn,
                branch_mode=branch_mode,
            )
            if use_turn_messages:
                envelopes.extend(
                    [
                        (args.channel_log, user_env),
                        (args.channel_log, assistant_env),
                    ]
                )
                counters["messages_published"] += 2
                if not samples:
                    samples.extend([user_env, assistant_env])
            if publish_turns:
                envelopes.append((args.channel_turn, turn_env))
                counters["turns_published"] += 1
                if use_turn_messages:
                    if len(samples) < 3:
                        samples.append(turn_env)
                elif not samples:
                    samples.append(turn_env)

    if publish_messages and not use_turn_messages:
        for message in messages:
            if message.role not in ("user", "assistant", "system"):
                continue
            if message.role in ("user", "assistant"):
                speaker = args.user_speaker if message.role == "user" else args.assistant_speaker
                env = build_message_envelope(
                    conversation_id=conversation_id,
                    conversation_title=title,
                    message=message,
                    speaker=speaker,
                )
                envelopes.append((args.channel_log, env))
                counters["messages_published"] += 1
                if len(samples) < 3:
                    samples.append(env)
                continue
            system_env = _build_system_envelope(
                conversation_id,
                message,
                title,
                args.include_system,
            )
            if system_env:
                envelopes.append((args.channel_log, system_env))
                counters["messages_published"] += 1
                if len(samples) < 3:
                    samples.append(system_env)

    if publish_messages and use_turn_messages and args.include_system:
        for message in messages:
            if message.role != "system":
                continue
            system_env = _build_system_envelope(
                conversation_id,
                message,
                title,
                args.include_system,
            )
            if system_env:
                envelopes.append((args.channel_log, system_env))
                counters["messages_published"] += 1
                if len(samples) < 3:
                    samples.append(system_env)

    return envelopes, counters, samples


async def _run(args: argparse.Namespace) -> int:
    if args.only_messages and args.only_turns:
        raise ValueError("Choose only one of --only-messages or --only-turns.")
    _validate_channel_schemas(
        args.channel_log,
        args.channel_turn,
        args.channel_import_run,
        args.channel_conversation,
        args.channel_example,
        embedding_channel=args.embedding_channel if args.emit_embeddings else None,
    )
    if args.include_branches and args.only_turns:
        raise ValueError("--only-turns is not supported with --include-branches.")
    publish_messages = not args.only_turns
    publish_turns = not args.only_messages
    if args.include_branches and publish_turns:
        print("warning: include-branches requested; defaulting to messages-only to avoid cross-branch turns.")
        publish_turns = False

    export_path = resolve_export_path(args.export)
    conversations = load_conversations(export_path)
    if args.limit:
        conversations = conversations[: args.limit]
    artifact_stat = export_path.stat()
    artifact_sha = file_sha256(export_path)
    import_run_id = build_import_run_id(
        artifact_sha256=artifact_sha,
        include_branches=args.include_branches,
        include_system=args.include_system,
        force_full=args.force_full,
        limit=args.limit,
    )

    state_path = (ROOT / args.state_file).resolve()
    state = _load_state(state_path)
    force_full = args.force_full
    per_conv_update = state.get("per_conversation_update_time", {})
    per_conv_max = state.get("per_conversation_max_message_time", {})

    started_at = datetime.utcnow().isoformat()
    counters: Dict[str, int] = {
        "conversations_total": len(conversations),
        "conversations_processed": 0,
        "messages_extracted": 0,
        "messages_published": 0,
        "turns_emitted": 0,
        "turns_published": 0,
        "examples_published": 0,
        "embedding_requests_published": 0,
        "skipped_by_state_conversations": 0,
        "skipped_by_state_messages": 0,
    }
    sample_envs: List[BaseEnvelope] = []

    bus = OrionBusAsync(url=args.bus_url)
    if not args.dry_run:
        await bus.connect()

    try:
        for conversation in conversations:
            conversation_id = conversation.get("id")
            update_time = conversation.get("update_time") or 0
            prev_update = per_conv_update.get(str(conversation_id))
            if not force_full and prev_update is not None and update_time <= prev_update:
                counters["skipped_by_state_conversations"] += 1
                continue

            raw_messages = list(iter_messages(conversation, include_branches=args.include_branches))
            filtered_messages = [
                msg for msg in raw_messages if _message_allowed(msg.role, args.include_system)
            ]
            max_seen_message_time = per_conv_max.get(conversation_id, None)
            eligible_messages = []
            for msg in filtered_messages:
                counters["messages_extracted"] += 1
                if not force_full and max_seen_message_time is not None and msg.create_time <= max_seen_message_time:
                    counters["skipped_by_state_messages"] += 1
                    continue
                eligible_messages.append(msg)

            turns: List[Any] = []
            if publish_turns or (publish_messages and not args.include_branches):
                turns = pair_turns(msg for msg in eligible_messages if msg.role in ("user", "assistant"))
            counters["turns_emitted"] += len(turns)

            envelopes, publish_counts, samples = _build_publish_envelopes(
                conversation=conversation,
                messages=eligible_messages,
                turns=turns,
                publish_messages=publish_messages,
                publish_turns=publish_turns,
                args=args,
            )
            counters["messages_published"] += publish_counts["messages_published"]
            counters["turns_published"] += publish_counts["turns_published"]
            convo_env = build_conversation_envelope(
                import_run_id=import_run_id,
                conversation=conversation,
                message_count=publish_counts["messages_published"],
                turn_count=publish_counts["turns_published"],
                include_branches=args.include_branches,
            )
            envelopes.append((args.channel_conversation, convo_env))
            for turn in turns:
                example_env = build_example_envelope(
                    import_run_id=import_run_id,
                    conversation_id=str(conversation.get("id") or ""),
                    conversation_title=str(conversation.get("title") or "Untitled"),
                    turn=turn,
                )
                envelopes.append((args.channel_example, example_env))
                counters["examples_published"] += 1

            if args.emit_embeddings:
                embedding_envelopes, embedding_counts = _build_embedding_envelopes(envelopes, args)
                envelopes.extend(embedding_envelopes)
                counters["embedding_requests_published"] += embedding_counts["embedding_requests_published"]

            if samples and not sample_envs:
                sample_envs = samples

            await publish_import(bus, envelopes, args.rate_limit, args.dry_run, counters)
            counters["conversations_processed"] += 1

            if not force_full:
                per_conv_update[str(conversation_id)] = update_time
                max_msg_time = max((msg.create_time for msg in eligible_messages), default=None)
                if max_msg_time is not None:
                    per_conv_max[str(conversation_id)] = max_msg_time

        completed_at = datetime.utcnow().isoformat()
        run_env = build_import_run_envelope(
            import_run_id=import_run_id,
            source_artifact_path=str(export_path),
            source_artifact_sha256=artifact_sha,
            source_artifact_bytes=artifact_stat.st_size,
            source_artifact_mtime=datetime.utcfromtimestamp(artifact_stat.st_mtime).isoformat(),
            include_branches=args.include_branches,
            include_system=args.include_system,
            force_full=args.force_full,
            dry_run=args.dry_run,
            state_file=args.state_file,
            counters=counters,
            started_at=started_at,
            completed_at=completed_at,
            metadata={"bus_url": args.bus_url},
        )
        await publish_import(
            bus,
            [(args.channel_import_run, run_env)],
            args.rate_limit,
            args.dry_run,
            counters,
        )
    finally:
        if not args.dry_run:
            await bus.close()

    state["last_run_utc"] = datetime.utcnow().isoformat()
    state["per_conversation_update_time"] = per_conv_update
    state["per_conversation_max_message_time"] = per_conv_max

    if not args.dry_run:
        _write_state(state_path, state)

    print(f"conversations_total={counters['conversations_total']}")
    print(f"conversations_processed={counters['conversations_processed']}")
    print(f"messages_extracted={counters['messages_extracted']}")
    print(f"messages_published={counters['messages_published']}")
    print(f"turns_emitted={counters['turns_emitted']}")
    print(f"turns_published={counters['turns_published']}")
    print(f"examples_published={counters['examples_published']}")
    print(f"embedding_requests_published={counters['embedding_requests_published']}")
    print(f"skipped_by_state_conversations={counters['skipped_by_state_conversations']}")
    print(f"skipped_by_state_messages={counters['skipped_by_state_messages']}")

    if args.dry_run:
        print("dry_run_sample_envelopes=")
        for env in sample_envs[:3]:
            print(json.dumps(env.model_dump(mode="json"), indent=2))

    return 0


def main() -> int:
    args = _parse_args(sys.argv[1:])
    print(f"{CHATGPT_IMPORTER_NAME}: starting import")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
