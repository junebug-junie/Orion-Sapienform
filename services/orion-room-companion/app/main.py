from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.room_claude import (
    ExternalRoomResponderV1,
    RoomClaudeRequestV1,
    RoomClaudeUtteranceV1,
)

from .claude_session import ClaudeTurnResult, run_room_turn
from .room_prompt import AUTO_INVITE_CLAUSE, SYSTEM_PROMPT, build_turn_prompt, filtered_summary, is_pass
from .session_store import forget_session, peek_or_mint_session, remember_session
from .settings import Settings, get_settings

logger = logging.getLogger("orion-room-companion")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[ROOM_COMPANION] %(asctime)s %(levelname)s - %(name)s - %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_heartbeat_chassis(settings: Settings) -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.ROOM_COMPANION_NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


# Env the subprocess is allowed to inherit. An ALLOWLIST, not a denylist:
# a denylist has to enumerate every way Claude Code can be redirected to a
# different provider or credential, and the first draft of this function
# already missed CLAUDE_CODE_OAUTH_TOKEN -- which is exactly the variable
# that, if it arrived from ambient os.environ rather than deliberately, would
# authenticate as someone else's credential entirely -- plus the
# Bedrock/Vertex switches and ANTHROPIC_MODEL. Anything genuinely needed gets
# added here deliberately; nothing arrives by accident. As of 2026-08-18 the
# ONE deliberate exception is CLAUDE_CODE_OAUTH_TOKEN itself: build_subprocess_env
# injects it explicitly from settings.ROOM_COMPANION_CLAUDE_OAUTH_TOKEN below,
# the same way it already does for CLAUDE_CONFIG_DIR -- never through this
# allowlist, so _ENV_DENY_PREFIXES still blocks any *ambient* CLAUDE_CODE_OAUTH_TOKEN
# that isn't the one this service was deliberately configured with.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM",
        "HOSTNAME", "PWD", "SHELL", "USER", "LOGNAME",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
        # Claude Code's own root-sandbox gate; without it the CLI refuses to
        # start under bypassPermissions as root. See
        # orion/fcc/claude_spawn.py::claude_permission_argv.
        "IS_SANDBOX", "CLAUDE_CODE_BUBBLEWRAP",
    }
)

# Names that must never reach the subprocess even if someone adds them to the
# allowlist by mistake. Belt to the allowlist's braces, and the thing the
# credential-isolation tests assert on directly.
_ENV_DENY_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_OAUTH", "AWS_", "GOOGLE_", "GCP_")


def build_subprocess_env(settings: Settings) -> Dict[str, str]:
    """Env for the `claude` subprocess, built from an allowlist.

    This is the load-bearing part, not hygiene. orion-hub's FCC lane sets
    ANTHROPIC_BASE_URL to a local gateway and ANTHROPIC_AUTH_TOKEN alongside
    it (services/orion-hub/scripts/fcc_claude_bridge.py), which turns `claude`
    into a harness driving LOCAL models. If either reached this subprocess the
    room would still produce fluent text -- it just would not be Claude, which
    is the hardest failure here to notice by eye. ANTHROPIC_API_KEY would open
    a second pay-per-token billing relationship instead of reusing the
    operator's subscription. An *ambient* CLAUDE_CODE_OAUTH_TOKEN (leaked from
    the host shell, a copy-paste mistake) would authenticate as someone else's
    credential entirely -- which is why it is still on _ENV_DENY_PREFIXES and
    never read out of os.environ here.

    The real, deliberately-configured token is injected below from
    settings.ROOM_COMPANION_CLAUDE_OAUTH_TOKEN -- the same explicit-injection
    pattern already used for CLAUDE_CONFIG_DIR, not the ambient allowlist path.
    """
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    env = {
        k: v for k, v in env.items()
        if not any(k.startswith(prefix) for prefix in _ENV_DENY_PREFIXES)
    }
    env["CLAUDE_CONFIG_DIR"] = settings.ROOM_COMPANION_CLAUDE_CONFIG_DIR
    if settings.ROOM_COMPANION_CLAUDE_OAUTH_TOKEN:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = settings.ROOM_COMPANION_CLAUDE_OAUTH_TOKEN
    return env


def room_key(request: RoomClaudeRequestV1) -> str:
    return f"{request.platform}:{request.room_id}"


def ensure_runtime_dirs(settings: Settings) -> None:
    """Create the workspace before spawning into it.

    Cannot be done in the Dockerfile: the workspace lives under /data, which
    is a volume mount, so any image-time mkdir is masked at runtime. Caught by
    the live smoke, which failed every turn with
    `FileNotFoundError: '/data/workspace'` -- subprocess cwd is resolved by
    the OS, so a missing dir kills the turn before `claude` ever starts.

    Called per turn rather than only at startup so the smoke (which drives
    run_turn directly, without the consumer loop) works the same way.
    """
    Path(settings.ROOM_COMPANION_WORKSPACE).mkdir(parents=True, exist_ok=True)


def run_turn(settings: Settings, request: RoomClaudeRequestV1) -> RoomClaudeUtteranceV1:
    """One Claude turn, always producing an utterance -- including on failure.

    A failed turn publishes ok=False with the error text rather than staying
    silent. A room whose token expired must say so out loud; a silent skip is
    indistinguishable from Claude choosing not to speak, which is exactly the
    confusion that makes an outage invisible.
    """
    ensure_runtime_dirs(settings)
    key = room_key(request)
    session_id, resume = peek_or_mint_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key)

    prompt = build_turn_prompt(
        prompt=request.prompt,
        invited_by_name=request.invited_by,
        transcript=request.transcript,
        # filtered_summary(), not the raw dict. This is the privacy boundary:
        # the summary is third-party-bound, so it goes through the allowlist
        # rather than whatever orion-social-memory happens to include today.
        social_memory_summary=filtered_summary(request.social_memory_summary),
        first_turn=not resume,
    )

    result = run_room_turn(
        prompt,
        claude_bin=settings.ROOM_COMPANION_CLAUDE_BIN,
        model=settings.ROOM_COMPANION_MODEL,
        effort=settings.ROOM_COMPANION_EFFORT,
        setting_sources_env_key=settings.ROOM_COMPANION_SETTING_SOURCES_ENV_KEY,
        session_id=session_id,
        resume=resume,
        timeout_sec=settings.ROOM_COMPANION_TIMEOUT_SEC,
        # EVERY turn, not just the first. An earlier version passed this only
        # when minting a session, on the assumption that `--resume` carried the
        # framing forward. It does not: verified live in this container with a
        # marker instruction, a resumed turn WITHOUT the flag reverted to
        # Claude Code's default CLI persona ("Goodbye! Let me know if you need
        # anything else.") -- the assistant-taking-orders voice this room is
        # explicitly not for. Re-passing it on resume restores the framing,
        # 3/3 runs.
        #
        # Do NOT "fix" this by moving the framing into the user-role text
        # instead: that was tried and Claude correctly refused it as a
        # prompt-injection pattern ("it's embedded in a user-turn message
        # formatted to look like a system directive"). System framing has to
        # arrive as system framing.
        # Auto-invites additionally get permission to stay quiet. A manual
        # invite is someone actually asking, so passing there would be rude.
        append_system_prompt=(
            SYSTEM_PROMPT + AUTO_INVITE_CLAUSE if request.trigger == "auto" else SYSTEM_PROMPT
        ),
        env=build_subprocess_env(settings),
        cwd=settings.ROOM_COMPANION_WORKSPACE,
    )

    # A resume against a session the CLI no longer holds fails identically on
    # every retry, so the room would be wedged forever. Drop the mapping and
    # let the next turn mint a fresh session, reseeded from the transcript.
    if not result.ok and resume and _looks_like_missing_session(result.error):
        logger.warning("room_companion_session_lost room=%s session=%s -- forgetting", key, session_id)
        forget_session(settings.ROOM_COMPANION_SESSION_STATE_PATH, key)
    elif result.ok:
        # Persist only after the turn that creates the session actually
        # succeeds. Writing at mint time meant a failed first turn left the map
        # pointing at a session the CLI never created, so the NEXT turn had to
        # burn a doomed --resume before self-healing.
        #
        # Record what the CLI reported rather than what was asked for: today
        # --session-id is honored exactly, but nothing would notice if that
        # changed, and a drifted map resumes the wrong conversation.
        remember_session(
            settings.ROOM_COMPANION_SESSION_STATE_PATH,
            key,
            result.claude_session_id or session_id,
        )

    return _to_utterance(settings, request, result)


def _looks_like_missing_session(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    # Parenthesised for readability; `and` already binds tighter than `or`,
    # so this was never a precedence bug -- but the second clause is broad
    # enough to matter, so it should be obvious what it means.
    return ("no conversation found" in lowered) or ("session" in lowered and "not found" in lowered)


def _to_utterance(
    settings: Settings, request: RoomClaudeRequestV1, result: ClaudeTurnResult
) -> RoomClaudeUtteranceV1:
    # A pass is a real, successful, BILLED turn in which Claude chose silence.
    # It keeps ok=True and its true cost_usd -- suppressing the event would
    # hide that spend from the meter, and deciding not to speak costs the same
    # tokens as speaking.
    passed = bool(result.ok and is_pass(result.text))
    return RoomClaudeUtteranceV1(
        request_id=request.request_id,
        correlation_id=request.correlation_id,
        platform=request.platform,
        room_id=request.room_id,
        session_id=request.session_id,
        responder=ExternalRoomResponderV1(
            participant_id=settings.ROOM_COMPANION_PARTICIPANT_ID,
            participant_name=settings.ROOM_COMPANION_PARTICIPANT_NAME,
            participant_kind="peer_ai",
        ),
        text="" if passed else result.text,
        passed=passed,
        claude_session_id=result.claude_session_id,
        model=result.model,
        cost_usd=result.cost_usd,
        duration_ms=result.duration_ms,
        ok=result.ok,
        error=result.error,
    )


async def publish_utterance(
    settings: Settings, bus: OrionBusAsync, utterance: RoomClaudeUtteranceV1
) -> None:
    if bus is None or not getattr(bus, "enabled", False):
        return
    envelope = BaseEnvelope(
        kind="room.claude.utterance.v1",
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.ROOM_COMPANION_NODE_NAME,
        ),
        payload=utterance.model_dump(mode="json"),
    )
    await bus.publish(settings.CHANNEL_ROOM_CLAUDE_UTTERANCE, envelope)


def parse_request(raw: Any) -> Optional[RoomClaudeRequestV1]:
    """Accept either a bare payload or a BaseEnvelope-wrapped one."""
    if not isinstance(raw, dict):
        return None
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
    try:
        return RoomClaudeRequestV1.model_validate(payload)
    except Exception:
        logger.exception("room_companion_bad_request_payload")
        return None


async def handle_message(settings: Settings, bus: OrionBusAsync, raw: Any) -> None:
    request = parse_request(raw)
    if request is None:
        return
    logger.info(
        "room_companion_turn_start room=%s request=%s invited_by=%s",
        request.room_id, request.request_id, request.invited_by,
    )
    # Blocking subprocess work off the event loop so the heartbeat keeps
    # beating and a slow turn does not stall the consumer.
    try:
        utterance = await asyncio.to_thread(run_turn, settings, request)
    except Exception as exc:
        # An unexpected crash must still be audible. Without this, the
        # consumer's blanket handler logs and drops the request, and the room
        # sees silence -- indistinguishable from Claude choosing not to speak,
        # which is exactly what makes an outage invisible.
        logger.exception("room_companion_turn_crashed request=%s", request.request_id)
        utterance = _to_utterance(
            settings,
            request,
            ClaudeTurnResult(
                ok=False, text="", claude_session_id=None, model=None,
                cost_usd=0.0, duration_ms=0, exit_code=-1,
                error=f"room companion crashed: {type(exc).__name__}: {exc}"[:2000],
            ),
        )
    logger.info(
        "room_companion_turn_done room=%s request=%s ok=%s passed=%s model=%s cost_usd=%.6f ms=%d chars=%d%s",
        request.room_id, request.request_id, utterance.ok, utterance.passed, utterance.model,
        utterance.cost_usd, utterance.duration_ms, len(utterance.text),
        f" error={utterance.error}" if utterance.error else "",
    )
    await publish_utterance(settings, bus, utterance)


async def run_consumer(settings: Settings, bus: OrionBusAsync, stop: asyncio.Event) -> None:
    channel = settings.CHANNEL_ROOM_CLAUDE_REQUEST
    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            if stop.is_set():
                break
            try:
                data = msg.get("data")
                raw = json.loads(data if isinstance(data, str) else data.decode("utf-8"))
            except Exception:
                logger.exception("room_companion_bad_envelope")
                continue
            try:
                await handle_message(settings, bus, raw)
            except Exception:
                # One bad turn must not kill the consumer loop.
                logger.exception("room_companion_handle_failed")


async def main_async() -> None:
    setup_logging()
    settings = get_settings()
    stop = asyncio.Event()

    def _handle_signal() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    import signal

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    heartbeat = build_heartbeat_chassis(settings)
    if settings.ORION_BUS_ENABLED:
        await heartbeat.start_background()

    bus = OrionBusAsync(
        settings.ORION_BUS_URL,
        enabled=settings.ORION_BUS_ENABLED,
        enforce_catalog=settings.ORION_BUS_ENFORCE_CATALOG,
    )
    try:
        if settings.ORION_BUS_ENABLED:
            await bus.connect()
            logger.info(
                "room_companion_listening channel=%s participant=%s model=%s",
                settings.CHANNEL_ROOM_CLAUDE_REQUEST,
                settings.ROOM_COMPANION_PARTICIPANT_NAME,
                settings.ROOM_COMPANION_MODEL,
            )
            await run_consumer(settings, bus, stop)
        else:
            logger.warning("orion_bus_disabled_idle")
            await stop.wait()
    finally:
        if settings.ORION_BUS_ENABLED:
            await heartbeat.stop()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
