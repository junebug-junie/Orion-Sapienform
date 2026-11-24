"""HelpRequest consumer — Claude budget → Claude once → persist PeerBrief."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union

from app.brief_prompt import parse_peer_brief_body
from app.claude_peer import (
    DEFAULT_TIMEOUT_SEC,
    claude_reason,
    describe_claude_refusal,
    run_claude_peer,
)
from app.claude_limit_backoff import (
    MEMORY as _CLAUDE_LIMIT_MEMORY,
    ClaudeLimitMemory,
    describe_hold,
    seed_memory_from_graph,
)
from app.settings import Settings
from orion.autonomy.ask_claude_trigger import _budget_refusal
from orion.core.bus.resilience import run_subscription
from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.curiosity.peer_briefs import peer_brief_consume_cypher
from orion.curiosity.agency_episode import commit_ask, offer_or_complete_cypher
from orion.dev_economics.rate_limit_events import LimitObservation, observe
from orion.schemas.curiosity_peer import (
    clip,
    PEER_BRIEF_CONSUMED_KIND,
    HelpRequestV1,
    PeerBriefConsumedV1,
    PeerBriefV1,
)
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomClaudeUtteranceV1

logger = logging.getLogger("orion-curiosity-peer.worker")

HelpInput = Union[HelpRequestV1, str, bytes, dict[str, Any]]
ClaudeFn = Callable[..., Union[str, PeerBriefV1]]
ObserveClaudeFn = Callable[[], Any]
PersistFn = Callable[[PeerBriefV1], Any]

_CLAUDE_CONVERSATION_PREFIX = (
    "[conversation-only; peer could not look at the repo] "
)


def parse_help_request(raw: HelpInput) -> HelpRequestV1:
    if isinstance(raw, HelpRequestV1):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        # Bus envelope: Hub publishes BaseEnvelope with HelpRequest under
        # `payload`; some paths nest under `data`.
        for key in ("payload", "data"):
            inner = raw.get(key)
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    continue
            if isinstance(inner, dict) and "help_id" in inner:
                raw = inner
                break
    return HelpRequestV1.model_validate(raw)


def _new_brief_id() -> str:
    return f"brief-{uuid.uuid4().hex[:12]}"


def _failed_brief(
    help_req: HelpRequestV1,
    *,
    reason: str,
    peer: str = "claude_room",
    summary: str = "",
) -> PeerBriefV1:
    return PeerBriefV1(
        brief_id=_new_brief_id(),
        help_id=help_req.help_id,
        run_id=help_req.run_id,
        prior_id=help_req.prior_id,
        peer=peer,  # type: ignore[arg-type]
        status="failed",
        summary=summary,
        refusal_reason=reason,
    )


def _refused_budget_brief(
    help_req: HelpRequestV1,
    reason: str,
    *,
    peer: str = "claude_room",
) -> PeerBriefV1:
    return PeerBriefV1(
        brief_id=_new_brief_id(),
        help_id=help_req.help_id,
        run_id=help_req.run_id,
        prior_id=help_req.prior_id,
        peer=peer,  # type: ignore[arg-type]
        status="refused_budget",
        summary="",
        refusal_reason=reason,
    )


def context_pack_from_help(help_req: HelpRequestV1) -> str:
    """Minimal context from HelpRequest fields so Claude is not empty-handed."""
    return (
        f"mode: {help_req.mode}\n"
        f"question: {help_req.question}\n"
        f"tried: {help_req.tried_summary}\n"
        f"success_criteria: {help_req.success_criteria}\n"
    )


def _map_claude_result(help_req: HelpRequestV1, result: Union[str, PeerBriefV1]) -> PeerBriefV1:
    """Map Claude room output; strip evidence claims (room tools read Orion state only, never the repo or graph)."""
    if isinstance(result, PeerBriefV1):
        brief = result
        if brief.peer != "claude_room":
            brief = brief.model_copy(update={"peer": "claude_room"})
    else:
        brief = parse_peer_brief_body(str(result), help=help_req, peer="claude_room")

    if brief.status != "ok":
        return brief if brief.peer == "claude_room" else brief.model_copy(
            update={"peer": "claude_room"}
        )

    summary = (brief.summary or "").strip()
    if summary and not summary.lower().startswith("[conversation-only"):
        summary = f"{_CLAUDE_CONVERSATION_PREFIX}{summary}"
    # Never emit evidence: lines for claude_room unless pointers are explicitly empty.
    return brief.model_copy(
        update={
            "peer": "claude_room",
            "summary": summary,
            "evidence_pointers": [],
        }
    )


def _graph_credentials_ready(settings: Settings) -> bool:
    host = (settings.ORION_CURIOSITY_GRAPH_HOST or "").strip()
    port = (settings.ORION_CURIOSITY_GRAPH_PORT or "").strip()
    user = (settings.ORION_CURIOSITY_GRAPH_USER or "").strip()
    secret = settings.ORION_CURIOSITY_GRAPH_PASSWORD
    password = secret.get_secret_value() if secret is not None else ""
    return bool(host and port and user and str(password).strip())


def _curiosity_graph_uri(settings: Settings) -> str:
    from urllib.parse import quote

    host = settings.ORION_CURIOSITY_GRAPH_HOST.strip()
    port = settings.ORION_CURIOSITY_GRAPH_PORT.strip()
    user = quote(settings.ORION_CURIOSITY_GRAPH_USER.strip(), safe="")
    secret = settings.ORION_CURIOSITY_GRAPH_PASSWORD
    password = quote(
        secret.get_secret_value() if secret is not None else "",
        safe="",
    )
    return f"redis://{user}:{password}@{host}:{port}/0"


def _build_curiosity_graph_client(settings: Settings) -> Any:
    from orion.graph.falkor_client import RedisGraphQueryClient

    return RedisGraphQueryClient(
        uri=_curiosity_graph_uri(settings),
        graph_name=(settings.ORION_CURIOSITY_GRAPH_OWN or "orion_worldview").strip()
        or "orion_worldview",
        read_only=False,
    )


def _unwrap_room_claude_utterance(body: Any) -> Optional[RoomClaudeUtteranceV1]:
    """Unwrap a Titanium envelope the same way Hub's room_claude_relay does.

    Redis pubsub ``msg['data']`` is the JSON envelope; the utterance lives under
    ``payload``, not ``data``.
    """
    if not isinstance(body, dict):
        return None
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else body
    if not isinstance(payload, dict):
        return None
    try:
        return RoomClaudeUtteranceV1.model_validate(payload)
    except Exception:
        return None


def _run_coro_threadsafe(
    coro: Any,
    loop: Optional[asyncio.AbstractEventLoop],
    *,
    timeout: float,
) -> Any:
    """Await a coroutine from a worker thread without asyncio.run on a foreign loop.

    OrionBusAsync is bound to the consumer loop. Calling ``asyncio.run`` from a
    worker thread on that client's awaitables is a silent landmine. Prefer
    ``run_coroutine_threadsafe`` when a running loop was captured; fall back to
    ``asyncio.run`` only when no loop is running (unit tests).
    """
    if loop is not None and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)
    return asyncio.run(coro)


def _default_persist(
    brief: PeerBriefV1,
    *,
    settings: Settings,
    bus: Any,
    graph_client: Any = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    def _graph_execute(cypher: str, params: Optional[dict[str, Any]] = None) -> None:
        client = graph_client
        if client is None:
            if not _graph_credentials_ready(settings):
                logger.warning(
                    "curiosity_peer_persist_graph_unconfigured brief_id=%s "
                    "(set ORION_CURIOSITY_GRAPH_HOST/PORT/USER/PASSWORD for "
                    "worldview PeerBrief MERGE; bus publish still attempted)",
                    brief.brief_id,
                )
                return
            try:
                client = _build_curiosity_graph_client(settings)
            except Exception:
                logger.exception(
                    "curiosity_peer_persist_graph_client_failed brief_id=%s",
                    brief.brief_id,
                )
                raise
        graph_query = getattr(client, "graph_query", None)
        if graph_query is None:
            raise RuntimeError("graph client has no graph_query()")
        graph_query(cypher, params)

    def _bus_publish(channel: str, payload: Any) -> None:
        if bus is None:
            logger.warning("curiosity_peer_persist_no_bus brief_id=%s", brief.brief_id)
            return
        publish = getattr(bus, "publish", None)
        if publish is None:
            raise RuntimeError("bus has no publish()")

        async def _pub() -> None:
            result = publish(channel, payload)
            if hasattr(result, "__await__"):
                await result

        _run_coro_threadsafe(_pub(), loop, timeout=30.0)

    persist_peer_brief(
        brief=brief,
        graph_execute=_graph_execute,
        bus_publish=_bus_publish,
    )


def make_bus_claude_peer(
    *,
    settings: Settings,
    bus: Any,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ClaudeFn:
    """Build a Claude callable that publishes + waits on the room channels.

    Safe to call from a worker thread when ``loop`` is the consumer event loop
    (``asyncio.to_thread`` → ``run_coroutine_threadsafe``). Tests should inject
    ``claude=`` instead of using this helper. Subscribes before publish so a
    fast utterance cannot race past an unsubscribed listener.
    """

    def _claude(help_req: HelpRequestV1, *, context_pack: str = "") -> PeerBriefV1:
        # Stash only — real bus publish is awaited inside wait after subscribe,
        # so we never nest a second loop inside a running loop.
        _pending_request: list[RoomClaudeRequestV1] = []

        def publish_request(req: RoomClaudeRequestV1) -> None:
            _pending_request.append(req)

        def wait_utterance(
            request_id: str,
            *,
            timeout_sec: float,
            publish_request: Optional[Callable[[], Any]] = None,
        ) -> str:
            async def _wait() -> str:
                async def _consume() -> str:
                    async with bus.subscribe(
                        settings.CHANNEL_ROOM_CLAUDE_UTTERANCE
                    ) as pubsub:
                        if publish_request is not None:
                            publish_request()
                        while _pending_request:
                            req = _pending_request.pop(0)
                            result = bus.publish(
                                settings.CHANNEL_ROOM_CLAUDE_REQUEST,
                                req.model_dump(mode="json"),
                            )
                            if hasattr(result, "__await__"):
                                await result
                        async for msg in bus.iter_messages(pubsub):
                            data = msg.get("data") if isinstance(msg, dict) else msg
                            if isinstance(data, bytes):
                                data = data.decode("utf-8")
                            if isinstance(data, str):
                                try:
                                    data = json.loads(data)
                                except json.JSONDecodeError:
                                    continue
                            utt = _unwrap_room_claude_utterance(data)
                            if utt is None:
                                continue
                            if utt.request_id != request_id:
                                continue
                            if not utt.ok:
                                raise RuntimeError(
                                    utt.error or "claude utterance not ok"
                                )
                            if utt.passed:
                                return ""
                            return utt.text or ""
                    raise TimeoutError(
                        f"claude utterance timeout after {timeout_sec}s"
                    )

                try:
                    return await asyncio.wait_for(_consume(), timeout=timeout_sec)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"claude utterance timeout after {timeout_sec}s"
                    ) from exc

            return _run_coro_threadsafe(
                _wait(), loop, timeout=float(timeout_sec) + 5.0
            )

        return run_claude_peer(
            help_req,
            context_pack=context_pack,
            publish_request=publish_request,
            wait_utterance=wait_utterance,
            timeout_sec=timeout_sec,
        )

    return _claude


def handle_help_request(
    raw: HelpInput,
    *,
    settings: Optional[Settings] = None,
    claude: Optional[ClaudeFn] = None,
    observe_claude_limit: Optional[ObserveClaudeFn] = None,
    persist: Optional[PersistFn] = None,
    context_pack: str = "",
    bus: Any = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    begin_episode: Optional[Callable[[HelpRequestV1], bool]] = None,
    limit_memory: Optional[ClaudeLimitMemory] = None,
    now: Optional[Callable[[], datetime]] = None,
) -> PeerBriefV1 | None:
    """Process one HelpRequest: limit back-off → Claude meter → episode →
    Claude once → persist.

    Claude is the only peer. Inject `claude` / `observe_claude_limit` /
    `persist` / `limit_memory` / `now` in tests.
    """
    help_req = parse_help_request(raw)
    settings = settings or Settings()
    if not (context_pack or "").strip():
        context_pack = context_pack_from_help(help_req)

    def _persist(brief: PeerBriefV1) -> None:
        if settings.CURIOSITY_PEER_EPISODES_ENABLED and (brief.help_id != help_req.help_id or brief.run_id != help_req.run_id):
            raise ValueError("peer returned a brief for another request")
        if settings.CURIOSITY_PEER_EPISODES_ENABLED and brief.status == "ok" and not (brief.summary.strip() or brief.evidence_pointers):
            raise ValueError("peer returned an empty successful brief")
        if persist is not None:
            persist(brief)
        else:
            _default_persist(brief, settings=settings, bus=bus, loop=loop)

    memory = limit_memory if limit_memory is not None else _CLAUDE_LIMIT_MEMORY
    clock = now or (lambda: datetime.now(timezone.utc))

    # Back-off first (#3655): Claude's own "you've hit your limit" on the last
    # real call holds every hire until the reset it named. No Claude call, no
    # transcript scan.
    hold = memory.active(clock())
    if hold is not None:
        brief = _refused_budget_brief(
            help_req, claude_reason(describe_hold(hold), "claude_limit_backoff")
        )
        logger.warning(
            "curiosity_peer_claude_limit_backoff help_id=%s resets_at=%s stated=%s",
            help_req.help_id,
            hold.resets_at.isoformat(),
            hold.reset_stated,
        )
        _persist(brief)
        return brief

    # Claude transcript meter. It reads Claude Code transcripts under
    # CURIOSITY_PEER_CLAUDE_PROJECTS_PATH (compose mounts the host tree
    # read-only). A transcript-observed `limited` still refuses. `unknown`
    # (no Claude Code session in the window -- the account is idle) no longer
    # refuses here: try, back off on refusal (#3655, Juniper 2026-11-24).
    # `_budget_refusal` itself is unchanged for its other callers.
    def _observe_claude_default() -> Any:
        return observe(root=settings.CURIOSITY_PEER_CLAUDE_PROJECTS_PATH)

    claude_observe = observe_claude_limit or _observe_claude_default
    claude_limit = claude_observe()
    claude_refusal = _budget_refusal(claude_limit)
    if claude_refusal == "budget_unobserved":
        logger.info(
            "curiosity_peer_claude_meter_unobserved_trying help_id=%s",
            help_req.help_id,
        )
        claude_refusal = None
    if claude_refusal is not None:
        brief = _refused_budget_brief(
            help_req,
            claude_reason(
                describe_claude_refusal(claude_refusal),
                f"claude_{claude_refusal}",
            ),
        )
        logger.warning(
            "curiosity_peer_claude_budget_refused help_id=%s reason=%s "
            "observed=%s state=%s",
            help_req.help_id,
            claude_refusal,
            getattr(claude_limit, "observed", None),
            getattr(claude_limit, "state", None),
        )
        _persist(brief)
        return brief

    def _record(brief: PeerBriefV1) -> None:
        recorded = memory.record_outcome(
            status=brief.status,
            text=f"{brief.refusal_reason or ''} {brief.summary or ''}",
            at=clock(),
            default_sec=settings.CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC,
        )
        if recorded is not None:
            logger.warning(
                "curiosity_peer_claude_limit_hit help_id=%s resets_at=%s stated=%s",
                help_req.help_id,
                recorded.resets_at.isoformat(),
                recorded.reset_stated,
            )

    if settings.CURIOSITY_PEER_EPISODES_ENABLED:
        if begin_episode is None:
            if not _graph_credentials_ready(settings):
                raise RuntimeError("episode commitment requires configured graph credentials")
            graph = _build_curiosity_graph_client(settings)
            begin_episode = lambda request: commit_ask(graph, request)
        if not begin_episode(help_req):
            logger.info("curiosity_peer_duplicate_commit help_id=%s", help_req.help_id)
            return None

    claude_fn = claude
    if claude_fn is None:
        if bus is None:
            brief = _failed_brief(
                help_req,
                reason=claude_reason(
                    "Claude helper not wired in this process (no bus / no claude callable).",
                    "claude_unwired",
                ),
            )
            _persist(brief)
            return brief
        claude_fn = make_bus_claude_peer(
            settings=settings, bus=bus, loop=loop
        )

    try:
        result = claude_fn(help_req, context_pack=context_pack)
        brief = _map_claude_result(help_req, result)
        _record(brief)
        _persist(brief)
        return brief
    except Exception as claude_exc:
        logger.warning(
            "curiosity_peer_claude_failed help_id=%s err=%s",
            help_req.help_id,
            claude_exc,
        )
        brief = _failed_brief(
            help_req,
            reason=claude_reason(
                "Claude helper was tried and failed: "
                f"{clip(str(claude_exc), 200).rstrip('.')}.",
                "claude_failed",
            ),
        )
        _record(brief)
        _persist(brief)
        return brief


def seed_claude_limit_memory(
    settings: Settings,
    *,
    graph_client: Any = None,
    memory: Optional[ClaudeLimitMemory] = None,
    now: Optional[datetime] = None,
) -> Any:
    """Boot: seed the limit back-off from the newest claude_room brief in the
    last 5 h (worldview PeerBrief nodes this service writes). Best effort: a
    graph error logs and leaves the memory empty (the next limit failure sets it).
    """
    memory = memory if memory is not None else _CLAUDE_LIMIT_MEMORY
    if graph_client is None and not _graph_credentials_ready(settings):
        logger.info("curiosity_peer_limit_seed_skipped reason=graph_unconfigured")
        return None
    try:
        client = graph_client or _build_curiosity_graph_client(settings)
        hold = seed_memory_from_graph(
            memory,
            client.graph_query,
            now=now or datetime.now(timezone.utc),
            default_sec=settings.CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_peer_limit_seed_failed err=%s", exc)
        return None
    logger.info(
        "curiosity_peer_limit_seeded active=%s resets_at=%s",
        hold is not None,
        hold.resets_at.isoformat() if hold is not None else None,
    )
    return hold


def apply_peer_brief_consumed(
    raw: Any,
    *,
    settings: Settings,
    graph_client: Any = None,
) -> list[str]:
    """MERGE consumed=true for brief_ids from a PeerBriefConsumed envelope."""
    body = raw
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        body = json.loads(body)
    if isinstance(body, dict):
        for key in ("payload", "data"):
            inner = body.get(key)
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    continue
            if isinstance(inner, dict) and "brief_ids" in inner:
                body = inner
                break
            if isinstance(inner, dict) and inner.get("schema_version") == (
                PEER_BRIEF_CONSUMED_KIND
            ):
                body = inner
                break
    consumed = PeerBriefConsumedV1.model_validate(body)
    if not consumed.brief_ids and consumed.phase != "completed":
        return []

    client = graph_client
    if client is None:
        if not _graph_credentials_ready(settings):
            logger.warning(
                "curiosity_peer_consume_graph_unconfigured n=%s",
                len(consumed.brief_ids),
            )
            return []
        client = _build_curiosity_graph_client(settings)
    if consumed.consumer_run_id is not None:
        cypher, params = offer_or_complete_cypher(consumed)
    else:
        cypher, params = peer_brief_consume_cypher(consumed.brief_ids)
    graph_query = getattr(client, "graph_query", None)
    if graph_query is None:
        raise RuntimeError("graph client has no graph_query()")
    graph_query(cypher, params)
    logger.info(
        "curiosity_peer_briefs_consumed n=%s ids=%s",
        len(consumed.brief_ids),
        consumed.brief_ids[:8],
    )
    return list(consumed.brief_ids)


async def handle_help_request_raw(settings: Settings, raw: Any, *, bus: Any = None) -> None:
    """Async consumer entry — scaffold-compatible wrapper."""
    try:
        loop = asyncio.get_running_loop()
        handle_help_request(raw, settings=settings, bus=bus, loop=loop)
    except Exception:
        logger.exception("curiosity_peer_handle_failed")


async def run_help_consumer(settings: Settings, bus: Any, stop: Any) -> None:
    """Subscribe to help requests and run the peer transport order."""
    channel = settings.CHANNEL_HELP_REQUEST
    loop = asyncio.get_running_loop()

    async def _handle(msg: dict) -> None:
        try:
            data = msg.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            logger.info(
                "curiosity_peer_help_seen channel=%s bytes=%s",
                channel,
                len(data) if isinstance(data, str) else 0,
            )
            # Sync peer path (Claude request + utterance wait) off
            # the consumer loop; pass ``loop`` so bus I/O uses
            # run_coroutine_threadsafe instead of asyncio.run.
            await asyncio.to_thread(
                handle_help_request,
                data,
                settings=settings,
                bus=bus,
                loop=loop,
            )
        except Exception:
            logger.exception("curiosity_peer_handle_failed")

    await run_subscription(
        bus, channel, _handle, stop_event=stop, name="curiosity-peer-help-request"
    )


async def run_consumed_consumer(settings: Settings, bus: Any, stop: Any) -> None:
    """Hub soft-nudge → consumed event → MERGE PeerBrief.consumed=true."""

    async def _handle(msg: dict) -> None:
        try:
            data = msg.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            await asyncio.to_thread(
                apply_peer_brief_consumed, data, settings=settings
            )
        except Exception:
            logger.exception("curiosity_peer_consume_failed")

    await run_subscription(
        bus,
        settings.CHANNEL_PEER_BRIEF_CONSUMED,
        _handle,
        stop_event=stop,
        name="curiosity-peer-brief-consumed",
    )


async def run_consumer(settings: Settings, bus: Any, stop: Any) -> None:
    """Run help + consumed consumers until stop."""
    await asyncio.gather(
        run_help_consumer(settings, bus, stop),
        run_consumed_consumer(settings, bus, stop),
    )


# Re-export for tests that assert LimitObservation typing stays available.
__all__ = [
    "LimitObservation",
    "apply_peer_brief_consumed",
    "context_pack_from_help",
    "handle_help_request",
    "handle_help_request_raw",
    "make_bus_claude_peer",
    "parse_help_request",
    "run_consumer",
    "run_consumed_consumer",
    "run_help_consumer",
    "_default_persist",
    "_map_claude_result",
    "_run_coro_threadsafe",
]
