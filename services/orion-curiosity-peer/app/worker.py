"""HelpRequest consumer — budget → Cursor → Claude-once → persist PeerBrief."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Optional, Union

from app.claude_fallback import DEFAULT_TIMEOUT_SEC, run_claude_fallback
from app.cursor_errors import TokenUnavailable, classify_cursor_failure
from app.cursor_invoker import build_sealed_prompt, parse_peer_brief_body, run_cursor_job
from app.settings import Settings
from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.dev_economics.cursor_limit_events import (
    CursorLimitObservation,
    decide_cursor_budget,
    observe_cursor_limit,
)
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomClaudeUtteranceV1

logger = logging.getLogger("orion-curiosity-peer.worker")

HelpInput = Union[HelpRequestV1, str, bytes, dict[str, Any]]
CursorFn = Callable[..., PeerBriefV1]
ClaudeFn = Callable[..., Union[str, PeerBriefV1]]
ObserveFn = Callable[[], CursorLimitObservation]
PersistFn = Callable[[PeerBriefV1], Any]


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
    peer: str,
    reason: str,
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


def _refused_budget_brief(help_req: HelpRequestV1, reason: str) -> PeerBriefV1:
    return PeerBriefV1(
        brief_id=_new_brief_id(),
        help_id=help_req.help_id,
        run_id=help_req.run_id,
        prior_id=help_req.prior_id,
        peer="cursor_auto",
        status="refused_budget",
        summary="",
        refusal_reason=reason,
    )


def _default_cursor(
    help_req: HelpRequestV1,
    *,
    settings: Settings,
    context_pack: str,
) -> PeerBriefV1:
    secret = settings.CURSOR_API_KEY
    api_key = secret.get_secret_value() if secret is not None else ""
    if not api_key.strip():
        raise TokenUnavailable("CURSOR_API_KEY missing")
    sealed = build_sealed_prompt(help_req, context_pack=context_pack)
    try:
        return run_cursor_job(
            help_req,
            sealed,
            api_key=api_key,
            cwd=settings.CURIOSITY_PEER_REPO_ROOT,
            model=settings.CURIOSITY_PEER_MODEL,
        )
    except TokenUnavailable:
        raise
    except Exception as exc:
        if classify_cursor_failure(exc) == "token_unavailable":
            raise TokenUnavailable(str(exc)) from exc
        raise


def _map_claude_result(help_req: HelpRequestV1, result: Union[str, PeerBriefV1]) -> PeerBriefV1:
    if isinstance(result, PeerBriefV1):
        if result.peer != "claude_room":
            return result.model_copy(update={"peer": "claude_room"})
        return result
    return parse_peer_brief_body(str(result), help=help_req, peer="claude_room")


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


def _default_persist(
    brief: PeerBriefV1,
    *,
    settings: Settings,
    bus: Any,
    graph_client: Any = None,
) -> None:
    def _graph_execute(cypher: str) -> None:
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
        graph_query(cypher)

    def _bus_publish(channel: str, payload: dict[str, Any]) -> None:
        if bus is None:
            logger.warning("curiosity_peer_persist_no_bus brief_id=%s", brief.brief_id)
            return
        publish = getattr(bus, "publish", None)
        if publish is None:
            raise RuntimeError("bus has no publish()")
        result = publish(channel, payload)
        if hasattr(result, "__await__"):
            import asyncio

            asyncio.run(result)

    persist_peer_brief(
        brief=brief,
        graph_execute=_graph_execute,
        bus_publish=_bus_publish,
    )


def make_bus_claude_fallback(
    *,
    settings: Settings,
    bus: Any,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> ClaudeFn:
    """Build a Claude callable that publishes + waits on the room channels.

    Safe to call from a worker thread (e.g. ``asyncio.to_thread``). Tests should
    inject ``claude=`` instead of using this helper. Subscribes before publish
    so a fast utterance cannot race past an unsubscribed listener.
    """

    def _claude(help_req: HelpRequestV1, *, context_pack: str = "") -> PeerBriefV1:
        import asyncio

        # Stash only — real bus publish is awaited inside wait after subscribe,
        # so we never nest asyncio.run inside a running loop.
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

            return asyncio.run(_wait())

        return run_claude_fallback(
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
    cursor: Optional[CursorFn] = None,
    claude: Optional[ClaudeFn] = None,
    observe_limit: Optional[ObserveFn] = None,
    persist: Optional[PersistFn] = None,
    context_pack: str = "",
    bus: Any = None,
) -> PeerBriefV1:
    """Process one HelpRequest: budget gate → Cursor → Claude once → persist.

    Inject `cursor` / `claude` / `observe_limit` / `persist` in tests.
    """
    help_req = parse_help_request(raw)
    settings = settings or Settings()

    def _persist(brief: PeerBriefV1) -> None:
        if persist is not None:
            persist(brief)
        else:
            _default_persist(brief, settings=settings, bus=bus)

    observe = observe_limit or observe_cursor_limit
    observation = observe()
    refusal = decide_cursor_budget(observation)
    if refusal is not None:
        brief = _refused_budget_brief(help_req, refusal)
        logger.warning(
            "curiosity_peer_budget_refused help_id=%s reason=%s observed=%s state=%s",
            help_req.help_id,
            refusal,
            getattr(observation, "observed", None),
            getattr(observation, "state", None),
        )
        _persist(brief)
        return brief

    cursor_fn = cursor or (
        lambda h, **_k: _default_cursor(h, settings=settings, context_pack=context_pack)
    )
    claude_fn = claude  # may stay None until token failure

    cursor_error: Optional[BaseException] = None
    try:
        brief = cursor_fn(help_req, context_pack=context_pack)
        _persist(brief)
        return brief
    except Exception as cursor_exc:
        cursor_error = cursor_exc
        kind = classify_cursor_failure(cursor_exc)
        logger.warning(
            "curiosity_peer_cursor_failed help_id=%s kind=%s err=%s",
            help_req.help_id,
            kind,
            cursor_exc,
        )
        if kind != "token_unavailable":
            brief = _failed_brief(
                help_req,
                peer="cursor_auto",
                reason=f"cursor_other: {cursor_exc}",
            )
            _persist(brief)
            return brief

    # Token / unavailable → exactly one Claude attempt.
    assert cursor_error is not None
    if claude_fn is None:
        if bus is None:
            brief = _failed_brief(
                help_req,
                peer="claude_room",
                reason=(
                    f"cursor_token_unavailable: {cursor_error}; "
                    "claude_unwired (no bus / no claude callable)"
                ),
            )
            _persist(brief)
            return brief
        claude_fn = make_bus_claude_fallback(settings=settings, bus=bus)

    try:
        result = claude_fn(help_req, context_pack=context_pack)
        brief = _map_claude_result(help_req, result)
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
            peer="claude_room",
            reason=(
                f"cursor_token_unavailable: {cursor_error}; "
                f"claude_failed: {claude_exc}"
            ),
        )
        _persist(brief)
        return brief


async def handle_help_request_raw(settings: Settings, raw: Any, *, bus: Any = None) -> None:
    """Async consumer entry — scaffold-compatible wrapper."""
    try:
        handle_help_request(raw, settings=settings, bus=bus)
    except Exception:
        logger.exception("curiosity_peer_handle_failed")


async def run_consumer(settings: Settings, bus: Any, stop: Any) -> None:
    """Subscribe to help requests and run the peer transport order."""
    import asyncio

    channel = settings.CHANNEL_HELP_REQUEST
    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            if stop.is_set():
                break
            try:
                data = msg.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                logger.info(
                    "curiosity_peer_help_seen channel=%s bytes=%s",
                    channel,
                    len(data) if isinstance(data, str) else 0,
                )
                # Sync peer path (Cursor SDK + optional Claude wait) off the
                # consumer loop so make_bus_claude_fallback can asyncio.run.
                await asyncio.to_thread(
                    handle_help_request, data, settings=settings, bus=bus
                )
            except Exception:
                logger.exception("curiosity_peer_handle_failed")
