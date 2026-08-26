import asyncio
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.vision import (
    VisionSceneInventoryV1,
    VisionArtifactPayload,
    VisionWindowPayload,
    VisionWindowRequestPayload,
    VisionWindowResultPayload,
)

from .projection import (
    build_window_payload,
    envelope_to_http_dict,
    identity_confidence_from_artifact,
    identity_hint_from_artifact,
    stream_key_from_artifact,
)
from .recovery_store import RecoveryStore
from .presence import PresenceRegistry, write_snapshot_to_postgres
from .scene_belief import SceneBeliefRegistry
from .settings import Settings

settings = Settings()


def _source_ref() -> ServiceRef:
    return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)


def _corr_uuid(value: Optional[Union[str, UUID]]) -> UUID:
    if value is None:
        return uuid.uuid4()
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _should_ignore_uncertain_reading(
    existing: Optional[Dict[str, Any]],
    new_confidence: Optional[str],
    *,
    now: float,
    max_age_sec: float,
) -> bool:
    """True when a new "uncertain" identity reading should be dropped rather
    than overwrite `_identity_by_stream`'s existing entry for this stream.

    Pure decision logic, split out of `_consume_identity`'s loop (2026-08-26
    review) so this can be tested directly rather than only through the bus
    subscription. A still-fresh CONFIRMED reading wins over a single
    flickery "unsure" frame (a bad angle, a turned head) -- one unsure frame
    moments after Orion has already settled on "this is Juniper" must not
    flip presence's `identity_uncertain` back on and trigger an awkward
    "is that you?" a beat later. Only "uncertain" readings are ever
    subject to this hold-off; a new "confirmed" reading always overwrites
    (see `_consume_identity` -- this function is only consulted for
    `new_confidence == "uncertain"`).
    """
    if new_confidence != "uncertain" or existing is None:
        return False
    if existing.get("confidence") != "confirmed":
        return False
    age = now - existing.get("ts", 0.0)
    return age <= max_age_sec


class WindowService:
    def __init__(self) -> None:
        self.bus = OrionBusAsync(
            url=settings.ORION_BUS_URL,
            enforce_catalog=settings.ORION_BUS_ENFORCE_CATALOG,
        )
        self._consumer_task: Optional[asyncio.Task] = None
        self._identity_consumer_task: Optional[asyncio.Task] = None
        self._rpc_task: Optional[asyncio.Task] = None
        self._emitter_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Per-stream ingest buffers: list of {artifact, ts, env}
        self._buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._buffer_lock = asyncio.Lock()

        # Latest identity_face hint per stream: {"hint": {...}, "ts": float}.
        # Separate from _buffers -- identity arrives on its own dedicated
        # channel (CHANNEL_WINDOW_IDENTITY_INTAKE), at its own (rate-
        # limited, opportunistic) cadence, not once per artifact like the
        # detection buffer. Read-then-gate-by-age in _flush_and_publish;
        # never grows unbounded (one entry per stream_id, overwritten).
        self._identity_by_stream: Dict[str, Dict[str, Any]] = {}
        self._identity_lock = asyncio.Lock()

        self._live_lock = asyncio.Lock()
        self._live_by_stream: Dict[str, VisionWindowPayload] = {}
        self._live_global: Optional[VisionWindowPayload] = None
        self._recent_by_stream: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=settings.VISION_WINDOW_RECOVERY_MAX_N)
        )
        self._recent_global: deque = deque(maxlen=settings.VISION_WINDOW_RECOVERY_MAX_N)

        self._recovery: Optional[RecoveryStore] = None
        self._recovery_ok = False
        self._bus_ready = False

        self._cursor_i = 0
        self._belief_registry = SceneBeliefRegistry(
            vote_n=settings.WINDOW_BELIEF_VOTE_N,
            enter_votes=settings.WINDOW_BELIEF_ENTER_VOTES,
            exit_votes=settings.WINDOW_BELIEF_EXIT_VOTES,
        )
        self._presence_registry = PresenceRegistry(
            grace_sec=settings.WINDOW_PRESENCE_GRACE_SEC,
            write_min_interval_sec=settings.WINDOW_PRESENCE_WRITE_MIN_INTERVAL_SEC,
        )
        # Background presence writes, tracked so they cannot be GC'd mid-flight
        # (asyncio.create_task holds only a weak reference -- same reasoning
        # as orion-vision-host's liveness alert task).
        self._presence_write_tasks: set = set()

        # Metrics counters (§12)
        self._m_ingest = 0
        self._m_snapshots = 0
        self._m_inventory_published = 0
        self._m_inventory_failed = 0
        self._m_recovery_ok = 0
        self._m_recovery_fail = 0
        self._m_catchup_expired = 0

    def _recovery_url(self) -> str:
        return (settings.VISION_WINDOW_RECOVERY_REDIS_URL or settings.ORION_BUS_URL).strip()

    def _next_cursor(self) -> str:
        self._cursor_i += 1
        return f"vw:{self._cursor_i:012d}:{uuid.uuid4().hex[:6]}"

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        logger.info(
            f"[WINDOW] Startup config: recovery_enabled={settings.VISION_WINDOW_RECOVERY_ENABLED} "
            f"redis_url_redacted=yes max_n={settings.VISION_WINDOW_RECOVERY_MAX_N} "
            f"ttl_sec={settings.VISION_WINDOW_RECOVERY_TTL_SEC} "
            f"flush_interval_ms={settings.FLUSH_INTERVAL_MS} "
            f"ready_requires_recovery={settings.VISION_WINDOW_READY_REQUIRES_RECOVERY}"
        )

        await self.bus.connect()
        self._bus_ready = True

        if settings.VISION_WINDOW_RECOVERY_ENABLED:
            self._recovery = RecoveryStore(
                self._recovery_url(),
                ttl_sec=settings.VISION_WINDOW_RECOVERY_TTL_SEC,
                max_n=settings.VISION_WINDOW_RECOVERY_MAX_N,
            )
            self._recovery_ok = await self._recovery.connect()
            if not self._recovery_ok:
                logger.warning("[WINDOW] Recovery disabled at runtime (connection failed)")
        else:
            self._recovery = None
            self._recovery_ok = False

        self._consumer_task = asyncio.create_task(self._consume())
        # WINDOW_IDENTITY_ENABLED alone is not sufficient (review finding,
        # 2026-08-26): the identity hint is only ever READ inside the
        # WINDOW_BELIEF_ENABLED branch of _flush_and_publish (presence and
        # council evidence both derive from belief's output). An operator
        # disabling belief (e.g. to debug scene-belief flapping) while
        # leaving identity's own flag at its true default would otherwise
        # get a consumer that keeps subscribing, keeps triggering real GPU
        # dispatch upstream, keeps writing _identity_by_stream -- and never
        # has any of it read by anything. CLAUDE.md section 0A names this
        # failure shape directly: "reducers alive but cursors stale."
        # Refusing to start the consumer is the deterministic gate instead
        # of a comment nobody reads.
        identity_consumer_enabled = settings.WINDOW_IDENTITY_ENABLED and settings.WINDOW_BELIEF_ENABLED
        if settings.WINDOW_IDENTITY_ENABLED and not settings.WINDOW_BELIEF_ENABLED:
            logger.warning(
                "[WINDOW] WINDOW_IDENTITY_ENABLED=true but WINDOW_BELIEF_ENABLED=false -- "
                "identity consumer NOT started (its output is only ever read inside the belief "
                "branch of _flush_and_publish; running it anyway would be pure waste)."
            )
        if identity_consumer_enabled:
            self._identity_consumer_task = asyncio.create_task(self._consume_identity())
        self._rpc_task = asyncio.create_task(self._consume_rpc())
        self._emitter_task = asyncio.create_task(self._emit_loop())
        logger.info(
            f"[WINDOW] Started. intake={settings.CHANNEL_WINDOW_INTAKE} "
            f"identity_intake={settings.CHANNEL_WINDOW_IDENTITY_INTAKE if identity_consumer_enabled else 'disabled'} "
            f"pub={settings.CHANNEL_WINDOW_PUB} rpc={settings.CHANNEL_WINDOW_REQUEST}"
        )

    async def stop(self) -> None:
        self._shutdown_event.set()
        for t in [self._consumer_task, self._identity_consumer_task, self._rpc_task, self._emitter_task]:
            if t:
                try:
                    t.cancel()
                    await t
                except asyncio.CancelledError:
                    pass
        await self.bus.close()
        self._bus_ready = False
        if self._recovery:
            await self._recovery.close()
            self._recovery = None

    def _should_flush_stream(self, stream_id: str, now: float) -> bool:
        buf = self._buffers.get(stream_id) or []
        if not buf:
            return False
        first_ts = buf[0]["ts"]
        age_ms = (now - first_ts) * 1000.0
        if len(buf) >= settings.MAX_ARTIFACTS_PER_WINDOW:
            return True
        if age_ms >= float(settings.FLUSH_INTERVAL_MS):
            return True
        if age_ms >= float(settings.MAX_WINDOW_AGE_MS):
            return True
        if age_ms >= settings.WINDOW_SIZE_SEC * 1000.0 and len(buf) > 0:
            return True
        return False

    async def _consume(self) -> None:
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_INTAKE) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break
                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue
                env = decoded.envelope
                try:
                    if isinstance(env.payload, dict):
                        payload = VisionArtifactPayload(**env.payload)
                    else:
                        payload = env.payload
                except Exception as e:
                    logger.warning(f"[WINDOW] Invalid artifact payload: {e}")
                    continue
                sk = stream_key_from_artifact(payload)
                logger.debug(f"[WINDOW] ingest accepted stream={sk} artifact={payload.artifact_id}")
                self._m_ingest += 1
                async with self._buffer_lock:
                    self._buffers[sk].append({"artifact": payload, "ts": time.time(), "env": env})

    async def _consume_identity(self) -> None:
        """Separate loop, separate channel from _consume() above --
        identity_face artifacts arrive on their own dedicated,
        single-consumer lane (CHANNEL_WINDOW_IDENTITY_INTAKE, see
        settings.py's docstring for why it is not the general
        CHANNEL_WINDOW_INTAKE broadcast). Stores only the latest usable hint
        per stream, overwriting the previous one -- this is presence-shaped
        state (what does the LATEST evidence say), not a window buffer, so
        it never accumulates and never needs a size cap.
        """
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_IDENTITY_INTAKE) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break
                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue
                env = decoded.envelope
                try:
                    if isinstance(env.payload, dict):
                        payload = VisionArtifactPayload(**env.payload)
                    else:
                        payload = env.payload
                except Exception as e:
                    logger.warning(f"[WINDOW] Invalid identity artifact payload: {e}")
                    continue
                hint = identity_hint_from_artifact(payload)
                confidence = identity_confidence_from_artifact(payload)
                if hint is None and confidence is None:
                    # No usable signal at all (no face detected, or a
                    # gallery-misconfig candidate) -- nothing to store.
                    # Never clears a still-fresh prior reading either way;
                    # the freshness gate in _flush_and_publish is what
                    # decides whether an older reading is still usable, not
                    # this arrival.
                    continue
                sk = stream_key_from_artifact(payload)
                async with self._identity_lock:
                    existing = self._identity_by_stream.get(sk)
                    if _should_ignore_uncertain_reading(
                        existing, confidence, now=time.time(), max_age_sec=settings.WINDOW_IDENTITY_MAX_AGE_SEC
                    ):
                        logger.debug(
                            f"[WINDOW] identity_hint stream={sk} unsure_reading_ignored "
                            f"still_fresh_confirmed=true"
                        )
                        continue
                    self._identity_by_stream[sk] = {"hint": hint, "confidence": confidence, "ts": time.time()}
                logger.debug(
                    f"[WINDOW] identity_hint stream={sk} "
                    f"subject={hint['subject'] if hint else None} confidence={confidence}"
                )

    async def _consume_rpc(self) -> None:
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_REQUEST) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break
                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue
                env = decoded.envelope
                asyncio.create_task(self._handle_rpc(env))

    async def _handle_rpc(self, env: BaseEnvelope) -> None:
        try:
            if isinstance(env.payload, dict):
                req = VisionWindowRequestPayload(**env.payload)
            else:
                req = env.payload
        except Exception as e:
            logger.error(f"[WINDOW] RPC invalid payload: {e}")
            return
        art = req.artifact
        sk = stream_key_from_artifact(art)
        now = time.time()
        entry = {"artifact": art, "ts": now, "env": env}
        # Single projection path: same flush/materialize as streaming (§3.3)
        await self._flush_and_publish(
            stream_id=sk,
            buffered=[entry],
            correlation_id=env.correlation_id,
            causality_chain=list(env.causality_chain),
        )
        async with self._live_lock:
            window_payload = self._live_by_stream.get(sk) or self._live_global
        if window_payload is None:
            return
        res_payload = VisionWindowResultPayload(window=window_payload)
        reply_env = BaseEnvelope(
            kind="vision.window.result",
            source=_source_ref(),
            correlation_id=env.correlation_id,
            causality_chain=[*env.causality_chain],
            payload=res_payload.model_dump(mode="json"),
            reply_to=None,
        )
        if env.reply_to:
            await self.bus.publish(env.reply_to, reply_env)

    async def _emit_loop(self) -> None:
        while not self._shutdown_event.is_set():
            now = time.time()
            async with self._buffer_lock:
                stream_ids = list(self._buffers.keys())
            for sid in stream_ids:
                flush = False
                async with self._buffer_lock:
                    if self._should_flush_stream(sid, now):
                        flush = True
                if flush:
                    await self._drain_stream(sid)
            await asyncio.sleep(0.25)

    async def _drain_stream(self, stream_id: str) -> None:
        async with self._buffer_lock:
            buf = self._buffers.get(stream_id)
            if not buf:
                return
            items = list(buf)
            buf.clear()
        await self._flush_and_publish(
            stream_id=stream_id,
            buffered=items,
            correlation_id=uuid.uuid4(),
            causality_chain=[],
        )

    async def _flush_and_publish(
        self,
        *,
        stream_id: str,
        buffered: List[Dict[str, Any]],
        correlation_id: Optional[Union[str, UUID]],
        causality_chain: List[Any],
    ) -> None:
        if not buffered:
            return
        items: List[Tuple[VisionArtifactPayload, float]] = [(b["artifact"], b["ts"]) for b in buffered]
        # Skip missing envelopes (direct test calls); production ingest always attaches env.
        envs: List[BaseEnvelope] = [b["env"] for b in buffered if b["env"] is not None]
        window_start = min(b["ts"] for b in buffered)
        window_end = time.time()
        cursor = self._next_cursor()
        payload = build_window_payload(
            stream_id=stream_id,
            items=items,
            envs=envs,
            window_start=window_start,
            window_end=window_end,
            cursor=cursor,
            stale_after_ms=settings.STALE_AFTER_MS,
        )
        identity_hint: Optional[Dict[str, Any]] = None
        identity_confidence: Optional[str] = None
        if settings.WINDOW_BELIEF_ENABLED and settings.WINDOW_IDENTITY_ENABLED:
            # Fetched HERE, before the belief block below, not inside it
            # (review finding, 2026-08-26): this is the one await in this
            # method, and inserting it between SceneBeliefRegistry.observe()
            # and _note_presence() broke what was previously an atomic
            # (no-await) stretch. _flush_and_publish is genuinely reachable
            # concurrently for the same stream_id (unserialized
            # asyncio.create_task per RPC message in _handle_rpc, plus the
            # independent periodic emit loop's _drain_stream), so two
            # flushes could interleave exactly at that await point and apply
            # belief/presence updates out of order. Fetching first keeps the
            # belief block itself await-free, exactly as it was before this
            # field existed. Both fetches read the same underlying entry and
            # the same age gate; two calls rather than one combined getter to
            # keep each independently testable, matching the rest of this
            # module's style.
            identity_hint = await self._get_fresh_identity_hint(stream_id, now=window_end)
            identity_confidence = await self._get_fresh_identity_confidence(stream_id, now=window_end)

        if settings.WINDOW_BELIEF_ENABLED:
            summary = dict(payload.summary or {})
            evidence = dict(summary.get("evidence") or {})
            observed = frozenset(str(x).strip().lower() for x in evidence.get("hard_labels") or [] if str(x).strip())
            result = self._belief_registry.observe(stream_id, observed)
            summary["evidence"] = self._belief_registry.enrich_evidence(stream_id, evidence)

            if identity_hint and "person" in observed:
                # Code-enforced, not just prompt-trusted (review finding,
                # 2026-08-26): council's hedging rule for this field
                # (interpretation.py) is a prompt instruction, not a
                # guarantee, and an identity hint can outlive the person it
                # was about by up to WINDOW_IDENTITY_MAX_AGE_SEC (default
                # 90s). Gating on THIS window's own raw hard_labels (not
                # believed_labels, which lags behind by design -- and
                # matches the existing "Activity verbs require person in
                # hard_labels" rule's own semantics one line below) means a
                # person-less window structurally cannot carry an
                # identity_hypothesis, regardless of what the LLM does with
                # a stray one. Only ever ADDS the key -- an absent/stale/
                # unpersoned hint means no identity_hypothesis field at all,
                # not a field asserting "unsure"/"unknown".
                summary["evidence"] = {**summary["evidence"], "identity_hypothesis": identity_hint}

            payload = payload.model_copy(update={"summary": summary})
            if settings.WINDOW_PRESENCE_ENABLED:
                self._note_presence(
                    stream_id,
                    result.believed_labels,
                    identity_hint=identity_hint,
                    identity_confidence=identity_confidence,
                )
            if result.added or result.removed:
                added = ",".join(sorted(result.added)) or "-"
                removed = ",".join(sorted(result.removed)) or "-"
                believed = ",".join(sorted(result.believed_labels)) or "-"
                logger.info(
                    f"[WINDOW] belief_transition stream={stream_id} "
                    f"added={added} removed={removed} believed={believed}"
                )
        env_dump = payload.model_dump(mode="json")

        async with self._live_lock:
            self._live_by_stream[stream_id] = payload
            self._live_global = payload
            self._recent_by_stream[stream_id].appendleft(env_dump)
            self._recent_global.appendleft(env_dump)

        if self._recovery and self._recovery.enabled:
            ok = await self._recovery.persist_snapshot(stream_id, env_dump, cursor)
            if ok:
                self._m_recovery_ok += 1
            else:
                self._m_recovery_fail += 1

        cid = _corr_uuid(correlation_id)
        envelope = BaseEnvelope(
            kind="vision.window",
            source=_source_ref(),
            correlation_id=cid,
            causality_chain=[*causality_chain],
            payload=payload.model_dump(mode="json"),
        )
        await self.bus.publish(settings.CHANNEL_WINDOW_PUB, envelope)
        self._m_snapshots += 1
        await self._publish_scene_inventory(payload, stream_id, cid, causality_chain)
        logger.info(
            f"[WINDOW] flush snapshot_id={payload.window_id} stream={stream_id} "
            f"artifacts={len(buffered)} cursor={cursor}"
        )

    async def _publish_scene_inventory(
        self,
        payload: VisionWindowPayload,
        stream_id: str,
        correlation_id: str,
        causality_chain: List[str],
    ) -> None:
        """Emit this window's scene census. Best effort; never blocks a flush.

        Written on EVERY window, deliberately unlike `vision_events`. The
        council only re-interprets when the observed label SET changes and logs
        `reason=stable_scene` otherwise, so a pure count change (two boxes
        become one) emits no event at all -- and a departure is a non-event by
        nature, since nothing fires when a thing stops being there. Object
        permanence needs a continuous record, so this one is unconditional and
        a timer-driven reducer reads it later.

        A failure here must never cost a window: the snapshot is already
        published and cached by the time this runs.
        """
        if not settings.WINDOW_SCENE_INVENTORY_ENABLED or not self.bus:
            return
        try:
            summary = payload.summary or {}
            evidence = summary.get("evidence") or {}
            inventory = VisionSceneInventoryV1(
                window_id=payload.window_id,
                stream_id=payload.stream_id or stream_id,
                camera_id=payload.camera_id,
                window_start_ts=payload.start_ts,
                window_end_ts=payload.end_ts,
                frame_count=int(summary.get("item_count") or 0),
                counts={str(k): int(v) for k, v in (summary.get("object_counts") or {}).items()},
                detections={
                    str(k): int(v) for k, v in (summary.get("label_detections") or {}).items()
                },
                believed_labels=[str(x) for x in (evidence.get("believed_hard_labels") or [])],
            )
            await self.bus.publish(
                settings.CHANNEL_SCENE_INVENTORY_PUB,
                BaseEnvelope(
                    kind="vision.scene.inventory.v1",
                    source=_source_ref(),
                    correlation_id=correlation_id,
                    causality_chain=[*causality_chain],
                    payload=inventory.model_dump(mode="json"),
                ),
            )
            self._m_inventory_published += 1
        except Exception as exc:
            self._m_inventory_failed += 1
            logger.warning(f"[WINDOW] scene inventory publish failed: {exc}")

    async def _get_fresh_identity_hint(self, stream_id: str, *, now: float) -> Optional[Dict[str, Any]]:
        """The latest identity_face hint for this stream, or None if there
        isn't one or it is too old to speak to "now" -- same staleness
        discipline as orion/situational/context.py's percept gate. Age is
        measured against ``now`` (the window's own end_ts), not wall-clock
        time.time(), so a slow flush cannot make an otherwise-fresh hint
        read as stale relative to the window it is being folded into.
        """
        async with self._identity_lock:
            entry = self._identity_by_stream.get(stream_id)
        if entry is None:
            return None
        age = now - entry["ts"]
        if age > settings.WINDOW_IDENTITY_MAX_AGE_SEC:
            return None
        return entry["hint"]

    async def _get_fresh_identity_confidence(self, stream_id: str, *, now: float) -> Optional[str]:
        """The latest identity_face confidence classification for this
        stream (``"confirmed"`` | ``"uncertain"`` | ``None``), or None if
        there isn't one or it is too old -- same staleness contract as
        `_get_fresh_identity_hint` above (same underlying entry, same
        `WINDOW_IDENTITY_MAX_AGE_SEC` age gate, same clock parameter)."""
        async with self._identity_lock:
            entry = self._identity_by_stream.get(stream_id)
        if entry is None:
            return None
        age = now - entry["ts"]
        if age > settings.WINDOW_IDENTITY_MAX_AGE_SEC:
            return None
        return entry.get("confidence")

    def _note_presence(
        self,
        stream_id: str,
        believed_labels: frozenset[str],
        *,
        identity_hint: Optional[Dict[str, Any]] = None,
        identity_confidence: Optional[str] = None,
    ) -> None:
        """Update presence and, if due, fire a best-effort Postgres write.

        Best-effort end to end: `PresenceRegistry.record` never raises, and a
        failed or slow write cannot delay this window's flush -- the same
        contract as the scene-belief transition it rides alongside.
        """
        snapshot = self._presence_registry.record(
            stream_id, believed_labels, identity_hint=identity_hint, identity_confidence=identity_confidence
        )
        if snapshot is None:
            return
        if not settings.POSTGRES_URI:
            return
        task = asyncio.create_task(
            asyncio.to_thread(
                write_snapshot_to_postgres,
                stream_id,
                snapshot,
                postgres_uri=settings.POSTGRES_URI,
            )
        )
        self._presence_write_tasks.add(task)
        task.add_done_callback(self._presence_write_tasks.discard)

    async def health_live(self) -> Dict[str, Any]:
        return {"status": "ok", "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}

    async def ready_probe(self) -> Tuple[bool, str]:
        if not self._bus_ready:
            return False, "bus_not_connected"
        if settings.VISION_WINDOW_READY_REQUIRES_RECOVERY:
            if not self._recovery or not self._recovery.enabled:
                return False, "recovery_required_unavailable"
            if not await self._recovery.ping():
                return False, "recovery_ping_failed"
        elif self._recovery and self._recovery.enabled:
            if not await self._recovery.ping():
                return True, "degraded_recovery_unavailable"
        return True, "ok"

    def _cap_limit(self, limit: Optional[int]) -> int:
        raw = limit or 20
        return max(1, min(int(raw), settings.VISION_WINDOW_HTTP_MAX_LIMIT))

    async def http_current(self, stream_id: Optional[str]) -> Dict[str, Any]:
        async with self._live_lock:
            if stream_id:
                live = self._live_by_stream.get(stream_id)
            else:
                live = self._live_global
        if live:
            body = envelope_to_http_dict(live, source="live_state")
            body["status"] = "ok"
            return body
        if self._recovery and self._recovery.enabled:
            data = await self._recovery.read_latest(stream_id)
            if data:
                try:
                    p = VisionWindowPayload(**data)
                    body = envelope_to_http_dict(p, source="recovery_state")
                    body["status"] = "ok"
                    return body
                except ValidationError:
                    pass
        return {
            "status": "empty",
            "source": "none",
            "snapshot_id": None,
            "stream_id": stream_id,
            "generated_at": None,
            "cursor": None,
            "age_ms": None,
            "envelope": None,
        }

    async def http_current_stale_check(self, stream_id: Optional[str]) -> Dict[str, Any]:
        body = await self.http_current(stream_id)
        if body.get("status") == "empty":
            return body
        env = body.get("envelope") or {}
        end_ts = float(env.get("end_ts") or 0)
        age_ms = int(max(0.0, (time.time() - end_ts) * 1000))
        stale_after = int((env.get("freshness") or {}).get("stale_after_ms") or settings.STALE_AFTER_MS)
        if age_ms > stale_after:
            body["status"] = "stale"
        return body

    async def http_recent(self, stream_id: Optional[str], limit: int) -> Dict[str, Any]:
        lim = self._cap_limit(limit)
        recovery_ok = bool(self._recovery and self._recovery.enabled and await self._recovery.ping())
        rows: List[Dict[str, Any]] = []
        if recovery_ok:
            rows = await self._recovery.read_last_n(stream_id, lim)
        if not rows:
            async with self._live_lock:
                dq = self._recent_by_stream[stream_id] if stream_id else self._recent_global
                rows = list(dq)[:lim]
        degraded = settings.VISION_WINDOW_RECOVERY_ENABLED and not recovery_ok
        return {"items": rows, "recovery_degraded": degraded, "limit": lim}

    async def http_catchup(
        self, stream_id: Optional[str], after_cursor: Optional[str], limit: int
    ) -> Any:
        lim = self._cap_limit(limit)
        recovery_ok = bool(self._recovery and self._recovery.enabled and await self._recovery.ping())
        rows: List[Dict[str, Any]] = []
        if recovery_ok:
            rows = await self._recovery.read_last_n(stream_id, settings.VISION_WINDOW_RECOVERY_MAX_N)
        if not rows:
            async with self._live_lock:
                dq = self._recent_by_stream[stream_id] if stream_id else self._recent_global
                rows = list(dq)
        degraded = settings.VISION_WINDOW_RECOVERY_ENABLED and not recovery_ok
        if not after_cursor:
            return {"items": rows[:lim], "recovery_degraded": degraded}
        sorted_rows = sorted(rows, key=lambda r: str(r.get("cursor") or ""))
        cursors = [str(r.get("cursor") or "") for r in sorted_rows if r.get("cursor")]
        if not cursors:
            return {"items": [], "recovery_degraded": degraded}
        earliest = min(cursors)
        latest = max(cursors)
        if after_cursor < earliest:
            self._m_catchup_expired += 1
            logger.info(f"[WINDOW] catch-up cursor_expired after={after_cursor}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "cursor_expired",
                    "message": "Requested cursor is outside the bounded recovery window.",
                    "latest_cursor": latest,
                    "earliest_available_cursor": earliest,
                },
            )
        out: List[Dict[str, Any]] = []
        for r in sorted_rows:
            c = str(r.get("cursor") or "")
            if c and c > after_cursor:
                out.append(r)
            if len(out) >= lim:
                break
        return {"items": out[:lim], "recovery_degraded": degraded}


service = WindowService()
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from `service.bus` above (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=True,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    await service.start()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            f"[WINDOW] system_health_heartbeat_started service={settings.SERVICE_NAME} "
            f"interval_sec={settings.HEARTBEAT_INTERVAL_SEC}"
        )
    except Exception as exc:
        logger.warning(f"[WINDOW] system_health_heartbeat_start_failed error={exc}")
        heartbeat_chassis = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning(f"[WINDOW] system_health_heartbeat_stop_error error={exc}")
        heartbeat_chassis = None
    await service.stop()


app = FastAPI(title="Orion Vision Window", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return await service.health_live()


@app.get("/readyz")
async def readyz() -> JSONResponse:
    ok, reason = await service.ready_probe()
    code = 200 if ok else 503
    return JSONResponse(status_code=code, content={"status": "ready" if ok else "not_ready", "detail": reason})


@app.get("/api/vision-window/current")
async def api_current() -> Dict[str, Any]:
    return await service.http_current_stale_check(None)


@app.get("/api/vision-window/streams/{stream_id}/current")
async def api_current_stream(stream_id: str) -> Dict[str, Any]:
    return await service.http_current_stale_check(stream_id)


@app.get("/api/vision-window/recent")
async def api_recent(limit: int = Query(default=20, ge=1, le=500)) -> Dict[str, Any]:
    return await service.http_recent(None, limit)


@app.get("/api/vision-window/streams/{stream_id}/recent")
async def api_recent_stream(stream_id: str, limit: int = Query(default=20, ge=1, le=500)) -> Dict[str, Any]:
    return await service.http_recent(stream_id, limit)


@app.get("/api/vision-window/catch-up")
async def api_catchup(
    after_cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
):
    return await service.http_catchup(None, after_cursor, limit)


@app.get("/api/vision-window/streams/{stream_id}/catch-up")
async def api_catchup_stream(
    stream_id: str,
    after_cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
):
    return await service.http_catchup(stream_id, after_cursor, limit)


@app.get("/api/vision-window/metrics")
async def api_metrics() -> Dict[str, Any]:
    return {
        "vision_window_ingest_events_total": service._m_ingest,
        "vision_window_snapshots_published_total": service._m_snapshots,
        "vision_window_recovery_writes_total": service._m_recovery_ok,
        "vision_window_recovery_write_failures_total": service._m_recovery_fail,
        "vision_window_cursor_expired_total": service._m_catchup_expired,
    }
