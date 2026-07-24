# orion/core/bus/async_service.py
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, AsyncIterator, Dict, Optional

from pydantic import BaseModel, ValidationError
from redis import asyncio as aioredis

from orion.schemas.registry import resolve as resolve_schema_id
from .bus_schemas import BaseEnvelope
from .codec import OrionCodec
from .enforce import enforcer
from .rpc_health import RpcHealthAggregator, RpcHealthSnapshot
from .velocity_keys import DEFAULT_BUCKET_TTL_SEC, velocity_key

logger = logging.getLogger("orion.bus.async")


class OrionBusAsync:
    """
    Async Redis bus client.
    """

    def __init__(
        self,
        url: str,
        *,
        enabled: bool = True,
        codec: Optional[OrionCodec] = None,
        enforce_catalog: bool | None = None,
        track_velocity: bool | None = None,
    ):
        self.url = url
        self.enabled = enabled
        self.codec = codec or OrionCodec()
        self._redis: Optional[aioredis.Redis] = None
        self._rpc_pubsub: Optional[aioredis.client.PubSub] = None
        self._rpc_worker_task: Optional[asyncio.Task] = None
        self._rpc_worker_running = False
        self._rpc_lock = asyncio.Lock()
        self._rpc_subscribed: set[str] = set()
        self._pending_rpc: dict[tuple[str, str], asyncio.Future] = {}
        # docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md
        # step 2: in-process only, not shared across fork() children, same as every
        # other per-instance RPC state above. No periodic self-publish wired yet --
        # see rpc_health.py's own module docstring for why that's a separate,
        # deliberately deferred decision.
        self._rpc_health = RpcHealthAggregator()
        if enforce_catalog is None:
            enforce_catalog = os.getenv("ORION_BUS_ENFORCE_CATALOG", "false").lower() == "true"
        self.enforce_catalog = bool(enforce_catalog)
        enforcer.enforce = enforce_catalog
        if track_velocity is None:
            track_velocity = (
                os.getenv("ORION_BUS_VELOCITY_TRACKING_ENABLED", "false").lower() == "true"
            )
        self.track_velocity = bool(track_velocity)

    def _pubsub_redis_kwargs(self) -> dict[str, Any]:
        """Pub/sub listen() blocks indefinitely; socket_timeout must be disabled."""
        return {
            "decode_responses": False,
            "socket_timeout": None,
            "socket_connect_timeout": 10.0,
            "health_check_interval": 30,
        }

    def _command_redis_kwargs(self) -> dict[str, Any]:
        return {
            "decode_responses": False,
            "socket_timeout": 60.0,
            "socket_connect_timeout": 10.0,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        }

    def _create_pubsub_redis(self) -> aioredis.Redis:
        return aioredis.from_url(self.url, **self._pubsub_redis_kwargs())

    async def reconnect(self) -> None:
        """Drop and re-open the command Redis connection after transport errors."""
        if self._redis is not None:
            with suppress(Exception):
                await self._redis.close()
            self._redis = None
        await self.connect()

    async def fork(self, *, start_rpc_worker: bool = False) -> "OrionBusAsync":
        """
        Create an independent bus client sharing config/codec with a fresh Redis connection.

        Useful for nested RPC in services that already maintain a long-lived subscriber on
        another bus instance.
        """
        child = OrionBusAsync(
            url=self.url,
            enabled=self.enabled,
            codec=self.codec,
            enforce_catalog=self.enforce_catalog,
        )
        if start_rpc_worker:
            await child.connect()
            child.start_rpc_worker()
        return child

    def get_rpc_health_snapshot(self) -> RpcHealthSnapshot:
        """Drain this instance's accumulated real rpc_request() outcomes since the
        last call (or since construction). Read-only w.r.t. the bus itself -- no I/O,
        no publish. No automatic periodic caller exists yet; see rpc_health.py's
        module docstring for why that's a separate, deliberately deferred decision."""
        return self._rpc_health.snapshot_and_reset()

    def start_rpc_worker(self) -> None:
        if not self.enabled:
            return
        if self._rpc_worker_task and not self._rpc_worker_task.done():
            return
        self._rpc_worker_running = True
        self._rpc_worker_task = asyncio.create_task(self._run_rpc_only())
        logger.info("[rpc-fork] worker started")

    async def _run_rpc_only(self) -> None:
        if not self.enabled:
            return
        rpc_redis = self._create_pubsub_redis()
        self._rpc_pubsub = rpc_redis.pubsub()
        try:
            while self._rpc_worker_running:
                # No reply-channel is subscribed yet (subscriptions happen lazily,
                # on-demand, from _rpc_subscribe() as RPC calls come in) — redis-py's
                # PubSub.get_message() -> parse_response() raises RuntimeError
                # ("pubsub connection not set") when self.connection is None, since a
                # subscribe()/psubscribe() call is what actually opens the underlying
                # connection. Without this guard the worker crashes on its very first
                # loop iteration, every single startup, before any turn ever runs.
                if self._rpc_pubsub.connection is None:
                    await asyncio.sleep(0.05)
                    continue
                try:
                    msg = await self._rpc_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg and msg.get("type") in ("message", "pmessage"):
                        # Handling (decode/dispatch) is inside the same try as the read:
                        # a bad payload must degrade to reconnect-and-retry like a
                        # transport error, not kill the worker — otherwise a malformed
                        # message reintroduces the exact "silently permanent fallback to
                        # ad-hoc subscribe" failure this fix exists to close, just via a
                        # different trigger.
                        await self._handle_rpc_result(msg)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Transport-level failure (e.g. a network blip severing the
                    # connection mid-read) — reconnect and re-subscribe to whatever
                    # reply channels were already registered, instead of letting the
                    # whole worker die silently and permanently degrading every future
                    # HarnessGovernorClient.run() to the ad-hoc per-turn fallback path.
                    logger.warning("[rpc-fork] pubsub read/handle failed, reconnecting: %s", exc)
                    # Hold the same lock every _rpc_subscribe() caller holds while
                    # touching self._rpc_pubsub/_rpc_subscribed — without it, a
                    # concurrent caller mid-subscribe() on the object we're about to
                    # close() can raise or land on a connection we're discarding.
                    async with self._rpc_lock:
                        with suppress(Exception):
                            await self._rpc_pubsub.close()
                        with suppress(Exception):
                            await rpc_redis.close()
                        rpc_redis = self._create_pubsub_redis()
                        self._rpc_pubsub = rpc_redis.pubsub()
                        if self._rpc_subscribed:
                            try:
                                await self._rpc_pubsub.subscribe(*sorted(self._rpc_subscribed))
                            except Exception as resub_exc:
                                # Do not suppress-and-forget: if Redis is still down,
                                # self._rpc_pubsub.connection stays None and the
                                # top-of-loop guard above retries on its own — but log
                                # it, or a sustained outage goes completely invisible.
                                logger.warning(
                                    "[rpc-fork] resubscribe failed after reconnect, will retry: %s",
                                    resub_exc,
                                )
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[rpc-fork] worker crashed unexpectedly, RPC calls will fall back to ad-hoc subscribe")
            raise
        finally:
            if self._rpc_pubsub is not None:
                with suppress(Exception):
                    if self._rpc_subscribed:
                        await self._rpc_pubsub.unsubscribe(*sorted(self._rpc_subscribed))
                with suppress(Exception):
                    await self._rpc_pubsub.close()
                self._rpc_pubsub = None
            with suppress(Exception):
                await rpc_redis.close()
            self._rpc_worker_running = False

    async def _handle_rpc_result(self, msg: dict) -> None:
        channel_raw = msg.get("channel")
        channel = channel_raw.decode("utf-8", "ignore") if isinstance(channel_raw, (bytes, bytearray)) else str(channel_raw)
        decoded = self.codec.decode(msg.get("data"))
        if not decoded.ok:
            return
        corr = str(getattr(decoded.envelope, "correlation_id", "") or "")
        logger.info("[rpc-fork] reply received corr_id=%s reply_channel=%s", corr, channel)
        key = (channel, corr)
        fut = self._pending_rpc.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(msg)
            logger.info("[rpc-fork] future resolved corr_id=%s reply_channel=%s", corr, channel)

    async def _rpc_subscribe(self, reply_channel: str) -> None:
        await self.connect()
        if self._rpc_pubsub is None and self._rpc_worker_running:
            for _ in range(50):
                if self._rpc_pubsub is not None:
                    break
                await asyncio.sleep(0.01)
        if self._rpc_pubsub is None:
            raise RuntimeError("RPC worker pubsub is not initialized")
        if reply_channel in self._rpc_subscribed:
            logger.info("[rpc] reply-listener already subscribed reply_channel=%s", reply_channel)
            return
        logger.info("[rpc] reply-listener creating reply_channel=%s", reply_channel)
        await self._rpc_pubsub.subscribe(reply_channel)
        self._rpc_subscribed.add(reply_channel)
        logger.info("[rpc] reply-listener ready reply_channel=%s", reply_channel)

    async def connect(self) -> None:
        if not self.enabled:
            return
        if self._redis is None:
            self._redis = aioredis.from_url(self.url, **self._command_redis_kwargs())
            await self._redis.ping()

    async def close(self) -> None:
        self._rpc_worker_running = False
        if self._rpc_worker_task is not None:
            self._rpc_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._rpc_worker_task
            self._rpc_worker_task = None
        for fut in self._pending_rpc.values():
            if not fut.done():
                fut.cancel()
        self._pending_rpc.clear()
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("OrionBusAsync not connected. Call await connect().")
        return self._redis

    async def publish(self, channel: str, msg: BaseEnvelope | Dict[str, Any]) -> None:
        if not self.enabled:
            return
        enforcer.validate(channel)
        self._validate_payload(channel, msg)
        data = self.codec.encode(msg)  # bytes
        await self.redis.publish(channel, data)
        if self.track_velocity:
            await self._record_velocity(channel)

    async def _record_velocity(self, channel: str) -> None:
        """Best-effort per-channel publish counter (see orion.bus.velocity).

        Runs after the real publish already succeeded above -- a broken
        counter must never fail or slow down an otherwise-successful
        publish, so any error here is logged and swallowed, not raised.
        """
        try:
            key = velocity_key(channel, datetime.now(timezone.utc))
            pipe = self.redis.pipeline(transaction=False)
            pipe.incr(key)
            pipe.expire(key, DEFAULT_BUCKET_TTL_SEC)
            await pipe.execute()
        except Exception:
            # WARNING, not debug: this is the only signal that Phase 1's
            # live-data gate (docs/superpowers/specs/2026-07-23-bus-channel-
            # velocity-census-design.md) would otherwise miss a persistently
            # broken counter (e.g. Redis ACL denial, eviction under memory
            # pressure) silently reading as "traffic is just quiet."
            logger.warning(
                "bus velocity tracking failed for channel=%s", channel, exc_info=True
            )

    def _validate_payload(self, channel: str, msg: BaseEnvelope | Dict[str, Any]) -> None:
        entry = enforcer.entry_for(channel)
        if not entry:
            return
        schema_id = entry.get("schema_id")
        if not schema_id:
            return
        payload = None
        if isinstance(msg, BaseEnvelope):
            payload = msg.payload
        elif isinstance(msg, dict) and msg.get("schema") == "orion.envelope":
            try:
                env = BaseEnvelope.model_validate(msg)
            except ValidationError as exc:
                raise ValueError(f"Envelope validation failed for channel {channel}") from exc
            payload = env.payload
        if payload is None:
            return
        # Boundary rule: payloads on the bus must be JSON-ish.
        # If a producer passes a Pydantic model, normalize to a dict before validation.
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")

        model = resolve_schema_id(schema_id)
        try:
            model.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Payload validation failed for channel {channel} (schema_id={schema_id})"
            ) from exc

    @asynccontextmanager
    async def subscribe(self, *channels: str, patterns: bool = False) -> AsyncIterator[aioredis.client.PubSub]:
        if not self.enabled:
            raise RuntimeError("Bus disabled")
        pubsub_redis = self._create_pubsub_redis()
        pubsub = pubsub_redis.pubsub()
        if patterns:
            await pubsub.psubscribe(*channels)
        else:
            await pubsub.subscribe(*channels)
        try:
            yield pubsub
        finally:
            try:
                if patterns:
                    await pubsub.punsubscribe(*channels)
                else:
                    await pubsub.unsubscribe(*channels)
            finally:
                with suppress(Exception):
                    await pubsub.close()
                with suppress(Exception):
                    await pubsub_redis.close()

    async def iter_messages(self, pubsub: aioredis.client.PubSub) -> AsyncIterator[dict]:
        """
        Unified async message iterator. Yields dicts with fields similar to redis-py's listen().
        """
        async for msg in pubsub.listen():
            mtype = msg.get("type")
            if mtype not in ("message", "pmessage"):
                continue
            yield msg

    async def _emit_rpc_timeout_grammar(
        self,
        *,
        request_channel: str,
        reply_channel: str,
        corr: str,
        timeout_sec: float,
        timeout_elapsed_ms: float,
    ) -> None:
        """Real, bus-wide RPC-timeout grammar marker for the transport metacog trigger
        (docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md, Option C).

        Generalizes the existing exec_turn_timeout/stance_timeout markers
        (services/orion-hub/scripts/grammar_emit.py, scoped to one harness/thought RPC
        each) to every rpc_request() timeout across all 37+ real call sites sharing this
        one client. Same trace-lane family as the pre-existing bus.transport: grammar
        producer (config/substrate-lattice/grammar_producer_registry.v1.yaml) but a
        distinct semantic_role, so this doesn't collide with or get consumed by the old
        transport_bus_reducer/bus_health_observed pipeline.

        Fire-and-forget, never raises past this boundary -- same guarantee as
        RpcHealthAggregator.record_timeout(), which this call always accompanies (see
        both call sites in rpc_request()). Only reached on the rare/exceptional timeout
        path, not the hot success path, so a real network publish here (vs. the
        in-memory-only aggregator) does not carry rpc_request()'s own hot-path overhead
        concern (PR #1299's benchmark scope).
        """
        try:
            from orion.grammar.publish import publish_grammar_event
            from orion.schemas.grammar import (
                GrammarAtomV1,
                GrammarEventV1,
                GrammarProvenanceV1,
            )

            now = datetime.now(timezone.utc)
            trace_id = f"bus.transport:rpc_timeout:{corr}"
            event_id = f"{trace_id}:{uuid.uuid4().hex[:12]}"
            provenance = GrammarProvenanceV1(
                source_service="orion-bus",
                source_component="rpc_request_timeout",
                source_event_id=corr,
                source_trace_id=trace_id,
            )
            atom = GrammarAtomV1(
                atom_id=event_id,
                trace_id=trace_id,
                atom_type="uncertainty_marker",
                semantic_role="rpc_transport_timeout",
                layer="transport",
                dimensions=["transport", "bus", "rpc", "liveness"],
                summary=(
                    f"RPC timeout: {request_channel} -> {reply_channel} "
                    f"after {timeout_sec:.1f}s (elapsed {timeout_elapsed_ms:.1f}ms)"
                ),
                text_value=request_channel,
                confidence=1.0,
                salience=0.9,
                source_event_id=corr,
            )
            event = GrammarEventV1(
                event_id=event_id,
                event_kind="atom_emitted",
                trace_id=trace_id,
                correlation_id=corr,
                emitted_at=now,
                layer="transport",
                dimensions=["transport", "bus", "rpc", "liveness"],
                atom=atom,
                provenance=provenance,
            )
            await publish_grammar_event(self, event, source_name="orion-bus", correlation_id=None)
        except Exception:
            logger.warning("rpc_timeout_grammar_publish_failed corr=%s", corr, exc_info=True)

    async def rpc_request(
        self,
        request_channel: str,
        envelope: BaseEnvelope,
        *,
        reply_channel: str,
        timeout_sec: float = 60.0,
    ) -> dict:
        """
        Publish `envelope` to request_channel and await first message on reply_channel.
        """
        started = perf_counter()
        corr = str(envelope.correlation_id)
        if self._rpc_worker_task and not self._rpc_worker_task.done():
            key = (reply_channel, corr)
            fut = asyncio.get_running_loop().create_future()
            self._pending_rpc[key] = fut
            try:
                async with self._rpc_lock:
                    await self._rpc_subscribe(reply_channel)
                logger.info(
                    "[rpc] publish begin corr_id=%s request_channel=%s reply_channel=%s path=worker elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    (perf_counter() - started) * 1000.0,
                )
                await self.publish(request_channel, envelope)
                logger.info(
                    "[rpc] publish success corr_id=%s request_channel=%s reply_channel=%s path=worker elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    (perf_counter() - started) * 1000.0,
                )
                logger.info(
                    "[rpc] waiting for reply corr_id=%s reply_channel=%s timeout_sec=%.2f path=worker",
                    corr,
                    reply_channel,
                    timeout_sec,
                )
                result = await asyncio.wait_for(fut, timeout=timeout_sec)
                # docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md:
                # the worker path previously had no completion log at all on success --
                # only the inline path (below) logged "reply received". Confirmed live
                # (2026-07-23): of 33 real worker/inline RPC calls sampled from
                # orion-cortex-orch's logs, only 6 used the inline path; the other 27
                # completed with zero observable latency, since success here just returned
                # `fut`'s result silently. This line closes that gap -- same log shape as
                # the inline path's own "reply received" line, so a parser doesn't need to
                # special-case the two RPC paths. Same log-call shape (str/str/float args,
                # %s/%.1f format) as the 4 other logger.info() calls already on this exact
                # path, so no new overhead profile is introduced -- a formal before/after
                # benchmark is scoped to the design spec's next patch step (the in-memory
                # aggregator), not this one, since that step is where a second per-call
                # code path actually gets added.
                success_elapsed_ms = (perf_counter() - started) * 1000.0
                logger.info(
                    "[rpc] reply received corr_id=%s reply_channel=%s path=worker elapsed_ms=%.1f",
                    corr,
                    reply_channel,
                    success_elapsed_ms,
                )
                self._rpc_health.record_success(request_channel=request_channel, latency_ms=success_elapsed_ms)
                return result
            except asyncio.TimeoutError:
                timeout_elapsed_ms = (perf_counter() - started) * 1000.0
                logger.error(
                    "[rpc] timeout waiting for reply corr_id=%s request_channel=%s reply_channel=%s timeout_sec=%.2f elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    timeout_sec,
                    timeout_elapsed_ms,
                )
                self._rpc_health.record_timeout(request_channel=request_channel, elapsed_ms=timeout_elapsed_ms)
                await self._emit_rpc_timeout_grammar(
                    request_channel=request_channel,
                    reply_channel=reply_channel,
                    corr=corr,
                    timeout_sec=timeout_sec,
                    timeout_elapsed_ms=timeout_elapsed_ms,
                )
                raise TimeoutError(f"RPC timeout waiting on {reply_channel}")
            finally:
                self._pending_rpc.pop(key, None)

        async with self.subscribe(reply_channel) as pubsub:
            try:
                logger.info(
                    "[rpc] publish begin corr_id=%s request_channel=%s reply_channel=%s path=inline elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    (perf_counter() - started) * 1000.0,
                )
                await self.publish(request_channel, envelope)
                logger.info(
                    "[rpc] publish success corr_id=%s request_channel=%s reply_channel=%s path=inline elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    (perf_counter() - started) * 1000.0,
                )

                async def _wait_one():
                    async for msg in self.iter_messages(pubsub):
                        success_elapsed_ms = (perf_counter() - started) * 1000.0
                        logger.info(
                            "[rpc] reply received corr_id=%s reply_channel=%s path=inline elapsed_ms=%.1f",
                            corr,
                            reply_channel,
                            success_elapsed_ms,
                        )
                        self._rpc_health.record_success(
                            request_channel=request_channel, latency_ms=success_elapsed_ms
                        )
                        return msg

                logger.info(
                    "[rpc] waiting for reply corr_id=%s reply_channel=%s timeout_sec=%.2f path=inline",
                    corr,
                    reply_channel,
                    timeout_sec,
                )
                msg = await asyncio.wait_for(_wait_one(), timeout=timeout_sec)
                return msg
            except asyncio.TimeoutError:
                timeout_elapsed_ms = (perf_counter() - started) * 1000.0
                logger.error(
                    "[rpc] timeout waiting for reply corr_id=%s request_channel=%s reply_channel=%s timeout_sec=%.2f elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    timeout_sec,
                    timeout_elapsed_ms,
                )
                self._rpc_health.record_timeout(request_channel=request_channel, elapsed_ms=timeout_elapsed_ms)
                await self._emit_rpc_timeout_grammar(
                    request_channel=request_channel,
                    reply_channel=reply_channel,
                    corr=corr,
                    timeout_sec=timeout_sec,
                    timeout_elapsed_ms=timeout_elapsed_ms,
                )
                raise TimeoutError(f"RPC timeout waiting on {reply_channel}")

    async def rpc_legacy_dict(
        self,
        request_channel: str,
        payload: Dict[str, Any],
        *,
        reply_prefix: str,
        timeout_sec: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Compatibility RPC helper for legacy dict-style services.

        Convention:
          - caller publishes a dict that includes `reply_channel` and `correlation_id`
          - callee replies with either a Titanium envelope OR a legacy json dict
        """
        if not self.enabled:
            raise RuntimeError("Bus disabled")

        from uuid import uuid4
        corr = uuid4()
        reply_channel = f"{reply_prefix}{corr}"

        msg_payload = dict(payload)
        msg_payload.setdefault("reply_channel", reply_channel)
        msg_payload.setdefault("correlation_id", str(corr))

        async with self.subscribe(reply_channel) as pubsub:
            await self.publish(request_channel, msg_payload)
            try:
                async def _wait_one():
                    async for msg in self.iter_messages(pubsub):
                        return msg
                msg = await asyncio.wait_for(_wait_one(), timeout=timeout_sec)
            except asyncio.TimeoutError as te:
                raise TimeoutError(f"RPC timeout waiting on {reply_channel}") from te

        decoded = self.codec.decode(msg.get("data"))
        if decoded.ok and isinstance(decoded.envelope.payload, dict):
            return decoded.envelope.payload

        # legacy fallback: raw json dict
        data = msg.get("data")
        if isinstance(data, (bytes, bytearray)):
            try:
                import json

                return json.loads(data.decode("utf-8", "ignore"))
            except Exception:
                pass
        return {"ok": False, "error": decoded.error or "decode_failed"}
