from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Optional

from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.codec import OrionCodec

from .settings import settings
from .substrate.bus_synaptic import BUS_SYNAPTIC_ZSCORE_SATURATION, query_real_bus_synaptic_raw_mean_abs_z
from .substrate.ensemble import EnsembleConfig, EnsembleH1ResultV1, EnsembleSubstrate
from .substrate.reconstruction import compute_h1_ensemble
from .substrate.routing import (
    ORGAN_SITE_MAP,
    SiteAssignment,
    UnroutableAtomTypeError,
    UnroutableOrganError,
    route_atom,
)

logger = logging.getLogger("orion-heartbeat.service")

_GRAMMAR_EVENT_KIND = "grammar.event.v1"  # orion/bus/channels.yaml's
# orion:grammar:event message_kind -- checked against the live channel
# registry, not guessed.


class HeartbeatService(BaseChassis):
    """Additive, read-only consumer of the existing orion:grammar:event
    stream (see design doc's "Current architecture" for why this reuses
    orion-substrate-runtime's already-solved standardization work instead of
    the 2026-05-01 charter's bespoke per-organ reducers). Publishes nothing
    in v0 -- no downstream consumer, no phi broadcast, ablation-safe by
    construction (nothing depends on this service's output yet).

    **2026-07-28: single HeartbeatSubstrate replaced with an N-trajectory
    EnsembleSubstrate** (docs/superpowers/specs/2026-07-28-precision-weighted-
    attention-organ-and-heartbeat-discrimination-design.md) -- see that
    module's docstring for why an ensemble, not a single trajectory. Message
    intake is now decoupled from the N-trajectory absorb() cost via
    `_absorb_queue`/`_absorb_worker_loop` (a burst previously would have
    blocked the single asyncio event loop this service runs on for the full
    N-trajectory absorb() duration on every message; now it blocks the
    queue's freshness, not the bus consumer's ability to keep accepting
    messages). Dissipation (decay + bus_synaptic-driven reheat) runs on its
    OWN wall-clock timer (`_decay_reheat_loop`), independent of message
    arrival -- if organs go quiet, real messages stop, but dissipation must
    keep running or "rest" could never actually be reached (Missing
    Question 7, resolved this way).

    **`_ensemble_lock` -- caught by review before this was ever merged.**
    `absorb()`/`decay_reheat_tick()` run via `asyncio.to_thread` (see those
    methods' docstrings for why -- CPU-bound quimb calls were blocking the
    event loop). Offloading them to threads without also serializing access
    to the shared, mutable `EnsembleSubstrate`/quimb `MatrixProductState`
    state is a real, separate bug: two independently-scheduled tasks (or a
    thread-offloaded call racing a same-thread read like `stats()`/
    `_h1_loop`) can mutate/read the same tensor network concurrently, with
    no guarantee numpy/BLAS operations are safe to interleave that way. All
    four places that touch `self.ensemble` -- absorb, decay/reheat, H1
    computation, and stats() -- now hold this lock, so the actual heavy
    quimb work still happens off the event-loop thread (the lock itself
    doesn't block other unrelated coroutines, like message intake, which
    never touches `self.ensemble` directly), but never concurrently with
    another ensemble read/write.
    """

    def __init__(self) -> None:
        super().__init__(
            ChassisConfig(
                service_name=settings.service_name,
                service_version=settings.service_version,
                node_name=settings.node_name or "unknown",
                instance_id=settings.instance_id,
                bus_url=settings.orion_bus_url,
                bus_enabled=settings.orion_bus_enabled,
                heartbeat_interval_sec=settings.heartbeat_interval_sec,
                health_channel=settings.health_channel,
            )
        )
        self.codec = OrionCodec()
        self.ensemble = EnsembleSubstrate(
            config=EnsembleConfig(
                n_trajectories=settings.n_trajectories,
                gamma=settings.decay_gamma,
                base_decay_prob=settings.base_decay_prob,
                decay_spread_sensitivity=settings.decay_spread_sensitivity,
                reheat_strength=settings.reheat_strength,
                reheat_prob_scale=settings.reheat_prob_scale,
            ),
            base_seed=settings.substrate_seed,
        )
        self.latest_h1: Optional[EnsembleH1ResultV1] = None
        self._absorb_queue: asyncio.Queue[SiteAssignment] = asyncio.Queue(
            maxsize=settings.absorb_queue_maxsize
        )
        # Serializes every touch of self.ensemble's mutable quimb state --
        # see class docstring's "_ensemble_lock" section for why this exists.
        self._ensemble_lock = asyncio.Lock()
        # Counters surfaced on /health -- not published anywhere, purely for
        # the debug HTTP surface (design doc's inspectability commitment).
        self.events_seen = 0
        self.events_queued = 0
        self.events_absorbed = 0
        self.events_dropped_queue_full = 0
        self.events_skipped_organ = 0
        self.events_skipped_atom_type = 0
        self.events_skipped_no_atom = 0
        self.events_skipped_malformed = 0

    def latest_h1_dict(self) -> dict[str, Any] | None:
        if self.latest_h1 is None:
            return None
        d = asdict(self.latest_h1)
        d["generated_at"] = self.latest_h1.generated_at.isoformat()
        return d

    async def stats(self) -> dict[str, Any]:
        """Async (not a plain method) -- reads self.ensemble's mutable state
        (tick_count/max_bond/norm), which must be lock-protected the same as
        every other touch point (see class docstring's "_ensemble_lock").
        """
        async with self._ensemble_lock:
            ensemble_stats = {
                "tick_count": self.ensemble.tick_count(),
                "n_trajectories": self.ensemble.config.n_trajectories,
                "seeds": list(self.ensemble.seeds),
                "max_bond": self.ensemble.max_bond(),
                "norm": self.ensemble.norm(),
            }
        return {
            "events_seen": self.events_seen,
            "events_queued": self.events_queued,
            "events_absorbed": self.events_absorbed,
            "events_dropped_queue_full": self.events_dropped_queue_full,
            "events_skipped_organ": self.events_skipped_organ,
            "events_skipped_atom_type": self.events_skipped_atom_type,
            "events_skipped_no_atom": self.events_skipped_no_atom,
            "events_skipped_malformed": self.events_skipped_malformed,
            **ensemble_stats,
            "absorb_queue_size": self._absorb_queue.qsize(),
            "allowlisted_organs": sorted(ORGAN_SITE_MAP.keys()),
        }

    def _handle_atom_payload(self, atom: dict[str, Any]) -> None:
        """One GrammarAtomV1 dict (already extracted from a decoded
        GrammarEventV1 payload). Skips (does not raise) on anything outside
        v0's organ/atom_type allowlist -- a malformed or out-of-scope event
        must not crash the tick loop; see routing.route_atom's own
        docstring for why routing itself still raises.

        Routes the atom and enqueues it for absorption -- does NOT call
        ensemble.absorb() directly (see class docstring: that's now
        _absorb_worker_loop's job, kept off the message-intake path).
        """
        provenance = atom.get("_provenance") or {}
        source_service = str(provenance.get("source_service") or "")
        atom_type = str(atom.get("atom_type") or "")

        try:
            assignment = route_atom(
                source_service=source_service,
                atom_type=atom_type,
                confidence=atom.get("confidence"),
                salience=atom.get("salience"),
                uncertainty=atom.get("uncertainty"),
            )
        except UnroutableOrganError:
            self.events_skipped_organ += 1
            return
        except UnroutableAtomTypeError:
            self.events_skipped_atom_type += 1
            logger.warning(
                "heartbeat_unroutable_atom_type atom_type=%s source_service=%s "
                "-- orion/schemas/grammar.py may have gained a new AtomType not "
                "yet covered in routing.ATOM_TYPE_OPERATOR_KIND",
                atom_type,
                source_service,
            )
            return
        except (ValueError, TypeError) as exc:
            # route_atom()'s confidence/salience/uncertainty float() coercion
            # can raise these for a malformed atom (heartbeat works directly
            # off the decoded dict, not a re-validated GrammarAtomV1, so a
            # non-numeric value isn't ruled out upstream). Found by review:
            # this was previously falling through to _run()'s generic
            # except-Exception handler under a bare "heartbeat_handle_failed"
            # log line, uncounted -- now has its own counter and log line,
            # same as the two named routing exceptions above.
            self.events_skipped_malformed += 1
            logger.warning(
                "heartbeat_malformed_atom_fields source_service=%s atom_type=%s err=%s",
                source_service,
                atom_type,
                exc,
            )
            return

        try:
            self._absorb_queue.put_nowait(assignment)
            self.events_queued += 1
        except asyncio.QueueFull:
            self.events_dropped_queue_full += 1
            logger.warning(
                "heartbeat_absorb_queue_full dropping atom source_service=%s -- "
                "absorb worker is falling behind real event rate",
                source_service,
            )

    async def _handle_grammar_message(self, payload_dict: dict[str, Any]) -> None:
        self.events_seen += 1
        if payload_dict.get("event_kind") != "atom_emitted":
            return
        atom = payload_dict.get("atom")
        if not isinstance(atom, dict):
            self.events_skipped_no_atom += 1
            return
        # provenance lives at the GrammarEventV1 level, not on the atom
        # itself (orion/schemas/grammar.py) -- stash it under the atom dict
        # so _handle_atom_payload has a single argument to work with.
        atom = dict(atom)
        atom["_provenance"] = payload_dict.get("provenance") or {}
        self._handle_atom_payload(atom)

    async def _absorb_worker_loop(self) -> None:
        """Drains _absorb_queue, applying each real atom to every trajectory
        in the ensemble. Kept off the bus-consumer coroutine on purpose (see
        class docstring) -- a burst of events queues up here instead of
        blocking message intake.

        **Deliberately checks the queue BEFORE checking self._stop**, not
        `while not self._stop.is_set(): ... queue.get() ...`. Found by test
        (test_run_loop_robustness.py's end-to-end tests): with the stop-first
        ordering, `_stop` can already be set by the time this task gets its
        very first scheduling turn (the bus-consumer coroutine in `_run()`
        processes a fast, non-yielding fake message stream and calls
        `self._stop.set()` before this task ever runs) -- the loop would
        then exit immediately without ever draining an already-queued item,
        defeating `_run()`'s own queue.join() drain-before-cancel logic.
        Checking the queue first guarantees every already-queued item gets
        processed before stop is honored, in both real and test operation.

        **`ensemble.absorb()` runs via `asyncio.to_thread`, not inline.**
        Confirmed live 2026-07-28: `ensemble.absorb()`/`decay_reheat_tick()`
        are synchronous CPU-bound quimb calls (~118ms for N=8) -- running
        them directly on this coroutine blocks the ENTIRE single-threaded
        event loop, not just this worker's own queue draining. Under real
        backlog (confirmed live: `absorb_queue_size` reached 30 under normal
        traffic) this starved `_decay_reheat_loop`'s own async socket I/O
        long enough to produce a spurious `bus_synaptic_query_failed
        err=Timeout connecting to server` -- FalkorDB itself was fine
        (reproduced the identical query manually with zero latency); the
        event loop simply never got a chance to service that socket's
        response while an absorb() burst was running. Offloading to a
        thread lets numpy/scipy's C-level linear algebra proceed while the
        event loop keeps servicing other coroutines (message intake, health
        endpoint, the dissipation loop's real I/O).
        """
        while True:
            try:
                assignment = await asyncio.wait_for(self._absorb_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._stop.is_set():
                    break
                continue
            try:
                async with self._ensemble_lock:
                    await asyncio.to_thread(self.ensemble.absorb, assignment)
                self.events_absorbed += 1
            except Exception as exc:  # noqa: BLE001 - one bad absorb must not kill the worker
                logger.warning("heartbeat_absorb_failed err=%s", exc)
            finally:
                self._absorb_queue.task_done()

    async def _drain_absorb_queue(self) -> int:
        """Process every currently-queued item synchronously, without the
        background _absorb_worker_loop task -- for tests (and any other
        caller) that invoke _handle_grammar_message()/_handle_atom_payload()
        directly rather than running the full _run() loop, where there is
        no background worker to drain the queue on its own. Not used by
        _run() itself (real operation always has the worker task running)."""
        processed = 0
        while not self._absorb_queue.empty():
            assignment = self._absorb_queue.get_nowait()
            async with self._ensemble_lock:
                self.ensemble.absorb(assignment)
            self.events_absorbed += 1
            self._absorb_queue.task_done()
            processed += 1
        return processed

    async def _decay_reheat_loop(self) -> None:
        """Wall-clock-driven dissipation tick, independent of message
        arrival -- see class docstring for why this can't be tick-driven.
        `ensemble.decay_reheat_tick()` runs via `asyncio.to_thread` -- see
        `_absorb_worker_loop`'s docstring for why (same CPU-bound-call-
        blocks-the-event-loop finding, confirmed live 2026-07-28)."""
        while not self._stop.is_set():
            await asyncio.sleep(settings.decay_reheat_interval_sec)
            try:
                raw_z = await query_real_bus_synaptic_raw_mean_abs_z(
                    settings.falkordb_uri, settings.falkordb_bus_graph
                )
                real_signal = min(1.0, raw_z / BUS_SYNAPTIC_ZSCORE_SATURATION)
                reheat_prob = settings.reheat_prob_scale * real_signal
                async with self._ensemble_lock:
                    await asyncio.to_thread(self.ensemble.decay_reheat_tick, reheat_prob)
            except Exception as exc:  # noqa: BLE001 - must not kill the dissipation loop
                logger.warning("heartbeat_decay_reheat_failed err=%s", exc)

    async def _h1_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(settings.h1_interval_sec)
            try:
                async with self._ensemble_lock:
                    self.latest_h1 = compute_h1_ensemble(self.ensemble)
                logger.info(
                    "heartbeat_h1_computed tick_count=%d mean_ratio=%.4f std_ratio=%.4f "
                    "verdict=%s seeds=%s",
                    self.latest_h1.tick_count,
                    self.latest_h1.mean_ratio,
                    self.latest_h1.std_ratio,
                    self.latest_h1.verdict,
                    self.latest_h1.seeds,
                )
            except Exception as exc:  # noqa: BLE001 - must not kill the loop
                logger.warning("heartbeat_h1_computation_failed err=%s", exc)

    async def _run(self) -> None:
        h1_task = asyncio.create_task(self._h1_loop(), name="heartbeat-h1-loop")
        absorb_task = asyncio.create_task(self._absorb_worker_loop(), name="heartbeat-absorb-worker")
        decay_reheat_task = asyncio.create_task(
            self._decay_reheat_loop(), name="heartbeat-decay-reheat-loop"
        )
        try:
            async with self.bus.subscribe(settings.channel_grammar_event) as pubsub:
                async for msg in self.bus.iter_messages(pubsub):
                    if self._stop.is_set():
                        break

                    decoded = self.codec.decode(msg.get("data"))
                    if not decoded.ok or decoded.envelope is None:
                        logger.warning(
                            "heartbeat_decode_failed channel=%s error=%s",
                            settings.channel_grammar_event,
                            decoded.error,
                        )
                        continue

                    env = decoded.envelope
                    if env.kind != _GRAMMAR_EVENT_KIND:
                        continue
                    payload_dict = env.payload if isinstance(env.payload, dict) else {}

                    try:
                        await self._handle_grammar_message(payload_dict)
                    except Exception as exc:  # noqa: BLE001 - one bad event must not kill the subscriber
                        logger.warning("heartbeat_handle_failed err=%s", exc)
        finally:
            # Drain whatever's already queued before cancelling the absorb
            # worker -- correct shutdown behavior (finish absorbing real
            # events already accepted rather than silently dropping them),
            # and also what makes this loop's own end-to-end tests
            # deterministic (a fake message stream with no genuine await
            # points never yields control to the background worker
            # otherwise; awaiting queue.join() is a real suspend point).
            try:
                await asyncio.wait_for(self._absorb_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "heartbeat_absorb_queue_drain_timeout size=%d", self._absorb_queue.qsize()
                )
            for task in (h1_task, absorb_task, decay_reheat_task):
                task.cancel()
            for task in (h1_task, absorb_task, decay_reheat_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
