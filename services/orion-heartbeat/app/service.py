from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.codec import OrionCodec
from orion.grammar.publish import publish_grammar_event
from orion.schemas.heartbeat_seat_rollup import (
    HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC,
    HEARTBEAT_SEAT_ROLLUP_CHANNEL,
    HEARTBEAT_SEAT_ROLLUP_KIND,
    HeartbeatSeatRollupV1,
)

from .seat_rollup import SeatRollupAccumulator, parse_emitted_epoch

from .settings import settings
from .substrate.ensemble import EnsembleConfig, EnsembleH1ResultV1, EnsembleSubstrate
from .substrate.reconstruction import compute_h1_ensemble, verdict_thresholds
from .substrate.proprioception import OrganFireWindow
from .substrate.routing import (
    ORGAN_SITE_MAP,
    SEAT_ORDER,
    SELF_SOURCE_SERVICE,
    SITE_ORGAN_MAP,
    UNROUTED_PRODUCERS,
    SiteAssignment,
    SkippedInputError,
    UnroutableAtomTypeError,
    UnroutableOrganError,
    catalog_grammar_producers,
    input_rule,
    route_atom,
    routing_version,
)
from .substrate.verdict_atoms import (
    MEAN_RATIO_MEANING,
    VERDICT_MEANING,
    H1Tick,
    HourlyWindow,
    build_summary_event,
    safe_token,
)

logger = logging.getLogger("orion-heartbeat.service")

_GRAMMAR_EVENT_KIND = "grammar.event.v1"  # orion/bus/channels.yaml's
# orion:grammar:event message_kind -- checked against the live channel
# registry, not guessed.
_MAX_UNROUTED_SOURCE_KEYS = 32  # bound on the lifetime per-producer counter
_MAX_COUNTER_KEYS = 64  # bound on the per-reason / per-role input counters
# Seat rollup (app/seat_rollup.py): how often the flush loop looks for closed
# buckets, and how many seat rows it holds for retry while the bus refuses
# publishes (12 buckets x 5 seats = 1 h).
SEAT_ROLLUP_FLUSH_CHECK_SEC = 10.0
SEAT_ROLLUP_MAX_PENDING = 12 * 5
SEAT_ROLLUP_PUBLISH_TIMEOUT_SEC = 5.0


def _bump(counter: dict[str, int], key: str, limit: int = _MAX_COUNTER_KEYS) -> None:
    if key not in counter and len(counter) >= limit:
        key = "other"
    counter[key] = counter.get(key, 0) + 1


class HeartbeatService(BaseChassis):
    """Additive, read-only consumer of the existing orion:grammar:event
    stream (see design doc's "Current architecture" for why this reuses
    orion-substrate-runtime's already-solved standardization work instead of
    the 2026-05-01 charter's bespoke per-organ reducers).

    **2026-10-07: publishes a bounded H1 summary atom back onto
    orion:grammar:event** (one summary per window; settled-transition atoms retired 2026-11-24 --
    the class is noise, app/substrate/verdict_atoms.py). Ledger-only: no reducer, field
    channel or prior consumes them. Its own atoms come back on the same
    subscription and are skipped by source_service before routing
    (events_skipped_self), so heartbeat can never feed on itself.

    **2026-07-28: single HeartbeatSubstrate replaced with an N-trajectory
    EnsembleSubstrate** (docs/superpowers/specs/2026-07-28-precision-weighted-
    attention-organ-and-heartbeat-discrimination-design.md) -- see that
    module's docstring for why an ensemble, not a single trajectory. Message
    intake is now decoupled from the N-trajectory absorb() cost via
    `_absorb_queue`/`_absorb_worker_loop` (a burst previously would have
    blocked the single asyncio event loop this service runs on for the full
    N-trajectory absorb() duration on every message; now it blocks the
    queue's freshness, not the bus consumer's ability to keep accepting
    messages). Dissipation (decay) runs on its OWN wall-clock timer
    (`_decay_loop`), independent of message
    arrival -- if organs go quiet, real messages stop, but dissipation must
    keep running or "rest" could never actually be reached (Missing
    Question 7, resolved this way).

    **`_ensemble_lock` -- caught by review before this was ever merged.**
    `absorb()`/`decay_tick()` run via `asyncio.to_thread` (see those
    methods' docstrings for why -- CPU-bound quimb calls were blocking the
    event loop). Offloading them to threads without also serializing access
    to the shared, mutable `EnsembleSubstrate`/quimb `MatrixProductState`
    state is a real, separate bug: two independently-scheduled tasks (or a
    thread-offloaded call racing a same-thread read like `stats()`/
    `_h1_loop`) can mutate/read the same tensor network concurrently, with
    no guarantee numpy/BLAS operations are safe to interleave that way. All
    four places that touch `self.ensemble` -- absorb, decay, H1
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
            ),
            base_seed=settings.substrate_seed,
        )
        self.latest_h1: Optional[EnsembleH1ResultV1] = None
        self.organ_fires = OrganFireWindow(window_sec=settings.organ_fire_window_sec)
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
        self.events_skipped_self = 0
        # Atoms from a routed service that its input rule skips (clockwork,
        # duplicate, structural floor, ...), counted per rule reason.
        self.events_skipped_input = 0
        self.events_skipped_input_by_reason: dict[str, int] = {}
        # Absorbed atoms per input kind (measured / presence).
        self.inputs_absorbed_by_kind: dict[str, int] = {}
        # Roles on a routed service that no ROLE_RULES entry names, so the
        # service default applied. A new emitter role shows up here.
        self.inputs_unlisted_role: dict[str, int] = {}
        # Per-producer count of atoms dropped as UnroutableOrganError --
        # previously one anonymous total. Bounded (overflow -> "other").
        self.events_skipped_organ_by_source: dict[str, int] = {}
        # orion:grammar:event's catalogued producers, read once at boot. Used
        # only to NAME what heartbeat does not route; routing stays
        # ORGAN_SITE_MAP (see routing.py for why).
        self.catalog_producers = catalog_grammar_producers()
        self.verdict_window = HourlyWindow(start=datetime.now(timezone.utc))
        self.verdict_atoms_published = 0
        self.verdict_atoms_publish_failed = 0
        self.last_verdict_atom: dict[str, Any] | None = None
        # Five-minute per-seat rollup (issue #3886). None when disabled.
        now = time.time()
        self.routing_version = routing_version()
        self.seat_rollup: Optional[SeatRollupAccumulator] = (
            SeatRollupAccumulator(
                node=settings.node_name or "unknown",
                seats=SEAT_ORDER,
                started_at_epoch=now,
                routing_version=self.routing_version,
                silence_gap_sec=settings.seat_rollup_silence_gap_sec,
            )
            if settings.seat_rollup_enabled
            else None
        )
        # When the heartbeat last stopped hearing orion:grammar:event: construction, or the end of
        # the previous _run (the chassis restarts _run after a bus drop). Every bucket from here
        # to the next subscribe is partial.
        self._rollup_unsubscribed_at: float = now
        self._rollup_subscribed = False
        self._seat_rollup_pending: deque[HeartbeatSeatRollupV1] = deque(maxlen=SEAT_ROLLUP_MAX_PENDING)
        self.seat_rollup_published = 0
        self.seat_rollup_publish_failed = 0
        self.seat_rollup_dropped_pending_full = 0
        # Atoms whose emitted_at was missing or unparseable: bucketed by receipt time instead.
        self.seat_rollup_emitted_at_fallback = 0
        self.last_seat_rollup: dict[str, Any] | None = None

    def latest_h1_dict(self) -> dict[str, Any] | None:
        if self.latest_h1 is None:
            return None
        d = asdict(self.latest_h1)
        d["generated_at"] = self.latest_h1.generated_at.isoformat()
        # 2026-11-08 replay labels (verdict_atoms.py): read verdict as noise
        # and mean_ratio as a very-short-memory sequence meter.
        d["verdict_meaning"] = VERDICT_MEANING
        d["mean_ratio_meaning"] = MEAN_RATIO_MEANING
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
            "events_skipped_self": self.events_skipped_self,
            "events_skipped_input": self.events_skipped_input,
            "events_skipped_input_by_reason": dict(self.events_skipped_input_by_reason),
            "inputs_absorbed_by_kind": dict(self.inputs_absorbed_by_kind),
            "inputs_unlisted_role": dict(self.inputs_unlisted_role),
            "events_skipped_organ_by_source": dict(self.events_skipped_organ_by_source),
            # Catalogued producers heartbeat does not route, and sources seen
            # on the wire that the catalog does not list (catalog drift).
            # None (not []) when channels.yaml could not be read: unknown is
            # not "nothing unrouted".
            "catalog_loaded": bool(self.catalog_producers),
            "catalog_producers_unrouted": (
                sorted(self.catalog_producers - set(ORGAN_SITE_MAP) - {SELF_SOURCE_SERVICE})
                if self.catalog_producers
                else None
            ),
            # Why each catalogued producer is left off the seats.
            "unrouted_producer_reasons": dict(UNROUTED_PRODUCERS),
            "uncatalogued_sources_seen": sorted(
                s for s in self.events_skipped_organ_by_source
                if self.catalog_producers and s not in self.catalog_producers and s != "other"
            ),
            "seat_rollup": self._seat_rollup_stats(),
            "verdict_atoms": {
                "enabled": settings.verdict_atoms_enabled,
                "published": self.verdict_atoms_published,
                "publish_failed": self.verdict_atoms_publish_failed,
                "verdict_meaning": VERDICT_MEANING,
                "window_start": self.verdict_window.start.isoformat(),
                "window_h1_ticks": self.verdict_window.h1_ticks,
                "last": self.last_verdict_atom,
            },
            **ensemble_stats,
            "absorb_queue_size": self._absorb_queue.qsize(),
            "absorb_queue_maxsize": settings.absorb_queue_maxsize,
            "allowlisted_organs": sorted(ORGAN_SITE_MAP.keys()),
            "organ_site_map": dict(ORGAN_SITE_MAP),
            "seat_order": list(SEAT_ORDER),
            # The live tuning actually in force, so a read-only surface can
            # explain the mechanism (decay rate, spread sensitivity,
            # scale, band edges) from the running service rather than from a
            # mirrored copy of .env_example that may not match this container.
            "config": {
                "n_trajectories": settings.n_trajectories,
                "gamma": settings.decay_gamma,
                "base_decay_prob": settings.base_decay_prob,
                "decay_spread_sensitivity": settings.decay_spread_sensitivity,
                "decay_interval_sec": settings.decay_interval_sec,
                "h1_interval_sec": settings.h1_interval_sec,
                **verdict_thresholds(),
            },
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
        semantic_role = str(atom.get("semantic_role") or "")

        if source_service == SELF_SOURCE_SERVICE:
            # Our own verdict atoms echoing back on the shared channel.
            self.events_skipped_self += 1
            return

        if source_service in ORGAN_SITE_MAP:
            _rule, listed = input_rule(source_service, semantic_role)
            if not listed:
                _bump(self.inputs_unlisted_role, f"{safe_token(source_service)}:{safe_token(semantic_role or 'none')}")

        try:
            assignment = route_atom(source_service=source_service, atom=atom)
        except UnroutableOrganError:
            self.events_skipped_organ += 1
            key = safe_token(source_service)
            if (
                key not in self.events_skipped_organ_by_source
                and len(self.events_skipped_organ_by_source) >= _MAX_UNROUTED_SOURCE_KEYS
            ):
                key = "other"
            self.events_skipped_organ_by_source[key] = self.events_skipped_organ_by_source.get(key, 0) + 1
            self.verdict_window.record_unrouted(source_service or "unknown")
            return
        except SkippedInputError as exc:
            self.events_skipped_input += 1
            _bump(self.events_skipped_input_by_reason, exc.reason)
            self._rollup_observe(atom, site_index=ORGAN_SITE_MAP[source_service], level=None)
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
            # route_atom()'s measured-field float() coercion
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

        # Before the queue: the rollup counts what the heartbeat routes, also
        # when the absorb queue is full (the replay counts those atoms too).
        self._rollup_observe(atom, site_index=assignment.site_index, level=assignment.level)

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
        if self.seat_rollup is not None:
            try:
                self.seat_rollup.observe_receipt(time.time())
            except Exception as exc:  # noqa: BLE001 - the rollup never blocks routing
                logger.warning("heartbeat_seat_rollup_receipt_failed err=%s", exc)
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
        # The seat rollup buckets by the event's emitted_at (the same column
        # grammar_events is bucketed by) and dedupes on event_id.
        atom["_emitted_at"] = payload_dict.get("emitted_at")
        atom["_event_id"] = payload_dict.get("event_id")
        self._handle_atom_payload(atom)

    def _rollup_observe(self, atom: dict[str, Any], *, site_index: int, level: Optional[float]) -> None:
        """Count one routed (level given) or skipped (level None) atom in the seat rollup. Fail-open."""
        rollup = self.seat_rollup
        if rollup is None:
            return
        try:
            now = time.time()
            emitted = parse_emitted_epoch(atom.get("_emitted_at"))
            real_emitted = emitted is not None
            if emitted is None:
                self.seat_rollup_emitted_at_fallback += 1
                emitted = now
            event_id = atom.get("_event_id")
            event_id = str(event_id) if event_id else None
            if level is None:
                outcome = rollup.observe_skip(
                    site_index=site_index, emitted_epoch=emitted, now_epoch=now, event_id=event_id,
                    record_lag=real_emitted,
                )
            else:
                outcome = rollup.observe_atom(
                    site_index=site_index, level=level, emitted_epoch=emitted, now_epoch=now, event_id=event_id,
                    record_lag=real_emitted,
                )
            if outcome == "late":
                bucket = int(emitted // HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC) * HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC
                if rollup.late_by_bucket.get(bucket) == 1:  # once per bucket
                    logger.warning(
                        "heartbeat_seat_rollup_late_atom bucket_start=%s lag_sec=%.1f -- the stored row for "
                        "this bucket undercounts it",
                        datetime.fromtimestamp(bucket, timezone.utc).isoformat(), now - emitted,
                    )
        except Exception as exc:  # noqa: BLE001 - the rollup never blocks routing
            logger.warning("heartbeat_seat_rollup_observe_failed err=%s", exc)

    def _rollup_mark_subscribed(self) -> None:
        """Called once the subscription is open: the span since the heartbeat last stopped hearing
        the channel was blind, so its buckets are partial. Fail-open."""
        self._rollup_subscribed = True
        if self.seat_rollup is None:
            return
        try:
            self.seat_rollup.mark_blind(self._rollup_unsubscribed_at, time.time())
        except Exception as exc:  # noqa: BLE001 - the rollup never blocks the subscriber
            logger.warning("heartbeat_seat_rollup_mark_blind_failed err=%s", exc)

    def _seat_rollup_stats(self) -> dict[str, Any]:
        rollup = self.seat_rollup
        return {
            "enabled": rollup is not None,
            "channel": HEARTBEAT_SEAT_ROLLUP_CHANNEL,
            "open_buckets": len(rollup) if rollup is not None else 0,
            "published": self.seat_rollup_published,
            "publish_failed": self.seat_rollup_publish_failed,
            "pending": len(self._seat_rollup_pending),
            "dropped_pending_full": self.seat_rollup_dropped_pending_full,
            # Atoms stamped in an already-published bucket (or before start), more than one
            # bucket in the future, or repeated in a bucket: not added to any row.
            "late_atoms_dropped": rollup.late if rollup is not None else 0,
            "late_by_bucket": (
                {datetime.fromtimestamp(b, timezone.utc).isoformat(): n for b, n in sorted(rollup.late_by_bucket.items())}
                if rollup is not None
                else {}
            ),
            "late_atoms_dropped_by_seat": (
                dict(zip(SEAT_ORDER, rollup.late_by_seat)) if rollup is not None else {}
            ),
            "future_atoms_dropped": rollup.future if rollup is not None else 0,
            # A repeated event_id is counted once, in the bucket of its first copy, like the
            # ledger. cross_bucket: the copies fell in different buckets (harness finalize, #3900).
            "duplicate_atoms_ignored": rollup.duplicates if rollup is not None else 0,
            "duplicate_atoms_cross_bucket": rollup.duplicates_cross_bucket if rollup is not None else 0,
            # Receive time minus emitted_at per seat, last 4096 first copies (#3901). An atom
            # later than the flush grace after its bucket ends is dropped as late.
            "arrival_lag_by_seat": rollup.lag_stats() if rollup is not None else {},
            "emitted_at_fallback": self.seat_rollup_emitted_at_fallback,
            "routing_version": self.routing_version,
            "last": self.last_seat_rollup,
        }

    def _seat_rollup_envelope(self, event: HeartbeatSeatRollupV1) -> BaseEnvelope:
        return BaseEnvelope(
            kind=HEARTBEAT_SEAT_ROLLUP_KIND,
            source=ServiceRef(
                name=settings.service_name, version=settings.service_version, node=settings.node_name
            ),
            payload=event.model_dump(mode="json"),
        )

    async def _flush_seat_rollup_once(self, now_epoch: float) -> int:
        """Move closed buckets into the pending queue and publish them oldest first. Returns the
        number published. Fail-open: a publish error leaves the rest queued for the next tick. The
        queue is bounded (SEAT_ROLLUP_MAX_PENDING), so a long bus outage drops the oldest rows --
        a missing row reads as unknown, never as zero."""
        rollup = self.seat_rollup
        if rollup is None:
            return 0
        pending = self._seat_rollup_pending
        for event in rollup.flush_closed(now_epoch, grace_sec=settings.seat_rollup_flush_grace_sec):
            if pending.maxlen is not None and len(pending) == pending.maxlen:
                self.seat_rollup_dropped_pending_full += 1
                logger.warning("heartbeat_seat_rollup_pending_full dropping rollup_id=%s", pending[0].rollup_id)
            pending.append(event)
        published = 0
        while pending:
            event = pending[0]
            try:
                await asyncio.wait_for(
                    self.bus.publish(HEARTBEAT_SEAT_ROLLUP_CHANNEL, self._seat_rollup_envelope(event)),
                    timeout=SEAT_ROLLUP_PUBLISH_TIMEOUT_SEC,
                )
            except Exception as exc:  # noqa: BLE001 - incl. timeout; retried on the next tick
                self.seat_rollup_publish_failed += 1
                logger.warning(
                    "heartbeat_seat_rollup_publish_failed rollup_id=%s pending=%d err=%s",
                    event.rollup_id, len(pending), exc,
                )
                break
            pending.popleft()
            published += 1
            self.seat_rollup_published += 1
            self.last_seat_rollup = {
                "rollup_id": event.rollup_id,
                "level_sum": event.level_sum,
                "atom_count": event.atom_count,
                "partial": event.partial,
            }
        return published

    async def _seat_rollup_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(SEAT_ROLLUP_FLUSH_CHECK_SEC)
            try:
                await self._flush_seat_rollup_once(time.time())
            except Exception as exc:  # noqa: BLE001 - must not kill the loop
                logger.warning("heartbeat_seat_rollup_flush_failed err=%s", exc)

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
        Confirmed live 2026-07-28: `ensemble.absorb()`/`decay_tick()`
        are synchronous CPU-bound quimb calls (~118ms for N=8) -- running
        them directly on this coroutine blocks the ENTIRE single-threaded
        event loop, not just this worker's own queue draining. Under real
        backlog (confirmed live: `absorb_queue_size` reached 30 under normal
        traffic) this starved the dissipation loop (then a FalkorDB reader)'s own async socket I/O
        long enough to produce a spurious `bus_synaptic_query_failed
        err=Timeout connecting to server` -- FalkorDB itself was fine
        (reproduced the identical query manually with zero latency); the
        event loop simply never got a chance to service that socket's
        response while an absorb() burst was running. Offloading to a
        thread lets numpy/scipy's C-level linear algebra proceed while the
        event loop keeps servicing other coroutines (message intake, health
        endpoint, the dissipation loop).
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
                organ = SITE_ORGAN_MAP.get(assignment.site_index)
                if organ:
                    self.organ_fires.record(organ)
                self.events_absorbed += 1
                _bump(self.inputs_absorbed_by_kind, assignment.input_kind)
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
            organ = SITE_ORGAN_MAP.get(assignment.site_index)
            if organ:
                self.organ_fires.record(organ)
            self.events_absorbed += 1
            _bump(self.inputs_absorbed_by_kind, assignment.input_kind)
            self._absorb_queue.task_done()
            processed += 1
        return processed

    async def _decay_loop(self) -> None:
        """Wall-clock-driven dissipation tick, independent of message
        arrival -- see class docstring for why this can't be tick-driven.
        `ensemble.decay_tick()` runs via `asyncio.to_thread` -- see
        `_absorb_worker_loop`'s docstring for why (same CPU-bound-call-
        blocks-the-event-loop finding, confirmed live 2026-07-28)."""
        while not self._stop.is_set():
            await asyncio.sleep(settings.decay_interval_sec)
            try:
                async with self._ensemble_lock:
                    await asyncio.to_thread(self.ensemble.decay_tick)
            except Exception as exc:  # noqa: BLE001 - must not kill the dissipation loop
                logger.warning("heartbeat_decay_failed err=%s", exc)

    async def _publish_verdict_event(self, event, kind: str) -> bool:
        try:
            await publish_grammar_event(self.bus, event, source_name=settings.service_name)
        except Exception as exc:  # noqa: BLE001 - a bus hiccup must not kill the H1 loop
            self.verdict_atoms_publish_failed += 1
            logger.warning("heartbeat_verdict_atom_publish_failed kind=%s err=%s", kind, exc)
            return False
        self.verdict_atoms_published += 1
        self.last_verdict_atom = {
            "kind": kind,
            "trace_id": event.trace_id,
            "summary": event.atom.summary if event.atom else None,
            "at": event.emitted_at.isoformat(),
        }
        logger.info("heartbeat_verdict_atom_published kind=%s trace_id=%s", kind, event.trace_id)
        return True

    async def _observe_verdict(self, now: datetime) -> None:
        """Feed the latest H1 result to the summary window. Settled-transition
        atoms were retired 2026-11-24 (#3838): the class is noise (11-08
        replay), see verdict_atoms.py."""
        h1 = self.latest_h1
        if h1 is None:
            return
        tick = H1Tick(
            at=now,
            verdict=h1.verdict,
            mean_ratio=h1.mean_ratio,
            std_ratio=h1.std_ratio,
            bulk_penetration_depth=h1.bulk_penetration_depth,
        )
        self.verdict_window.record_tick(tick)

    async def _maybe_flush_summary(self, now: datetime) -> None:
        window = self.verdict_window
        if (now - window.start).total_seconds() < settings.verdict_summary_interval_sec:
            return
        event = build_summary_event(node=settings.node_name or "unknown", window=window, end=now)
        # Reset even if the publish fails: the next window must not silently
        # absorb this one's counts and claim a 2 h span as 1 h.
        self.verdict_window = HourlyWindow(start=now)
        await self._publish_verdict_event(event, "summary")

    async def _h1_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(settings.h1_interval_sec)
            h1_ok = False
            try:
                async with self._ensemble_lock:
                    fire_snapshot = self.organ_fires.snapshot()
                    self.latest_h1 = compute_h1_ensemble(
                        self.ensemble,
                        fire_counts=fire_snapshot.counts,
                        fire_snapshot=fire_snapshot,
                    )
                logger.info(
                    "heartbeat_h1_computed tick_count=%d mean_ratio=%.4f std_ratio=%.4f "
                    "verdict=%s seeds=%s",
                    self.latest_h1.tick_count,
                    self.latest_h1.mean_ratio,
                    self.latest_h1.std_ratio,
                    self.latest_h1.verdict,
                    self.latest_h1.seeds,
                )
                h1_ok = True
            except Exception as exc:  # noqa: BLE001 - must not kill the loop
                logger.warning("heartbeat_h1_computation_failed err=%s", exc)
                self.verdict_window.h1_failures += 1
            if not settings.verdict_atoms_enabled:
                continue
            try:
                now = datetime.now(timezone.utc)
                if h1_ok:
                    await self._observe_verdict(now)
                await self._maybe_flush_summary(now)
            except Exception as exc:  # noqa: BLE001 - must not kill the loop
                logger.warning("heartbeat_verdict_atom_failed err=%s", exc)

    async def _run(self) -> None:
        h1_task = asyncio.create_task(self._h1_loop(), name="heartbeat-h1-loop")
        absorb_task = asyncio.create_task(self._absorb_worker_loop(), name="heartbeat-absorb-worker")
        decay_task = asyncio.create_task(self._decay_loop(), name="heartbeat-decay-loop")
        tasks = [h1_task, absorb_task, decay_task]
        if self.seat_rollup is not None:
            # The open bucket is not published on shutdown: it is partial.
            tasks.append(asyncio.create_task(self._seat_rollup_loop(), name="heartbeat-seat-rollup"))
        try:
            async with self.bus.subscribe(settings.channel_grammar_event) as pubsub:
                self._rollup_mark_subscribed()
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
            # Only a real subscription's end starts a new blind span; a failed subscribe attempt
            # must not move the start of the span that is already running.
            if self._rollup_subscribed:
                self._rollup_subscribed = False
                self._rollup_unsubscribed_at = time.time()
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
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
