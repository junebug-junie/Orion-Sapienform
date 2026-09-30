"""One real RPC timeout -> one transport trigger, while the baseline gate emits.

Two things see the same ``rpc_request()`` timeout:

1. the ``rpc_transport_timeout`` grammar atom, emitted by the calling process on
   every timeout (``orion/core/bus/async_service.py::_emit_rpc_timeout_grammar``),
   in every service, published or not;
2. the per-hop transport baseline gate, which sees the same timeout as
   ``channel_latency[hop].timeout_count`` in the caller's next rpc_health
   snapshot -- but only for services that publish snapshots.

While ``EQUILIBRIUM_TRANSPORT_BASELINE_EMIT`` is off the atom is the only owner of
timeouts (the pooled rpc_health legacy branch was retired 2026-09-29), so this
module is not used. While EMIT is effective, the gate's ``timeout``/
``zero_success`` episodes own every timeout they actually saw, and the atom must
not fire a second trigger for it.

**The atom does not say which service emitted it** (provenance is always
``orion-bus``), so ownership is decided by evidence, not by a list of
"covered" services that would silently drift as publishers are added:

- A gate-folded window whose hop saw ``timeout_count = n`` becomes ``n`` credits
  for that hop's request channel (the part before ``#``), valid over that
  window's ``[window_start, window_end]``.
- An atom whose ``emitted_at`` falls inside a credit's window (both timestamps
  come from the emitting process's own clock) consumes one credit and is
  dropped: the gate owns it. The window is widened backwards by
  ``lookback_s``: short-lived buses (orion-mind, execution-dispatch,
  orion-thought) fold their timeouts into the publishing bus via
  ``SharedRpcHealthSink``/``absorb()``, which keeps the *absorbing* window's
  ``window_start`` -- so a timeout can be counted in a window that starts
  after it happened.
- An atom with no credit is held for ``grace_s`` (long enough for the caller's
  next snapshot to arrive) and then fires as before. **Coverage can only fail
  open**: a service that does not publish snapshots, a snapshot that was
  skipped, a gate that cold-started, a short-lived bus that never reached the
  publisher -- every one of these just leaves the atom un-credited, and it fires.

Matching is count-conserving, so two services timing out on the same channel in
the same window yield exactly as many atom triggers as the gate did not see.
A credit on an excluded hop (e.g. ``#log_orion_metacognition``) also owns its
atom: the exclusion exists so metacog's own traffic cannot start metacog, and
the atom path must not reopen that loop. Every decision is logged.

Bounded: at most ``max_pending`` held atoms; on overflow the oldest fires
early (fails open, never dropped). Credits and held atoms are indexed by
request channel, so a snapshot only re-scans atoms on the channels it credits.

Pure: no clock reads, no I/O. The service passes ``now`` in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger("orion.equilibrium.transport_timeout_owner")


def request_channel_of(hop: str) -> str:
    """The rpc_request() channel a hop key was recorded under (``hop_key()``
    appends ``#<health_label>``). Hand-rolled hops (``verb:``, ``governor:``,
    ``http:``, ``fcc:``) never match an atom's request channel, by design."""
    return str(hop).split("#", 1)[0]


@dataclass
class _Credit:
    channel: str
    key: str
    service: str
    instance: str | None
    excluded: bool
    start_ts: float
    end_ts: float
    remaining: int
    received_ts: float


@dataclass
class PendingAtom:
    atom: dict[str, Any]
    correlation_id: str
    request_channel: str
    emitted_ts: float
    received_ts: float
    zen_state: str
    pressure: float


@dataclass
class TimeoutAtomOwner:
    grace_s: float = 75.0
    skew_s: float = 2.0
    lookback_s: float = 60.0
    credit_ttl_s: float = 300.0
    max_pending: int = 5000
    _credits: dict[str, list[_Credit]] = field(default_factory=dict)
    _pending: dict[str, list[PendingAtom]] = field(default_factory=dict)
    _overflow: list[PendingAtom] = field(default_factory=list)

    # ------------------------------------------------------------- matching

    def _take_credit(self, channel: str, emitted_ts: float) -> _Credit | None:
        # Prefer a non-excluded credit, so an ambiguous atom on a channel that
        # carries both labelled and unlabelled traffic never hides behind the
        # exclusion when a real episode also saw a timeout there.
        best: _Credit | None = None
        for c in self._credits.get(channel, ()):
            if c.remaining <= 0:
                continue
            if not (c.start_ts - self.lookback_s <= emitted_ts <= c.end_ts + self.skew_s):
                continue
            if best is None or (best.excluded and not c.excluded):
                best = c
        if best is not None:
            best.remaining -= 1
        return best

    @staticmethod
    def _log_owned(p: PendingAtom, c: _Credit, *, when: str) -> None:
        logger.info(
            "transport_timeout_owner owner=baseline_gate when=%s channel=%s key=%s service=%s "
            "instance=%s excluded=%s corr=%s",
            when, p.request_channel, c.key, c.service, c.instance, c.excluded, p.correlation_id,
        )

    # --------------------------------------------------------------- inputs

    def offer_atom(self, pending: PendingAtom) -> bool:
        """True -> the gate already owns this timeout (drop the atom).
        False -> held; it fires from ``expire()`` unless a credit arrives."""
        credit = self._take_credit(pending.request_channel, pending.emitted_ts)
        if credit is not None:
            self._log_owned(pending, credit, when="on_atom")
            return True
        if self.pending_count >= self.max_pending:
            oldest_ch = min(
                (ch for ch, lst in self._pending.items() if lst),
                key=lambda ch: self._pending[ch][0].received_ts,
            )
            evicted = self._pending[oldest_ch].pop(0)
            logger.warning(
                "transport_timeout_owner owner=atom reason=pending_overflow channel=%s corr=%s max=%d",
                evicted.request_channel, evicted.correlation_id, self.max_pending,
            )
            self._overflow.append(evicted)
        self._pending.setdefault(pending.request_channel, []).append(pending)
        return False

    def add_credits(
        self,
        observations: Iterable[Any],
        *,
        window_start_ts: float | None,
        window_end_ts: float,
        now: float,
    ) -> int:
        """Register the timeouts one gate-folded window saw, then settle any
        held atoms they cover. Returns how many held atoms the gate took."""
        start = window_start_ts if window_start_ts is not None else window_end_ts - 30.0
        touched: set[str] = set()
        for ob in observations:
            n = int(getattr(ob, "timeout_count", 0) or 0)
            if n <= 0:
                continue
            key = str(ob.key)
            channel = request_channel_of(key)
            self._credits.setdefault(channel, []).append(
                _Credit(
                    channel=channel, key=key, service=str(ob.service),
                    instance=ob.instance, excluded=bool(ob.excluded),
                    start_ts=start, end_ts=window_end_ts, remaining=n, received_ts=now,
                )
            )
            touched.add(channel)
        taken = 0
        for channel in touched:
            held = self._pending.get(channel)
            if not held:
                continue
            still: list[PendingAtom] = []
            for p in held:
                credit = self._take_credit(channel, p.emitted_ts)
                if credit is not None:
                    self._log_owned(p, credit, when="on_snapshot")
                    taken += 1
                else:
                    still.append(p)
            self._pending[channel] = still
        return taken

    def expire(self, now: float) -> list[PendingAtom]:
        """Atoms no gate window claimed within ``grace_s``: the caller fires them.
        Also ages out stale credits."""
        due, self._overflow = list(self._overflow), []
        for ch in list(self._pending):
            held = self._pending[ch]
            expired = [p for p in held if now - p.received_ts >= self.grace_s]
            if expired:
                self._pending[ch] = [p for p in held if now - p.received_ts < self.grace_s]
                for p in expired:
                    logger.info(
                        "transport_timeout_owner owner=atom reason=no_gate_window channel=%s corr=%s held_s=%.1f",
                        p.request_channel, p.correlation_id, now - p.received_ts,
                    )
                due.extend(expired)
            if not self._pending[ch]:
                del self._pending[ch]
        for ch in list(self._credits):
            kept = [c for c in self._credits[ch] if c.remaining > 0 and now - c.received_ts < self.credit_ttl_s]
            if kept:
                self._credits[ch] = kept
            else:
                del self._credits[ch]
        return due

    def drain(self) -> list[PendingAtom]:
        """Shutdown: everything still held fires rather than being lost."""
        out = list(self._overflow) + [p for lst in self._pending.values() for p in lst]
        self._pending, self._overflow = {}, []
        return out

    @property
    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())

    @property
    def credit_count(self) -> int:
        return sum(c.remaining for lst in self._credits.values() for c in lst)
