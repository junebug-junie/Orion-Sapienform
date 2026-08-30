"""Vision liveness watcher: notice when the eye stops working and say so.

**Why this exists.** On 2026-08-20 22:00 UTC orion-vision-host began refusing
every task with ``gpu_hard_floor`` and did not stop for ~21 hours. The container
reported ``Up``, ``/health`` was fine, the process was alive, the models were
resident -- and Orion was blind. Nothing anywhere noticed. It was found by hand,
a day later, while looking at something else.

That is the failure mode this guards: **not a crash, a refusal.** A crashed
container is loud and has other ways of being noticed. A healthy container
serving nothing is silent, and silence is indistinguishable from a quiet room.

**What it watches.** Only what this service already knows for certain: the
outcome of its own tasks (``app.main._log_task_completion``). It deliberately
does NOT read ``vision_events`` to look for a write gap. A write gap is the more
general signal -- it would also catch a crash -- but it is ambiguous in the
wrong direction: an empty table is exactly what a genuinely quiet camera
produces, and this repo has already burned itself once by reading ordinary
silence as a stall. Task outcomes have no such ambiguity: a task that ran and
returned ``ok: false`` is unambiguously a failure, not a quiet room.

**Residual gap, stated rather than papered over.** A watcher inside the service
cannot report that the service is gone. If the container dies outright, nothing
here fires. That case needs an external heartbeat consumer and is not covered by
this patch.

**Hysteresis is not optional.** A bare threshold on a trailing window flaps: the
rate crosses, an alert fires, one success drops it back, the next failure fires
again. So arming and clearing use *different* thresholds, and a cleared alert
still has to wait out a cooldown before it can fire again.

**Best effort, always.** Every path here is wrapped so that a notify outage, a
DNS failure, or a bad token can never fail a vision task. Losing an alert is
bad; losing the pipeline because the alerting broke is worse.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Tuple

logger = logging.getLogger("orion-vision-host.liveness")


@dataclass(frozen=True)
class LivenessDecision:
    """What the watcher wants to do, given the outcomes it has seen."""

    alert: bool = False
    recovered: bool = False
    reason: str = ""
    fail_rate: float = 0.0
    sample_count: int = 0
    failing_for_sec: float = 0.0
    top_error_code: Optional[str] = None


class VisionLivenessWatcher:
    """Rolling-window failure tracker with hysteresis and a cooldown.

    Pure decision logic -- ``record()`` takes an outcome and returns a
    ``LivenessDecision``. It performs no I/O, so the whole thing is testable
    without a bus, a database, or a clock. The caller does the sending.

    ``now`` is injectable on every method for the same reason; production
    passes nothing and gets ``time.monotonic()``.
    """

    def __init__(
        self,
        *,
        window_sec: float = 300.0,
        min_samples: int = 10,
        arm_fail_rate: float = 0.8,
        clear_fail_rate: float = 0.2,
        sustain_sec: float = 180.0,
        cooldown_sec: float = 3600.0,
        state_store: Any = None,
    ) -> None:
        if not 0.0 <= clear_fail_rate < arm_fail_rate <= 1.0:
            # A clear threshold at or above the arm threshold is not hysteresis,
            # it is a flapping bug with extra config. Refuse it at construction
            # rather than let it ship looking configured.
            raise ValueError(
                f"clear_fail_rate ({clear_fail_rate}) must be >= 0 and strictly "
                f"below arm_fail_rate ({arm_fail_rate}), which must be <= 1.0"
            )
        self._window_sec = float(window_sec)
        self._min_samples = int(min_samples)
        self._arm_fail_rate = float(arm_fail_rate)
        self._clear_fail_rate = float(clear_fail_rate)
        self._sustain_sec = float(sustain_sec)
        self._cooldown_sec = float(cooldown_sec)

        # (timestamp, ok, error_code)
        self._samples: Deque[Tuple[float, bool, Optional[str]]] = deque()
        self._failing_since: Optional[float] = None
        self._alerting: bool = False
        self._last_alert_at: Optional[float] = None
        self._alert_unconfirmed: bool = False
        # Which transition is awaiting confirmation: an alert rolls back to
        # not-alerting, a recovery rolls back to STILL-alerting so it retries.
        self._pending_kind: Optional[str] = None

        # Durable arm/clear state. Without it a restart between the alert and the
        # clear loses the recovery forever -- live on 2026-08-29, eight
        # `vision_blind` records since 08-21 and zero `vision_recovered`, ever.
        # See app/liveness_state.py for the evidence and the design.
        self._state_store = state_store
        self._last_now: Optional[float] = None
        # True while the arm state came from disk rather than from failures this
        # process actually observed. A restored watcher has an EMPTY sample
        # deque, so the thin-sample clear branch would otherwise declare
        # recovery on the first single success -- posting "Vision is working
        # again" off one sample while the GPU is still broken.
        self._restored_alerting: bool = False
        # Surfaced on /health via snapshot(): on a read-only or full volume the
        # whole fix is inert and nothing anywhere would have said so.
        self._state_write_ok: Optional[bool] = None
        self._restore_state()

    def _restore_state(self) -> None:
        """Load persisted state, translating wall clock back to monotonic.

        `_failing_since` / `_last_alert_at` are `time.monotonic()` values, which
        are process-relative: restoring them raw would seed this process's clock
        from another process's epoch. They are stored as wall clock and shifted
        back here by the current wall/monotonic offset.
        """
        if self._state_store is None:
            return
        try:
            state = self._state_store.load()
        except Exception:  # a store must never stop the service from seeing
            return
        if state is None or not getattr(state, "alerting", False):
            return
        now_wall = time.time()
        now_mono = time.monotonic()

        def _to_mono(wall: Optional[float]) -> Optional[float]:
            if wall is None:
                return None
            # Ages in the future (clock step) clamp to "just now" rather than
            # producing a deadline this process can never reach.
            return now_mono - max(0.0, now_wall - float(wall))

        self._alerting = True
        self._restored_alerting = True
        self._failing_since = _to_mono(getattr(state, "failing_since_wall", None))

        # `_last_alert_at` is deliberately NOT restored (review finding, HIGH).
        # Restoring it suppressed the next alert for the whole cooldown, and it
        # bought nothing: `_alerting=True` is restored too, and the arm path
        # short-circuits on `if self._alerting` long before the cooldown check,
        # so duplicate-alert suppression during the incident is already covered.
        # Keeping it turned a 225s detection gap into 3590s -- measured, with a
        # single fake clock driving both time.monotonic() and time.time().
        self._last_alert_at = None

    def _persist_state(self) -> None:
        """Write current arm state. Called only on transitions, not per sample."""
        if self._state_store is None:
            return
        try:
            from .liveness_state import PersistedLivenessState

            now_wall = time.time()
            now_mono = self._last_now if self._last_now is not None else time.monotonic()

            def _to_wall(mono: Optional[float]) -> Optional[float]:
                return None if mono is None else now_wall - (now_mono - float(mono))

            self._state_write_ok = bool(self._state_store.save(
                PersistedLivenessState(
                    alerting=self._alerting,
                    failing_since_wall=_to_wall(self._failing_since),
                    last_alert_at_wall=_to_wall(self._last_alert_at),
                )
            ))
        except Exception:
            self._state_write_ok = False
            return

    # -- introspection, for /health and tests -------------------------------

    def snapshot(self, now: Optional[float] = None) -> dict[str, Any]:
        ts = float(now if now is not None else time.monotonic())
        self._evict(ts)
        rate, count, top = self._stats()
        return {
            "alerting": self._alerting,
            "fail_rate": round(rate, 4),
            "sample_count": count,
            "top_error_code": top,
            "failing_for_sec": (
                round(ts - self._failing_since, 1)
                if self._failing_since is not None
                else 0.0
            ),
            "window_sec": self._window_sec,
            "arm_fail_rate": self._arm_fail_rate,
            "clear_fail_rate": self._clear_fail_rate,
            "state_path": getattr(self._state_store, "path", None),
            "state_write_ok": self._state_write_ok,
            "arm_restored_from_disk": self._restored_alerting,
        }

    # -- internals ----------------------------------------------------------

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_sec
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _stats(self) -> Tuple[float, int, Optional[str]]:
        count = len(self._samples)
        if count == 0:
            return 0.0, 0, None
        failures = [s for s in self._samples if not s[1]]
        rate = len(failures) / count
        top: Optional[str] = None
        if failures:
            counts: dict[str, int] = {}
            for _, _, code in failures:
                key = code or "unknown"
                counts[key] = counts.get(key, 0) + 1
            top = max(counts.items(), key=lambda kv: kv[1])[0]
        return rate, count, top

    # -- the one real entry point -------------------------------------------

    def record(
        self,
        *,
        ok: bool,
        error_code: Optional[str] = None,
        now: Optional[float] = None,
    ) -> LivenessDecision:
        ts = float(now if now is not None else time.monotonic())
        # Remembered so `_persist_state` converts monotonic->wall against the SAME
        # clock `record` is using. Mixing an injected clock with real
        # `time.monotonic()` produced wall timestamps days off -- harmless in
        # production, which never injects, but it made the persisted value
        # untestable and would silently rot.
        self._last_now = ts
        self._samples.append((ts, bool(ok), error_code))
        self._evict(ts)
        rate, count, top = self._stats()

        # Below the sample floor the rate is too thin to ARM on -- but clearing
        # must stay reachable, which the first version of this got wrong.
        #
        # Review finding (HIGH): returning early here skipped the clear branch
        # entirely, so a watcher that armed and then recovered at a low task
        # cadence stayed `alerting=True` forever and short-circuited every
        # later alert -- deaf to the next real outage. The margin was thin
        # enough to hit in practice: measured healthy cadence is 13 samples per
        # 300s window against a floor of 10.
        #
        # A full window with zero failures is unambiguous recovery no matter
        # how few samples it holds, so that clears. Anything else below the
        # floor holds state and holds the clock (a stream that merely thinned
        # has not thereby recovered).
        if count < self._min_samples:
            if count > 0 and rate == 0.0:
                self._failing_since = None
                # A restored arm has an empty deque, so "a full window with zero
                # failures" is not what this branch is seeing -- it is seeing one
                # or two samples since boot. Require the sample floor before
                # believing a restored incident is over.
                if self._alerting and not self._restored_alerting:
                    self._alerting = False
                    self._alert_unconfirmed = True
                    self._pending_kind = "recovery"
                    self._persist_state()
                    return LivenessDecision(
                        recovered=True,
                        reason="vision tasks succeeding again",
                        fail_rate=rate, sample_count=count, top_error_code=top,
                    )
            return LivenessDecision(fail_rate=rate, sample_count=count, top_error_code=top)

        if rate >= self._arm_fail_rate:
            if self._failing_since is None:
                # The sustain clock starts when there is enough evidence to
                # trust the rate, NOT at the first failure. Below min_samples
                # the rate is one or two data points and is meaningless -- a
                # single failed task would read as 100%. Consequence, worth
                # knowing when reading the tests: time-to-alert is
                # (time to accumulate min_samples) + sustain_sec, not
                # sustain_sec, and it therefore scales with task cadence.
                # Measured live 2026-08-21: 13 samples per 300s window in
                # healthy operation (~23s apart, since a real task spends
                # ~2.5s on the GPU) -> ~410s to alert. During the actual
                # outage tasks arrived every ~5s, because a refused task
                # returns instantly without doing any work -> ~225s. The
                # cadence speeds up exactly when things break, so the worst
                # case for detection is the healthy case, not the broken one.
                self._failing_since = ts
            failing_for = ts - self._failing_since

            if self._alerting or failing_for < self._sustain_sec:
                return LivenessDecision(
                    fail_rate=rate, sample_count=count,
                    failing_for_sec=failing_for, top_error_code=top,
                )

            if self._last_alert_at is not None and (ts - self._last_alert_at) < self._cooldown_sec:
                return LivenessDecision(
                    fail_rate=rate, sample_count=count,
                    failing_for_sec=failing_for, top_error_code=top,
                )

            # Provisional. The caller confirms via note_alert_delivered();
            # see that method for why this is not committed here.
            self._alerting = True
            self._restored_alerting = False   # observed in-process now, not restored
            self._last_alert_at = ts
            self._alert_unconfirmed = True
            self._pending_kind = "alert"
            return LivenessDecision(
                alert=True,
                reason=f"{int(rate * 100)}% of vision tasks failing ({top or 'unknown'})",
                fail_rate=rate, sample_count=count,
                failing_for_sec=failing_for, top_error_code=top,
            )

        # Review finding (MEDIUM): the clock used to reset ONLY at or below the
        # clear threshold, so a service parked in the hysteresis band (chronic
        # partial failure) kept a clock set by some blip hours earlier. The
        # sustain gate then did not gate -- a fresh outage alerted at 175s
        # against a 180s requirement -- and the message told the operator the
        # outage was 169 minutes old when it was under three.
        #
        # The band is meant to hold state for an ALREADY-ARMED alert, not to
        # accumulate sustain credit for one that has not fired. So while not
        # alerting, any rate below the arm threshold resets the clock.
        if not self._alerting and rate < self._arm_fail_rate:
            self._failing_since = None

        if rate <= self._clear_fail_rate:
            self._failing_since = None
            if self._alerting:
                self._alerting = False
                self._restored_alerting = False
                self._alert_unconfirmed = True
                self._pending_kind = "recovery"
                self._persist_state()
                return LivenessDecision(
                    recovered=True,
                    reason="vision tasks succeeding again",
                    fail_rate=rate, sample_count=count, top_error_code=top,
                )

        # Between the two thresholds: the hysteresis band. Hold current state,
        # and hold the sustain clock -- a rate that sags to 0.5 and climbs back
        # has not recovered and must not restart the clock.
        return LivenessDecision(fail_rate=rate, sample_count=count, top_error_code=top)


    def note_alert_delivered(self, delivered: bool) -> None:
        """Confirm or roll back the alert state set by the last ``record()``.

        Review finding (MEDIUM): ``record()`` used to commit ``_alerting`` and
        ``_last_alert_at`` at the moment it *decided* to alert, before the
        caller had any idea whether the POST succeeded. If notify was down, the
        token was rejected, or ``NOTIFY_BASE_URL`` was unset, the send failed
        and nothing retried -- but the watcher believed it had alerted, so
        ``if self._alerting`` suppressed every later attempt. The incident was
        lost for its entire duration, not for one attempt.

        On failure this rolls the state back so the next ``record()`` retries.
        The sustain clock is deliberately NOT rolled back: the failure run is
        still real and still ongoing: only the *notification* failed, so the
        retry should be immediate rather than waiting out sustain_sec again.
        """
        if not self._alert_unconfirmed:
            return
        kind = self._pending_kind
        self._alert_unconfirmed = False
        self._pending_kind = None
        if not delivered:
            if kind == "recovery":
                # The clear was decided but never announced. Roll back to
                # alerting so the next healthy sample re-emits it, rather than
                # persisting a silent close nobody was ever told about.
                self._alerting = True
            else:
                self._alerting = False
                self._last_alert_at = None
        # Persist either way: a confirmed arm must survive a restart (that is the
        # whole point), and a rollback must not leave a stale armed file behind
        # that would resurrect an alert this process decided never happened.
        self._persist_state()


def build_watcher_or_default(cfg: Any) -> VisionLivenessWatcher:
    """Construct a watcher from settings, degrading loudly on a bad config.

    Review finding (HIGH): ``VisionHostService`` is constructed at module
    import, so a ``ValueError`` from the watcher's constructor -- raised for a
    non-hysteretic ``VISION_LIVENESS_*`` value -- crashlooped the entire vision
    service. The alerting subsystem became a brand-new way to cause the exact
    21-hour blackout it was written to detect, and
    ``VISION_LIVENESS_ALERT_ENABLED=false`` did not bypass it.

    The ``ValueError`` stays in ``__init__``: refusing a flapping config is
    correct for a library, and the tests assert it. This wrapper is the policy
    layer -- **nothing in the alerting path may stop this service from
    seeing**, so a bad value degrades to the known-good defaults with a loud
    log instead.

    Lives here rather than in ``app.main`` so it is importable and testable
    without pulling in torch/PIL: the fix for a HIGH finding should not be
    covered only by a test that skips on the host venv.
    """
    store = None
    try:
        path = str(getattr(cfg, "VISION_LIVENESS_STATE_PATH", "") or "").strip()
        if path:
            from .liveness_state import LivenessStateStore

            store = LivenessStateStore(path)
    except Exception as exc:
        # Same policy as the config fallback below: no part of alerting may stop
        # this service from seeing. Degrade to in-memory state, loudly.
        print(f"[LIVENESS] state store unavailable ({exc}); arm state is in-memory only", flush=True)
        store = None

    try:
        return VisionLivenessWatcher(
            state_store=store,
            window_sec=cfg.VISION_LIVENESS_WINDOW_SEC,
            min_samples=cfg.VISION_LIVENESS_MIN_SAMPLES,
            arm_fail_rate=cfg.VISION_LIVENESS_ARM_FAIL_RATE,
            clear_fail_rate=cfg.VISION_LIVENESS_CLEAR_FAIL_RATE,
            sustain_sec=cfg.VISION_LIVENESS_SUSTAIN_SEC,
            cooldown_sec=cfg.VISION_LIVENESS_COOLDOWN_SEC,
        )
    except Exception as exc:
        # print(), not loguru: this runs at import, before start() has
        # configured the sink. stdout is what `docker logs` shows.
        print(
            f"[LIVENESS] invalid VISION_LIVENESS_* config ({exc}); falling back "
            f"to defaults. Vision serving is UNAFFECTED -- check "
            f"services/orion-vision-host/.env.",
            flush=True,
        )
        return VisionLivenessWatcher(state_store=store)


def build_attention_request(
    decision: LivenessDecision,
    *,
    node_name: str,
    service_version: str = "orion-vision-host",
) -> dict[str, Any]:
    """Render a ``ChatAttentionRequest`` body (orion/schemas/notify.py:79).

    Plain dict rather than the pydantic model: orion-vision-host is a thin GPU
    service and should not take a dependency on the notify schema package to
    post one JSON body. The field names are pinned by
    ``test_liveness_alert.py::test_attention_body_matches_notify_schema``, which
    reads the real model, so a schema change breaks a test rather than silently
    posting a body the notify service drops.
    """
    if decision.recovered:
        return {
            "source_service": "vision-host",
            "reason": "vision_recovered",
            "severity": "info",
            "message": (
                f"Vision is working again on {node_name}: "
                f"{int((1 - decision.fail_rate) * 100)}% of recent tasks succeeding."
            ),
            "context": {
                "node": node_name, "service": service_version,
                "fail_rate": decision.fail_rate, "sample_count": decision.sample_count,
            },
            "require_ack": False,
        }
    return {
        "source_service": "vision-host",
        "reason": "vision_blind",
        "severity": "warning",
        "message": (
            f"Orion cannot see. {decision.reason} on {node_name} for "
            f"{int(decision.failing_for_sec // 60)}m "
            f"({decision.sample_count} recent tasks). "
            f"The container is probably still reporting healthy -- check "
            f"`docker logs {node_name and 'orion-' + node_name + '-vision-host' or 'vision-host'} "
            f"| grep VISION_TASK` and the VRAM budget in "
            f"services/orion-vision-host/.env against the card actually installed."
        ),
        "context": {
            "node": node_name, "service": service_version,
            "fail_rate": decision.fail_rate, "sample_count": decision.sample_count,
            "failing_for_sec": decision.failing_for_sec,
            "top_error_code": decision.top_error_code,
        },
        "require_ack": True,
    }


def post_attention_request(
    body: dict[str, Any],
    *,
    base_url: str,
    token: Optional[str] = None,
    timeout_sec: float = 5.0,
) -> bool:
    """POST one attention request. Returns True on success. Never raises.

    Uses ``urllib.request`` rather than adding httpx/requests to this service's
    requirements: it is a single JSON POST, and AGENTS.md section 10 says not to
    take a dependency for something the standard library already does. The
    caller runs this in a thread (``asyncio.to_thread``) because urllib is
    blocking and this is called from the task-completion path -- a slow or
    hanging notify service must never stall the vision event loop.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = f"{str(base_url).rstrip('/')}/attention/request"
    try:
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("X-Orion-Notify-Token", token)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            ok = 200 <= int(resp.status) < 300
            if not ok:
                logger.warning("liveness alert rejected: HTTP %s", resp.status)
            return ok
    except Exception as exc:
        # Losing an alert is bad. Losing the vision pipeline because the
        # alerting broke is worse. Swallow, log, continue.
        logger.warning("liveness alert POST failed (%s): %s", url, exc)
        return False
