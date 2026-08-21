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
            "failing_for_sec": round(ts - self._failing_since, 1) if self._failing_since else 0.0,
            "window_sec": self._window_sec,
            "arm_fail_rate": self._arm_fail_rate,
            "clear_fail_rate": self._clear_fail_rate,
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
        self._samples.append((ts, bool(ok), error_code))
        self._evict(ts)
        rate, count, top = self._stats()

        # Not enough evidence yet. Deliberately does not reset failing_since:
        # a low-traffic stream whose sample count dips below the floor has not
        # thereby recovered, and treating it as recovered would reset the
        # sustain clock every time traffic thinned.
        if count < self._min_samples:
            return LivenessDecision(fail_rate=rate, sample_count=count, top_error_code=top)

        if rate >= self._arm_fail_rate:
            if self._failing_since is None:
                # The sustain clock starts when there is enough evidence to
                # trust the rate, NOT at the first failure. Below min_samples
                # the rate is one or two data points and is meaningless -- a
                # single failed task would read as 100%. Consequence, worth
                # knowing when reading the tests: time-to-alert is
                # (time to accumulate min_samples) + sustain_sec, not
                # sustain_sec. At the observed 5s task cadence with
                # min_samples=10 and sustain_sec=180 that is 45 + 180 = 225s.
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

            self._alerting = True
            self._last_alert_at = ts
            return LivenessDecision(
                alert=True,
                reason=f"{int(rate * 100)}% of vision tasks failing ({top or 'unknown'})",
                fail_rate=rate, sample_count=count,
                failing_for_sec=failing_for, top_error_code=top,
            )

        if rate <= self._clear_fail_rate:
            self._failing_since = None
            if self._alerting:
                self._alerting = False
                return LivenessDecision(
                    recovered=True,
                    reason="vision tasks succeeding again",
                    fail_rate=rate, sample_count=count, top_error_code=top,
                )

        # Between the two thresholds: the hysteresis band. Hold current state,
        # and hold the sustain clock -- a rate that sags to 0.5 and climbs back
        # has not recovered and must not restart the clock.
        return LivenessDecision(fail_rate=rate, sample_count=count, top_error_code=top)


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
