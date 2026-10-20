"""Back off from a GPU pool that is shedding work, instead of asking again every few seconds.

The pool's U4 shed (orion/gpu_pool/scheduler.py, D3 one-shot refusal) refuses a one-shot request
of a shed priority at once with reason ``shed:<name>`` (``shed:cabinet_hot`` while the cabinet is
hot). The pool gives no retry-after hint: it cannot know when the heat clears, and its shed
signals are refreshed every hardware-watch tick, so their ``valid_until`` says nothing about it.
Callers that do one LLM call per work item (vision-council: one per camera window, ~every 6 s;
topic-foundry: one per boundary/topic in a batch) used to ask again for the next item at once,
so a heat episode of hours became thousands of refused leases (2026-10-20: 14,257 in 14 h).

``ShedBackoff`` is the one rule, shared by the pool client (keyed by lease priority, inside the
gateway) and by callers of the gateway (one key per call path):

- a shed refusal opens an *episode*. The next attempt is allowed only after a delay that starts at
  ``BASE_SEC`` and doubles on every further shed refusal, up to ``CAP_SEC``, with +-``JITTER``.
- while the delay runs, ``check`` refuses locally (no RPC) and counts the skip.
- when the delay is over, exactly ONE attempt (the probe, identified by a token) goes through;
  others still wait. Probe refused again -> longer delay. Probe granted (or refused for a reason
  the pool decided after judging the lease) -> the episode ends at once, so work resumes on the
  very next call after the heat clears. A neutral outcome (deadline, cancel, pool unreachable)
  only releases the probe. Only the probe moves the episode; calls already in flight do not.
- one log line when an episode opens and one when it closes (with the skip count), never one
  per attempt. Skipped work is therefore never silent, but also never a log flood.

Why these numbers: hardware-watch re-decides the reflex shed every ``HARDWARE_WATCH_TICK_SEC``
(30 s), so probing faster than 30 s can not see a change. The shed only lifts after the cabinet
cools past the rearm line, which takes minutes. A 180 s cap (6 reflex ticks) holds a sustained
episode to ~20 probes per hour per key, and bounds the idle time after the heat clears to at
most ~3.6 min (180 s + 20 % jitter) -- not the 30 min a longer cap would allow.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

SHED_REASON_PREFIX = "shed:"
BASE_SEC = 30.0
CAP_SEC = 180.0
JITTER = 0.2
# A probe whose caller never reports back (crashed) stops blocking other probes after this long.
# The pool client does not rely on it: a probe the pool QUEUED instead of refusing ends the episode
# after QUEUED_PROBE_CLEARS_SEC (orion/gpu_pool/client.py).
PROBE_STALE_SEC = 60.0
# Outcomes that say nothing about whether the pool is shedding: the probe is released, the episode
# neither grows nor ends. "deadline": the caller's own deadline ran out (can be shorter than a pool
# tick); "cancelled"/"dead_lettered": the lease ended for another reason; pool_*: the pool (or the
# gateway's bus to it) was not reachable.
NEUTRAL_REASONS = frozenset({"deadline", "cancelled", "dead_lettered", "backlogged"})
NEUTRAL_PREFIXES = ("pool_", "gpu_pool_")

_log = logging.getLogger("orion.gpu_pool.shed_backoff")


def is_shed_reason(reason: Any) -> bool:
    return isinstance(reason, str) and reason.startswith(SHED_REASON_PREFIX)


def is_neutral_reason(reason: Any) -> bool:
    return isinstance(reason, str) and (reason in NEUTRAL_REASONS or reason.startswith(NEUTRAL_PREFIXES))


def gateway_shed_refusal(payload: Any) -> tuple[str, float | None] | None:
    """(reason, retry_after_sec or None) when an orion-llm-gateway chat result says the GPU pool
    refused the call because it is shedding; else None. Reads the gateway's documented error
    shape: ``raw.error == "gpu_pool_unavailable"``, ``raw.details.reason`` (and the optional
    ``raw.details.retry_after_sec`` the gateway adds from its own backoff)."""
    if payload is not None and hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("raw")
    if not isinstance(raw, Mapping) or raw.get("error") != "gpu_pool_unavailable":
        return None
    details = raw.get("details")
    if not isinstance(details, Mapping) or not is_shed_reason(details.get("reason")):
        return None
    hint = details.get("retry_after_sec")
    try:
        hint_f = float(hint) if hint is not None else None
    except (TypeError, ValueError):
        hint_f = None
    return str(details["reason"]), (hint_f if hint_f is not None and hint_f > 0 else None)


def gateway_error_reason(payload: Any) -> str | None:
    """The pool refusal reason in a gateway chat result (``raw.details.reason`` when
    ``raw.error == "gpu_pool_unavailable"``), else None (an answer, or an upstream error after a grant)."""
    if payload is not None and hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    raw = payload.get("raw") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping) or raw.get("error") != "gpu_pool_unavailable":
        return None
    details = raw.get("details")
    reason = details.get("reason") if isinstance(details, Mapping) else None
    return str(reason) if reason else "gpu_pool_unavailable"


@dataclass
class _Episode:
    reason: str
    started_at: float
    not_before: float
    delay: float
    refusals: int = 1
    skipped: int = 0
    probe_token: int | None = None
    probing_since: float | None = None


class ShedBackoff:
    def __init__(self, *, name: str, base_sec: float = BASE_SEC, cap_sec: float = CAP_SEC,
                 jitter: float = JITTER, probe_stale_sec: float = PROBE_STALE_SEC,
                 clock: Callable[[], float] = time.monotonic, rng: Callable[[], float] = random.random,
                 log: Callable[[str], None] | None = None) -> None:
        if not (0 < base_sec <= cap_sec) or not (0 <= jitter < 1):
            raise ValueError("need 0 < base_sec <= cap_sec and 0 <= jitter < 1")
        self.name = name
        self.base_sec, self.cap_sec, self.jitter = float(base_sec), float(cap_sec), float(jitter)
        self.probe_stale_sec = float(probe_stale_sec)
        self._clock, self._rng = clock, rng
        self._log = log or _log.warning
        self._lock = threading.Lock()
        self._episodes: dict[str, _Episode] = {}
        self._next_token = 0

    # --- the gate ---------------------------------------------------------------------------
    def admit(self, key: str) -> tuple[float | None, int | None]:
        """(wait, probe_token).

        - (None, None): no episode, go ahead (an ordinary call).
        - (None, token): the window is over and this call is THE probe; report its outcome with
          ``token`` so only the probe moves the episode.
        - (seconds, None): do not call, the pool would refuse; the skip is counted for the close log.
        """
        now = self._clock()
        with self._lock:
            ep = self._episodes.get(key)
            if ep is None:
                return None, None
            probing = ep.probe_token is not None and now - (ep.probing_since or now) < self.probe_stale_sec
            if now >= ep.not_before and not probing:
                self._next_token += 1
                ep.probe_token, ep.probing_since = self._next_token, now
                return None, ep.probe_token
            ep.skipped += 1
            return max(ep.not_before - now, 1.0), None

    def check(self, key: str) -> float | None:
        """``admit`` without the token, for callers that only need yes/no (tests, eval)."""
        return self.admit(key)[0]

    def reason(self, key: str) -> str | None:
        with self._lock:
            ep = self._episodes.get(key)
            return ep.reason if ep else None

    def remaining(self, key: str) -> float | None:
        now = self._clock()
        with self._lock:
            ep = self._episodes.get(key)
            return max(0.0, ep.not_before - now) if ep else None

    # --- outcomes ---------------------------------------------------------------------------
    def resolve(self, key: str, reason: str | None, *, token: int | None = None,
                hint_sec: float | None = None) -> None:
        """Report an attempt's outcome, with the token ``admit`` gave it.

        - a shed reason: no episode -> open one. The probe -> back off further. Any other call (it
          was already in flight) -> counted, no growth.
        - a neutral reason (deadline, cancelled, pool unreachable: says nothing about the shed) ->
          the probe is released so the next call probes; the episode neither grows nor ends.
        - anything else (None = granted/answered, or a refusal the pool made after judging the lease)
          from the probe -> the pool is not shedding this key: the episode ends now. From a call that
          is not the outstanding probe it is ignored (it may predate the episode)."""
        if is_shed_reason(reason):
            self._refused(key, str(reason), token, hint_sec)
        elif is_neutral_reason(reason):
            self.abandon(key, token=token)
        else:
            self._clear(key, token)

    def abandon(self, key: str, *, token: int | None = None) -> None:
        """The probe ended without saying anything about the shed (cancelled, transport error):
        let the next call probe again, without growing or ending the episode."""
        with self._lock:
            ep = self._episodes.get(key)
            if ep is not None and token is not None and token == ep.probe_token:
                ep.probe_token = ep.probing_since = None

    def _jittered(self, delay: float) -> float:
        return delay * (1.0 + self.jitter * (2.0 * self._rng() - 1.0))

    def _refused(self, key: str, reason: str, token: int | None, hint_sec: float | None) -> None:
        now = self._clock()
        opened = False
        with self._lock:
            ep = self._episodes.get(key)
            if ep is None:
                ep = _Episode(reason=reason, started_at=now, not_before=now, delay=self.base_sec)
                self._episodes[key] = ep
                opened = True
            elif token is not None and token == ep.probe_token:
                ep.delay = min(self.cap_sec, ep.delay * 2.0)   # the probe was refused: back off further
                ep.refusals += 1
                ep.probe_token = ep.probing_since = None
            else:
                ep.refusals += 1                               # in flight before/while the probe ran
                ep.reason = reason
                return
            ep.reason = reason
            wait = self._jittered(ep.delay)
            if hint_sec is not None and hint_sec > 0:
                wait = max(wait, min(float(hint_sec), self.cap_sec * (1.0 + self.jitter)))
            ep.not_before = now + wait
        if opened:
            self._log(f"gpu_shed_backoff_start name={self.name} key={key} reason={reason} "
                      f"next_try_in_sec={wait:.0f} (later calls are skipped until then; one probe per window, "
                      f"delay doubles to {self.cap_sec:.0f}s)")

    def _clear(self, key: str, token: int | None) -> None:
        now = self._clock()
        with self._lock:
            ep = self._episodes.get(key)
            if ep is None or token is None or token != ep.probe_token:
                return
            self._episodes.pop(key, None)
        self._log(f"gpu_shed_backoff_end name={self.name} key={key} reason={ep.reason} "
                  f"lasted_sec={now - ep.started_at:.0f} refusals={ep.refusals} skipped={ep.skipped}")

    def reset(self) -> None:
        with self._lock:
            self._episodes.clear()
