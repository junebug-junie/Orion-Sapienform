"""Rolling record of background-admission decisions, so a deferral can be perceived.

ROADMAP step A5 (`docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md`), designed in
`docs/superpowers/specs/2026-08-19-A5-deferral-perceptible-proposal.md`.

WHY THIS EXISTS
---------------
A4 gave every admission decision a log line. A log line is a durable record for an operator and
is invisible to Orion. This module holds the same decisions in a bounded rolling window so the
one fact the scarcity arc exists to surface -- *I wanted to think and had to wait, this long* --
can be read back and put into Orion's own context.

THE DEFINITION THAT MATTERS
---------------------------
A first-poll admit is NOT a deferral.

`wait_for_slack` asks `/slots` whether there is room. That question costs an HTTP round trip,
measured live at 0.012-0.091s. If the answer is yes on the first ask, the request was never
deferred and the elapsed time is the cost of *asking*, not the cost of *waiting*. Over four
hours of live traffic on 2026-08-19, 294 of 294 admissions cleared on the first poll -- so a
ledger that counted `waited > 0` as a deferral would report 294 phantom waits a day and teach
Orion a false fact about its own constraint.

    deferral := polls > 1  (a poll interval was actually slept through)
             or outcome == "timeout_forwarded"

Every decision still lands in the ledger; `checked` is the denominator. That is what makes
"nothing waited" distinguishable from "nothing was measured", which are different facts and
would otherwise be the same silence.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No prompt text, no response text, no user or session identity. This module is called from
`priority_admission`, which only ever sees a `RouteTarget` -- it has no access to content and
must not acquire any. Pinned by `test_ledger_holds_no_request_content`.

No attribution of *who held the slot*. llama.cpp's `/slots` reports occupancy, not ownership, so
"while that ran instead" cannot be answered from here without confabulating it.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

# Bounded so a long-lived gateway process cannot grow this without limit. At the observed rate
# (~74 background admissions/hour) this holds well over a day; the window is enforced by
# timestamp in `snapshot`, and this cap is only the memory backstop.
_MAX_RECORDS = 4096

_DEFERRED_OUTCOMES = frozenset({"timeout_forwarded"})


@dataclass(frozen=True)
class AdmissionRecord:
    """One admission decision. Timings only -- see the module docstring on content."""

    ts: float          # wall-clock epoch seconds, for windowing across a snapshot call
    route_key: str
    url: str
    waited_s: float
    polls: int
    reserved: int
    outcome: str       # "admitted" | "unchecked" | "timeout_forwarded"

    @property
    def is_deferral(self) -> bool:
        """True only if a poll interval was actually slept through, or the wait timed out.

        `polls > 1` is the honest test: the first poll is the question, every poll after it
        means the answer was "no room" and the caller slept. See the module docstring.
        """
        return self.polls > 1 or self.outcome in _DEFERRED_OUTCOMES


class AdmissionLedger:
    """Thread-safe bounded ledger. One instance per gateway process.

    Thread-safe rather than asyncio-safe on purpose: `wait_for_slack` (async, used by the
    OpenAI passthrough) and `wait_for_slack_sync` (blocking, used by `run_llm_chat` under
    `asyncio.to_thread`) both record here, and the sync one genuinely runs on a worker thread.
    """

    def __init__(self, *, max_records: int = _MAX_RECORDS) -> None:
        self._records: Deque[AdmissionRecord] = deque(maxlen=max(1, max_records))
        self._lock = threading.Lock()

    def record(
        self,
        *,
        route_key: str,
        url: str,
        waited_s: float,
        polls: int,
        reserved: int,
        outcome: str,
        ts: Optional[float] = None,
    ) -> AdmissionRecord:
        rec = AdmissionRecord(
            ts=time.time() if ts is None else float(ts),
            route_key=str(route_key or ""),
            url=str(url or ""),
            waited_s=max(0.0, float(waited_s)),
            polls=max(0, int(polls)),
            reserved=max(0, int(reserved)),
            outcome=str(outcome or ""),
        )
        with self._lock:
            self._records.append(rec)
        return rec

    def snapshot(self, *, window_s: float, now: Optional[float] = None) -> Dict[str, Any]:
        """Counters over the trailing `window_s`.

        `checked` is the denominator and is what makes a quiet window legible: `deferrals == 0`
        with `checked == 294` is "Orion asked 294 times and was never made to wait", which is a
        real observation. `deferrals == 0` with `checked == 0` is "nothing asked", which is not.
        A consumer that reads only `deferrals` cannot tell those apart, so both ship.
        """
        cutoff = (time.time() if now is None else float(now)) - max(0.0, float(window_s))
        with self._lock:
            recent = [r for r in self._records if r.ts >= cutoff]

        deferrals = [r for r in recent if r.is_deferral]
        # max() over an empty sequence raises; an empty window is the normal case, not an error.
        longest = max((r.waited_s for r in deferrals), default=0.0)
        last_ts = max((r.ts for r in deferrals), default=None)
        return {
            "window_s": float(window_s),
            "checked": len(recent),
            "deferrals": len(deferrals),
            "timeouts": sum(1 for r in recent if r.outcome == "timeout_forwarded"),
            "unchecked": sum(1 for r in recent if r.outcome == "unchecked"),
            # Sum over deferrals only. Summing every record's `waited_s` would total up the
            # /slots round-trip cost and call it waiting -- the same phantom the deferral
            # definition exists to prevent, just aggregated.
            "deferred_s_total": round(float(sum(r.waited_s for r in deferrals)), 3),
            "longest_wait_s": round(longest, 3),
            "last_deferral_ts": last_ts,
            "routes": sorted({r.route_key for r in recent if r.route_key}),
        }


_LEDGER = AdmissionLedger()


def get_ledger() -> AdmissionLedger:
    """The process-wide ledger. A function, not a bare import, so tests can reach through it."""
    return _LEDGER
