"""orion-topic-foundry's SHADOW bid into the body economy (Patch C3b). Logged, never obeyed.

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Organ 4: Choice" and
"Execution seam" (topic_foundry.run, Tier 2). The market is orion-durable-runs
``POST /physiology/bid``; the client is ``orion.physiology.client.bid`` (0.5 s timeout, ``None`` on
any failure).

Producer here: a training run (app/services/training.py ``_run_training``), at run start. The run
has no gate of its own at that point (the route already created it), so the real outcome is
``ran``. want_id = ``act_emit.run_correlation_id(run_id)``, the run's ``maintain`` act correlation
id, so scripts/analysis/body_economy_shadow_vs_real.py can join the decision (want_ledger) to the
act (grammar_acts).

Synchronous: ``_run_training`` runs in a FastAPI BackgroundTasks worker thread with no event loop.
``place()`` starts the bid on a daemon thread and returns at once; ``settle(real)`` records the real
outcome. Whichever finishes second logs one line::

    physiology_shadow_bid family=topic_foundry.run want_id=<id> shadow=funded reason=net_positive
        lane=fast real=ran act_correlation_id=-

SHADOW means byte-identical behaviour: nothing here returns a value the run reads, and every
failure is swallowed. A funded shadow decision whose real act was blocked releases its reservation
(``POST /physiology/bid/<want_id>/release``).

Flag: ``TOPIC_FOUNDRY_PHYSIOLOGY_SHADOW_BID_ENABLED`` (default true). Off: no bid, no log line.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from typing import Any, Optional

logger = logging.getLogger("topic-foundry.physiology_shadow")

PRODUCER = "orion-topic-foundry"


def _enabled() -> bool:
    try:
        from app.settings import settings

        return bool(getattr(settings, "topic_foundry_physiology_shadow_bid_enabled", True))
    except Exception:  # noqa: BLE001
        return True


def bid(want: Any) -> Any:
    """The client call, behind one name so tests can stub it."""
    from orion.physiology.client import bid as client_bid

    return client_bid(want)


def release(want_id: str) -> None:
    """Best-effort: free a funded shadow reservation whose act will not run."""
    try:
        from orion.physiology.client import DEFAULT_URL

        base = (os.environ.get("ORION_PHYSIOLOGY_BID_URL") or DEFAULT_URL).rstrip("/")
        req = urllib.request.Request(f"{base}/{want_id}/release", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=0.5):  # noqa: S310 -- internal service URL
            pass
    except Exception as exc:  # noqa: BLE001
        logger.info("physiology_shadow_release_failed want_id=%s err=%s", want_id, type(exc).__name__)


class ShadowBid:
    """One want's shadow bid, for a synchronous producer. Every method is best-effort."""

    def __init__(self, *, family: str, verb: str, tier: int, want_id: str, object_ref: Optional[str] = None,
                 component: str = "", enabled: Optional[bool] = None):
        self.family = family
        self.verb = verb
        self.tier = tier
        self.want_id = str(want_id)
        self.object_ref = object_ref
        self.component = component
        self.enabled = _enabled() if enabled is None else bool(enabled)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._answered = False
        self._decision: Any = None
        self._real: Optional[str] = None
        self._act_corr = ""
        self._logged = False

    def place(self) -> None:
        """Start the bid on a daemon thread (once). Returns immediately."""
        if self._thread is not None or not self.enabled:
            return
        try:
            from orion.schemas.physiology_want import WantV1

            want = WantV1(
                want_id=self.want_id, family=self.family,
                producer=f"{PRODUCER}.{self.component}" if self.component else PRODUCER,
                verb=self.verb, tier=self.tier, object_ref=self.object_ref, correlation_id=self.want_id,
            )
            self._thread = threading.Thread(target=self._run, args=(want,), name="physiology-shadow-bid",
                                            daemon=True)
            self._thread.start()
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_bid_not_placed family=%s err=%s", self.family, type(exc).__name__)

    def _run(self, want: Any) -> None:
        try:
            decision = bid(want)
        except Exception:  # noqa: BLE001 -- no answer
            decision = None
        with self._lock:
            self._answered, self._decision = True, decision
        self._maybe_log()

    def settle(self, real: str, *, act_correlation_id: Optional[str] = None) -> None:
        """Record the producer's real outcome beside the shadow decision (once)."""
        if self._thread is None:
            return
        with self._lock:
            if self._real is not None:
                return
            self._real = real
            corr = str(act_correlation_id) if act_correlation_id else ""
            self._act_corr = corr if corr != self.want_id else ""
        self._maybe_log()

    def _maybe_log(self) -> None:
        try:
            with self._lock:
                if self._logged or not self._answered or self._real is None:
                    return
                self._logged = True
                decision, real, act_corr = self._decision, self._real, self._act_corr
            outcome = getattr(decision, "outcome", None) or "none"
            logger.info(
                "physiology_shadow_bid family=%s want_id=%s shadow=%s reason=%s lane=%s real=%s act_correlation_id=%s",
                self.family, self.want_id, outcome, getattr(decision, "reason", None) or "-",
                getattr(decision, "lane", None) or "-", real, act_corr or "-",
            )
            if outcome == "funded" and real.startswith("blocked"):
                release(self.want_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_log_failed family=%s err=%s", self.family, type(exc).__name__)

    def join(self, timeout: Optional[float] = None) -> None:
        """Tests only: wait for the bid thread."""
        if self._thread is not None:
            self._thread.join(timeout)


__all__ = ["ShadowBid", "bid", "release"]
