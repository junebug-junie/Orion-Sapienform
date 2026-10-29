"""orion-thought's SHADOW bids into the body economy (Patch C3b). Decisions are logged, never obeyed.

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Organ 4: Choice" and
"Execution seam". The market is orion-durable-runs ``POST /physiology/bid``; the client is
``orion.physiology.client.abid`` (0.5 s timeout, ``None`` on any failure).

What a producer does:

1. ``shadow = ShadowBid(family=..., verb=..., tier=..., want_id=<the act's correlation id>)``.
2. ``shadow.place()`` at the point the producer would act. This starts the bid as a background
   task and returns at once: the producer never waits on the economy.
3. ``shadow.settle(real)`` once the producer's OWN gate has decided: ``"ran"`` or
   ``"blocked:<reason>"``. When the bid answers, one line is logged::

       physiology_shadow_bid family=thought.painting want_id=<id> shadow=funded reason=net_positive
           lane=diffusion real=blocked:thermal_refused act_correlation_id=-

   ``want_id`` is the correlation id of the act the producer opens (grammar_acts), so
   scripts/analysis/body_economy_shadow_vs_real.py can join the shadow decision (want_ledger) to
   the real act. ``act_correlation_id`` is set only when the act ended up on a different id.

Producer here: the ``generate`` step of a ``reverie.visual`` run (app/visual_steps.py), the one
GPU stage, whose first real generation opens the run's ``make`` act. Family ``thought.painting``,
or ``thought.dream_hop`` for a dream-carry image hop. want_id = the generate step's correlation id,
which is the act's correlation id when this step opens it.

SHADOW means byte-identical behaviour: nothing here returns a value the producer reads, and every
failure is swallowed. A funded shadow decision whose real act was blocked releases its
reservation (``POST /physiology/bid/<want_id>/release``), so a want that never ran does not hold
shadow capacity until it expires.

Flag: ``THOUGHT_PHYSIOLOGY_SHADOW_BID_ENABLED`` (default true). Off: no bid, no log line.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("orion-thought.physiology_shadow")

PRODUCER = "orion-thought"
# Strong references to in-flight bid and release tasks (asyncio keeps only weak ones).
_TASKS: set[asyncio.Task] = set()


def _enabled() -> bool:
    try:
        from .settings import settings

        return bool(getattr(settings, "thought_physiology_shadow_bid_enabled", True))
    except Exception:  # noqa: BLE001
        return True


async def abid(want: Any) -> Any:
    """The client call, behind one name so tests can stub it."""
    from orion.physiology.client import abid as client_abid

    return await client_abid(want)


def _release_url(want_id: str) -> str:
    from orion.physiology.client import DEFAULT_URL

    base = (os.environ.get("ORION_PHYSIOLOGY_BID_URL") or DEFAULT_URL).rstrip("/")
    return f"{base}/{want_id}/release"


def release(want_id: str) -> None:
    """Best-effort: free a funded shadow reservation whose act will not run."""
    try:
        req = urllib.request.Request(_release_url(want_id), data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=0.5):  # noqa: S310 -- internal service URL
            pass
    except Exception as exc:  # noqa: BLE001
        logger.info("physiology_shadow_release_failed want_id=%s err=%s", want_id, type(exc).__name__)


def _track(task: asyncio.Task) -> None:
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


class ShadowBid:
    """One want's shadow bid. Every method is best-effort and never raises."""

    def __init__(self, *, family: str, verb: str, tier: int, want_id: Optional[str] = None,
                 object_ref: Optional[str] = None, component: str = "", enabled: Optional[bool] = None):
        self.family = family
        self.verb = verb
        self.tier = tier
        self.want_id = str(want_id) if want_id else str(uuid4())
        self.object_ref = object_ref
        self.component = component
        self.enabled = _enabled() if enabled is None else bool(enabled)
        # Set by the producer when the real outcome is not the one settle() would derive.
        self.real: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._settled = False

    @property
    def placed(self) -> bool:
        return self._task is not None

    def place(self) -> None:
        """Start the bid in the background (once). Returns immediately."""
        if self._task is not None or not self.enabled:
            return
        try:
            from orion.schemas.physiology_want import WantV1

            want = WantV1(
                want_id=self.want_id, family=self.family,
                producer=f"{PRODUCER}.{self.component}" if self.component else PRODUCER,
                verb=self.verb, tier=self.tier, object_ref=self.object_ref, correlation_id=self.want_id,
            )
            self._task = asyncio.get_running_loop().create_task(abid(want))
            _track(self._task)
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_bid_not_placed family=%s err=%s", self.family, type(exc).__name__)

    def settle(self, real: str, *, act_correlation_id: Optional[str] = None) -> None:
        """Record the producer's real outcome beside the shadow decision (once)."""
        if self._task is None or self._settled:
            return
        self._settled = True
        if self.real:
            # The producer named the outcome itself (a hold this want never got past): no act of
            # this want opened, so any act id the caller saw belongs to something else.
            real, act_correlation_id = self.real, None
        act_corr = str(act_correlation_id) if act_correlation_id and str(act_correlation_id) != self.want_id else ""
        try:
            self._task.add_done_callback(lambda t: self._log(t, real, act_corr))
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_settle_failed family=%s err=%s", self.family, type(exc).__name__)

    def _log(self, task: asyncio.Task, real: str, act_corr: str) -> None:
        try:
            decision = None if task.cancelled() or task.exception() is not None else task.result()
            outcome = getattr(decision, "outcome", None) or "none"
            logger.info(
                "physiology_shadow_bid family=%s want_id=%s shadow=%s reason=%s lane=%s real=%s act_correlation_id=%s",
                self.family, self.want_id, outcome, getattr(decision, "reason", None) or "-",
                getattr(decision, "lane", None) or "-", real, act_corr or "-",
            )
            if outcome == "funded" and real.startswith("blocked"):
                _track(task.get_loop().create_task(asyncio.to_thread(release, self.want_id)))
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_log_failed family=%s err=%s", self.family, type(exc).__name__)


__all__ = ["ShadowBid", "abid", "release"]
