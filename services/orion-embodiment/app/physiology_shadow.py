"""orion-embodiment's SHADOW bid into the body economy: the ``aitown.session`` want (2026-10-30).

Same contract as services/orion-hub/scripts/physiology_shadow.py (Patch C3b): the bid is placed in
the background when an AI Town session start is asked for, the controller's OWN gate decides, and
one line logs both. Nothing here returns a value the controller reads, and every failure is
swallowed. The market is orion-durable-runs ``POST /physiology/bid`` via
``orion.physiology.client``. Patch D (after ~2026-11-03) makes the decision binding.

Flag: ``EMBODIMENT_PHYSIOLOGY_SHADOW_BID_ENABLED`` (default true). Off: no bid, no log line.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("orion.embodiment.physiology_shadow")

PRODUCER = "orion-embodiment.town_sessions"
_TASKS: set[asyncio.Task] = set()


async def abid(want: Any) -> Any:
    """The client call, behind one name so tests can stub it."""
    from orion.physiology.client import abid as client_abid

    return await client_abid(want)


async def arelease(want_id: str) -> Any:
    from orion.physiology.client import arelease as client_arelease

    return await client_arelease(want_id)


def _track(task: asyncio.Task) -> None:
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


class ShadowBid:
    """One want's shadow bid. Every method is best-effort and never raises."""

    def __init__(self, *, family: str, verb: str, tier: int, want_id: str, object_ref: Optional[str],
                 enabled: bool = True):
        self.family, self.verb, self.tier = family, verb, tier
        self.want_id, self.object_ref = str(want_id), object_ref
        self.enabled = bool(enabled)
        self._task: Optional[asyncio.Task] = None

    def place(self) -> None:
        if self._task is not None or not self.enabled:
            return
        try:
            from orion.schemas.physiology_want import WantV1

            want = WantV1(want_id=self.want_id, family=self.family, producer=PRODUCER, verb=self.verb,
                          tier=self.tier, object_ref=self.object_ref, correlation_id=self.want_id)
            self._task = asyncio.get_running_loop().create_task(abid(want))
            _track(self._task)
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_bid_not_placed family=%s err=%s", self.family, type(exc).__name__)

    def settle(self, real: str) -> None:
        """Log the shadow decision beside the controller's real one (``ran`` | ``blocked:<why>``)."""
        if self._task is None:
            return
        try:
            self._task.add_done_callback(lambda t: self._log(t, real))
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_settle_failed family=%s err=%s", self.family, type(exc).__name__)

    def _log(self, task: asyncio.Task, real: str) -> None:
        try:
            decision = None if task.cancelled() or task.exception() is not None else task.result()
            outcome = getattr(decision, "outcome", None) or "none"
            logger.info("physiology_shadow_bid family=%s want_id=%s shadow=%s reason=%s lane=%s real=%s",
                        self.family, self.want_id, outcome, getattr(decision, "reason", None) or "-",
                        getattr(decision, "lane", None) or "-", real)
            if outcome == "funded" and real.startswith("blocked"):
                _track(task.get_loop().create_task(arelease(self.want_id)))
        except Exception as exc:  # noqa: BLE001
            logger.info("physiology_shadow_log_failed family=%s err=%s", self.family, type(exc).__name__)


async def release(want_id: str) -> None:
    """The want's act closed: end its reservation now instead of at expiry. Best-effort."""
    try:
        await arelease(want_id)
    except Exception as exc:  # noqa: BLE001
        logger.info("physiology_shadow_release_failed want_id=%s err=%s", want_id, type(exc).__name__)


__all__ = ["PRODUCER", "ShadowBid", "abid", "arelease", "release"]
