"""Curiosity reads Orion's now at kickoff and records what the run's finding cited (integration patch 3, #3425).

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A (A4, "Curiosity candidate
selection"). This module is curiosity's registered ``reads_now`` (config/integration/organs.yaml).

* **Read** (``read_for_run``): at an investigate run's kickoff, ``GET /temporal-self/now?organ=curiosity`` with no
  ``since`` (it moves curiosity's cursor) and ``run_ref=curiosity_run:<run_id>``. The kickoff shows the items as
  candidates (``orion.curiosity.kickoff_prompt`` WHAT YOUR OTHER ORGANS LIVED ...), each with its
  ``[experience:<id>]``.
* **Keep the shown row** in Redis (``orion:curiosity:now_shown:<run_id>``, 7 days): a durable run can wait hours for
  a GPU hold and outlive a Hub restart, and its cites are checked against what its kickoff was shown.
* **Record** (``record_run_cites``): when the run's turn ends with text, the shown items the finding names as
  ``experience:<id>`` are posted (``citing_ref=curiosity_run:<run_id>``). Durable-runs keeps only shown ids, at most
  3, idempotent per (shown row, event), so a retried turn stores nothing twice. No cites, no call.

Blind to the three-arm experiment (#3183): nothing here takes, reads or sends the offer arm, its propensities, the
replica's plan or the worldview view. The read happens and renders the same way under every arm.

Never raises; every call is bounded (``timeout_sec``); a failure is a kickoff without the section, logged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from orion.integration import now_client
from orion.temporal_self.now import cites_in_text, now_lines

logger = logging.getLogger("orion-hub.curiosity_now")

ORGAN = "curiosity"
SHOWN_KEY_PREFIX = "orion:curiosity:now_shown:"
SHOWN_TTL_SEC = 7 * 86400
LABEL_SHOWN = 160
DEFAULT_TIMEOUT_SEC = 3.0

# In-process copy, for a Hub without Redis (tests, a degraded bus). Bounded: oldest dropped first.
_LOCAL: dict[str, dict[str, Any]] = {}
_LOCAL_MAX = 64


def run_ref(run_id: str) -> str:
    return f"curiosity_run:{run_id}"


async def _keep_shown(redis: Any, run_id: str, shown_id: str, shown_ids: list[str]) -> None:
    row = {"shown_id": shown_id, "shown_ids": shown_ids}
    _LOCAL[run_id] = row
    while len(_LOCAL) > _LOCAL_MAX:
        _LOCAL.pop(next(iter(_LOCAL)))
    if redis is None:
        return
    try:
        await redis.setex(f"{SHOWN_KEY_PREFIX}{run_id}", SHOWN_TTL_SEC, json.dumps(row))
    except Exception as exc:  # noqa: BLE001 -- the in-process copy still serves this Hub
        logger.warning("curiosity_now_keep_failed run=%s err=%s", run_id, type(exc).__name__)


async def _shown_for(redis: Any, run_id: str) -> Optional[dict[str, Any]]:
    if redis is not None:
        try:
            raw = await redis.get(f"{SHOWN_KEY_PREFIX}{run_id}")
            if raw:
                row = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                if isinstance(row, dict) and row.get("shown_id"):
                    return row
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_now_shown_read_failed run=%s err=%s", run_id, type(exc).__name__)
    return _LOCAL.get(run_id)


async def read_for_run(durable_runs_url: str, run_id: str, *, enabled: bool, redis: Any = None,
                       timeout_sec: float = DEFAULT_TIMEOUT_SEC, get: Any = None) -> list[str]:
    """The now's lines for this run's kickoff ([] when off, failed or empty). Keeps the shown row. Never raises."""
    if not enabled or not run_id:
        return []
    try:
        got = await now_client.aread_now(durable_runs_url, ORGAN, run_ref=run_ref(run_id), timeout_sec=timeout_sec,
                                         get=get)
        if got.now is None:
            logger.info("curiosity_now_unread run=%s reason=%s", run_id, got.error)
            return []
        await _keep_shown(redis, run_id, got.now.shown_id, got.shown_ids)
        lines = now_lines(got.now, label_cap=LABEL_SHOWN)
        logger.info("curiosity_now_offered run=%s shown_id=%s items=%d candidates=%d", run_id, got.now.shown_id,
                    len(lines), got.now.candidates)
        return lines
    except Exception as exc:  # noqa: BLE001 -- a kickoff never fails on the now
        logger.warning("curiosity_now_unread run=%s reason=%s", run_id, type(exc).__name__)
        return []


async def record_run_cites(durable_runs_url: str, run_id: str, text: Optional[str], *, enabled: bool,
                           redis: Any = None, timeout_sec: float = DEFAULT_TIMEOUT_SEC, post: Any = None) -> int:
    """Post the shown items the finding names; returns how many were stored. Never raises."""
    if not enabled or not run_id or not (text or "").strip():
        return 0
    try:
        row = await _shown_for(redis, run_id)
        if row is None:
            return 0  # this run's kickoff read no now (off, failed, or before this patch)
        cites = cites_in_text(text, list(row.get("shown_ids") or []))
        if not cites:
            logger.info("curiosity_now_cites run=%s shown_id=%s cited=0", run_id, row["shown_id"])
            return 0
        result, err = await now_client.arecord_use(durable_runs_url, str(row["shown_id"]), cites, run_ref(run_id),
                                                   timeout_sec=timeout_sec, post=post)
        stored = len(result.uses) if result else 0
        logger.info("curiosity_now_cites run=%s shown_id=%s cited=%d stored=%d rejected=%d err=%s", run_id,
                    row["shown_id"], len(cites), stored,
                    (len(result.rejected_not_shown) + len(result.rejected_over_cap)) if result else 0, err or "-")
        return stored
    except Exception as exc:  # noqa: BLE001 -- a finished run never fails on its cites
        logger.warning("curiosity_now_cites_failed run=%s err=%s", run_id, type(exc).__name__)
        return 0
