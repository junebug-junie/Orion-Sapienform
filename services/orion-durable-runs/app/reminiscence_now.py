"""The memory organ reads Orion's now in its "where I am now" hop and records its cites (integration patch 7b).

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A ("Who reads the now",
patch 7: "... and memory read the now"). The organ registry (config/integration/organs.yaml) names this module as
the memory organ's ``reads_now``.

Which run: ``reminiscence.episode`` (app/reminiscence_graph.py), the daily turn in which Orion writes five hops
about one past week. The hop ``where_i_am_now`` is the one about the present ("a [now] sentence with a live ref"),
so it is the one hop shown the now. The other hops are about a past week; the now is not their material.

1. **Before the turn** (``read``): ``GET /temporal-self/now?organ=memory&run_ref=reminiscence:<run_id>`` on this
   same service, with no ``since`` (memory's own cursor; #3498). The items go into the hop's prompt with their
   ``[experience:<id>]`` tags (``render``).
2. **After the turn** (``close``), only when the turn produced text: the tags in the reply, shown ids first, at
   most 3 (``orion.curiosity.now_section.extract_cites``, the same extractor curiosity uses), go to
   ``POST /temporal-self/now/use`` once the hop's save receipt is proven. A reply whose receipt fails is closed
   with no cites (the model ran, but no hop was written for its cites to have shaped). No tag: the post is still sent with none, so the shown row reads "read the
   now, cited nothing". The endpoint keeps only ids that were shown (``record_use``).

A turn that never answered (refusal, timeout) leaves the read open, so the next hop attempt or the next day is
offered the same experiences. Never fails a hop: every failure is None plus one log line. Flag:
``REMINISCENCE_NOW_ENABLED`` (default true). Off: no read, no block, no post.
Log lines: ``reminiscence_now_read``, ``reminiscence_now_unavailable``, ``reminiscence_now_cites``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import httpx

from orion.curiosity.now_section import extract_cites, parse_now
from orion.schemas.temporal_self import NOW_MAX_CITES, TemporalNowV1

logger = logging.getLogger("orion-durable-runs.reminiscence_now")

ORGAN = "memory"
HOP = "where_i_am_now"
# This service's own HTTP port inside its container (app/dream_day_store.py SELF_URL): the now is read through the
# same route every other organ uses, so it is stored and cursored the same way.
SELF_URL = "http://127.0.0.1:8121"
TIMEOUT_SEC = 5.0


def citing_ref(run_id: str) -> str:
    return f"reminiscence:{run_id}"


def render(now: Optional[TemporalNowV1]) -> str:
    """The prompt block: one line per item with its tag, organ and code-written label. "" when there is none."""
    if now is None or not now.items:
        return ""
    lines = ["What your other organs lived since memory last looked (Orion's now; each line starts with its tag):"]
    for item in now.items:
        lines.append(f"- [experience:{item.event_id}] {item.organ}, {item.occurred_at:%Y-%m-%d %H:%MZ}: {item.label}")
    lines.append(
        "If one of these is part of where you are now and shaped this hop, add one more line to your reply after "
        f"the receipt with its tag exactly as shown, e.g. [experience:<id>]. At most {NOW_MAX_CITES}, only tags "
        "listed here. No tag is a real answer too. The tags never go inside the note.")
    return "\n".join(lines)


class ReminiscenceNow:
    def __init__(self, *, enabled: bool, base_url: str = SELF_URL, timeout_sec: float = TIMEOUT_SEC,
                 client_factory: Optional[Callable[[], Any]] = None) -> None:
        self.enabled = bool(enabled)
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=self.timeout_sec))

    async def _get(self, run_id: str) -> Any:
        async with self._client_factory() as client:
            return await client.get(f"{self.base_url}/temporal-self/now",
                                    params={"organ": ORGAN, "run_ref": citing_ref(run_id)})

    async def _post(self, body: dict[str, Any]) -> Any:
        async with self._client_factory() as client:
            return await client.post(f"{self.base_url}/temporal-self/now/use", json=body)

    async def read(self, run_id: str) -> Optional[TemporalNowV1]:
        """What the hop is shown, or None (off, unreachable, slow, malformed). Never raises, never takes longer than
        ``timeout_sec`` in all (httpx's own timeout is per phase)."""
        if not self.enabled:
            return None
        try:
            resp = await asyncio.wait_for(self._get(run_id), self.timeout_sec)
            now = parse_now(resp.json()) if resp.status_code == 200 else None
            status = resp.status_code
        except Exception as exc:  # noqa: BLE001 -- the hop runs without the now
            logger.info("reminiscence_now_unavailable run=%s err=%s", run_id, f"{type(exc).__name__}: {exc}"[:200])
            return None
        if now is None:
            logger.info("reminiscence_now_unavailable run=%s status=%s", run_id, status)
            return None
        logger.info("reminiscence_now_read run=%s shown_id=%s items=%d", run_id, now.shown_id, len(now.items))
        return now

    @staticmethod
    def cites_in(now: TemporalNowV1, text: str) -> list[str]:
        """The reply's tags, shown ids first, at most 3. Invented ids are kept so the endpoint counts them."""
        return extract_cites(text, shown=[i.event_id for i in now.items])

    async def close(self, run_id: str, now: TemporalNowV1, cites: list[str]) -> Optional[list[str]]:
        """Post ``cites`` (possibly none) against ``now``; this closes the read. The SHOWN ids among them (what the
        endpoint can store), or None when the post failed. Never raises, bounded by ``timeout_sec``."""
        try:
            resp = await asyncio.wait_for(
                self._post({"shown_id": now.shown_id, "citing_ref": citing_ref(run_id), "cites": list(cites)}),
                self.timeout_sec)
            if resp.status_code != 200:
                logger.warning("reminiscence_now_cites run=%s posted=false status=%s", run_id, resp.status_code)
                return None
            result = resp.json()
        except Exception as exc:  # noqa: BLE001 -- a cite is never worth a failed hop
            logger.warning("reminiscence_now_cites run=%s posted=false err=%s", run_id,
                           f"{type(exc).__name__}: {exc}"[:200])
            return None
        logger.info("reminiscence_now_cites run=%s shown_id=%s cites=%d stored=%d not_shown=%d", run_id, now.shown_id,
                    len(cites), len(result.get("uses") or []), len(result.get("rejected_not_shown") or []))
        shown = {i.event_id for i in now.items}
        return [c for c in cites if c in shown]


__all__ = ["HOP", "ORGAN", "ReminiscenceNow", "citing_ref", "render"]
