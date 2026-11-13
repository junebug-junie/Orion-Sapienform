"""The market's personality values reader (personality rev 4, Move 2). Binding, no shadow.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md, rev 4 "Move 2".
Pure math: ``orion.physiology.personality_values`` (care, holdback, runaway guard) and
``orion.physiology.weights`` (source (c)). This module does the reads and the embedding, for
``app.physiology_market.Market``:

* ``refresh`` (a background task, at most once per ``ttl_sec``; never on a bid's path): reads the active
  tendencies (``personality_disposition``, ``origin = 'choice'``, ``status = 'active'``) and the last 30 days
  of ``personality_choice`` rows. The rows become source (c)'s ``ChoiceRecord``s. The last 7 days' SHOWN
  options are embedded (cached by text) and each is matched to the tendency it most cares about: the
  runaway guard's daily shares. A capped tendency is logged on every refresh
  (``personality_values_tendency_capped``), never silent.
* ``term`` (per bid): the want's arm (20% ``holdback``, deterministic per want id), and in the ``care`` arm
  its care: the want's object text embedded at vector-host (cached; at most ``embed_timeout_sec``) against
  the tendencies. Any failure is care 0 with its reason on the decision (fail-open: care 0 is exactly
  the value without personality).

Read-only on Postgres. Embeddings are compute-only at vector-host (the same ``/embedding`` the personality
reducer uses); nothing is stored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional, Sequence

from orion.physiology import personality_values as pv
from orion.physiology.weights import ChoiceRecord, choice_record

logger = logging.getLogger("orion-durable-runs.personality_values")

EmbedFn = Callable[[str, str], Awaitable[list[float]]]

CHOICE_WINDOW = timedelta(days=30)
EMBED_CACHE_MAX = 4000
GUARD_MAX_TEXTS = 2000           # shown options embedded per refresh (the cache keeps the rest)
GUARD_EMBED_CONCURRENCY = 2      # leaves vector-host room for the per-bid care embeds
# More guard embeds failed than this share of shown options: the shares are unknown, so the last caps are kept
# (a failed embed would otherwise read as "matched nothing" and silently lift every cap).
GUARD_MAX_FAILED_SHARE = 0.2

TENDENCIES_SQL = """
SELECT disposition_id, label, centroid, weight
  FROM personality_disposition
 WHERE origin = 'choice' AND status = 'active'
"""

CHOICES_SQL = """
SELECT choice_id, lane, occurred_at, chosen, alternatives, declined_all, care_promoted, act_id
  FROM personality_choice
 WHERE occurred_at >= %(since)s
 ORDER BY occurred_at
"""


def _json(v: Any) -> Any:
    if isinstance(v, (bytes, str)):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def _utc(v: Any) -> Optional[datetime]:
    if not isinstance(v, datetime):
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def _options(v: Any) -> list[dict]:
    v = _json(v)
    if isinstance(v, dict):
        return [v]
    return [o for o in v if isinstance(o, dict)] if isinstance(v, list) else []


@dataclass(frozen=True)
class ShownChoice:
    occurred_at: datetime
    texts: tuple[str, ...]


def parse_choice_row(r: Any) -> Optional[tuple[ChoiceRecord, ShownChoice]]:
    at = _utc(r.get("occurred_at"))
    if at is None or not r.get("choice_id") or not r.get("lane"):
        return None
    chosen = _options(r.get("chosen"))
    alts = _options(r.get("alternatives"))
    rec = choice_record(str(r["choice_id"]), str(r["lane"]), at, chosen[0].get("key") if chosen else None,
                        [str(o.get("key")) for o in alts if o.get("key")],
                        declined_all=bool(r.get("declined_all")), care_promoted=bool(r.get("care_promoted")),
                        act_id=r.get("act_id"))
    texts = tuple(str(o.get("text")) for o in [*chosen, *alts] if o.get("text"))
    return rec, ShownChoice(at, texts)


def parse_tendency_row(r: Any) -> Optional[pv.Tendency]:
    d = pv.unit(r.get("centroid") or [])
    try:
        w = float(r.get("weight") or 0.0)
    except (TypeError, ValueError):
        return None
    if d is None or not r.get("disposition_id"):
        return None
    return pv.Tendency(str(r["disposition_id"]), str(r.get("label") or ""), d, w)


@dataclass
class Snapshot:
    at: datetime
    tendencies: list[pv.Tendency] = field(default_factory=list)
    choices: list[ChoiceRecord] = field(default_factory=list)
    caps: dict[str, pv.Cap] = field(default_factory=dict)
    shown_options: int = 0
    shown_matched: int = 0
    guard_embed_failed: int = 0
    caps_kept: bool = False      # the guard could not read this refresh; the previous caps stand


class PersonalityValues:
    """One per Market. ``enabled`` False: ``term`` is None and ``choices`` is None (decided as before)."""

    def __init__(self, *, enabled: bool, pool_getter: Callable[[], Any], embed: Optional[EmbedFn],
                 ttl_sec: float = 600.0, embed_timeout_sec: float = 1.0,
                 holdback_fraction: float = pv.HOLDBACK_FRACTION) -> None:
        self.enabled = bool(enabled)
        self._pool_getter = pool_getter
        self._embed = embed
        self._ttl = ttl_sec
        self._embed_timeout = embed_timeout_sec
        self.holdback_fraction = holdback_fraction
        self.snapshot: Optional[Snapshot] = None
        self._refreshing: Optional[asyncio.Task] = None
        self._last_attempt: Optional[datetime] = None
        self._close: Optional[Callable[[], Awaitable[None]]] = None
        self._vec: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self.counts = {"refresh_ok": 0, "refresh_failed": 0, "care": 0, "care_zero": 0, "holdback": 0,
                       "embed_failed": 0}
        self.last_error: Optional[str] = None

    # --- reads ---------------------------------------------------------------------------------------

    def choices(self) -> Optional[list[ChoiceRecord]]:
        """Source (c)'s choice points, or None (flag off, or never read: no (c), never a rate of 0)."""
        if not self.enabled or self.snapshot is None:
            return None
        return self.snapshot.choices

    def snapshot_at(self) -> Optional[datetime]:
        return self.snapshot.at if self.snapshot is not None else None

    def kick(self, now: datetime) -> Optional[asyncio.Task]:
        """Start a background refresh when the snapshot is missing or older than the TTL. Never awaits."""
        if not self.enabled:
            return None
        if self._refreshing is not None and not self._refreshing.done():
            return None
        # The TTL runs from the last ATTEMPT: a database outage retries once per TTL, not on every bid.
        if self._last_attempt is not None and 0 <= (now - self._last_attempt).total_seconds() <= self._ttl:
            return None
        self._last_attempt = now
        try:
            self._refreshing = asyncio.get_running_loop().create_task(self.refresh(now))
        except RuntimeError:
            return None
        return self._refreshing

    async def refresh(self, now: datetime) -> Optional[Snapshot]:
        try:
            async with self._pool_getter().connection() as conn:
                t_rows = await (await conn.execute(TENDENCIES_SQL)).fetchall()
                c_rows = await (await conn.execute(CHOICES_SQL, {"since": now - CHOICE_WINDOW})).fetchall()
        except Exception as exc:  # noqa: BLE001 -- keep the last snapshot; none: care 0, no (c)
            self.counts["refresh_failed"] += 1
            self.last_error = f"read_failed:{type(exc).__name__}"
            logger.warning("personality_values_refresh_failed reason=%s", self.last_error, exc_info=True)
            return self.snapshot
        tendencies = [t for t in (parse_tendency_row(r) for r in t_rows) if t is not None]
        parsed = [p for p in (parse_choice_row(r) for r in c_rows) if p is not None]
        snap = Snapshot(at=now, tendencies=tendencies, choices=[p[0] for p in parsed])
        await self._guard(snap, [p[1] for p in parsed], now)
        if snap.caps_kept and self.snapshot is not None:
            live = {t.disposition_id for t in tendencies}
            snap.caps = {k: v for k, v in self.snapshot.caps.items() if k in live}
        snap.tendencies = pv.apply_caps(tendencies, snap.caps)
        self.snapshot = snap
        self.counts["refresh_ok"] += 1
        self.last_error = None
        for cap in snap.caps.values():
            label = next((t.label for t in tendencies if t.disposition_id == cap.disposition_id), "")
            logger.warning("personality_values_tendency_capped tendency=%s label=%r share_7d=%.3f factor=%.3f",
                           cap.disposition_id, label[:80], cap.mean_share, cap.factor)
        logger.info("personality_values_refresh tendencies=%d choices=%d shown_options_7d=%d matched=%d "
                    "guard_embed_failed=%d caps_kept=%s capped=%d", len(tendencies), len(snap.choices),
                    snap.shown_options, snap.shown_matched, snap.guard_embed_failed, snap.caps_kept, len(snap.caps))
        return snap

    async def _guard(self, snap: Snapshot, shown: Sequence[ShownChoice], now: datetime) -> None:
        """The runaway guard's 7-day shares (module docstring). An option whose embed failed is still a
        shown option (matched to nothing), so a failed embed can only lower a share, never raise it."""
        if not snap.tendencies:
            return
        # The guard judges CAP_DAYS COMPLETE UTC days ending yesterday (pv.runaway_caps), so read from the
        # start of the oldest one. ``now - 7 days`` cut the oldest day short after noon and broke every run.
        today = now.astimezone(timezone.utc).date()
        since = datetime.combine(today - timedelta(days=pv.CAP_DAYS), datetime.min.time(), tzinfo=timezone.utc)
        recent = [s for s in shown if s.occurred_at >= since]
        texts = list(dict.fromkeys(t for s in recent for t in s.texts))[:GUARD_MAX_TEXTS]
        vecs = await self._embed_many(texts)
        entries = []
        failed = 0
        for s in recent:
            d = s.occurred_at.astimezone(timezone.utc).date()
            for t in s.texts:
                v = vecs.get(t)
                tid = None
                if v is None:
                    failed += 1
                else:
                    best, _c, score = pv.best_tendency(v, snap.tendencies, with_cap=False)
                    tid = best.disposition_id if best is not None and score > 0 else None
                entries.append((d, tid))
        snap.shown_options = len(entries)
        snap.shown_matched = sum(1 for _d, tid in entries if tid)
        snap.guard_embed_failed = failed
        if entries and failed / len(entries) > GUARD_MAX_FAILED_SHARE:
            snap.caps_kept = True
            logger.warning("personality_values_guard_unread failed=%d of %d shown options; previous caps kept",
                           failed, len(entries))
            return
        snap.caps = pv.runaway_caps(pv.daily_shown_shares(entries), now.astimezone(timezone.utc).date())

    # --- embedding -----------------------------------------------------------------------------------

    async def _embed_one(self, text: str) -> Optional[tuple[float, ...]]:
        if text in self._vec:
            self._vec.move_to_end(text)
            return self._vec[text]
        if self._embed is None:
            return None
        doc = "physiology-care-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        vec = pv.unit(await self._embed(doc, text))
        if vec is not None:
            self._vec[text] = vec
            while len(self._vec) > EMBED_CACHE_MAX:
                self._vec.popitem(last=False)
        return vec

    async def _embed_many(self, texts: Sequence[str]) -> dict[str, Optional[tuple[float, ...]]]:
        sem = asyncio.Semaphore(GUARD_EMBED_CONCURRENCY)

        async def one(t: str) -> tuple[str, Optional[tuple[float, ...]]]:
            async with sem:
                try:
                    return t, await self._embed_one(t)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("personality_values_guard_embed_failed error=%s", type(exc).__name__)
                    return t, None

        return dict(await asyncio.gather(*(one(t) for t in texts)))

    # --- per bid -------------------------------------------------------------------------------------

    async def term(self, want_id: str, text: Optional[str], *, endorsement_rate: Optional[float] = None,
                   endorsement_n: int = 0, endorsement_excluded: str = "",
                   embed_wait_sec: Optional[float] = None) -> Optional[pv.PersonalityTerm]:
        """The want's personality term, or None when the flag is off. Never raises."""
        if not self.enabled:
            return None
        arm = pv.holdback_arm(want_id, self.holdback_fraction)
        if arm == pv.ARM_HOLDBACK:
            self.counts["holdback"] += 1
            return pv.PersonalityTerm(arm=arm, reason=pv.REASON_HOLDBACK, endorsement_excluded="holdback")
        endorse = dict(endorsement_rate=endorsement_rate, endorsement_n=endorsement_n,
                       endorsement_excluded=endorsement_excluded)
        res = await self._care(pv.object_text(text), embed_wait_sec)
        self.counts["care" if res.care > 0 else "care_zero"] += 1
        return pv.PersonalityTerm(arm=arm, care=res.care, reason=res.reason, tendency_id=res.tendency_id,
                                  cosine=res.cosine, **endorse)

    async def _care(self, text: Optional[str], wait: Optional[float] = None) -> pv.CareResult:
        if text is None:
            return pv.CareResult(0.0, pv.REASON_NO_TEXT)
        snap = self.snapshot
        if snap is None:
            return pv.CareResult(0.0, pv.REASON_UNREAD)
        if not snap.tendencies:
            return pv.CareResult(0.0, pv.REASON_NO_TENDENCY)
        try:
            wait = self._embed_timeout if wait is None else min(wait, self._embed_timeout)
            # Shielded: a short wait gives up on THIS bid, but the embed finishes and fills the cache, so the
            # same text's next bid has it (the embedder's own HTTP timeout bounds the background call).
            task = asyncio.ensure_future(self._embed_one(text))
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            vec = await asyncio.wait_for(asyncio.shield(task), wait)
        except Exception as exc:  # noqa: BLE001 -- fail-open: care 0, said on the decision and in the log
            self.counts["embed_failed"] += 1
            logger.warning("personality_values_care_embed_failed error=%s", type(exc).__name__)
            return pv.CareResult(0.0, pv.REASON_EMBED_FAILED)
        if vec is None:
            self.counts["embed_failed"] += 1
            return pv.CareResult(0.0, pv.REASON_EMBED_FAILED)
        return pv.care_for(vec, snap.tendencies)

    def set_closer(self, close: Optional[Callable[[], Awaitable[None]]]) -> None:
        self._close = close

    async def aclose(self) -> None:
        if self._refreshing is not None and not self._refreshing.done():
            self._refreshing.cancel()
        if self._close is not None:
            try:
                await self._close()
            except Exception:  # noqa: BLE001
                logger.warning("personality_values_close_failed", exc_info=True)

    def health(self) -> dict:
        s = self.snapshot
        return {"enabled": self.enabled, "snapshot_at": s.at.isoformat() if s else None,
                "tendencies": len(s.tendencies) if s else 0, "choices_30d": len(s.choices) if s else 0,
                "shown_options_7d": s.shown_options if s else 0,
                "guard_embed_failed": s.guard_embed_failed if s else 0, "caps_kept": s.caps_kept if s else False,
                "capped": sorted(s.caps) if s else [], "last_error": self.last_error, **self.counts}


__all__ = ["PersonalityValues", "Snapshot", "parse_choice_row", "parse_tendency_row"]
