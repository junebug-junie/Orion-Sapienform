"""Orion tells Juniper when a read changed them: one message per read, in Orion's own words.

Trigger: ``orion/world_pulse_read/findings.py`` (a link a read made, accepted and projected; a
belief a curiosity run moved while citing the read). Delivery: the EXISTING outreach path --
``EndogenousOutreach.compose`` (the real unified turn, PASS allowed) then ``offer_message`` (the
decision log, the closed-vocabulary grounding guard, sockets + chat history + notification) -- not
a parallel one.

Gates (Juniper, 2026-10-13: "reach out as much as they want while learning"): no daily cap and no
cooldown for this kind (``skip_rate_gates``). Disabled, turn-in-flight, recent-chat, quiet hours and
arousal ``strained`` still apply. A finding blocked by them is retried on a later tick (it stays
pending until told), so quiet hours delay a message instead of losing it.

Dedupe: findings of one read are bundled into one message; a finding already told (decision row
``sent`` or ``orion_passed`` naming its ``finding_id``) is never told again. A later finding of the
same read (a belief moved days after its link) is a new bundle. A bundle whose composition fails
``max_attempts`` times is dropped and logged.

Evidence: the decision row carries ``seed_id``, ``finding_ids``, ``finding_kinds`` and
``evidence_refs`` (the way curiosity Door-A rows carry ``run_id``/``line``), and the delivered
message carries an ``outreach_provenance.v1`` capsule with the exact prompt (the evidence).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.schemas.reading_finding import ReadingFindingBundleV1, ReadingFindingV1
from orion.world_pulse_read.finding_prompt import article_appendix, build_reading_finding_prompt
from orion.world_pulse_read.findings import (
    PriorLookup,
    ReadRefV1,
    findings_for_reads,
    first_sentences,
)
from orion.world_pulse_read.introspect import reading_items_between
from orion.world_pulse_read.origin import read_origins

logger = logging.getLogger("orion-hub.reading_finding_outreach")

READING_FINDING_TAG = "reading_finding_outreach"
TOLD_REASONS = frozenset({"sent", "orion_passed"})
# Composition failures that count toward ``max_attempts``. A gate block is not a failure.
FAILED_REASONS = frozenset({"empty_generation", "named_ungrounded_signal"})

# Reads this old are still watched for a NEW finding (a belief a run moves days later).
READ_WINDOW_DAYS = 7

DECISIONS_SQL = """
SELECT reason, result_json->'finding_ids' AS finding_ids
FROM endogenous_outreach_decisions
WHERE decided_at >= $1 AND reason = ANY($2::text[])
  AND result_json->>'source' = 'reading_finding_outreach'
"""


def bundle_key(bundle: ReadingFindingBundleV1) -> str:
    """Stable id of one bundle, for joins. Never a correlation id: an abandoned compose attempt
    is cancelled by correlation id, and a retry reusing it would be killed on spawn."""
    return str(uuid5(NAMESPACE_URL, f"{READING_FINDING_TAG}:{bundle.seed_id}:{'|'.join(bundle.finding_ids)}"))


def reportable(finding: ReadingFindingV1) -> bool:
    """A link counts once it is in the graph (the Assertion projected); a moved belief always."""
    return finding.kind != "link_made" or finding.projected is True


def pending_bundles(
    reads: list[ReadRefV1],
    by_seed: dict[str, list[ReadingFindingV1]],
    *,
    told: set[str],
    failures: dict[str, int],
    max_attempts: int,
) -> list[ReadingFindingBundleV1]:
    """One bundle per read with reportable findings not yet told. Pure."""
    out = []
    for read in reads:
        fresh = [f for f in by_seed.get(read.seed_id, [])
                 if reportable(f) and f.finding_id not in told and failures.get(f.finding_id, 0) < max_attempts]
        if fresh:
            out.append(ReadingFindingBundleV1(seed_id=read.seed_id, title=read.title, url=read.url,
                                              learned=read.learned, article_summary=read.article_summary,
                                              findings=fresh))
    return out


def decision_meta(bundle: ReadingFindingBundleV1, decision_id: str) -> dict[str, Any]:
    return {
        "decision_id": decision_id,
        "bundle_key": bundle_key(bundle),
        "seed_id": bundle.seed_id,
        "title": bundle.title,
        "url": bundle.url,
        "finding_ids": bundle.finding_ids,
        "finding_kinds": sorted({f.kind for f in bundle.findings}),
        "evidence_refs": sorted({r for f in bundle.findings for r in f.evidence_refs}),
    }


def _ids(raw: Any) -> list[str]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    return [str(v) for v in value] if isinstance(value, list) else []


class ReadingFindingOutreach:
    def __init__(
        self,
        *,
        enabled: bool,
        pool_provider: Callable[[], Any],
        outreach_provider: Callable[[], Any],
        prior_lookup: Optional[PriorLookup] = None,
        tick_interval_sec: float = 300.0,
        lookback_hours: float = 48.0,
        max_attempts: int = 3,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._pool_provider = pool_provider
        self._outreach_provider = outreach_provider
        self._prior_lookup = prior_lookup
        self.tick_interval_sec = max(30.0, float(tick_interval_sec))
        self.lookback_hours = float(lookback_hours)
        self.max_attempts = max(1, int(max_attempts))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Process-local mirror of the decision log: a row is written on a daemon thread, so the
        # next tick may not see it yet. Never the only record (restart -> the table answers).
        self._told: set[str] = set()
        self._failures: dict[str, int] = {}
        self._blocked_logged: set[tuple[str, str]] = set()
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.enabled:
            logger.info("reading_finding_outreach disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("reading_finding_outreach started tick=%ss lookback_h=%s max_attempts=%s",
                    self.tick_interval_sec, self.lookback_hours, self.max_attempts)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("reading_finding_outreach_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def _pending(self, now: datetime) -> list[ReadingFindingBundleV1]:
        pool = self._pool_provider()
        if pool is None:
            return []
        # A finding counts when it was RECORDED inside the lookback; its read may be older
        # (up to READ_WINDOW_DAYS), so a belief a run moves days after the read is still told.
        since = now - timedelta(hours=self.lookback_hours)
        async with pool.acquire() as conn:
            items = await reading_items_between(
                conn, since=now - timedelta(days=READ_WINDOW_DAYS), until=now, text_cap=None)
            reads = [ReadRefV1(seed_id=i.id, occurred_at=i.occurred_at, url=i.extra.get("url") or "",
                               request_id=i.extra.get("request_id") or "", title=i.extra.get("title") or "",
                               learned=i.text, article_summary=i.extra.get("article_summary") or "")
                     for i in items]
            result = await findings_for_reads(conn, reads, until=now, prior_lookup=self._prior_lookup)
            if result.beliefs_unchecked:
                logger.warning("reading_finding_beliefs_unchecked reason=%s", result.beliefs_unchecked)
            by_seed = {seed: [f for f in fs if f.observed_at >= since] for seed, fs in result.by_seed.items()}
            rows = await conn.fetch(DECISIONS_SQL, now - timedelta(days=READ_WINDOW_DAYS + 7),
                                    sorted(TOLD_REASONS | FAILED_REASONS))
            bundles = self._bundles(reads, by_seed, rows)
            # How each read started, so the prompt states it from the record instead of asserting it.
            origins = await read_origins(conn, [b.seed_id for b in bundles]) if bundles else {}
        return [b.model_copy(update={"origin": origins.get(b.seed_id, [])}) for b in bundles]

    def _bundles(self, reads: list[ReadRefV1], by_seed: dict[str, list[ReadingFindingV1]],
                 rows: list[Any]) -> list[ReadingFindingBundleV1]:
        told, logged_failures = set(self._told), {}
        for row in rows:
            for fid in _ids(row["finding_ids"]):
                if row["reason"] in TOLD_REASONS:
                    told.add(fid)
                elif row["reason"] in FAILED_REASONS:
                    logged_failures[fid] = logged_failures.get(fid, 0) + 1
        # The in-process count and the logged rows describe the same attempts (the row may lag):
        # the larger one, never the sum.
        failures = {fid: max(logged_failures.get(fid, 0), self._failures.get(fid, 0))
                    for fid in set(logged_failures) | set(self._failures)}
        return pending_bundles(reads, by_seed, told=told, failures=failures, max_attempts=self.max_attempts)

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        bundles = await self._pending(now)
        summary: dict[str, Any] = {"at": now.isoformat(), "pending": len(bundles), "results": []}
        self.last_tick = summary
        if not bundles:
            return summary
        outreach = self._outreach_provider() if self._outreach_provider else None
        if outreach is None:
            logger.warning("reading_finding_outreach_no_outreach_loop pending=%d", len(bundles))
            summary["blocked"] = "no_outreach_loop"
            return summary
        for bundle in bundles:
            summary["results"].append(await self._tell(outreach, bundle))
            if summary["results"][-1]["reason"] not in TOLD_REASONS | FAILED_REASONS:
                break  # a gate is closed: every later bundle would hit it too
        return summary

    async def _tell(self, outreach: Any, bundle: ReadingFindingBundleV1) -> dict[str, Any]:
        correlation_id = str(uuid4())  # per attempt: see bundle_key
        decision_id = str(uuid4())
        meta = decision_meta(bundle, decision_id)
        await outreach.refresh_human_gates()
        blocked = outreach.blocked_reason(skip_rate_gates=True)
        if blocked:
            # Retried next tick. Logged/recorded once per bundle+reason, not every tick of a
            # quiet-hours night.
            key = (meta["bundle_key"], str(blocked))
            if key not in self._blocked_logged:
                if len(self._blocked_logged) > 1000:
                    self._blocked_logged.clear()  # a log-spam guard, not state
                self._blocked_logged.add(key)
                logger.info("reading_finding_outreach_blocked seed=%s reason=%s findings=%d",
                            bundle.seed_id, blocked, len(bundle.findings))
                outreach.record_blocked(str(blocked), correlation_id=correlation_id,
                                        source=READING_FINDING_TAG, extra=meta)
            return {"seed_id": bundle.seed_id, "reason": str(blocked)}
        prompt = build_reading_finding_prompt(bundle)
        text, debug = await outreach.compose(
            prompt, correlation_id, retrieval_query=first_sentences(bundle.learned, 2) or None)
        if not text:
            self._count_failure(bundle)
            outreach.record_blocked("empty_generation", correlation_id=correlation_id,
                                    source=READING_FINDING_TAG, extra=meta)
            logger.info("reading_finding_outreach_no_text seed=%s debug=%s", bundle.seed_id, debug)
            return {"seed_id": bundle.seed_id, "reason": "empty_generation"}
        from scripts.endogenous_outreach import build_outreach_provenance

        capsule = build_outreach_provenance(
            prompt_text=prompt,
            lanes={"reading_findings": len(bundle.findings), "seed_id": bundle.seed_id,
                   "finding_kinds": meta["finding_kinds"]},
            correlation_id=correlation_id, decision_id=decision_id,
        )
        result = await outreach.offer_message(
            text=text, correlation_id=correlation_id, tag=READING_FINDING_TAG,
            model=debug.get("fcc_model_label"), meta=meta, skip_rate_gates=True, provenance=capsule,
            record_appendix=article_appendix(bundle),
        )
        reason = str(result.get("reason") or ("sent" if result.get("outreach") else "not_sent"))
        if reason in TOLD_REASONS:
            self._told.update(bundle.finding_ids)
        elif reason in FAILED_REASONS:
            self._count_failure(bundle)
        logger.info("reading_finding_outreach_result seed=%s reason=%s findings=%s",
                    bundle.seed_id, reason, ",".join(bundle.finding_ids))
        return {"seed_id": bundle.seed_id, "reason": reason, "text": text if reason == "sent" else None}

    def _count_failure(self, bundle: ReadingFindingBundleV1) -> None:
        for fid in bundle.finding_ids:
            self._failures[fid] = self._failures.get(fid, 0) + 1
            if self._failures[fid] >= self.max_attempts:
                logger.warning("reading_finding_outreach_gave_up finding=%s attempts=%d", fid, self._failures[fid])
