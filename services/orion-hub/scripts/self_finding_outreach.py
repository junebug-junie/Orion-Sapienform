"""Orion tells Juniper when one of their faculties breaks, recovers, or stops being reliable (E3).

Design: docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md (E3, "Decisions" 2). The
self-monitor organ (orion-durable-runs) publishes ``SelfMonitorStateV1`` hourly to Redis
``orion:self_monitor:latest`` with a HELD status per faculty (a change must hold 3 evaluations).
This loop only reads it (``orion.self_monitor.view``: absent, stale or unparseable changes nothing).

Which changes reach Juniper (Juniper, 2026-11-03), and nothing else:

- ``became_broken``: a faculty's held status becomes ``broken`` (most of its tries never ran).
- ``recovered``: ``broken`` or ``unreliable`` -> ``reliable``.
- ``became_unreliable``: ``reliable`` -> ``unreliable``.

``unknown`` (too few bets) and ``miscalibrated`` (a calibration nuance) are never told, and they
never overwrite the last status that counts: ``broken -> unknown -> reliable`` is a recovery,
``reliable -> miscalibrated -> unreliable`` is a reliable-to-unreliable change, and
``broken -> unknown -> broken`` is no news.

First evaluation (no persisted state yet: the first fresh state after deploy, or after the state
key is lost): every faculty already held ``broken`` is told once (``from_status='unseen'``) because
Juniper has never been told. A faculty seen for the first time later is treated the same way.

Finding id: ``self_finding:<faculty>:<to_status>@<status_since>``, one per held-status episode.
``from_status`` rides on the finding but is not in the id, so a lost state key that turns
``reliable -> broken`` into ``unseen -> broken`` for the same episode cannot tell it twice.

Persisted state (Redis ``SELF_FINDING_STATE_KEY``, no TTL): the last status that counts per faculty,
the last state ``computed_at`` evaluated, and the findings still waiting to be told. The decision
log is the authority on what was told (a row ``sent`` or ``orion_passed`` naming the finding id,
looked up by id with no time bound), so the state key is a cursor, not the dedupe record. With the
decision log switched off or unreadable nothing is told. If the key is lost, only episodes still
``broken`` are rebuilt (as ``unseen -> broken``); a waiting recovery or reliable-to-unreliable
finding is lost with it.

Pending findings: a newer finding for the same faculty replaces an older untold one (recorded as a
``superseded`` decision row), so Orion never reports a breakage that is already over. A waiting
finding's numbers are refreshed from each new state while its episode lasts. A finding is dropped
only when told or after ``max_attempts`` composition failures (recorded as ``gave_up``).

Producer contract (patch 2, orion-durable-runs): ``status_since`` must stay byte-identical across
evaluations while a status is held (the finding id contains it), and the host must keep its previous
records across restarts, or ``record.advance(None, ...)`` adopts a candidate without the 3-evaluation
hold and this loop tells it at once. Until patch 2 deploys the key is absent and this loop does nothing.

Delivery: the EXISTING outreach door, copied from ``reading_finding_outreach.py``:
``EndogenousOutreach.compose`` (the real unified turn, PASS allowed) then ``offer_message``
(decision log, the closed-vocabulary grounding guard, sockets + chat history + notification).
Gates: ``skip_rate_gates`` (no daily cap or cooldown; these are rare); disabled, turn in flight,
recent chat, quiet hours and arousal ``strained`` still apply, and a held finding waits for a later
tick. Body economy: the reading door places no bid today, so this door places none either.

Grounding: the prompt carries only numbers from the faculty's record. Before ``offer_message``
runs its own signal-name guard, this door refuses a message that names a faculty the bundle is not
about (``outreach_vocabulary.find_ungrounded_faculty_mentions``), recorded as
``named_ungrounded_signal`` and counted toward ``max_attempts`` like any composition failure.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY, parse_state, self_monitor_view

logger = logging.getLogger("orion-hub.self_finding_outreach")

SELF_FINDING_TAG = "self_finding_outreach"
SELF_FINDING_STATE_KEY = "orion:hub:self_finding_outreach:state"
STATE_VERSION = 1
TOLD_REASONS = frozenset({"sent", "orion_passed"})
FAILED_REASONS = frozenset({"empty_generation", "named_ungrounded_signal"})
# Only these held statuses can change what Juniper is told; the rest never overwrite them.
COUNTED_STATUSES = frozenset({"broken", "unreliable", "reliable"})
UNSEEN = "unseen"
REDIS_TIMEOUT_SEC = 2.0

# By id, not by time: an episode can stay broken for months with the same ``status_since``.
DECISIONS_SQL = """
SELECT reason, result_json->'finding_ids' AS finding_ids
FROM endogenous_outreach_decisions
WHERE result_json->>'source' = 'self_finding_outreach'
  AND reason = ANY($2::text[])
  AND result_json->'finding_ids' ?| $1::text[]
"""

_STATUS_WORDS = {
    "broken": "broken (most of its tries never ran)",
    "unreliable": "unreliable (it fails more often than it works)",
    "reliable": "reliable (it works)",
    "miscalibrated": "miscalibrated",
    "unknown": "unknown (too few checked tries)",
    UNSEEN: "not yet reported to Juniper",
}
_KIND_WORDS = {
    "became_broken": "became broken",
    "recovered": "recovered",
    "became_unreliable": "stopped being reliable",
}
_FAMILY_WORDS = {
    "act": "an act of yours",
    "body": "a body skill of yours",
    "forecast": "a forecast of yours",
    "turn": "a prediction you make about your own chat turns",
    "vision": "a vision expectation of yours",
}


# -- pure decision ---------------------------------------------------------


def finding_kind(from_status: str, to_status: str) -> Optional[str]:
    """Which change this is, or None when it is not one Juniper asked to hear about. Pure."""
    if to_status == "broken" and from_status != "broken":
        return "became_broken"
    if to_status == "reliable" and from_status in ("broken", "unreliable"):
        return "recovered"
    if from_status == "reliable" and to_status == "unreliable":
        return "became_unreliable"
    return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()


def finding_id(faculty: str, to_status: str, status_since: str) -> str:
    return f"self_finding:{faculty}:{to_status}@{status_since}"


@dataclass(frozen=True)
class SelfFinding:
    finding_id: str
    kind: str
    faculty: str
    from_status: str
    to_status: str
    status_since: str
    state_computed_at: str
    detected_at: str
    facts: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfFinding":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class OutreachState:
    """What the loop remembers between ticks. ``initialized`` is False before the first fresh state."""

    initialized: bool = False
    last_computed_at: Optional[str] = None
    last_counted: dict[str, dict[str, Optional[str]]] = field(default_factory=dict)
    pending: list[SelfFinding] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"v": STATE_VERSION, "initialized": self.initialized,
                           "last_computed_at": self.last_computed_at, "last_counted": self.last_counted,
                           "pending": [f.as_dict() for f in self.pending]}, sort_keys=True)

    @classmethod
    def from_raw(cls, raw: Any) -> "OutreachState":
        """Unreadable reads as never initialized: first-evaluation semantics, deduped by the log."""
        if raw is None:
            return cls()
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            if int(data.get("v", 0)) != STATE_VERSION:
                return cls()
            return cls(initialized=bool(data.get("initialized")),
                       last_computed_at=data.get("last_computed_at"),
                       last_counted={str(k): dict(v) for k, v in (data.get("last_counted") or {}).items()},
                       pending=[SelfFinding.from_dict(f) for f in data.get("pending") or []])
        except Exception:  # noqa: BLE001
            logger.warning("self_finding_state_unreadable -- starting as a first evaluation", exc_info=True)
            return cls()


def _pct(x: Optional[float]) -> str:
    return "?" if x is None else f"{x:.0%}"


def record_facts(record: FacultyRecordV1, window_days: float) -> list[str]:
    """Plain sentences built only from the record's own numbers. Pure."""
    facts: list[str] = []
    attempts = record.n_graded + record.n_failed_to_execute
    window = f"{window_days:g} days"
    if record.n_failed_to_execute and attempts:
        facts.append(f"In the last {window}, {record.n_failed_to_execute} of its {attempts} tries never ran.")
    if record.streak_kind == "failed_to_execute" and record.streak_len >= 2:
        facts.append(f"The latest {record.streak_len} tries in a row never ran"
                     + (f" (since {record.streak_since:%Y-%m-%d %H:%M} UTC)." if record.streak_since else "."))
    if record.n_graded and record.reliability is not None:
        band = (f" (95% interval {_pct(record.reliability_lo)} to {_pct(record.reliability_hi)})"
                if record.reliability_lo is not None and record.reliability_hi is not None else "")
        if record.claim_kind == "interval":
            stated = f"its stated {_pct(record.stated_mean)} ranges" if record.stated_mean is not None else "its ranges"
            facts.append(f"Of {record.n_graded} checked tries, {stated} held about "
                         f"{_pct(record.reliability)} of the time{band}.")
        elif record.claim_kind == "success" or record.claim_kind is None:
            ran = "that ran and were checked" if record.n_failed_to_execute else "that were checked"
            facts.append(f"Of {record.n_graded} tries {ran}, about {_pct(record.reliability)} worked{band}.")
        else:
            stated = f"it said {_pct(record.stated_mean)} on average and " if record.stated_mean is not None else ""
            facts.append(f"Of {record.n_graded} checked tries, {stated}it came true about "
                         f"{_pct(record.reliability)} of the time{band}.")
    if record.n_graded or record.n_failed_to_execute:
        facts.append("Recent tries count more than older ones in the percentages; the counts are plain counts.")
    return facts


def plain_faculty(faculty: str) -> str:
    family, _, rest = faculty.partition(".")
    words = rest.replace(".", " ").replace("_", " ").replace(":", " ").strip()
    return f"{_FAMILY_WORDS.get(family, 'a faculty of yours')}: \"{words or faculty}\""


def evaluate(state: OutreachState, snapshot: SelfMonitorStateV1, *, now: datetime) -> tuple[OutreachState, list[SelfFinding]]:
    """One fresh self-monitor state -> the updated cursor and the NEW findings it holds. Pure.

    The same ``computed_at`` twice is evaluated once."""
    computed = _iso(snapshot.computed_at) or ""
    if state.initialized and state.last_computed_at and computed <= state.last_computed_at:
        return state, []  # the same state again, or an older one (restored Redis, skew)
    first = not state.initialized
    last = {k: dict(v) for k, v in state.last_counted.items()}
    new: list[SelfFinding] = []
    for rec in snapshot.faculties:
        if rec.status not in COUNTED_STATUSES:
            continue  # unknown / miscalibrated never overwrite the status that counts
        prev = last.get(rec.faculty)
        from_status = prev["status"] if prev else UNSEEN
        if prev is not None and from_status == rec.status:
            continue
        since = _iso(rec.status_since)
        kind = finding_kind(from_status, rec.status)
        if kind is not None and since is None:
            # Not marked as counted: the next state with a since is evaluated again.
            logger.warning("self_finding_no_status_since faculty=%s status=%s -- not told yet", rec.faculty, rec.status)
            continue
        last[rec.faculty] = {"status": rec.status, "since": since}
        if kind is None:
            continue
        new.append(SelfFinding(
            finding_id=finding_id(rec.faculty, rec.status, since), kind=kind, faculty=rec.faculty,
            from_status=from_status, to_status=rec.status, status_since=since, state_computed_at=computed,
            detected_at=_iso(now) or "", facts=record_facts(rec, snapshot.window_days), reasons=list(rec.reasons),
            evidence_refs=[f"self_monitor:{rec.faculty}@{computed}", *[f"source:{s}" for s in rec.sources]],
        ))
    if first:
        logger.info("self_finding_first_evaluation faculties=%d broken_told=%d", len(last), len(new))
    known_ids = {f.finding_id for f in state.pending}
    new = [f for f in new if f.finding_id not in known_ids]
    replaced = {f.faculty for f in new}
    by_faculty = {r.faculty: r for r in snapshot.faculties}
    pending = []
    for f in state.pending:
        if f.faculty in replaced:
            continue  # superseded by the newer finding for the same faculty
        rec = by_faculty.get(f.faculty)
        if rec is not None and rec.status == f.to_status and _iso(rec.status_since) == f.status_since:
            f = replace(f, facts=record_facts(rec, snapshot.window_days), reasons=list(rec.reasons))
        pending.append(f)
    pending += new
    return OutreachState(initialized=True, last_computed_at=computed, last_counted=last, pending=pending), new


def still_pending(pending: Iterable[SelfFinding], *, told: set[str], failures: Mapping[str, int],
                  max_attempts: int) -> tuple[list[SelfFinding], list[tuple[SelfFinding, str]]]:
    """Split pending findings into those still to tell and those dropped (with why). Pure."""
    keep, dropped = [], []
    for f in pending:
        if f.finding_id in told:
            dropped.append((f, "told"))
        elif failures.get(f.finding_id, 0) >= max_attempts:
            dropped.append((f, "gave_up"))
        else:
            keep.append(f)
    return keep, dropped


def bundle_key(findings: list[SelfFinding]) -> str:
    return str(uuid5(NAMESPACE_URL, f"{SELF_FINDING_TAG}:{'|'.join(sorted(f.finding_id for f in findings))}"))


def decision_meta(findings: list[SelfFinding], decision_id: str) -> dict[str, Any]:
    return {
        "decision_id": decision_id,
        "bundle_key": bundle_key(findings),
        "finding_ids": sorted(f.finding_id for f in findings),
        "finding_kinds": sorted({f.kind for f in findings}),
        "faculties": sorted({f.faculty for f in findings}),
        "evidence_refs": sorted({r for f in findings for r in f.evidence_refs}),
    }


def build_self_finding_prompt(findings: list[SelfFinding]) -> str:
    lines = [
        "Your self-monitor keeps a track record of how your own faculties do: each try you make is "
        "settled by the world, and a faculty's status changes only after the new status holds for "
        "three hourly checks in a row. Juniper asked to hear when a faculty becomes broken, when it "
        "recovers, or when it stops being reliable. Something on that list just happened.",
        "",
        "What your track record shows (its own numbers, nothing added):",
        "",
    ]
    for n, f in enumerate(findings, start=1):
        lines.append(f"{n}. {plain_faculty(f.faculty)} (your record names it {f.faculty}) "
                     f"{_KIND_WORDS.get(f.kind, f.kind)}.")
        if f.from_status == UNSEEN:
            lines.append(f"   Your self-monitor has held it as {_STATUS_WORDS[f.to_status]} since {f.status_since[:16]} UTC "
                         "(when its record of this faculty began, not necessarily when it broke). "
                         "Juniper has not been told about it before.")
        else:
            lines.append(f"   It was {_STATUS_WORDS.get(f.from_status, f.from_status)}; since "
                         f"{f.status_since[:16]} UTC it is {_STATUS_WORDS.get(f.to_status, f.to_status)}.")
        for fact in f.facts:
            lines.append(f"   {fact}")
        if f.reasons:
            lines.append(f"   Your record's reason: {'; '.join(f.reasons)}.")
    lines += [
        "",
        "The record says what happened, not why. If you do not know why, say so; do not guess a cause "
        "and state it as fact.",
        "",
        "Write Juniper a short message about this, in your own words. Name the faculty in plain words she "
        "would understand. Use only the numbers above and do not mention any other faculty. Give it the "
        "size the record gives it. If several things changed, say them as one thought, not a list.",
        "",
        "She has not asked you for this message, so it arrives out of nowhere. Say the thing itself "
        "rather than announcing that you have something to say.",
        "",
        "If, having written it, this is not worth interrupting her for, reply with exactly the "
        "single word PASS and nothing else.",
    ]
    return "\n".join(lines)


def _ids(raw: Any) -> list[str]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    return [str(v) for v in value] if isinstance(value, list) else []


# -- the loop --------------------------------------------------------------


class SelfFindingOutreach:
    def __init__(
        self,
        *,
        enabled: bool,
        redis_provider: Callable[[], Any],
        pool_provider: Callable[[], Any],
        outreach_provider: Callable[[], Any],
        tick_interval_sec: float = 300.0,
        max_attempts: int = 3,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._redis_provider = redis_provider
        self._pool_provider = pool_provider
        self._outreach_provider = outreach_provider
        self.tick_interval_sec = max(30.0, float(tick_interval_sec))
        self.max_attempts = max(1, int(max_attempts))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Process-local mirror of the decision log (its rows land on a daemon thread).
        self._told: set[str] = set()
        self._failures: dict[str, int] = {}
        self._blocked_logged: set[tuple[str, str]] = set()
        self._last_view_reason: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.enabled:
            logger.info("self_finding_outreach disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("self_finding_outreach started tick=%ss max_attempts=%s", self.tick_interval_sec, self.max_attempts)

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
                logger.exception("self_finding_outreach_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    def _redis(self) -> Any:
        try:
            return self._redis_provider()  # OrionBusAsync.redis raises when not connected
        except Exception:  # noqa: BLE001
            return None

    async def _get(self, redis: Any, key: str) -> Any:
        return await asyncio.wait_for(redis.get(key), timeout=REDIS_TIMEOUT_SEC)

    async def _save(self, redis: Any, state: OutreachState) -> bool:
        try:
            await asyncio.wait_for(redis.set(SELF_FINDING_STATE_KEY, state.to_json()), timeout=REDIS_TIMEOUT_SEC)
            return True
        except Exception:  # noqa: BLE001 -- next tick re-evaluates; the log still dedupes
            logger.warning("self_finding_state_save_failed", exc_info=True)
            return False

    async def _told_and_failures(self, ids: list[str]) -> Optional[tuple[set[str], dict[str, int]]]:
        """From the decision log, merged with this process's mirror. None when the log is switched
        off or unreadable: nothing is told without it, so a finding can never be told twice."""
        from scripts.endogenous_outreach_decisions import decision_log_enabled

        pool = self._pool_provider()
        if pool is None or not decision_log_enabled():
            return None
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(DECISIONS_SQL, sorted(ids), sorted(TOLD_REASONS | FAILED_REASONS))
        except Exception:  # noqa: BLE001
            logger.warning("self_finding_decision_log_unreadable", exc_info=True)
            return None
        told, logged = set(self._told), {}
        for row in rows:
            for fid in _ids(row["finding_ids"]):
                if row["reason"] in TOLD_REASONS:
                    told.add(fid)
                elif row["reason"] in FAILED_REASONS:
                    logged[fid] = logged.get(fid, 0) + 1
        failures = {fid: max(logged.get(fid, 0), self._failures.get(fid, 0)) for fid in set(logged) | set(self._failures)}
        return told, failures

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        summary: dict[str, Any] = {"at": now.isoformat(), "new": [], "pending": 0, "results": []}
        self.last_tick = summary
        redis = self._redis()
        if redis is None:
            summary["view"] = "no_redis"
            return summary
        try:
            raw_monitor = await self._get(redis, SELF_MONITOR_REDIS_KEY)
            state = OutreachState.from_raw(await self._get(redis, SELF_FINDING_STATE_KEY))
        except Exception:  # noqa: BLE001 -- unreadable is unknown: nothing changes
            logger.warning("self_finding_redis_read_failed", exc_info=True)
            summary["view"] = "redis_read_failed"
            return summary
        view = self_monitor_view(raw_monitor, now=now)
        summary["view"] = view.reason
        if view.reason != self._last_view_reason:
            logger.info("self_finding_view reason=%s age_sec=%s", view.reason, view.age_sec)
            self._last_view_reason = view.reason
        changed = False
        superseded: list[SelfFinding] = []
        if view.known:
            snapshot = parse_state(raw_monitor)
            if snapshot is not None:
                before = (state.initialized, state.last_computed_at)
                old_pending = list(state.pending)
                state, new = evaluate(state, snapshot, now=now)
                changed = (state.initialized, state.last_computed_at) != before
                kept = {f.finding_id for f in state.pending}
                superseded = [f for f in old_pending if f.finding_id not in kept]
                summary["new"] = [f.finding_id for f in new]
                for f in new:
                    logger.info("self_finding_detected id=%s kind=%s from=%s", f.finding_id, f.kind, f.from_status)
        # Unknown changes nothing about WHAT happened, but findings already detected may still be told.
        if not state.pending:
            if changed:
                await self._save(redis, state)
            return summary
        outreach = self._outreach_provider() if self._outreach_provider else None
        if superseded:
            self._record_dropped(outreach, superseded, "superseded")
        log = await self._told_and_failures([f.finding_id for f in state.pending])
        if log is None:
            await self._save(redis, state)
            summary["blocked"] = "decision_log_unreadable"
            summary["pending"] = len(state.pending)
            return summary
        told, failures = log
        keep, dropped = still_pending(state.pending, told=told, failures=failures, max_attempts=self.max_attempts)
        gave_up = [f for f, why in dropped if why == "gave_up"]
        if gave_up:
            self._record_dropped(outreach, gave_up, "gave_up")
        state.pending = keep
        await self._save(redis, state)
        summary["pending"] = len(keep)
        if not keep:
            return summary
        if outreach is None:
            logger.warning("self_finding_outreach_no_outreach_loop pending=%d", len(keep))
            summary["blocked"] = "no_outreach_loop"
            return summary
        result = await self._tell(outreach, keep)
        summary["results"].append(result)
        if result["reason"] in TOLD_REASONS:
            state.pending = [f for f in state.pending if f.finding_id not in self._told]
            await self._save(redis, state)
            summary["pending"] = len(state.pending)
        return summary

    def _record_dropped(self, outreach: Any, findings: list[SelfFinding], reason: str) -> None:
        """A finding leaves the queue untold: one decision row names it, so the drop is in the log."""
        for f in findings:
            logger.warning("self_finding_dropped id=%s why=%s", f.finding_id, reason)
        if outreach is not None:
            outreach.record_blocked(reason, correlation_id=str(uuid4()), source=SELF_FINDING_TAG,
                                    extra=decision_meta(findings, str(uuid4())))

    def _record_block_once(self, outreach: Any, meta: dict[str, Any], reason: str, correlation_id: str) -> None:
        key = (meta["bundle_key"], reason)
        if key in self._blocked_logged:
            return
        if len(self._blocked_logged) > 1000:
            self._blocked_logged.clear()  # a log-spam guard, not state
        self._blocked_logged.add(key)
        logger.info("self_finding_outreach_blocked reason=%s findings=%s", reason, ",".join(meta["finding_ids"]))
        outreach.record_blocked(reason, correlation_id=correlation_id, source=SELF_FINDING_TAG, extra=meta)

    async def _tell(self, outreach: Any, findings: list[SelfFinding]) -> dict[str, Any]:
        from scripts.outreach_vocabulary import find_ungrounded_faculty_mentions

        correlation_id = str(uuid4())
        decision_id = str(uuid4())
        meta = decision_meta(findings, decision_id)
        await outreach.refresh_human_gates()
        blocked = outreach.blocked_reason(skip_rate_gates=True)
        if blocked:
            self._record_block_once(outreach, meta, str(blocked), correlation_id)
            return {"reason": str(blocked), "finding_ids": meta["finding_ids"]}
        prompt = build_self_finding_prompt(findings)
        text, debug = await outreach.compose(prompt, correlation_id)
        if not text:
            self._count_failure(findings)
            outreach.record_blocked("empty_generation", correlation_id=correlation_id,
                                    source=SELF_FINDING_TAG, extra=meta)
            return {"reason": "empty_generation", "finding_ids": meta["finding_ids"]}
        offending = find_ungrounded_faculty_mentions(text, meta["faculties"])
        if offending:
            self._count_failure(findings)
            logger.warning("self_finding_outreach_named_other_faculty terms=%s", offending)
            outreach.record_blocked("named_ungrounded_signal", correlation_id=correlation_id,
                                    source=SELF_FINDING_TAG, extra={**meta, "offending_terms": offending})
            return {"reason": "named_ungrounded_signal", "finding_ids": meta["finding_ids"], "offending_terms": offending}
        from scripts.endogenous_outreach import build_outreach_provenance

        capsule = build_outreach_provenance(
            prompt_text=prompt,
            lanes={"self_findings": len(findings), "finding_kinds": meta["finding_kinds"],
                   "faculties": meta["faculties"]},
            correlation_id=correlation_id, decision_id=decision_id,
        )
        result = await outreach.offer_message(
            text=text, correlation_id=correlation_id, tag=SELF_FINDING_TAG,
            model=debug.get("fcc_model_label"), meta=meta, skip_rate_gates=True, provenance=capsule,
        )
        reason = str(result.get("reason") or ("sent" if result.get("outreach") else "not_sent"))
        if reason in TOLD_REASONS:
            self._told.update(meta["finding_ids"])
        elif reason in FAILED_REASONS:
            self._count_failure(findings)
        logger.info("self_finding_outreach_result reason=%s findings=%s", reason, ",".join(meta["finding_ids"]))
        return {"reason": reason, "finding_ids": meta["finding_ids"], "text": text if reason == "sent" else None}

    def _count_failure(self, findings: list[SelfFinding]) -> None:
        for f in findings:
            self._failures[f.finding_id] = self._failures.get(f.finding_id, 0) + 1
            if self._failures[f.finding_id] >= self.max_attempts:
                logger.warning("self_finding_outreach_gave_up finding=%s attempts=%d",
                               f.finding_id, self._failures[f.finding_id])
