"""The cooling world act's own loop: the market decides, the pool acts (ladder retirement PR 5, 2026-11-05).

Spec: docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md, "PR 5" and "Decided" 1.
The pure rules are ``orion.autonomy.shed_act`` (want, binding rule, admission, draw, episode row) and
``orion.autonomy.shed_settlement`` (settle from the pool's record). This module only reads, bids and writes.

One tick (``ORION_COOLING_ACT_TICK_SEC``, default 60 s):

1. settle treated sheds still ``shed_pending`` from the pool's ``gpu_pool_orion_shed`` row (or orphan
   them), emit the outcome once (``ActionOutcomeEmitV1``, as dispatch did) and record the terminal on the
   episode -- the scorer in orion-sql-writer then finds a settled episode, exactly as before;
2. a want is held (deferred or no answer) and its retry time is not here: nothing else;
3. read the world (cabinet points, hardware-watch /health, background leases, the episode ledger) and
   run ``evaluate_shed_eligibility`` + ``admission_refusal``. Not eligible: an open want closes (its event
   is over); otherwise nothing;
4. post (or re-post) the event want with the cabinet's salience and OBEY ``shed_verdict``: funded ->
   draw the arm, precommit the episode, and (treated) send the one shed RPC.

Never raises out of a tick. Every read failure is a refusal on the snapshot, never a silent allow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from urllib.request import urlopen
from uuid import uuid4

from orion.autonomy import shed_act
from orion.autonomy.cabinet_heat import read_cabinet_heat
from orion.autonomy.self_shed import HardwareWatchView, evaluate_shed_eligibility
from orion.autonomy.shed_settlement import SHED_PENDING, settle_shed_result
from orion.hardware_watch.rules import TempPoint
from orion.physiology.event_wants import SOURCE_HISTORY

logger = logging.getLogger("orion-durable-runs.cooling_act")

# The binding bid waits this long (the run-bids binding timeout): the decision is the shed's only gate.
BID_TIMEOUT_SEC = 5.0
HISTORY_REFRESH_SEC = 3600.0         # the warm-band history moves by minutes a day; re-read hourly
CABINET_LOOKBACK = timedelta(hours=2)
SETTLE_BATCH = 20
ACTION_OUTCOME_SUBJECT = "orion"     # the subject dispatch used, so chat's "recent actions" read still finds it

def sqlalchemy_url(uri: str) -> str:
    """This image ships psycopg 3, not psycopg2: name the driver SQLAlchemy should use."""
    for prefix in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
        if uri.startswith(prefix):
            return "postgresql+psycopg://" + uri[len(prefix):]
    return uri


class CoolingActStore:
    """Sync SQLAlchemy reads and writes (the shared world-episode helpers take a SQLAlchemy connection)."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._ready = False

    @classmethod
    def from_uri(cls, uri: str) -> "CoolingActStore":
        from sqlalchemy import create_engine

        return cls(create_engine(sqlalchemy_url(uri), pool_pre_ping=True, pool_size=2, max_overflow=1))

    def load_cabinet_points(self, *, since: datetime, until: datetime) -> list[TempPoint]:
        from orion.autonomy.cabinet_heat import load_cabinet_points

        with self._engine.connect() as conn:
            return load_cabinet_points(conn, since=since, until=until)

    def load_warming_history(self, *, now: datetime, rise_threshold_c: float) -> list[tuple[datetime, float]]:
        """The cabinet's warming minutes of the last 30 days (``shed_act.warming_minutes``), from the raw
        readings (~86k rows, re-read hourly), so history and the live rise are the same measure."""
        points = self.load_cabinet_points(since=now - SOURCE_HISTORY - timedelta(minutes=20), until=now)
        return shed_act.warming_minutes(points, rise_threshold_c=rise_threshold_c)

    def load_background_occupancy(self) -> tuple[int, int]:
        """(granted, queued) background leases in the GPU pool's projection (moved from orion-proposal-runtime)."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT status, count(*) FROM gpu_pool_leases WHERE priority = 'background' "
                "AND status IN ('granted', 'recalling', 'queued', 'backlogged', 'retry_wait') GROUP BY status"
            )).fetchall()
        counts = {str(r[0]): int(r[1]) for r in rows}
        return counts.get("granted", 0) + counts.get("recalling", 0), sum(
            counts.get(k, 0) for k in ("queued", "backlogged", "retry_wait"))

    def _ledger_exists(self, conn) -> bool:
        from sqlalchemy import text

        from orion.autonomy import world_episodes

        return conn.execute(text("SELECT to_regclass(:t)"), {"t": world_episodes.TABLE}).scalar() is not None

    def admission_inputs(self, *, now: datetime) -> tuple[list[str], list[dict]]:
        """(episodes still in flight, treated episodes of the last 24 h). ([], []) while no ledger exists."""
        from orion.autonomy import world_episodes

        with self._engine.connect() as conn:
            if not self._ledger_exists(conn):
                return [], []
            return (world_episodes.in_flight(conn, template=shed_act.TEMPLATE, now=now),
                    _decode(world_episodes.recent_treated(conn, template=shed_act.TEMPLATE,
                                                          since=now - timedelta(days=1))))

    def insert_episode(self, row: dict) -> bool:
        """Precommit. Creates the ledger lazily under a short lock timeout (never blocks a boot)."""
        from orion.autonomy import world_episodes

        with self._engine.begin() as conn:
            if not self._ready:
                world_episodes.ensure_table(conn)
            inserted = world_episodes.insert_decision(conn, row)
        self._ready = True
        return inserted

    def episode_arm(self, episode_id: str) -> Optional[str]:
        from orion.autonomy import world_episodes

        with self._engine.connect() as conn:
            return world_episodes.arm_of(conn, episode_id)

    def record_settlement(self, *, episode_id: str, shed_id: Optional[str], state: str, settlement: dict) -> None:
        from orion.autonomy import world_episodes

        with self._engine.begin() as conn:
            world_episodes.record_settlement(conn, episode_id=episode_id, shed_id=shed_id, state=state,
                                             settlement=settlement)

    def load_pending_settlements(self, *, limit: int = SETTLE_BATCH) -> list[dict]:
        """Treated episodes still ``shed_pending`` (last 6 h), each with the pool's orion_self_shed row (or
        None while the pool has no row: they settle as settlement_timeout orphans)."""
        from sqlalchemy import text

        from orion.autonomy import world_episodes

        with self._engine.connect() as conn:
            if not self._ledger_exists(conn):
                return []
            has_pool = conn.execute(text("SELECT to_regclass('gpu_pool_orion_shed')")).scalar() is not None
            pool = ("(SELECT to_jsonb(s) FROM (SELECT shed_id, state, refusal, started_at, ended_at, drained_at, "
                    "grants_withheld, delayed_grant_sec, background_live_at_start, valid_until, ttl_sec "
                    "FROM gpu_pool_orion_shed g WHERE g.dispatch_id = e.episode_id LIMIT 1) s)"
                    if has_pool else "NULL::jsonb")
            rows = conn.execute(text(
                f"SELECT e.episode_id, e.dispatch_kind, e.settlement, {pool} AS pool_row "
                f"FROM {world_episodes.TABLE} e WHERE e.arm = 'treated' AND e.template = :t "
                "AND e.settlement->>'state' = :pending AND e.decided_at > now() - interval '6 hours' "
                "ORDER BY e.decided_at LIMIT :limit"),
                {"t": shed_act.TEMPLATE, "pending": SHED_PENDING, "limit": limit}).mappings().all()
        return _decode([dict(r) for r in rows], keys=("settlement", "pool_row"))

    def load_effect_posteriors(self) -> dict:
        from orion.autonomy.world_settlement_store import WorldSettlementStore

        return WorldSettlementStore(self._engine).load_effect_posteriors()

    def latest_surprise(self) -> Optional[float]:
        from orion.substrate.bus_synaptic_surprise import latest_bus_synaptic_prediction_error

        return latest_bus_synaptic_prediction_error(self._engine)


def _decode(rows: list[dict], keys: tuple[str, ...] = ("settlement",)) -> list[dict]:
    for row in rows:
        for key in keys:
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
    return rows


def read_hardware_watch(url: str, now: datetime) -> HardwareWatchView:
    try:
        with urlopen(url, timeout=2.0) as response:
            return HardwareWatchView.from_health(json.load(response), now=now)
    except Exception as exc:  # noqa: BLE001 -- unreachable is "reflex not known idle", a refusal
        return HardwareWatchView.from_health(None, now=now, error=f"{type(exc).__name__}"[:80])


@dataclass
class OpenWant:
    reading_at: datetime
    opened_at: datetime
    end: datetime
    retry_at: datetime
    want_id: str
    why: str
    bids: int = 0


class CoolingAct:
    """``market_getter``/``bus_getter`` return the process's Market and long-lived bus (or None)."""

    def __init__(self, *, settings: Any, store: Any, market_getter: Callable[[], Any], bus_getter: Callable[[], Any],
                 hardware_watch: Optional[Callable[[datetime], HardwareWatchView]] = None,
                 clock: Optional[Callable[[], datetime]] = None) -> None:
        self._s = settings
        self._store = store
        self._market = market_getter
        self._bus = bus_getter
        self._hw = hardware_watch or (lambda now: read_hardware_watch(settings.hardware_watch_health_url, now))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._open: Optional[OpenWant] = None
        self._previous = None                       # the last CabinetHeatReading (seeds hysteresis)
        self._history: list[tuple[datetime, float]] = []
        self._history_at: Optional[datetime] = None
        self.counts: dict[str, int] = {}
        self.last: dict[str, Any] = {}

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def health(self) -> dict[str, Any]:
        o = self._open
        return {"enabled": True, "counts": dict(self.counts), "last": dict(self.last),
                "open_want": None if o is None else {"want_id": o.want_id, "why": o.why, "bids": o.bids,
                                                     "retry_at": o.retry_at.isoformat(), "end": o.end.isoformat()}}

    async def run(self, stop: asyncio.Event) -> None:
        logger.info("cooling_act_started tick_sec=%.0f holdback=%.2f", self._s.cooling_act_tick_sec,
                    self._s.cooling_act_holdback_fraction)
        while not stop.is_set():
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 -- a tick never kills the loop
                self._count("tick_failed")
                logger.warning("cooling_act_tick_failed", exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=float(self._s.cooling_act_tick_sec))
            except asyncio.TimeoutError:
                continue

    # --- one tick ---------------------------------------------------------------------------------
    async def tick(self, now: Optional[datetime] = None) -> str:
        now = now or self._clock()
        try:
            await self.reconcile(now)
        except Exception:  # noqa: BLE001
            self._count("settle_failed")
            logger.warning("cooling_act_settle_failed", exc_info=True)
        o = self._open
        if o is not None and now < o.retry_at:
            return self._note(now, "holding", want_id=o.want_id, why=o.why)
        elig, cabinet = await asyncio.to_thread(self._eligibility, now)
        if not elig["eligible"]:
            if o is not None:
                self._close(f"event_over:{','.join(elig['refusals'])}"[:200], now)
            return self._note(now, "ineligible", refusals=elig["refusals"])
        if o is None and not shed_act.is_warming(cabinet):
            # Eligible but not warming: no event. (A held want keeps its event until its window ends.)
            return self._note(now, "not_warming", rise_c=cabinet.rise_c, state=cabinet.thermal_state)
        try:
            in_flight, treated = await asyncio.to_thread(self._store.admission_inputs, now=now)
            refusal = shed_act.admission_refusal(
                in_flight=in_flight, treated=treated, now=now, ttl_sec=float(self._s.shed_ttl_sec),
                min_gap_sec=float(self._s.shed_min_gap_sec), max_sec_per_day=float(self._s.shed_max_sec_per_day))
        except Exception:  # noqa: BLE001 -- fail closed: no ledger read, no decision
            logger.warning("cooling_act_admission_unreadable", exc_info=True)
            refusal = "world_ledger_unavailable"
        if refusal is not None:
            if o is not None:
                self._close(f"admission:{refusal}", now)
            return self._note(now, "not_admitted", refusal=refusal)
        salience = await self._salience(cabinet, now)
        if salience is None:
            if o is not None:
                self._close("salience_unknown", now)
            return self._note(now, "no_salience", rise_c=cabinet.rise_c, history=len(self._history))
        market = self._market_or_none()
        if o is None:
            reading_at = cabinet_reading_at(cabinet, now)
            want = shed_act.cooling_want(reading_at=reading_at, salience=salience, opened_at=now)
            o = OpenWant(reading_at=reading_at, opened_at=now, end=want.window.latest, retry_at=now,
                         want_id=want.want_id, why="new")
            self._open = o
        else:
            want = shed_act.cooling_want(reading_at=o.reading_at, salience=salience, opened_at=o.opened_at)
        decision = None
        if market is not None:
            try:
                decision = await asyncio.wait_for(market.bid(want, now), BID_TIMEOUT_SEC)
            except Exception:  # noqa: BLE001 -- incl. timeout: "no answer", never "funded"
                logger.warning("cooling_act_bid_failed want=%s", want.want_id, exc_info=True)
        o.bids += 1
        verdict = shed_act.shed_verdict(decision, now, o.end)
        self._release(market, want.want_id)          # the act holds no lane: nothing of it stays booked
        detail = dict(verdict.detail or {})
        logger.info("cooling_act_bid want=%s event_ref=%s salience=%.3f verdict=%s decision_id=%s outcome=%s reason=%s "
                    "net=%s obeyed=true", want.want_id, want.event_ref, salience, verdict.why, detail.get("decision_id"),
                    detail.get("outcome"), detail.get("reason"), detail.get("net"))
        if verdict.proceed:
            self._open = None
            arm = await self._act(want=want, detail=detail, elig=elig, salience=salience, now=now)
            return self._note(now, f"acted:{arm}", want_id=want.want_id, salience=salience)
        if verdict.close:
            self._close(verdict.why, now)
            return self._note(now, verdict.why, want_id=want.want_id)
        o.retry_at, o.why = verdict.retry_at or now, verdict.why
        self._count("held" if verdict.why.startswith("deferred") else "no_answer")
        return self._note(now, verdict.why, want_id=want.want_id, retry_at=o.retry_at.isoformat())

    def _note(self, now: datetime, result: str, **kw: Any) -> str:
        self.last = {"at": now.isoformat(), "result": result, **kw}
        return result

    def _close(self, why: str, now: datetime) -> None:
        o = self._open
        self._open = None
        if o is not None:
            self._count("closed")
            logger.info("cooling_act_want_closed want=%s why=%s bids=%d", o.want_id, why, o.bids)

    def _market_or_none(self) -> Any:
        try:
            return self._market()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _release(market: Any, want_id: str) -> None:
        if market is None:
            return
        try:
            market.release(want_id)
        except Exception:  # noqa: BLE001
            logger.warning("cooling_act_release_failed want=%s", want_id, exc_info=True)

    def _eligibility(self, now: datetime):
        points = self._store.load_cabinet_points(since=now - CABINET_LOOKBACK, until=now)
        cabinet = read_cabinet_heat(points, now, rise_threshold_c=float(self._s.shed_rise_threshold_c),
                                    previous=self._previous)
        self._previous = cabinet
        hw = self._hw(now)
        try:
            granted, queued = self._store.load_background_occupancy()
        except Exception:  # noqa: BLE001
            logger.warning("cooling_act_pool_occupancy_unavailable", exc_info=True)
            granted, queued = None, None
        try:
            in_flight = self._store.admission_inputs(now=now)[0]
        except Exception:  # noqa: BLE001 -- no ledger read means we cannot prove nothing is in flight
            logger.warning("cooling_act_episode_ledger_unavailable", exc_info=True)
            in_flight = ["episode_ledger_unavailable"]
        elig = evaluate_shed_eligibility(
            cabinet=cabinet, hardware_watch=hw, background_granted=granted, background_queued=queued,
            in_flight_episode_ids=in_flight, holdback_fraction=float(self._s.cooling_act_holdback_fraction), now=now)
        return elig, cabinet

    async def _salience(self, cabinet, now: datetime) -> Optional[float]:
        # On the loop's own clock (``now``), so a replay refreshes on simulated time as the live loop does.
        if self._history_at is None or (now - self._history_at).total_seconds() >= HISTORY_REFRESH_SEC \
                or now < self._history_at:
            try:
                self._history = await asyncio.to_thread(self._store.load_warming_history, now=now,
                                                        rise_threshold_c=float(self._s.shed_rise_threshold_c))
                self._history_at = now
            except Exception:  # noqa: BLE001 -- keep the last history; none -> unknown salience
                logger.warning("cooling_act_history_unavailable", exc_info=True)
        return shed_act.cabinet_salience(cabinet.rise_c, self._history, now)

    # --- the experiment ---------------------------------------------------------------------------
    async def _act(self, *, want, detail: dict, elig: dict, salience: float, now: datetime) -> str:
        episode_id = shed_act.episode_id_for(want.want_id)
        holdback = float(self._s.cooling_act_holdback_fraction)
        arm = shed_act.arm_for(episode_id, holdback)
        eligibility = {**elig, "want": {"want_id": want.want_id, "event_ref": want.event_ref, "salience": salience,
                                        "decision_id": detail.get("decision_id"), "net": detail.get("net"),
                                        "reason": detail.get("reason"), "family": want.family}}
        try:
            effect = shed_act.expected_effect(await asyncio.to_thread(self._store.load_effect_posteriors))
        except Exception:  # noqa: BLE001 -- the posterior is unreadable: a cold claim, said so on the row
            logger.warning("cooling_act_posterior_unavailable", exc_info=True)
            effect = shed_act.expected_effect(None)
        ttl = float(self._s.shed_ttl_sec)
        if arm == "control":
            # The control arm's precommit row is the ONLY record a held-back decision leaves.
            row = shed_act.episode_row(episode_id=episode_id, arm="control", now=now, eligibility=eligibility,
                                       effect=effect, decision_id=detail.get("decision_id"), ttl_sec=ttl)
            try:
                await asyncio.to_thread(self._store.insert_episode, row)
                self._count("control")
            except Exception:  # noqa: BLE001
                self._count("control_unrecorded")
                logger.warning("cooling_act_control_unrecorded episode=%s", episode_id, exc_info=True)
                return "control_unrecorded"
            logger.info("cooling_act_holdback episode=%s fraction=%.2f", episode_id, holdback)
            return "control"
        settlement = {"state": SHED_PENDING, "kind": "orion_self_shed", "decided_at": now.isoformat(),
                      "ttl_sec": ttl, "episode_id": episode_id}
        row = shed_act.episode_row(episode_id=episode_id, arm="treated", now=now, eligibility=eligibility,
                                   effect=effect, decision_id=detail.get("decision_id"), ttl_sec=ttl,
                                   settlement=settlement)
        try:
            inserted = await asyncio.to_thread(self._store.insert_episode, row)
            if not inserted and await asyncio.to_thread(self._store.episode_arm, episode_id) != "treated":
                return "recorded_as_control"        # never shed on a control row
            if not inserted:
                return "already_decided"           # a replay: the first decision already sent its RPC
        except Exception:  # noqa: BLE001 -- no precommit, no action: an unrecorded treatment teaches nothing
            self._count("precommit_failed")
            logger.warning("cooling_act_precommit_failed episode=%s", episode_id, exc_info=True)
            return "precommit_failed"
        out, error, rpc_ms = await self._shed_rpc(episode_id, want, detail, ttl)
        settlement = {**settlement, "rpc_ms": round(rpc_ms, 1)}
        if out is not None:
            settlement.update({"shed_id": out.shed_id, "pool_state": out.state, "refusal": out.refusal,
                               "valid_until": out.valid_until.isoformat() if out.valid_until else None})
        if error:
            settlement["rpc_error"] = error       # the pool may still have started it: settle from its ledger
        try:
            await asyncio.to_thread(self._store.record_settlement, episode_id=episode_id,
                                    shed_id=out.shed_id if out else None,
                                    state=(out.state if out else "rpc_error"), settlement=settlement)
        except Exception:  # noqa: BLE001
            logger.warning("cooling_act_settlement_record_failed episode=%s", episode_id, exc_info=True)
        self._count("treated")
        logger.info("cooling_act_shed_sent episode=%s shed_id=%s state=%s refusal=%s rpc_ms=%.0f",
                    episode_id, out.shed_id if out else None, out.state if out else "rpc_error",
                    out.refusal if out else error, rpc_ms)
        return "treated"

    async def _shed_rpc(self, episode_id: str, want, detail: dict, ttl: float):
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
        from orion.schemas.gpu_pool import (GPU_POOL_SHED_REPLY_PREFIX, GPU_POOL_SHED_REQUEST_CHANNEL,
                                            GPU_POOL_SHED_REQUEST_KIND, GpuPoolShedReasonRequestV1,
                                            GpuPoolShedResultV1)

        req = GpuPoolShedReasonRequestV1(
            action="set", dispatch_id=episode_id, ttl_sec=ttl,
            correlation={"want_id": want.want_id, "event_ref": want.event_ref,
                         "decision_id": detail.get("decision_id"), "trigger": "want_market"})
        started = time.perf_counter()
        bus = self._bus()
        if bus is None:
            return None, "no_bus", 0.0
        reply = f"{GPU_POOL_SHED_REPLY_PREFIX}{uuid4().hex}"
        env = BaseEnvelope(kind=GPU_POOL_SHED_REQUEST_KIND, source=ServiceRef(name=self._s.service_name),
                           correlation_id=str(uuid4()), reply_to=reply, payload=req.model_dump(mode="json"))
        try:
            raw = await bus.rpc_request(GPU_POOL_SHED_REQUEST_CHANNEL, env, reply_channel=reply,
                                        timeout_sec=float(self._s.gpu_pool_shed_rpc_timeout_sec))
            decoded = bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
            if not decoded.ok:
                raise RuntimeError(f"shed_reply_decode_failed:{decoded.error}")
            out = GpuPoolShedResultV1.model_validate(decoded.envelope.payload)
            return out, None, (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("cooling_act_shed_rpc_failed episode=%s error=%s", episode_id, error)
            return None, error, (time.perf_counter() - started) * 1000.0

    # --- settlement -------------------------------------------------------------------------------
    async def reconcile(self, now: datetime) -> int:
        """Settle shed_pending treated episodes from the pool's ledger (or orphan them). The outcome is
        published BEFORE the terminal is recorded (the render path's crash-safety rule)."""
        rows = await asyncio.to_thread(self._store.load_pending_settlements)
        settled = 0
        for row in rows:
            s = settle_shed_result(result_json={"settlement": row.get("settlement") or {}},
                                   pool_row=row.get("pool_row"), now=now)
            if s is None:
                continue
            if not await self._publish_outcome(episode_id=str(row["episode_id"]),
                                               kind=str(row.get("dispatch_kind") or shed_act.DISPATCH_KIND),
                                               summary=s.summary, success=s.success, now=now):
                continue
            block = s.result_json["settlement"]
            await asyncio.to_thread(self._store.record_settlement, episode_id=str(row["episode_id"]),
                                    shed_id=(row.get("pool_row") or {}).get("shed_id"), state=s.terminal,
                                    settlement=block)
            settled += 1
            self._count("settled")
            logger.info("cooling_act_shed_settled episode=%s terminal=%s drain=%s withheld=%s", row["episode_id"],
                        s.terminal, block["manipulation_check"].get("drain"),
                        block["manipulation_check"].get("grants_withheld"))
        return settled

    async def _publish_outcome(self, *, episode_id: str, kind: str, summary: str, success: bool,
                               now: datetime) -> bool:
        """The one ActionOutcomeEmitV1 per settled shed (as dispatch emitted it). True only when it went out."""
        from orion.autonomy.models import ActionOutcomeEmitV1
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        bus = self._bus()
        if bus is None:
            return False
        try:
            surprise = await asyncio.to_thread(self._store.latest_surprise)
        except Exception:  # noqa: BLE001
            surprise = None
        try:
            emit = ActionOutcomeEmitV1(subject=ACTION_OUTCOME_SUBJECT, action_id=episode_id, kind=kind,
                                       summary=summary[:500], success=success, visual_outcome=None,
                                       surprise=surprise if surprise is not None else 0.0, observed_at=now)
            await bus.publish(self._s.action_outcome_channel, BaseEnvelope(
                kind="action.outcome.emit.v1", source=ServiceRef(name=self._s.service_name),
                correlation_id=str(uuid4()), payload=emit.model_dump(mode="json")))
        except Exception:  # noqa: BLE001
            logger.warning("cooling_act_outcome_emit_failed episode=%s", episode_id, exc_info=True)
            return False
        return True


def cabinet_reading_at(cabinet, now: datetime) -> datetime:
    """When the newest cabinet reading was taken (``now - age``), to the second: the event's row."""
    age = cabinet.age_sec if cabinet.age_sec is not None else 0.0
    return (now - timedelta(seconds=float(age))).replace(microsecond=0)


__all__ = ["CoolingAct", "CoolingActStore", "OpenWant", "cabinet_reading_at",
           "read_hardware_watch", "sqlalchemy_url"]
