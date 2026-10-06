"""The watcher: every tick, read telemetry, run the pure rules, act on transitions.

Order on a cooling incident opening (Decisions locked, "AC order"): persist the incident, send the
critical alert, publish the urgent investigation request, publish the incident event. Each side
effect is recorded on the row; one that failed is retried on the next tick, never re-sent once it
succeeded (a restart reads the row, so it does not re-fire either).

Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md.

Thermal controller v2 (HARDWARE_WATCH_HEAT_CONTROLLER=v2, the default; spec
docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md):
- D1/D2 every tick: read the cabinet once (``read_cabinet_heat``; a failed query re-uses the last good
  readings within grace), and while it is critical (>= 34 C) or unreadable past grace publish one
  ``HardwareWatchReflexShedV1`` (``cabinet_hot`` / ``cabinet_unknown``, valid 3 ticks). No latch: when
  the state drops, one ``active=false`` clear is sent and the signal stops.
- D5 the cooling rule is ``cooling_verdict_v2``: AC power opens an incident only while the cabinet is
  warm; every v2 incident is alert-only (``shed=None`` on the event).
- D6 alerts and urgent investigations: at most one per rule+subject per sliding window; no urgent run
  for a p95 heat outlier. D7 heat opens on fixed ceilings, p95 is an annotation, a silent sensor
  resolves ``sensor_lost``. D9 an operator resolve snoozes only the reason it resolved.
v1 keeps the 2026-09-29 cooling + latched shed path for the one-week rollback window.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from orion.autonomy.cabinet_heat import CabinetHeatReading, read_cabinet_heat
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.hardware_watch.rules import (
    AcLowConfig, Baseline, CoolingRuleConfig, HeatRuleConfig, ShedRuleConfig, ShedVerdict, cooling_verdict,
    cooling_verdict_v2, heat_verdict, shed_verdict,
)
from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL, URGENT_REQUEST_KIND, CuriosityUrgentRequestV1
from orion.schemas.hardware_watch import (
    HARDWARE_WATCH_INCIDENT_CHANNEL, HARDWARE_WATCH_INCIDENT_KIND, HARDWARE_WATCH_REFLEX_SHED_CHANNEL,
    HARDWARE_WATCH_REFLEX_SHED_KIND, HardwareWatchIncidentV1, HardwareWatchReflexShedV1, HardwareWatchShedV1,
)
from orion.schemas.notify import NotificationRequest

from app.settings import Settings

logger = logging.getLogger("orion-hardware-watch.watcher")

COOLING_SUBJECT = "cabinet_ac"
CPU_KEY = "temp_c_max"
CABINET_KEY = "cabinet_temp_c"
EVIDENCE_POINTS = 40
# D2: a reflex signal lives this many ticks past its emit; the watcher re-sends it every tick.
REFLEX_VALID_TICKS = 3
# How much cabinet history one read covers (the 15-min rise window plus margin; hysteresis is seeded
# from the previous reading, so the window does not need to reach back to the trip).
CABINET_WINDOW_SEC = 1800
HEAT_RULES = ("cpu_heat", "gpu_heat")

Publish = Callable[[str, BaseEnvelope], Awaitable[None]]
Notify = Callable[[NotificationRequest], Any]   # sync, returns NotificationAccepted-like (.ok, .detail)

REASON_TEXT = {
    "low_power": "the AC plug has read under {low:.0f} W for 3+ minutes (compressor not running)",
    "no_fresh_sample": "the AC plug has sent no fresh reading for 5+ minutes (stale samples only)",
    "device_offline": "the AC plug has been offline to Z-Wave for 5+ minutes",
    "controller_not_ready": "the Z-Wave controller has not been ready for 5+ minutes",
    "no_samples": "no AC reading has arrived at all for 5+ minutes (orion-zwave or sql-writer down?)",
    "frozen": "the AC plug has reported the exact same wattage for 60+ minutes (a stuck or lying reading)",
    "low_power_v2": ("the AC plug's {window:.0f}-minute mean draw is {mean} W, under the {low:.0f} W duty-cycle "
                     "floor, while the cabinet is {state} ({temp} C)"),
    "simulated": "SIMULATED incident from the hardware-watch test hook (a drill; the AC is probably fine)",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _BaselineCache:
    base: Baseline
    computed_at: datetime


@dataclass
class TickReport:
    at: datetime | None = None
    ok: bool = True
    errors: dict[str, str] = field(default_factory=dict)
    verdicts: dict[str, dict] = field(default_factory=dict)


class Watcher:
    def __init__(self, *, settings: Settings, store: Any, publish: Publish, notify: Notify,
                 clock: Callable[[], datetime] = _utcnow, source: ServiceRef | None = None,
                 run_sync: Callable[..., Awaitable[Any]] | None = None):
        self.s = settings
        self.store = store
        self.publish = publish
        self.notify = notify
        self.clock = clock
        self.source = source or ServiceRef(name=settings.service_name, version=settings.service_version,
                                           node=settings.node_name)
        # store/notify calls are blocking; production runs them in a thread, tests inline.
        self._sync = run_sync or (lambda fn, *a, **kw: asyncio.to_thread(fn, *a, **kw))
        self.cooling_cfg = CoolingRuleConfig(
            low_watts=settings.ac_low_w, low_sec=settings.ac_low_sec, stale_sec=settings.ac_stale_sec,
            frozen_sec=settings.ac_frozen_sec, resolve_watts=settings.ac_resolve_w,
            resolve_sec=settings.ac_resolve_sec)
        self.shed_cfg = ShedRuleConfig(rise_c=settings.shed_rise_c, window_sec=settings.shed_rise_window_sec)
        self.ac_cfg = AcLowConfig(low_mean_w=settings.ac_low_mean_w, window_sec=settings.ac_low_window_sec,
                                  lookahead_min=settings.heat_lookahead_min)
        self.v2 = settings.heat_controller == "v2"
        # D7: fixed ceilings open heat incidents; the p95 rides along as an annotation.
        self.cpu_cfg = HeatRuleConfig(sustain_sec=settings.heat_sustain_sec,
                                      min_history_sec=settings.cpu_min_history_sec,
                                      ceiling_c=settings.cpu_ceiling_c, ceiling_sustain_sec=settings.heat_sustain_sec,
                                      ceiling_rearm_c=settings.cpu_ceiling_rearm_c, p95_opens=False,
                                      lost_sec=settings.heat_sensor_lost_sec)
        self.gpu_cfg = HeatRuleConfig(sustain_sec=settings.heat_sustain_sec,
                                      min_history_sec=settings.gpu_min_history_sec,
                                      ceiling_c=settings.gpu_ceiling_c,
                                      ceiling_sustain_sec=settings.gpu_ceiling_sustain_sec,
                                      ceiling_rearm_c=settings.gpu_ceiling_rearm_c, p95_opens=False,
                                      lost_sec=settings.heat_sensor_lost_sec)
        # v2 cabinet state (D1/D2)
        self._cab: CabinetHeatReading | None = None
        self._cab_points: list = []
        self._cab_points_at: datetime | None = None
        self._ac_low: bool | None = None
        self._reflex_sent: str | None = None          # reason last asserted (None = cleared/never)
        self._reflex: dict[str, Any] = {"controller": settings.heat_controller, "active": False, "reason": None}
        self._baselines: dict[str, _BaselineCache] = {}
        self._last_refresh: dict[str, datetime] = {}
        self._last_shed: dict[str, Any] = {}          # incident_id -> latest ShedVerdict (for the event)
        self._pending_resolved: dict[str, dict] = {}  # resolved rows whose event failed to publish
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None
        self.last = TickReport()

    def _guard(self) -> asyncio.Lock:
        """One lock per event loop: a tick, an operator resolve and a simulate never interleave, so
        a tick can never act on a row the operator just closed (re-shedding a resolved incident)."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    # --- the tick -----------------------------------------------------------------------------
    async def tick(self) -> TickReport:
        async with self._guard():
            return await self._tick()

    async def _tick(self) -> TickReport:
        now = self.clock()
        report = TickReport(at=now)
        if not self.s.enabled:
            self.last = report
            return report
        try:
            open_rows = {(r["rule"], r["subject"]): r for r in await self._sync(self.store.open_incidents)}
        except Exception as exc:  # noqa: BLE001 -- no incident memory: act on nothing this tick
            report.ok = False
            report.errors["open_incidents"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("hardware_watch_store_unreadable err=%s", exc)
            self.last = report
            return report
        steps = (("cabinet", self._cabinet), ("cooling", self._cooling), ("reflex", self._reflex_tick),
                 ("cpu_heat", self._cpu_heat), ("gpu_heat", self._gpu_heat)) if self.v2 else (
                 ("cooling", self._cooling), ("cpu_heat", self._cpu_heat), ("gpu_heat", self._gpu_heat))
        for name, step in steps:
            try:
                await step(now, open_rows, report)
            except Exception as exc:  # noqa: BLE001 -- one rule failing must not stop the others
                report.ok = False
                report.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
                logger.exception("hardware_watch_rule_failed rule=%s", name)
        try:
            await self._retry_and_refresh(now, report)
        except Exception as exc:  # noqa: BLE001
            report.ok = False
            report.errors["refresh"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.exception("hardware_watch_refresh_failed")
        self.last = report
        return report

    # --- rules --------------------------------------------------------------------------------
    async def _cabinet(self, now: datetime, open_rows: dict, report: TickReport) -> None:
        """D1: one cabinet read per tick. A failed query re-uses the last good readings, re-judged at
        ``now`` -- within grace that is the last state; past grace it is unknown (C10)."""
        try:
            pts = await self._sync(self.store.temp_points, self.s.cabinet_node, CABINET_KEY,
                                   now - timedelta(seconds=CABINET_WINDOW_SEC))
            self._cab_points, self._cab_points_at = list(pts), now
        except Exception as exc:  # noqa: BLE001 -- recorded; the reading below decides what it means
            report.errors["cabinet_read"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("hardware_watch_cabinet_read_failed err=%s (re-using readings from %s)", str(exc)[:200],
                           self._cab_points_at.isoformat() if self._cab_points_at else None)
        self._cab = read_cabinet_heat(self._cab_points, now, grace_sec=self.s.reading_grace_sec, previous=self._cab)
        report.verdicts["cabinet"] = self._cab.as_dict()

    def _cabinet_now(self, now: datetime) -> CabinetHeatReading:
        return self._cab if self._cab is not None else read_cabinet_heat([], now, grace_sec=self.s.reading_grace_sec)

    async def _reflex_tick(self, now: datetime, open_rows: dict, report: TickReport) -> None:
        """D2: assert the reflex reason every tick while it holds; one clear when it stops."""
        reading = dataclasses.replace(self._cabinet_now(now), ac_low=self._ac_low)
        reason = reading.reflex
        valid_until = now + timedelta(seconds=REFLEX_VALID_TICKS * self.s.tick_sec)
        self._reflex = {"controller": "v2", "active": bool(reason) and self.s.shed_enabled,
                        "reason": reason if self.s.shed_enabled else None, "would_shed": reason,
                        "shed_enabled": self.s.shed_enabled, "evaluated_at": now.isoformat(),
                        "valid_until": valid_until.isoformat() if reason and self.s.shed_enabled else None,
                        "cabinet": reading.as_dict()}
        report.verdicts["reflex"] = {k: v for k, v in self._reflex.items() if k != "cabinet"}
        if reason and self.s.shed_enabled:
            sig = HardwareWatchReflexShedV1(source_id=self._reflex_source(), active=True, reason=reason,
                                            valid_until=valid_until, cabinet=reading.as_dict(), emitted_at=now)
        elif self._reflex_sent is not None:
            sig = HardwareWatchReflexShedV1(source_id=self._reflex_source(), active=False,
                                            cabinet=reading.as_dict(), emitted_at=now)
        else:
            return
        try:
            await self.publish(HARDWARE_WATCH_REFLEX_SHED_CHANNEL, BaseEnvelope(
                kind=HARDWARE_WATCH_REFLEX_SHED_KIND, source=self.source, payload=sig.model_dump(mode="json")))
        except Exception as exc:  # noqa: BLE001 -- re-sent next tick; a lost one lapses at valid_until
            report.errors["reflex_publish"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.error("hardware_watch_reflex_publish_failed reason=%s err=%s", reason, exc)
            return
        if (reason if sig.active else None) != self._reflex_sent:
            logger.warning("hardware_watch_reflex_%s reason=%s temp_c=%s state=%s age_sec=%s ac_low=%s",
                           "on" if sig.active else "off", reason or self._reflex_sent, reading.temp_c,
                           reading.thermal_state, reading.age_sec, reading.ac_low)
        self._reflex_sent = reason if sig.active else None

    def _reflex_source(self) -> str:
        return f"{self.s.service_name}:{self.s.cabinet_node}:cabinet"[:120]

    def reflex_snapshot(self) -> dict[str, Any]:
        """What /health shows (and Orion's learned shed reads, D8): the reflex's own current claim."""
        return dict(self._reflex)

    async def _cooling(self, now: datetime, open_rows: dict, report: TickReport) -> None:
        if self.v2:
            return await self._cooling_v2(now, open_rows, report)
        lookback = max(self.cooling_cfg.frozen_sec, self.cooling_cfg.resolve_sec) + 300
        points = await self._sync(self.store.cooling_points, now - timedelta(seconds=lookback))
        v = cooling_verdict(points, now, self.cooling_cfg)
        report.verdicts["cooling"] = {"open_reason": v.open_reason, "resolve": v.resolve, **v.detail}
        row = open_rows.get(("cooling", COOLING_SUBJECT))
        if row is None:
            if v.open_reason:
                evidence = {"verdict": v.detail, "recent": _compact_cooling(points)}
                await self._open("cooling", COOLING_SUBJECT, v.open_reason, now, evidence)
            return
        if row["open_reason"] == "simulated":
            # A drill ends only by operator resolve -- unless the AC really fails meanwhile: then the
            # drill is closed and the real incident opens (real alert, real investigation).
            if v.open_reason:
                await self._resolve(row, now, "superseded", "rule")
                evidence = {"verdict": v.detail, "recent": _compact_cooling(points)}
                await self._open("cooling", COOLING_SUBJECT, v.open_reason, now, evidence)
                return
        elif v.resolve:
            await self._resolve(row, now, "recovered", "rule")
            return
        await self._update_shed(row, now)

    async def _cooling_v2(self, now: datetime, open_rows: dict, report: TickReport) -> None:
        """D5: AC power is a diagnosis. Alert-only incidents; the shed is the reflex's (D2)."""
        lookback = max(self.cooling_cfg.frozen_sec, self.ac_cfg.window_sec) + 300
        row = open_rows.get(("cooling", COOLING_SUBJECT))
        try:
            points = await self._sync(self.store.cooling_points, now - timedelta(seconds=lookback))
        except Exception:
            self._ac_low = None     # an unreadable AC is not "AC low" (D1: the outage case is elevated)
            raise
        cabinet = self._cabinet_now(now)
        v = cooling_verdict_v2(points, now, cabinet, cfg=self.cooling_cfg, ac=self.ac_cfg,
                               open_reason=row["open_reason"] if row else None)
        self._ac_low = v.detail.get("ac_low")
        report.verdicts["cooling"] = {"open_reason": v.open_reason, "resolve": v.resolve,
                                      **{k: x for k, x in v.detail.items() if k != "cabinet"}}
        evidence = {"verdict": v.detail, "recent": _compact_cooling(points)}
        if row is None:
            if v.open_reason:
                await self._open("cooling", COOLING_SUBJECT, v.open_reason, now, evidence)
            return
        if row["open_reason"] == "simulated":
            if v.open_reason:
                await self._resolve(row, now, "superseded", "rule")
                await self._open("cooling", COOLING_SUBJECT, v.open_reason, now, evidence)
            return
        if v.resolve:
            await self._resolve(row, now, "recovered", "rule")

    async def _cpu_heat(self, now: datetime, open_rows: dict, report: TickReport) -> None:
        nodes = list(dict.fromkeys(self.s.heat_node_list + [s for (r, s) in open_rows if r == "cpu_heat"]))
        for node in nodes:
            await self._heat("cpu_heat", node, node, CPU_KEY, self.cpu_cfg, now, open_rows, report)

    async def _gpu_heat(self, now: datetime, open_rows: dict, report: TickReport) -> None:
        # An open incident is always re-evaluated, even if its card stopped reporting or its node left
        # the config (a silent sensor neither opens nor resolves: it waits for data or the operator).
        subjects: dict[str, tuple[str, str]] = {}
        for (rule, subject) in open_rows:
            if rule == "gpu_heat" and "/gpu" in subject:
                node, card = subject.split("/gpu", 1)
                subjects[subject] = (node, f"gpu{card}_temp_c")
        for node in self.s.gpu_node_list:
            keys = await self._sync(self.store.gpu_keys, node, now - timedelta(hours=1))
            for key in keys:
                subjects[f"{node}/gpu{key[3:-len('_temp_c')]}"] = (node, key)
        for subject, (node, key) in subjects.items():
            await self._heat("gpu_heat", subject, node, key, self.gpu_cfg, now, open_rows, report)

    async def _heat(self, rule: str, subject: str, node: str, key: str, cfg: HeatRuleConfig, now: datetime,
                    open_rows: dict, report: TickReport) -> None:
        base = await self._baseline(node, key, now)
        window = max(cfg.sustain_sec, cfg.ceiling_sustain_sec) + 300
        points = await self._sync(self.store.temp_points, node, key, now - timedelta(seconds=window))
        v = heat_verdict(points, now, base, cfg)
        report.verdicts[f"{rule}:{subject}"] = {"open_reason": v.open_reason, "resolve": v.resolve, **v.detail}
        row = open_rows.get((rule, subject))
        if row is None:
            if v.open_reason:
                evidence = {"verdict": v.detail, "recent": [[p.ts.isoformat(), p.value] for p in points][-EVIDENCE_POINTS:]}
                await self._open(rule, subject, v.open_reason, now, evidence)
        elif v.resolve:
            await self._resolve(row, now, v.detail.get("resolve_reason") or "recovered", "rule")

    async def _baseline(self, node: str, key: str, now: datetime) -> Baseline:
        ck = f"{node}:{key}"
        cached = self._baselines.get(ck)
        if cached and (now - cached.computed_at).total_seconds() < self.s.heat_baseline_refresh_sec:
            return cached.base
        pts = await self._sync(self.store.temp_points, node, key, now - timedelta(days=self.s.heat_baseline_days))
        base = Baseline.from_points(pts, now)
        self._baselines[ck] = _BaselineCache(base, now)
        logger.info("hardware_watch_baseline node=%s key=%s n=%s p75=%s p95=%s history_h=%.1f",
                    node, key, base.n, base.p75, base.p95, base.history_sec / 3600)
        return base

    # --- transitions --------------------------------------------------------------------------
    async def _open(self, rule: str, subject: str, reason: str, now: datetime, evidence: dict,
                    *, by: str = "rule") -> dict | None:
        snooze = await self._sync(self.store.snoozed_until, rule, subject, reason)   # D9: this reason only
        if by == "rule" and snooze is not None and snooze > now:
            logger.info("hardware_watch_open_snoozed rule=%s subject=%s reason=%s until=%s", rule, subject, reason,
                        snooze.isoformat())
            return None
        row: dict[str, Any] = {"incident_id": uuid.uuid4().hex, "rule": rule, "subject": subject, "status": "open",
                               "open_reason": reason, "opened_at": now, "evidence": evidence, "updated_at": now}
        sv = None
        if rule == "cooling" and not self.v2:
            sv = await self._shed_now(now)
            row.update(self._shed_fields(sv, now))
        if not await self._sync(self.store.insert_incident, row):
            logger.info("hardware_watch_already_open rule=%s subject=%s", rule, subject)
            return None
        if sv is not None:
            self._last_shed[row["incident_id"]] = sv
        logger.warning("hardware_watch_incident_opened id=%s rule=%s subject=%s reason=%s shed=%s",
                       row["incident_id"], rule, subject, reason, row.get("shed_reason"))
        if rule == "cooling":
            await self._send_alert(row, now)            # alert first
        await self._request_urgent(row, now)            # investigate second
        await self._emit(row, "opened", now)
        return row

    async def _resolve(self, row: dict, now: datetime, reason: str, by: str) -> dict:
        fields: dict[str, Any] = {"status": "resolved", "resolved_at": now, "resolve_reason": reason,
                                  "resolved_by": by}
        # A drill's resolve never snoozes the real rule (the snooze is per rule+subject).
        if reason == "operator" and self.s.operator_snooze_sec > 0 and row["open_reason"] != "simulated":
            fields["snooze_until"] = now + timedelta(seconds=self.s.operator_snooze_sec)
        if not await self._sync(self.store.resolve_incident, row["incident_id"], **fields):
            logger.info("hardware_watch_already_resolved id=%s", row["incident_id"])
            return await self._sync(self.store.get_incident, row["incident_id"]) or {**row, **fields}
        row = {**row, **fields}
        self._last_shed.pop(row["incident_id"], None)
        logger.warning("hardware_watch_incident_resolved id=%s rule=%s subject=%s reason=%s by=%s",
                       row["incident_id"], row["rule"], row["subject"], reason, by)
        if not await self._emit(row, "resolved", now):
            self._pending_resolved[row["incident_id"]] = row   # retried next tick
        self._last_refresh.pop(row["incident_id"], None)
        if row["rule"] == "cooling" and row.get("alert_sent_at"):
            await self._send_recovered(row, now)    # no "closed" notice for an incident whose alert was deduped
        return row

    async def resolve_by_operator(self, incident_id: str, by: str = "juniper") -> dict | None:
        async with self._guard():
            row = await self._sync(self.store.get_incident, incident_id)
            if row is None or row["status"] != "open":
                return row
            return await self._resolve(row, self.clock(), "operator", by[:64] or "juniper")

    async def simulate_cooling(self) -> dict | None:
        async with self._guard():
            now = self.clock()
            return await self._open("cooling", COOLING_SUBJECT, "simulated", now,
                                    {"simulated": True, "note": "POST /incidents/simulate test hook"}, by="operator")

    # --- shed ---------------------------------------------------------------------------------
    async def _shed_now(self, now: datetime) -> ShedVerdict:
        """The shed verdict; a failed cabinet read counts as unreadable (warming). It must never
        block the AC alert, which is opened right after this."""
        try:
            pts = await self._sync(self.store.temp_points, self.s.cabinet_node, CABINET_KEY,
                                   now - timedelta(seconds=max(self.shed_cfg.window_sec, self.shed_cfg.max_age_sec) + 60))
        except Exception as exc:  # noqa: BLE001
            logger.warning("hardware_watch_cabinet_unreadable err=%s", str(exc)[:300])
            return ShedVerdict(True, "cabinet_unreadable", None, None)
        return shed_verdict(pts, now, self.shed_cfg)

    def _shed_fields(self, sv, now: datetime) -> dict[str, Any]:
        if not sv.requested:
            return {"shed_requested": False, "shed_reason": None}
        if not self.s.shed_enabled:
            return {"shed_requested": False, "shed_reason": "disabled"}
        return {"shed_requested": True, "shed_reason": sv.reason, "shed_requested_at": now}

    async def _update_shed(self, row: dict, now: datetime) -> None:
        """Latched: once a cooling incident requested shedding it keeps requesting until it resolves.
        Before that, re-evaluate every tick and publish at once when it starts."""
        sv = await self._shed_now(now)
        self._last_shed[row["incident_id"]] = sv        # current temp/rise for the event
        if row.get("shed_requested"):
            return
        fields = self._shed_fields(sv, now)
        if fields.get("shed_requested"):
            await self._sync(self.store.update_incident, row["incident_id"], **fields)
            row.update(fields)
            logger.warning("hardware_watch_shed_requested id=%s reason=%s", row["incident_id"], fields["shed_reason"])
            await self._emit(row, "refresh", now)

    # --- side effects -------------------------------------------------------------------------
    async def _deduped(self, column: str, row: dict, now: datetime) -> datetime | None:
        """D6: the time an alert/urgent request already went out for this rule+subject within the
        sliding window (from another incident), else None."""
        if self.s.alert_dedupe_window_sec <= 0 or row["open_reason"] == "simulated":
            return None
        last = await self._sync(self.store.last_side_effect_at, column, row["rule"], row["subject"])
        if last is not None and (now - last).total_seconds() < self.s.alert_dedupe_window_sec:
            return last
        return None

    async def _send_alert(self, row: dict, now: datetime) -> None:
        if row.get("alert_sent_at") or (row.get("alert_attempts") or 0) >= self.s.alert_max_attempts:
            return
        if str(row.get("alert_error") or "").startswith("deduped:"):
            return
        prior = await self._deduped("alert_sent_at", row, now)
        if prior is not None:
            fields = {"alert_error": f"deduped:alert_sent_at={prior.isoformat()}"[:300]}
            await self._sync(self.store.update_incident, row["incident_id"], **fields)
            row.update(fields)
            logger.warning("hardware_watch_alert_deduped id=%s rule=%s subject=%s prior=%s", row["incident_id"],
                           row["rule"], row["subject"], prior.isoformat())
            return
        attempts = (row.get("alert_attempts") or 0) + 1
        req = NotificationRequest(
            source_service=self.s.service_name, event_kind="hardware.watch.cooling.alert", severity="critical",
            title=_alert_title(row), body_text=_alert_body(row, self.cooling_cfg, self.ac_cfg if self.v2 else None),
            context={"incident_id": row["incident_id"], "rule": row["rule"], "subject": row["subject"],
                     "open_reason": row["open_reason"]},
            tags=["hardware-watch", "cooling"], channels_requested=["in_app", "email"],
            dedupe_key=f"cooling:{row['incident_id']}:alert", correlation_id=row["incident_id"])
        try:
            result = await self._sync(self.notify, req)
            ok, detail = bool(getattr(result, "ok", False)), getattr(result, "detail", None)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        fields: dict[str, Any] = {"alert_attempts": attempts}
        if ok:
            fields.update(alert_sent_at=now, alert_error=None)
            logger.warning("hardware_watch_alert_sent id=%s attempt=%s", row["incident_id"], attempts)
        else:
            fields["alert_error"] = str(detail)[:300]
            logger.error("hardware_watch_alert_failed id=%s attempt=%s err=%s", row["incident_id"], attempts, detail)
        await self._sync(self.store.update_incident, row["incident_id"], **fields)
        row.update(fields)

    async def _send_recovered(self, row: dict, now: datetime) -> None:
        req = NotificationRequest(
            source_service=self.s.service_name, event_kind="hardware.watch.cooling.resolved", severity="info",
            title=f"AC incident closed ({row['resolve_reason']}) - cabinet_ac",
            body_text=(f"Incident {row['incident_id']} ({row['open_reason']}) opened "
                       f"{row['opened_at'].isoformat()} closed {now.isoformat()} by {row['resolved_by']}. "
                       "The GPU pool resumes background/system work."),
            context={"incident_id": row["incident_id"]}, tags=["hardware-watch", "cooling"],
            channels_requested=["in_app"], dedupe_key=f"cooling:{row['incident_id']}:resolved",
            correlation_id=row["incident_id"])
        try:
            await self._sync(self.notify, req)
        except Exception as exc:  # noqa: BLE001 -- the recovery notice is best-effort
            logger.warning("hardware_watch_recovered_notice_failed id=%s err=%s", row["incident_id"], exc)

    async def _request_urgent(self, row: dict, now: datetime) -> None:
        if row.get("urgent_requested_at") or not self.s.urgent_enabled:
            return
        if str(row.get("urgent_error") or "").startswith(("deduped:", "skipped:")):
            return
        if row["rule"] in HEAT_RULES and row["open_reason"] == "above_p95":
            # D6/C9: a p95 outlier is true ~5 % of the time by construction; it is not worth an agent run.
            await self._sync(self.store.update_incident, row["incident_id"], urgent_error="skipped:p95_outlier")
            row["urgent_error"] = "skipped:p95_outlier"
            return
        prior = await self._deduped("urgent_requested_at", row, now)
        if prior is not None:
            err = f"deduped:urgent_requested_at={prior.isoformat()}"[:300]
            await self._sync(self.store.update_incident, row["incident_id"], urgent_error=err)
            row["urgent_error"] = err
            logger.warning("hardware_watch_urgent_deduped id=%s rule=%s subject=%s prior=%s", row["incident_id"],
                           row["rule"], row["subject"], prior.isoformat())
            return
        try:
            req = CuriosityUrgentRequestV1(
                incident_id=row["incident_id"], question=_question(row, self.cooling_cfg, self.ac_cfg if self.v2 else None),
                trigger="cooling" if row["rule"] == "cooling" else "heat", subject=row["subject"],
                evidence=_bounded({"rule": row["rule"], "open_reason": row["open_reason"],
                                   "opened_at": row["opened_at"].isoformat(), **(row.get("evidence") or {})}),
                requested_at=now, requested_by=self.s.service_name[:64])
            await self.publish(URGENT_REQUEST_CHANNEL, BaseEnvelope(
                kind=URGENT_REQUEST_KIND, source=self.source, correlation_id=uuid.UUID(row["incident_id"]),
                payload=req.model_dump(mode="json")))
        except Exception as exc:  # noqa: BLE001 -- retried next tick
            await self._sync(self.store.update_incident, row["incident_id"], urgent_error=f"{type(exc).__name__}: {exc}"[:300])
            logger.error("hardware_watch_urgent_request_failed id=%s err=%s", row["incident_id"], exc)
            return
        await self._sync(self.store.update_incident, row["incident_id"], urgent_requested_at=now, urgent_error=None)
        row["urgent_requested_at"] = now
        logger.warning("hardware_watch_urgent_requested id=%s rule=%s subject=%s",
                       row["incident_id"], row["rule"], row["subject"])

    async def _emit(self, row: dict, transition: str, now: datetime) -> bool:
        shed = None
        if row["rule"] == "cooling" and not self.v2:
            # The watcher's switch also stops a request latched before it was turned off.
            requested = bool(row.get("shed_requested")) and row["status"] == "open" and self.s.shed_enabled
            sv = self._last_shed.get(row["incident_id"])
            shed = HardwareWatchShedV1(
                requested=requested,
                reason=row.get("shed_reason") if self.s.shed_enabled or not row.get("shed_requested") else "disabled",
                requested_at=row.get("shed_requested_at"),
                cabinet_temp_c=getattr(sv, "temp_c", None), cabinet_rise_c=getattr(sv, "rise_c", None),
                valid_until=now + timedelta(seconds=self.s.shed_valid_sec) if requested else None)
        ev = HardwareWatchIncidentV1(
            incident_id=row["incident_id"], rule=row["rule"], subject=row["subject"], transition=transition,
            status=row["status"], open_reason=row["open_reason"], opened_at=row["opened_at"],
            resolved_at=row.get("resolved_at"), resolve_reason=row.get("resolve_reason"),
            resolved_by=row.get("resolved_by"), shed=shed,
            evidence=_bounded(row.get("evidence") or {}, 15_000), emitted_at=now)
        try:
            await self.publish(HARDWARE_WATCH_INCIDENT_CHANNEL, BaseEnvelope(
                kind=HARDWARE_WATCH_INCIDENT_KIND, source=self.source, correlation_id=uuid.UUID(row["incident_id"]),
                payload=ev.model_dump(mode="json")))
            self._last_refresh[row["incident_id"]] = now
            return True
        except Exception as exc:  # noqa: BLE001 -- the refresh loop re-publishes open incidents
            logger.error("hardware_watch_emit_failed id=%s transition=%s err=%s", row["incident_id"], transition, exc)
            return False

    async def _retry_and_refresh(self, now: datetime, report: TickReport) -> None:
        """Retry a failed alert / urgent request; re-publish every open incident each refresh_sec so
        the pool's shed signal never lapses while the incident is open (and a restarted pool
        re-learns it)."""
        for iid, row in list(self._pending_resolved.items()):
            if await self._emit(row, "resolved", now):
                self._pending_resolved.pop(iid, None)
        for row in await self._sync(self.store.open_incidents):
            if row["rule"] == "cooling" and not row.get("alert_sent_at"):
                await self._send_alert(row, now)        # returns at once for a deduped one
            if not row.get("urgent_requested_at"):
                await self._request_urgent(row, now)    # likewise
            last = self._last_refresh.get(row["incident_id"])
            if last is None or (now - last).total_seconds() >= self.s.refresh_sec:
                await self._emit(row, "refresh", now)


# --- text ---------------------------------------------------------------------------------------

def _compact_cooling(points) -> list[list]:
    return [[p.ts.isoformat(), p.watts, p.stale, p.device_online, p.controller_ready] for p in points][-EVIDENCE_POINTS:]


def _bounded(evidence: dict, limit: int = 30_000) -> dict:
    """Drop the recent-readings list first, then everything but the verdict, to fit ``limit``."""
    from pydantic_core import to_json

    if len(to_json(evidence)) <= limit:
        return evidence
    trimmed = {k: v for k, v in evidence.items() if k != "recent"}
    if len(to_json(trimmed)) <= limit:
        return {**trimmed, "recent": "trimmed"}
    return {"verdict": "trimmed", "keys": sorted(evidence)[:20]}


def _reason_text(row: dict, cfg: CoolingRuleConfig, ac: AcLowConfig | None = None) -> str:
    if ac is not None and row["open_reason"] == "low_power":
        v = (row.get("evidence") or {}).get("verdict") or {}
        cab = v.get("cabinet") or {}
        return REASON_TEXT["low_power_v2"].format(window=ac.window_sec / 60, mean=v.get("ac_mean_w"), low=ac.low_mean_w,
                                                  state=cab.get("effective_state"), temp=cab.get("temp_c"))
    return REASON_TEXT.get(row["open_reason"], row["open_reason"]).format(low=cfg.low_watts)


def _alert_title(row: dict) -> str:
    prefix = "SIMULATED " if row["open_reason"] == "simulated" else ""
    return f"{prefix}AC FAILURE: cabinet AC {row['open_reason']} - investigating"


def _alert_body(row: dict, cfg: CoolingRuleConfig, ac: AcLowConfig | None = None) -> str:
    v = (row.get("evidence") or {}).get("verdict") or {}
    shed = row.get("shed_reason")
    if ac is not None:   # v2: incidents never shed; the reflex follows the cabinet itself
        pool = ("GPU pool: no shed from this alert. The reflex sheds on its own when the cabinet reaches 34 C "
                "(or its sensor goes silent); Orion may hold back background work while it is warm.")
    elif row.get("shed_requested"):
        pool = f"GPU pool: shedding new background/system work ({shed})."
    else:
        pool = f"GPU pool: not shedding ({shed or 'cabinet not warming yet'})."
    lines = [
        f"The hardware watcher opened cooling incident {row['incident_id']} at {row['opened_at'].isoformat()}:",
        f"  {_reason_text(row, cfg, ac)}.",
        f"Last live reading: {v.get('last_live_watts')} W at {v.get('last_live_ts')}; "
        f"no live reading for {v.get('silent_sec')} s.",
        "Check the portable AC now (power, the Shelly plug, the Z-Wave stick).",
        pool,
        "An urgent investigation follows (at most one per 6 h); its report arrives separately.",
        f"Close it by hand: POST /incidents/{row['incident_id']}/resolve on orion-hardware-watch.",
    ]
    return "\n".join(lines)


def _question(row: dict, cfg: CoolingRuleConfig, ac: AcLowConfig | None = None) -> str:
    v = (row.get("evidence") or {}).get("verdict") or {}
    when = row["opened_at"].strftime("%H:%M UTC")
    if row["rule"] == "cooling":
        return (f"The cabinet AC watcher opened an incident at {when}: {_reason_text(row, cfg, ac)}. "
                "Is the cabinet actually losing cooling, or is this a sensor/Z-Wave/plumbing fault? "
                "Find the cause, rate the severity, and name the one thing Juniper should do now.")
    what = "hottest CPU/board sensor" if row["rule"] == "cpu_heat" else "GPU die temperature"
    arm = (f"above its fixed ceiling (its own 7-day p95 is {v.get('p95')} C)" if row["open_reason"] == "above_ceiling"
           else f"above its own 7-day p95 ({v.get('p95')} C) for 10+ minutes")
    return (f"{row['subject']}'s {what} has been {arm} as of {when} (latest {v.get('newest')} C). "
            "Is this real heat or a sensor fault? What load or condition is driving it, how severe is it, "
            "and what one thing should Juniper do?")
