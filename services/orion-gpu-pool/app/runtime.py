"""The pool runtime: the single writer that joins discovery, the scheduler, the lease graph,
the fenced projection and the bus.

Every entrypoint (RPC verb, announcement, tick) runs under one asyncio lock, so decisions are
serial and the scheduler always sees a consistent world. Only the Postgres advisory-lock holder
runs a runtime at all (see app/main.py), so this lock is the whole concurrency story.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.gpu_pool.client import WAIT_HOP_LABEL
from orion.gpu_pool.config import PoolConfig
from orion.gpu_pool.discovery import Probe, resolve_roles
from orion.gpu_pool.lease_graph import FINAL, InvalidTransition, initial_state
from orion.gpu_pool.scheduler import (
    Abort, Backlog, CardLive, DeadLetter, Expire, Grant, LeaseView, Recall, Requeue, RoleLive,
    SwapLoad, SwapUnload, Unavailable, schedule,
)
from orion.schemas.gpu_pool import (
    GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND, GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_KIND,
    DiscoveredRoleV1, GpuCardStateV1, GpuLeaseGrantV1, GpuLeaseReplyV1, GpuLeaseRequestV1,
    GpuLeaseRowV1, GpuPoolControlReplyV1, GpuPoolControlV1, GpuPoolEventV1, GpuPoolStateV1,
    LlmWorkerAnnounceV1,
)

logger = logging.getLogger("orion-gpu-pool.runtime")

Prober = Callable[[str, str, str, str], Awaitable[Probe]]

# scheduler decision -> lease-graph event type
_EVENT_FOR = {Grant: "grant", Recall: "recall", Abort: "abort", Expire: "expire",
              Unavailable: "unavailable", Backlog: "backlog", Requeue: "requeue", DeadLetter: "dead_letter"}
# lease-graph status reached -> public event name
_PUBLIC = {"granted": "granted", "recalling": "recalled", "backlogged": "backlogged",
           "unavailable": "unavailable", "dead_letter": "dead_lettered", "released": "released",
           "retry_wait": "retried", "queued": "queued"}
_GRAMMAR_EVENTS = {"granted", "recalled", "aborted", "expired", "backlogged", "unavailable",
                   "dead_lettered", "swap_requested", "discovery_mismatch"}


def _ts(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PoolRuntime:
    def __init__(self, *, cfg: PoolConfig, profiles: dict[str, Any], store: Any, graph: Any,
                 bus: Any = None, prober: Prober | None = None, now: Callable[[], datetime] = _utcnow,
                 mode: str = "observe", operator_token: str = "", service_name: str = "orion-gpu-pool",
                 announce_stale_sec: float = 120.0, probe_interval_sec: float = 15.0,
                 state_publish_sec: float = 5.0, replay_payload_max_bytes: int = 262144):
        import asyncio

        self.cfg, self.profiles, self.store, self.graph, self.bus = cfg, profiles, store, graph, bus
        self.prober, self.now, self.mode = prober, now, mode
        self.operator_token, self.service_name = operator_token, service_name
        self.announce_stale_sec, self.probe_interval_sec = announce_stale_sec, probe_interval_sec
        self.state_publish_sec, self.replay_payload_max_bytes = state_publish_sec, replay_payload_max_bytes
        self.lock = asyncio.Lock()
        self.cards: dict[str, CardLive] = {}
        self.announcements: dict[str, LlmWorkerAnnounceV1] = {}
        self.probes: dict[str, Probe] = {}
        self.discovered: list[DiscoveredRoleV1] = []
        self.roles: dict[str, RoleLive] = {}
        self.unclaimed: list[str] = []
        self._last_probe: datetime | None = None
        self._last_state: datetime | None = None
        self._swap_requested: set[tuple[str, str]] = set()

    # --- lifecycle --------------------------------------------------------------------
    async def start(self) -> None:
        stored = {r["card"]: r for r in await self.store.cards()}
        for card in self.cfg.cards:
            row = stored.get(card)
            if row is None:
                await self.store.upsert_card({"card": card, "lent": False, "swapped_in": [],
                                              "swap_state": "idle", "updated_at": self.now(),
                                              "updated_by": "boot"})
                self.cards[card] = CardLive(card)
            else:
                self.cards[card] = CardLive(card, bool(row["lent"]), set(row["swapped_in"] or []),
                                            row["swap_state"], _ts(row.get("cooldown_until")),
                                            _ts(row.get("last_active_at")))
        self._resolve()

    def _thread(self, lease_id: str) -> dict:
        return {"configurable": {"thread_id": f"gpu_pool:{lease_id}"}}

    # --- discovery --------------------------------------------------------------------
    async def on_announce(self, ann: LlmWorkerAnnounceV1) -> None:
        async with self.lock:
            self.announcements[ann.role] = ann

    async def _probe_all(self) -> None:
        if self.prober is None:
            return
        for role, spec in self.cfg.roles.items():
            url = self.cfg.url(role)
            try:
                self.probes[role] = await self.prober(role, url, spec.kind, spec.health)
            except Exception as exc:  # noqa: BLE001
                self.probes[role] = Probe(False, error=f"{type(exc).__name__}: {exc}", checked_at=self.now())
        self._last_probe = self.now()

    def _resolve(self) -> None:
        before = {d.role: d.status for d in self.discovered}
        self.discovered, self.roles, self.unclaimed = resolve_roles(
            self.cfg, self.profiles, self.announcements, self.probes, self.cards, self.now(),
            self.announce_stale_sec)
        self._discovery_changes = [
            d for d in self.discovered
            if d.status in ("confirmed", "mismatch") and before.get(d.role) != d.status
        ]

    # --- RPC verbs --------------------------------------------------------------------
    async def acquire(self, req: GpuLeaseRequestV1, *, operator: bool = False) -> GpuLeaseReplyV1:
        async with self.lock:
            if not req.work_class or req.work_class not in self.cfg.classes:
                return GpuLeaseReplyV1(status="unavailable", reason=f"unknown_class:{req.work_class}")
            roles = self.cfg.classes[req.work_class].roles
            if any(self.cfg.roles[r].operator_only for r in roles) and not operator:
                return GpuLeaseReplyV1(status="unavailable", reason="operator_only_class")
            request_id = req.request_id or uuid.uuid4().hex
            existing = await self.store.lease_by_request(request_id)
            if existing is not None:
                return await self._reply_for(existing)
            if req.replay_payload is not None and \
                    len(json.dumps(req.replay_payload)) > self.replay_payload_max_bytes:
                return GpuLeaseReplyV1(status="unavailable", reason="replay_payload_too_large")
            lease_id = uuid.uuid4().hex
            now = self.now()
            request = {**req.model_dump(mode="json", exclude={"verb", "lease_id", "outcome", "detail"}),
                       "request_id": request_id, "holder": req.holder or "unknown", "operator": operator}
            await self._start_thread(lease_id, request, now)
            await self._schedule_and_apply()
            return await self._reply_for(await self.store.lease(lease_id))

    async def heartbeat(self, lease_id: str) -> GpuLeaseReplyV1:
        async with self.lock:
            row = await self.store.lease(lease_id)
            if row is None:
                return GpuLeaseReplyV1(status="unknown_lease", lease_id=lease_id)
            if row["status"] in ("granted", "recalling"):
                row = await self._resume(lease_id, {"type": "heartbeat"}) or row
            return await self._reply_for(row)

    async def release(self, lease_id: str, outcome: str = "ok", detail: str | None = None) -> GpuLeaseReplyV1:
        async with self.lock:
            row = await self.store.lease(lease_id)
            if row is None:
                return GpuLeaseReplyV1(status="unknown_lease", lease_id=lease_id)
            kind = "release_ok" if outcome == "ok" else "cancel" if outcome == "cancelled" else "release_failed"
            if row["status"] not in ("granted", "recalling") and kind != "cancel":
                return await self._reply_for(row)
            row = await self._resume(lease_id, {"type": kind, "reason": detail or outcome}) or row
            await self._schedule_and_apply()
            return await self._reply_for(row)

    async def cancel(self, lease_id: str) -> GpuLeaseReplyV1:
        return await self.release(lease_id, "cancelled", "cancelled")

    async def control(self, ctl: GpuPoolControlV1) -> GpuPoolControlReplyV1:
        if not self.operator_token or ctl.operator_token != self.operator_token:
            return GpuPoolControlReplyV1(ok=False, reason="operator_token_rejected")
        if ctl.verb in ("lend", "unlend"):
            async with self.lock:
                card = self.cards.get(ctl.card or "")
                if card is None or not self.cfg.cards[card.card].lendable:
                    return GpuPoolControlReplyV1(ok=False, reason="card_not_lendable")
                card.lent = ctl.verb == "lend"
                await self.store.upsert_card({"card": card.card, "lent": card.lent,
                                              "updated_at": self.now(), "updated_by": ctl.actor})
                await self._emit(GpuPoolEventV1(event="lent" if card.lent else "unlent",
                                                cards=[card.card], holder=ctl.actor))
                await self._schedule_and_apply()
            return GpuPoolControlReplyV1(ok=True, detail={"card": card.card, "lent": card.lent})
        if ctl.verb in ("replay", "cancel"):
            async with self.lock:
                row = await self.store.lease(ctl.lease_id or "")
                if row is None:
                    return GpuPoolControlReplyV1(ok=False, reason="unknown_lease")
                event = {"type": "replay" if ctl.verb == "replay" else "cancel", "reason": f"operator:{ctl.actor}"}
                new = await self._resume(row["lease_id"], event)
                if new is None:
                    return GpuPoolControlReplyV1(ok=False, reason=f"not_{ctl.verb}able_from_{row['status']}")
                if ctl.verb == "replay":
                    await self._emit_row("replayed", new)
                await self._schedule_and_apply()
            return GpuPoolControlReplyV1(ok=True, detail={"lease_id": row["lease_id"], "status": new["status"]})
        if ctl.verb == "backfill":
            return await self._backfill(ctl)
        return GpuPoolControlReplyV1(ok=False, reason="unknown_verb")

    async def _backfill(self, ctl: GpuPoolControlV1) -> GpuPoolControlReplyV1:
        """Replay a filtered set of past leases as linked child leases. ``preview`` counts only."""
        spec = ctl.backfill or {}
        limit = min(int(spec.get("limit", 100)), 1000)
        async with self.lock:
            rows = await self.store.find_leases(
                work_class=spec.get("work_class"), holder=spec.get("holder"), status=spec.get("status"),
                since=_ts(spec.get("since")), until=_ts(spec.get("until")), limit=limit)
            if spec.get("preview", True):
                return GpuPoolControlReplyV1(ok=True, detail={"would_replay": len(rows)})
            children = []
            for parent in rows:
                snap = await self.graph.aget_state(self._thread(parent["lease_id"]))
                request = dict((snap.values or {}).get("request") or {})
                if not request:
                    continue
                request.update(request_id=uuid.uuid4().hex, parent_lease_id=parent["lease_id"],
                               deadline_at=None)
                child = uuid.uuid4().hex
                await self._start_thread(child, request, self.now(), parent_lease_id=parent["lease_id"])
                children.append(child)
            await self._schedule_and_apply()
        return GpuPoolControlReplyV1(ok=True, detail={"replayed": len(children), "children": children})

    # --- the tick ---------------------------------------------------------------------
    async def tick(self) -> None:
        async with self.lock:
            now = self.now()
            if self._last_probe is None or (now - self._last_probe).total_seconds() >= self.probe_interval_sec:
                await self._probe_all()
            self._resolve()
            for d in self._discovery_changes:
                await self._emit(GpuPoolEventV1(
                    event="discovery_confirmed" if d.status == "confirmed" else "discovery_mismatch",
                    role=d.role, cards=d.cards, reason=d.detail,
                    detail={"profile_name": d.profile_name, "model_file": d.model_file}))
            await self._schedule_and_apply()
            if self._last_state is None or (now - self._last_state).total_seconds() >= self.state_publish_sec:
                await self.publish_state()

    async def _schedule_and_apply(self) -> None:
        rows = await self.store.live_leases()
        decisions = schedule(self.cfg, self.roles, self.cards, [self._view(r) for r in rows], self.now())
        for d in decisions:
            if isinstance(d, (SwapLoad, SwapUnload)):
                await self._swap(d)
                continue
            event: dict[str, Any] = {"type": _EVENT_FOR[type(d)], "reason": getattr(d, "reason", None)}
            if isinstance(d, Grant):
                event["role"] = d.role
            if isinstance(d, Recall):
                event["recall_by"] = d.recall_by.isoformat()
            await self._resume(d.lease_id, event)
        await self._touch_swap_seats(rows)

    async def _swap(self, d: SwapLoad | SwapUnload) -> None:
        """Stage 1 is observe mode: swap decisions are published, never actuated."""
        key = (type(d).__name__, d.role)
        if key in self._swap_requested:
            return
        self._swap_requested.add(key)
        await self._emit(GpuPoolEventV1(
            event="swap_requested", role=d.role, cards=list(self.cfg.roles[d.role].cards), reason=d.reason,
            detail={"action": "load" if isinstance(d, SwapLoad) else "unload", "actuated": False,
                    "mode": self.mode}))

    async def _touch_swap_seats(self, rows: list[dict]) -> None:
        now = self.now()
        for row in rows:
            role = row.get("role")
            if row["status"] in ("granted", "recalling") and role and self.cfg.roles[role].swap:
                for card in self.cfg.roles[role].cards:
                    self.cards[card].last_active_at = now

    # --- lease-graph plumbing ---------------------------------------------------------
    async def _start_thread(self, lease_id: str, request: dict, now: datetime,
                            parent_lease_id: str | None = None) -> None:
        state = initial_state(lease_id, request, now)
        await self.graph.ainvoke(state, self._thread(lease_id))
        row = self._row(state, request, parent_lease_id)
        await self.store.upsert_lease(row)
        await self._emit_row("admitted", row)

    async def _resume(self, lease_id: str, event: dict[str, Any]) -> dict | None:
        from langgraph.types import Command

        cfg = self._thread(lease_id)
        snap = await self.graph.aget_state(cfg)
        if not snap.values or not snap.next:
            return None
        event = {**event, "at": self.now().isoformat()}
        try:
            values = await self.graph.ainvoke(Command(resume=event), cfg)
        except InvalidTransition as exc:
            logger.info("gpu_pool_transition_rejected lease=%s %s", lease_id, exc)
            return None
        except Exception as exc:  # noqa: BLE001 -- LangGraph wraps node errors
            if isinstance(exc.__cause__, InvalidTransition) or "-/->" in str(exc):
                logger.info("gpu_pool_transition_rejected lease=%s %s", lease_id, exc)
                return None
            raise
        prior = await self.store.lease(lease_id)
        row = self._row(values, values["request"], (prior or {}).get("parent_lease_id"))
        await self.store.upsert_lease(row)
        public = _PUBLIC.get(row["status"])
        if event["type"] == "abort":
            public = "aborted"
        elif event["type"] == "expire":
            public = "expired"
        elif event["type"] == "cancel":
            public = "cancelled"
        elif event["type"] == "heartbeat":
            public = None
        if public:
            await self._emit_row(public, row, prior=prior, reason=event.get("reason"))
        return row

    def _row(self, values: dict, request: dict, parent_lease_id: str | None) -> dict[str, Any]:
        return {
            "lease_id": values["lease_id"], "request_id": request["request_id"],
            "holder": request.get("holder") or "unknown", "work_class": request["work_class"],
            "priority": request.get("priority", "system"), "kind": request.get("kind", "request"),
            "status": values["status"], "role": values.get("role"), "attempt": values.get("attempt", 1),
            "generation": values.get("generation", 0), "operator": bool(request.get("operator")),
            "min_ctx_tokens": int(request.get("min_ctx_tokens") or 0),
            "needs_vision": bool(request.get("needs_vision")),
            "created_at": _ts(values["created_at"]), "queued_since": _ts(values.get("queued_since")),
            "granted_at": _ts(values.get("granted_at")), "recall_by": _ts(values.get("recall_by")),
            "not_before": _ts(values.get("not_before")), "deadline_at": _ts(request.get("deadline_at")),
            "expires_at": _ts(values.get("expires_at")),
            "turn_correlation_id": request.get("turn_correlation_id"),
            "parent_lease_id": parent_lease_id or request.get("parent_lease_id"),
            "reason": values.get("reason"), "updated_at": self.now(),
        }

    @staticmethod
    def _view(row: dict) -> LeaseView:
        return LeaseView(
            lease_id=row["lease_id"], work_class=row["work_class"], priority=row["priority"],
            status=row["status"], created_at=row["created_at"], role=row.get("role"),
            min_ctx_tokens=row.get("min_ctx_tokens") or 0, needs_vision=bool(row.get("needs_vision")),
            deadline_at=row.get("deadline_at"), recall_by=row.get("recall_by"),
            not_before=row.get("not_before"), queued_since=row.get("queued_since"),
            granted_at=row.get("granted_at"), expires_at=row.get("expires_at"),
            operator=bool(row.get("operator")),
        )

    # --- replies ------------------------------------------------------------------------
    def grant_for(self, role: str, lease_id: str, generation: int) -> GpuLeaseGrantV1:
        disc = next((d for d in self.discovered if d.role == role), None)
        return GpuLeaseGrantV1(
            lease_id=lease_id, generation=max(1, generation), role=role,
            cards=list(self.cfg.roles[role].cards), url=self.cfg.url(role),
            profile_name=disc.profile_name if disc else None, model_file=disc.model_file if disc else None,
            ctx_per_slot=disc.ctx_per_slot if disc else None, served_by=f"{self.cfg.host.name}-{role}")

    async def _reply_for(self, row: dict) -> GpuLeaseReplyV1:
        status = row["status"]
        if status == "granted":
            return GpuLeaseReplyV1(status="granted", lease_id=row["lease_id"],
                                   grant=self.grant_for(row["role"], row["lease_id"], row["generation"]))
        if status == "recalling":
            return GpuLeaseReplyV1(status="recall", lease_id=row["lease_id"], recall_by=row["recall_by"],
                                   grant=self.grant_for(row["role"], row["lease_id"], row["generation"]))
        if status in ("queued", "retry_wait"):
            return GpuLeaseReplyV1(status="queued", lease_id=row["lease_id"], position=await self._position(row))
        if status == "backlogged":
            return GpuLeaseReplyV1(status="backlogged", lease_id=row["lease_id"], reason=row.get("reason"))
        if status == "released":
            return GpuLeaseReplyV1(status="ok", lease_id=row["lease_id"], reason=row.get("reason"))
        return GpuLeaseReplyV1(status="unavailable", lease_id=row["lease_id"], reason=row.get("reason") or status)

    async def _position(self, row: dict) -> int:
        rank = self.cfg.priority_rank
        mine = (rank(row["priority"]), row["created_at"])
        return 1 + sum(1 for r in await self.store.live_leases()
                       if r["status"] == "queued" and r["work_class"] == row["work_class"]
                       and (rank(r["priority"]), r["created_at"]) < mine)

    # --- state + events ---------------------------------------------------------------
    async def snapshot(self, include_leases: bool = True) -> GpuPoolStateV1:
        rows = await self.store.live_leases()
        queue: dict[str, int] = {}
        backlog: dict[str, int] = {}
        for r in rows:
            if r["status"] == "queued":
                queue[r["work_class"]] = queue.get(r["work_class"], 0) + 1
            elif r["status"] == "backlogged":
                backlog[r["work_class"]] = backlog.get(r["work_class"], 0) + 1
        return GpuPoolStateV1(
            mode="enforce" if self.mode == "enforce" else "observe", config_digest=self.cfg.digest,
            cards=[GpuCardStateV1(card=c.card, vram_gb=self.cfg.cards[c.card].vram_gb,
                                  lendable=self.cfg.cards[c.card].lendable, lent=c.lent,
                                  swapped_in=sorted(c.swapped_in), swap_state=c.swap_state,
                                  cooldown_until=c.cooldown_until) for c in self.cards.values()],
            roles=self.discovered, unclaimed_servers=self.unclaimed,
            leases=[GpuLeaseRowV1(
                lease_id=r["lease_id"], request_id=r["request_id"], holder=r["holder"],
                work_class=r["work_class"], priority=r["priority"], kind=r["kind"], status=r["status"],
                role=r.get("role"), attempt=r.get("attempt", 1), created_at=r["created_at"],
                granted_at=r.get("granted_at"), recall_by=r.get("recall_by"),
                turn_correlation_id=r.get("turn_correlation_id")) for r in rows] if include_leases else [],
            queue_depth=queue, backlog_depth=backlog)

    async def publish_state(self) -> None:
        self._last_state = self.now()
        if self.bus is None:
            return
        state = await self.snapshot()
        await self._publish(GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_KIND, state.model_dump(mode="json"), None)

    async def history(self, lease_id: str) -> list[dict[str, Any]]:
        """The walker's path: the lease's own history as its last checkpoint recorded it."""
        snap = await self.graph.aget_state(self._thread(lease_id))
        return list((snap.values or {}).get("history") or [])

    async def _emit_row(self, event: str, row: dict, *, prior: dict | None = None, reason: str | None = None) -> None:
        waited_ms = held_ms = None
        now = self.now()
        if event == "granted" and row.get("queued_since"):
            waited_ms = (now - row["queued_since"]).total_seconds() * 1000
        if event in ("released", "aborted", "expired", "retried") and prior and prior.get("granted_at"):
            held_ms = (now - prior["granted_at"]).total_seconds() * 1000
        detail: dict[str, Any] = {}
        if event == "granted" and row.get("role"):
            detail["grant"] = self.grant_for(row["role"], row["lease_id"], row["generation"]).model_dump(mode="json")
        if event == "recalled":
            detail["recall_by"] = row["recall_by"].isoformat() if row.get("recall_by") else None
        await self._emit(GpuPoolEventV1(
            event=event, lease_id=row["lease_id"], holder=row["holder"], work_class=row["work_class"],
            priority=row["priority"], role=row.get("role") or (prior or {}).get("role"),
            cards=list(self.cfg.roles[row["role"]].cards) if row.get("role") else [],
            turn_correlation_id=row.get("turn_correlation_id"), attempt=row.get("attempt"),
            waited_ms=waited_ms, held_ms=held_ms, reason=reason or row.get("reason"), detail=detail))
        if self.bus is not None:
            hop = f"gpu_pool:{row['work_class']}#{WAIT_HOP_LABEL}"
            if event == "granted" and waited_ms is not None:
                self.bus.record_hop_success(hop, waited_ms)
            elif event == "unavailable" and (reason or row.get("reason")) == "deadline":
                self.bus.record_hop_timeout(hop, None)

    async def _emit(self, event: GpuPoolEventV1) -> None:
        if self.bus is None:
            return
        await self._publish(GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND, event.model_dump(mode="json"),
                            event.turn_correlation_id)
        if event.event in _GRAMMAR_EVENTS:
            await self._grammar(event)

    async def _publish(self, channel: str, kind: str, payload: dict, corr: str | None) -> None:
        try:
            cid = uuid.UUID(str(corr)) if corr else uuid.uuid4()
        except ValueError:
            cid = uuid.uuid4()
        try:
            await self.bus.publish(channel, BaseEnvelope(
                kind=kind, source=ServiceRef(name=self.service_name), correlation_id=cid, payload=payload))
        except Exception:  # noqa: BLE001 -- telemetry must never break scheduling
            logger.warning("gpu_pool_publish_failed channel=%s", channel, exc_info=True)

    async def _grammar(self, event: GpuPoolEventV1) -> None:
        """Lease facts land in grammar_events via sql-writer. Layer "capacity", never "transport":
        waiting in line is not transport. No reducer reads these until the metric gate (stage 6)."""
        try:
            from orion.grammar.publish import publish_grammar_event
            from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1

            trace_id = f"gpu_pool.lease:{event.lease_id or event.role or 'pool'}"
            event_id = f"{trace_id}:{event.event_id[:12]}"
            dims = ["capacity", "gpu", event.work_class or "pool"]
            atom = GrammarAtomV1(
                atom_id=event_id, trace_id=trace_id, atom_type="observation",
                semantic_role=f"gpu_lease_{event.event}", layer="capacity", dimensions=dims,
                summary=(f"{event.event} class={event.work_class} role={event.role} "
                         f"waited_ms={event.waited_ms} reason={event.reason}"),
                text_value=event.role or event.work_class or "pool", confidence=1.0, salience=0.5,
                source_event_id=event.event_id)
            await publish_grammar_event(self.bus, GrammarEventV1(
                event_id=event_id, event_kind="atom_emitted", trace_id=trace_id,
                correlation_id=event.turn_correlation_id, emitted_at=event.generated_at, layer="capacity",
                dimensions=dims, atom=atom,
                provenance=GrammarProvenanceV1(source_service=self.service_name, source_component="lease_graph",
                                               source_event_id=event.event_id, source_trace_id=trace_id)),
                source_name=self.service_name)
        except Exception:  # noqa: BLE001
            logger.warning("gpu_pool_grammar_failed event=%s", event.event, exc_info=True)
