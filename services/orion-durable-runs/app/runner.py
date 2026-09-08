"""DurableRunner -- owns the checkpointer, executes and resumes runs, and
publishes one `DurableRunStateV1` per node transition (plus a row on the
attention surface, so the sequencing is observable on the same table as the
processes it belongs to).

Resume rule, stated once: a thread is unfinished when the compiled graph's
state snapshot still has a `next` node. That is LangGraph's own signal, not a
status we keep separately, so a crash between two nodes and a node that
raised look the same to the sweep -- both are "re-invoke with `None` input on
this thread", which continues from the next node with every earlier node's
result intact. A thread older than `max_age_hours` is abandoned instead
(one `abandoned` state event), never resumed into a different day's material.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.curiosity.attention_schema import (
    read_attended_priors,
    to_attention_schema as curiosity_to_attention_schema,
)
from orion.curiosity.self_inquiry import read_self_definition, self_definition_to_detail
from orion.curiosity.worldview import (
    TurnOutcome,
    WorldviewReader,
    read_finding_connectivity,
    read_hop_notes,
    read_run_footprint,
    read_turn_outcome,
)
from orion.journaler.schemas import JournalEntryWriteV1
from orion.schemas.attention_schema import (
    ATTENTION_SCHEMA_CHANNEL,
    ATTENTION_SCHEMA_KIND,
    AttentionSchemaV1,
    bind_correlation,
)
from orion.schemas.durable_run import (
    CURIOSITY_NODES,
    CURIOSITY_TURN_REPLY_PREFIX,
    CURIOSITY_TURN_REQUEST_CHANNEL,
    CURIOSITY_TURN_REQUEST_KIND,
    DURABLE_RUN_STATE_KIND,
    CuriosityTurnRequestV1,
    CuriosityTurnResultV1,
    DurableRunRequestV1,
    DurableRunStateV1,
)

from app.graph import CuriosityRunState, Deps, build_curiosity_graph, finish_detail
from app.settings import Settings

logger = logging.getLogger("orion-durable-runs.runner")

JOURNAL_WRITE_CHANNEL = "orion:journal:write"


def _corr_uuid(raw: str) -> UUID:
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return uuid4()


class DurableRunner:
    def __init__(self, settings: Settings, *, bus: OrionBusAsync | None, checkpointer: Any) -> None:
        self._settings = settings
        self._bus = bus
        self._checkpointer = checkpointer
        self._reader: WorldviewReader | None = (
            WorldviewReader(host=settings.graph_host, port=settings.graph_port, graph_name=settings.graph_own)
            if settings.graph_host
            else None
        )
        self._graph = build_curiosity_graph(self._deps(), checkpointer)
        self._active: dict[str, asyncio.Task[None]] = {}
        # thread_id -> node the resume picked up at; consumed by the first
        # state event after a resume so `resumed_from_node` is stamped once.
        self._resumed_from: dict[str, str] = {}

    # --- deps: the real world behind each node ------------------------------

    def _deps(self) -> Deps:
        return Deps(
            run_turn=self._run_turn,
            read_turn_result=self._read_turn_result,
            publish_attention_row=self._publish_attention_row,
            publish_journal=self._publish_journal,
        )

    def _source(self) -> ServiceRef:
        s = self._settings
        return ServiceRef(name=s.service_name, version=s.service_version, node=s.node_name)

    async def _run_turn(self, request: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        """RPC to Hub. Any transport failure surfaces as ok=False so the node
        raises and the thread stays resumable at `harness_turn`."""
        if self._bus is None:
            return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=False, error="no_bus")
        reply_channel = f"{CURIOSITY_TURN_REPLY_PREFIX}:{request.correlation_id}"
        envelope = BaseEnvelope(
            kind=CURIOSITY_TURN_REQUEST_KIND,
            source=self._source(),
            correlation_id=_corr_uuid(request.correlation_id),
            reply_to=reply_channel,
            payload=request.model_dump(mode="json"),
        )
        try:
            raw = await self._bus.rpc_request(
                CURIOSITY_TURN_REQUEST_CHANNEL,
                envelope,
                reply_channel=reply_channel,
                timeout_sec=self._settings.turn_rpc_timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced as a failed node, retried by the sweep
            logger.warning("durable_run_turn_rpc_failed run=%s attempt=%s err=%s", request.run_id, request.attempt, exc)
            return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=False, error=f"rpc:{type(exc).__name__}")
        # rpc_request hands back the raw pubsub message; the envelope is in
        # `data` and must be decoded (live finding 2026-09-06: reading a
        # `payload` key off the raw message reads nothing).
        try:
            decoded = self._bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
            payload = decoded.envelope.payload if decoded.ok else None
            return CuriosityTurnResultV1.model_validate(payload or {})
        except Exception as exc:  # noqa: BLE001
            return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=False, error=f"bad_reply:{exc}")

    async def _read_turn_result(self, run_id: str) -> dict[str, Any]:
        reader = self._reader
        if reader is None:
            return {"outcome": None, "footprint": None, "hops": [], "evidence_summary": None, "graph_readable": False, "self_definition": None}

        def _read() -> dict[str, Any]:
            outcome = read_turn_outcome(reader, run_id)
            footprint = read_run_footprint(reader, run_id)
            hops = read_hop_notes(reader, run_id)
            evidence = read_finding_connectivity(reader, run_id)
            # Read for every run, not only self_inquiry: an investigation run
            # that also wrote a definition is still a definition Orion wrote.
            # Hub decides whether to mirror it, keyed on the run's line.
            self_definition = read_self_definition(reader, run_id)
            return {
                "outcome": (
                    {
                        "run_id": outcome.run_id,
                        "continue_line": outcome.continue_line,
                        "continue_note": outcome.continue_note,
                        "reach_out": outcome.reach_out,
                        "reach_out_why": outcome.reach_out_why,
                    }
                    if outcome is not None
                    else None
                ),
                "footprint": footprint,
                "hops": [[n, note] for n, note in hops],
                "evidence_summary": evidence.summary() if evidence is not None else None,
                "graph_readable": footprint is not None,
                "self_definition": self_definition_to_detail(self_definition),
            }

        try:
            return await asyncio.wait_for(asyncio.to_thread(_read), timeout=30.0)
        except Exception as exc:  # noqa: BLE001 -- unreadable graph is a state, not a crash
            logger.warning("durable_run_graph_read_failed run=%s err=%s", run_id, exc)
            return {"outcome": None, "footprint": None, "hops": [], "evidence_summary": None, "graph_readable": False, "self_definition": None}

    async def _publish_attention_row(self, facts: dict[str, Any]) -> bool:
        """The same curiosity-lane row Hub used to publish, from the same
        adapter (orion/curiosity/attention_schema.py)."""
        if self._bus is None:
            return False
        run_id = str(facts["run_id"])
        outcome_raw = facts.get("outcome")
        outcome = TurnOutcome(**outcome_raw) if isinstance(outcome_raw, dict) else None
        priors = None
        if self._reader is not None:
            try:
                priors = await asyncio.wait_for(asyncio.to_thread(read_attended_priors, self._reader, run_id), timeout=6.0)
            except Exception:  # noqa: BLE001
                priors = None
        row, corr = bind_correlation(
            curiosity_to_attention_schema(
                run_id=run_id,
                outcome=outcome,
                priors=priors,
                correlation_id=str(facts["correlation_id"]),
                generated_at=datetime.now(timezone.utc),
            )
        )
        return await self._publish(ATTENTION_SCHEMA_CHANNEL, ATTENTION_SCHEMA_KIND, row, corr)

    async def _publish_journal(self, entry: JournalEntryWriteV1) -> str | None:
        if self._bus is None:
            return None
        ok = await self._publish(JOURNAL_WRITE_CHANNEL, "journal.entry.write.v1", entry, _corr_uuid(entry.correlation_id or ""))
        return entry.entry_id if ok else None

    async def _publish(self, channel: str, kind: str, model: Any, corr: UUID) -> bool:
        try:
            await publish_with_reconnect(
                self._bus,
                channel,
                BaseEnvelope(kind=kind, source=self._source(), correlation_id=corr, payload=model.model_dump(mode="json")),
                log_label="durable_runs_publish",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("durable_run_publish_failed channel=%s err=%s", channel, exc)
            return False

    # --- state events -------------------------------------------------------

    async def _emit_state(
        self,
        state: CuriosityRunState,
        *,
        node: str,
        status: str,
        detail: dict[str, Any] | None = None,
        resumed_from: str | None = None,
    ) -> None:
        run_id = state["run_id"]
        idx = CURIOSITY_NODES.index(node) if node in CURIOSITY_NODES else -1
        next_node = CURIOSITY_NODES[idx + 1] if 0 <= idx < len(CURIOSITY_NODES) - 1 else None
        if status in ("completed", "failed", "abandoned"):
            next_node = None if status != "failed" else node
        event = DurableRunStateV1(
            run_id=run_id,
            workflow="curiosity.investigate",
            thread_id=run_id,
            node=node,
            next_node=next_node,
            status=status,  # type: ignore[arg-type]
            # The immediate `resumed` receipt passes resumed_from explicitly and
            # leaves the marker in place, so the first node that completes
            # after the resume carries it too; that completion pops it.
            resumed_from_node=resumed_from if resumed_from is not None else self._resumed_from.pop(run_id, None),
            correlation_id=state["correlation_id"],
            detail=detail or {},
        )
        if self._bus is not None:
            await self._publish(self._settings.state_channel, DURABLE_RUN_STATE_KIND, event, _corr_uuid(event.correlation_id))
        # The surface sees the sequencing: one row per transition, this lane's
        # own vocabulary is the node name.
        # `generated_at` (microsecond precision, one per call) keeps this
        # unique across repeats of the same node+status -- a run that fails
        # and resumes at `harness_turn` several times hits `resumed`/`failed`
        # at that node every time. Without it every retry after the first
        # collided on the same entry_id and sql-writer's PK dedup silently
        # dropped the row -- confirmed live 2026-09-07: a run with 8
        # `harness_turn` attempts left only one `resumed` surface row.
        ts_suffix = event.generated_at.strftime("%Y%m%dT%H%M%S%f")
        row, corr = bind_correlation(
            AttentionSchemaV1(
                entry_id=f"durable-{run_id}-{node}-{status}-{ts_suffix}",
                process="durable_run",
                correlation_id=event.correlation_id,
                attended_id=run_id,
                attended_label="curiosity.investigate",
                attention_reason=f"{node}:{status}",
                reason_narrative=f"durable run {run_id} {status} at {node}" + (f", next {next_node}" if next_node else ""),
                narrative_kind="computed",
                predicted_next=next_node,
            )
        )
        if self._bus is not None:
            await self._publish(ATTENTION_SCHEMA_CHANNEL, ATTENTION_SCHEMA_KIND, row, corr)
        logger.info("durable_run_state run=%s node=%s status=%s next=%s resumed_from=%s", run_id, node, status, next_node, event.resumed_from_node)

    # --- execution ----------------------------------------------------------

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def start_run(self, request: DurableRunRequestV1) -> None:
        """New run: seed the thread and drive it. Idempotent on run_id -- a
        duplicate request for a thread that already exists is a resume."""
        run_id = request.run_id
        if run_id in self._active:
            logger.info("durable_run_already_active run=%s", run_id)
            return
        snapshot = await self._graph.aget_state(self._config(run_id))
        if snapshot and snapshot.values:
            logger.info("durable_run_request_for_existing_thread run=%s -> resume", run_id)
            self._spawn(run_id, None, resumed_from=snapshot.next[0] if snapshot.next else None)
            return
        initial: CuriosityRunState = {
            "run_id": run_id,
            "correlation_id": request.correlation_id,
            "brief": request.brief.model_dump(mode="json"),
            "attempt": 0,
        }
        self._spawn(run_id, initial, resumed_from=None)

    def _spawn(self, run_id: str, initial: CuriosityRunState | None, *, resumed_from: str | None) -> None:
        if resumed_from:
            self._resumed_from[run_id] = resumed_from
        task = asyncio.create_task(self._drive(run_id, initial), name=f"durable-run-{run_id}")
        self._active[run_id] = task
        task.add_done_callback(lambda _t: self._active.pop(run_id, None))

    async def _drive(self, run_id: str, initial: CuriosityRunState | None) -> None:
        config = self._config(run_id)
        last_state: CuriosityRunState = initial or {}
        try:
            # astream with stream_mode="updates" yields one item per completed
            # node: {node_name: {returned keys}}. The checkpoint for that node
            # is written by the compiled graph before the next node starts.
            async for update in self._graph.astream(initial, config, stream_mode="updates"):
                for node, delta in (update or {}).items():
                    snap = await self._graph.aget_state(config)
                    last_state = dict(snap.values) if snap and snap.values else last_state
                    if node == CURIOSITY_NODES[-1]:
                        await self._emit_state(last_state, node=node, status="completed", detail=finish_detail(last_state))
                    else:
                        status = "resumed" if run_id in self._resumed_from else "running"
                        await self._emit_state(last_state, node=node, status=status)
        except Exception as exc:  # noqa: BLE001 -- the thread stays resumable at its next node
            snap = None
            try:
                snap = await self._graph.aget_state(config)
            except Exception:  # noqa: BLE001
                pass
            node = (snap.next[0] if snap and snap.next else CURIOSITY_NODES[0])
            state = dict(snap.values) if snap and snap.values else (initial or {"run_id": run_id, "correlation_id": ""})
            if node == CURIOSITY_NODES[0]:
                # A failed harness turn cannot record its own attempt (the node
                # raised before returning), so stamp it on the thread as if
                # START had written it: `next` stays harness_turn, the sweep
                # re-issues the turn as attempt+1. Verified against the saver.
                try:
                    from langgraph.graph import START

                    attempt = int(state.get("attempt") or 0) + 1
                    await self._graph.aupdate_state(config, {"attempt": attempt}, as_node=START)
                    state["attempt"] = attempt
                except Exception:  # noqa: BLE001
                    logger.warning("durable_run_attempt_stamp_failed run=%s", run_id, exc_info=True)
            logger.warning("durable_run_node_failed run=%s node=%s err=%s -- resumable", run_id, node, exc)
            await self._emit_state(state, node=node, status="failed", detail={"error": f"{type(exc).__name__}: {exc}"[:500]})

    # --- resume -------------------------------------------------------------

    async def unfinished_threads(self) -> list[tuple[str, str, datetime | None]]:
        """(thread_id, next_node, checkpoint_ts) for every thread whose latest
        checkpoint still has a next node. Newest checkpoint per thread wins
        (`alist` yields newest first)."""
        # MATERIALISE the listing before asking for any state. The Postgres
        # saver serialises its cursor use behind one asyncio.Lock; `alist` is
        # an async generator that holds that lock while it yields, and
        # `aget_state` needs the same lock -- calling one inside the other
        # deadlocked the runner at boot the first time a checkpoint existed
        # (live, 2026-09-07 01:16Z: "Waiting for application startup" forever,
        # zero Postgres activity). Newest checkpoint per thread wins.
        newest_ts: dict[str, datetime | None] = {}
        async for cp in self._checkpointer.alist(None):
            thread_id = str(((cp.config or {}).get("configurable") or {}).get("thread_id") or "")
            if not thread_id or thread_id in newest_ts:
                continue
            ts_raw = (cp.checkpoint or {}).get("ts")
            ts = None
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    ts = None
            newest_ts[thread_id] = ts
        out: list[tuple[str, str, datetime | None]] = []
        for thread_id, ts in newest_ts.items():
            snap = await self._graph.aget_state(self._config(thread_id))
            if not snap or not snap.next:
                continue
            out.append((thread_id, str(snap.next[0]), ts))
        return out

    async def resume_unfinished(self) -> dict[str, int]:
        counts = {"resumed": 0, "abandoned": 0, "active": 0}
        now = datetime.now(timezone.utc)
        for thread_id, next_node, ts in await self.unfinished_threads():
            if thread_id in self._active:
                counts["active"] += 1
                continue
            # An unparseable/missing checkpoint timestamp is UNKNOWN age, not
            # zero age -- treating it as brand new disabled the one guard that
            # stops a stale checkpoint from resuming into a different day's
            # material. Unknown is the unsafe case, so it fails toward
            # abandonment (bounded by max_age_hours), never toward a silent
            # immediate resume.
            age_h = ((now - ts).total_seconds() / 3600.0) if ts is not None else float("inf")
            if age_h > self._settings.max_age_hours:
                snap = await self._graph.aget_state(self._config(thread_id))
                state = dict(snap.values) if snap and snap.values else {"run_id": thread_id, "correlation_id": ""}
                await self._emit_state(state, node=next_node, status="abandoned", detail={"age_hours": round(age_h, 1)})
                # Mark terminal so the sweep stops seeing it: drive the thread
                # to END by updating state to a terminal status and clearing next.
                await self._graph.aupdate_state(self._config(thread_id), {"status": "abandoned"}, as_node=CURIOSITY_NODES[-1])
                counts["abandoned"] += 1
                continue
            logger.info("durable_run_resume run=%s from=%s age_h=%.1f", thread_id, next_node, age_h)
            snap = await self._graph.aget_state(self._config(thread_id))
            state = dict(snap.values) if snap and snap.values else {"run_id": thread_id, "correlation_id": ""}
            # Receipt at the moment of pickup -- a resumed harness_turn takes
            # 10-40 minutes to complete, and the table should say "resumed"
            # before then, not after.
            await self._emit_state(state, node=next_node, status="resumed", resumed_from=next_node)
            self._spawn(thread_id, None, resumed_from=next_node)
            counts["resumed"] += 1
        return counts

    async def sweep_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._settings.resume_sweep_sec)
                break
            except asyncio.TimeoutError:
                pass
            try:
                counts = await self.resume_unfinished()
                if counts["resumed"] or counts["abandoned"]:
                    logger.info("durable_run_sweep %s", counts)
            except Exception:  # noqa: BLE001
                logger.exception("durable_run_sweep_failed")

    @property
    def active_run_ids(self) -> list[str]:
        return sorted(self._active)
