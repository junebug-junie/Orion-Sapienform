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

Workflow registry (2026-09-21): this runner drives more than one compiled
graph. `DurableRunRequestV1.workflow` selects a `WorkflowSpec` from
`self._workflows`; an unregistered workflow name is rejected at `start_run`
rather than silently defaulting to curiosity's graph. All registered graphs
share ONE checkpointer/Postgres pool -- LangGraph's saver keys purely by
`thread_id` (run_id), not by graph identity, so this is safe as long as
run_ids stay unique across workflows (they already are: callers generate
them, same as today).

The one place this needs care: the resume sweep discovers unfinished threads
BEFORE it knows which graph each one belongs to (`unfinished_threads`).
`_peek_workflow` reads the raw checkpoint's `channel_values["workflow"]`
directly off the shared checkpointer (`aget_tuple`, langgraph's own stable
Checkpoint TypedDict field -- verified against the installed langgraph
version, not assumed) BEFORE calling any compiled graph's `aget_state`,
so a thread is never read through the wrong graph's node/edge schema. A
missing `workflow` key (any checkpoint written before this patch) reads as
`"curiosity.investigate"`, this file's own original workflow name, so every
already-in-flight run resumes exactly as it would have before this change.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.curiosity.attention_schema import (
    read_attended_priors,
    to_attention_schema as curiosity_to_attention_schema,
)
from orion.curiosity.self_inquiry import (
    lived_answer_to_detail,
    read_lived_answer,
    read_self_definition,
    self_definition_to_detail,
)
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
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective
from orion.schemas.durable_run import (
    CURIOSITY_NODES,
    CURIOSITY_TURN_REPLY_PREFIX,
    CURIOSITY_TURN_REQUEST_CHANNEL,
    CURIOSITY_TURN_REQUEST_KIND,
    CURIOSITY_TURN_RESULT_KIND,
    DURABLE_RUN_STATE_KIND,
    SELF_SENSE_EVAL_NODES,
    SELF_STUDY_REFLECT_NODES,
    CuriosityTurnRequestV1,
    CuriosityTurnResultV1,
    DurableRunRequestV1,
    DurableRunStateV1,
)
from orion.schemas.self_sense import CHANNEL_SELF_SENSE_EVAL_WRITE, KIND_SELF_SENSE_EVAL_WRITE, SelfSenseEvalV1
from orion.evals.self_sense_runner import envelope_correlation_id as self_sense_envelope_correlation_id

from app.graph import CuriosityRunState, Deps, build_curiosity_graph, failed_turn_correlation_id, finish_detail
from app.self_sense_graph import Deps as SelfSenseDeps
from app.self_sense_graph import build_self_sense_graph, finish_detail as self_sense_finish_detail
from app.reflect_graph import (
    SELF_STUDY_REFLECT_VERB,
    Deps as ReflectDeps,
    build_reflect_graph,
    finish_detail as reflect_finish_detail,
    parse_reflect_findings,
)
from app.settings import Settings

logger = logging.getLogger("orion-durable-runs.runner")

JOURNAL_WRITE_CHANNEL = "orion:journal:write"

DEFAULT_WORKFLOW = "curiosity.investigate"
SELF_SENSE_EVAL_WORKFLOW = "self_sense_eval"
SELF_STUDY_REFLECT_WORKFLOW = "self_study.reflect"


def _corr_uuid(raw: str) -> UUID:
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return uuid4()


@dataclass(frozen=True)
class WorkflowSpec:
    """One registered graph. `nodes` is that graph's own node order (used for
    next-node bookkeeping in state events); `finish_detail` extracts the
    workflow-specific payload Hub/cortex-exec read off a `completed` event."""

    workflow: str
    graph: Any
    nodes: list[str]
    finish_detail: Callable[[dict[str, Any]], dict[str, Any]]
    # Optional: the correlation the workflow's harness turn ran under, read
    # off the failing thread's state for the `failed` detail. None for
    # graphs that cannot know it (self-sense mints one per question inside
    # the node; reflect has no harness turn).
    failed_turn_correlation_id: Callable[[dict[str, Any]], str | None] | None = None


class DurableRunner:
    _corr_for_admission = staticmethod(_corr_uuid)
    def __init__(self, settings: Settings, *, bus: OrionBusAsync | None, checkpointer: Any) -> None:
        self._settings = settings
        self._bus = bus
        self._checkpointer = checkpointer
        self._reader: WorldviewReader | None = (
            WorldviewReader(host=settings.graph_host, port=settings.graph_port, graph_name=settings.graph_own)
            if settings.graph_host
            else None
        )
        self._workflows: dict[str, WorkflowSpec] = {
            DEFAULT_WORKFLOW: WorkflowSpec(
                workflow=DEFAULT_WORKFLOW,
                graph=build_curiosity_graph(self._curiosity_deps(), checkpointer),
                nodes=list(CURIOSITY_NODES),
                finish_detail=finish_detail,
                failed_turn_correlation_id=failed_turn_correlation_id,
            ),
            SELF_SENSE_EVAL_WORKFLOW: WorkflowSpec(
                workflow=SELF_SENSE_EVAL_WORKFLOW,
                graph=build_self_sense_graph(self._self_sense_deps(), checkpointer),
                nodes=list(SELF_SENSE_EVAL_NODES),
                finish_detail=self_sense_finish_detail,
            ),
            SELF_STUDY_REFLECT_WORKFLOW: WorkflowSpec(
                workflow=SELF_STUDY_REFLECT_WORKFLOW,
                graph=build_reflect_graph(self._reflect_deps(), checkpointer),
                nodes=list(SELF_STUDY_REFLECT_NODES),
                finish_detail=reflect_finish_detail,
            ),
        }
        self._active: dict[str, asyncio.Task[None]] = {}
        # thread_id -> node the resume picked up at; consumed by the first
        # state event after a resume so `resumed_from_node` is stamped once.
        self._resumed_from: dict[str, str] = {}

    def register_workflow(self, spec: WorkflowSpec) -> None:
        """Add a workflow after construction (used by graphs whose Deps need
        a reference back to this runner, e.g. a turn-executor built from
        `self._source()`). Must be called before any run for that workflow
        is submitted -- there is no dynamic re-registration mid-run, and
        registering the same name twice (a hot-reload path, a bug) would
        swap the graph object out from under any task in `self._active`
        still holding a reference to the old spec, so it's refused (review
        finding, 2026-09-21)."""
        if spec.workflow in self._workflows:
            raise ValueError(f"workflow already registered: {spec.workflow}")
        self._workflows[spec.workflow] = spec

    def _spec_for(self, workflow: str) -> WorkflowSpec | None:
        return self._workflows.get(workflow)

    async def _peek_workflow(self, thread_id: str) -> str | None:
        """The `workflow` a checkpointed thread belongs to, or ``None`` when
        no checkpoint exists for it at all (or the read failed) -- kept
        distinct from "checkpoint exists but predates the `workflow` key"
        (which resolves to `DEFAULT_WORKFLOW`) because `start_run` needs to
        tell a genuinely new run apart from a resume; see its own docstring.
        Reads directly off the shared checkpointer -- BEFORE calling any
        compiled graph's `aget_state`, since calling the wrong graph reads
        that thread through the wrong node/edge schema."""
        try:
            tup = await self._checkpointer.aget_tuple(self._config(thread_id))
        except Exception:  # noqa: BLE001
            return None
        if tup is None or not tup.checkpoint:
            return None
        values = tup.checkpoint.get("channel_values") or {}
        workflow = values.get("workflow")
        return str(workflow) if isinstance(workflow, str) and workflow else DEFAULT_WORKFLOW

    # --- deps: the real world behind each node ------------------------------

    def _curiosity_deps(self) -> Deps:
        return Deps(
            run_turn=self._run_turn,
            read_turn_result=self._read_turn_result,
            publish_attention_row=self._publish_attention_row,
            publish_journal=self._publish_journal,
        )

    def _self_sense_deps(self) -> SelfSenseDeps:
        # `run_turn` reuses the SAME Hub RPC curiosity's harness_turn uses --
        # `_run_turn` only cares about the CuriosityTurnRequestV1 it's given,
        # never which graph built it.
        return SelfSenseDeps(run_turn=self._run_turn, publish_rows=self._publish_self_sense_rows)

    async def _publish_self_sense_rows(self, rows: list[SelfSenseEvalV1]) -> tuple[int, int]:
        published = failed = 0
        for row in rows:
            ok = await self._publish(
                CHANNEL_SELF_SENSE_EVAL_WRITE, KIND_SELF_SENSE_EVAL_WRITE, row, self_sense_envelope_correlation_id(row)
            )
            if ok:
                published += 1
            else:
                failed += 1
        return published, failed

    def _reflect_deps(self) -> ReflectDeps:
        return ReflectDeps(call_reflect_llm=self._call_reflect_llm)

    async def _call_reflect_llm(
        self, self_study_reflect_input: dict[str, Any], llm_route: str
    ) -> list[dict[str, Any]] | None:
        """The real verb-dispatch RPC to cortex-orch -- the exact same
        request shape cortex-exec's own `_call_self_study_reflect_llm` built
        directly before this patch (`verb="self_study.reflect"`,
        `options={"policy_dispatch_only": True, ...}`); only WHERE it's sent
        from moved, not its shape. Returns a list of raw finding dicts on
        success, or None on ANY failure (bad input, RPC error/timeout,
        non-ok result, empty/unparseable text, wrong JSON shape) -- same
        "produce nothing on failure" contract, never raises for those."""
        from orion.cognition.cortex_payload_extract import extract_cortex_payload_text

        request = CortexClientRequest(
            mode="brain",
            route_intent="none",
            verb=SELF_STUDY_REFLECT_VERB,
            options={
                "policy_dispatch_only": True,
                **({"llm_route": llm_route} if llm_route else {}),
            },
            recall=RecallDirective(enabled=False, required=False),
            context=CortexClientContext(
                messages=[LLMMessage(role="user", content="Reflect on self-study snapshot.")],
                raw_user_text="Reflect on self-study snapshot.",
                metadata={"self_study_reflect_input": self_study_reflect_input},
            ),
        )
        rpc_correlation_id = uuid4()
        reply_channel = f"orion:cortex:result:self-study-reflect:{rpc_correlation_id}"
        envelope = BaseEnvelope(
            kind="cortex.orch.request",
            source=self._source(),
            correlation_id=rpc_correlation_id,
            reply_to=reply_channel,
            payload=request.model_dump(mode="json"),
        )
        try:
            msg = await self._bus.rpc_request(
                self._settings.cortex_request_channel,
                envelope,
                reply_channel=reply_channel,
                timeout_sec=self._settings.reflect_llm_call_timeout_sec,
            )
            decoded = self._bus.codec.decode(msg.get("data"))
            if not decoded.ok or decoded.envelope is None:
                logger.warning("self_study_reflect_llm_decode_failed corr=%s err=%s", rpc_correlation_id, decoded.error)
                return None
            payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        except Exception as exc:  # noqa: BLE001 -- construction/RPC/decode failure degrades to "no reflection"
            logger.warning("self_study_reflect_llm_call_failed corr=%s err=%s", rpc_correlation_id, exc)
            return None

        if not payload.get("ok", False):
            logger.warning(
                "self_study_reflect_llm_not_ok corr=%s status=%s error=%s",
                rpc_correlation_id, payload.get("status"), payload.get("error"),
            )
            return None
        text = extract_cortex_payload_text(payload)
        if not text:
            logger.warning("self_study_reflect_llm_empty_text corr=%s", rpc_correlation_id)
            return None
        findings = parse_reflect_findings(text)
        if findings is None:
            logger.warning("self_study_reflect_llm_unparseable_or_bad_shape corr=%s", rpc_correlation_id)
            return None
        return findings

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
            payload=request.model_dump(mode="json", exclude_none=True),
        )
        try:
            # The admitted graph owns the inference deadline. Its declared
            # budget may exceed the legacy Hub RPC ceiling; transport must not
            # turn that valid long attempt into an early retry. Queue waiting
            # never reaches this RPC at all.
            rpc_timeout = self._settings.turn_rpc_timeout_sec
            if request.lease is not None:
                rpc_timeout = max(rpc_timeout, request.timeout_sec)
            raw = await self._bus.rpc_request(
                CURIOSITY_TURN_REQUEST_CHANNEL,
                envelope,
                reply_channel=reply_channel,
                timeout_sec=rpc_timeout,
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
            result = CuriosityTurnResultV1.model_validate(payload or {})
            if request.lease is not None:
                if decoded.envelope.kind != CURIOSITY_TURN_RESULT_KIND:
                    raise ValueError("unexpected admitted turn reply kind")
                if (decoded.envelope.correlation_id != envelope.correlation_id
                        or result.run_id != request.run_id
                        or result.correlation_id != request.correlation_id):
                    raise ValueError("admitted turn reply identity mismatch")
            return result
        except Exception as exc:  # noqa: BLE001
            return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=False, error=f"bad_reply:{exc}")

    async def _read_turn_result(self, run_id: str) -> dict[str, Any]:
        reader = self._reader
        empty = {
            "outcome": None,
            "footprint": None,
            "hops": [],
            "evidence_summary": None,
            "graph_readable": False,
            "self_definition": None,
            "lived_answer": None,
        }
        if reader is None:
            return empty

        def _read() -> dict[str, Any]:
            outcome = read_turn_outcome(reader, run_id)
            footprint = read_run_footprint(reader, run_id)
            hops = read_hop_notes(reader, run_id)
            evidence = read_finding_connectivity(reader, run_id)
            # Read for every run, not only self_inquiry: an investigation run
            # that also wrote a definition is still a definition Orion wrote.
            # Hub decides whether to mirror it, keyed on the run's line.
            # Lived self-inquiry writes `:LivedAnswer` instead of `:SelfDefinition`.
            self_definition = read_self_definition(reader, run_id)
            lived_answer = read_lived_answer(reader, run_id)
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
                "lived_answer": lived_answer_to_detail(lived_answer),
            }

        try:
            return await asyncio.wait_for(asyncio.to_thread(_read), timeout=30.0)
        except Exception as exc:  # noqa: BLE001 -- unreadable graph is a state, not a crash
            logger.warning("durable_run_graph_read_failed run=%s err=%s", run_id, exc)
            return empty

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
        spec: WorkflowSpec,
        node: str,
        status: str,
        detail: dict[str, Any] | None = None,
        resumed_from: str | None = None,
    ) -> None:
        run_id = state["run_id"]
        nodes = spec.nodes
        idx = nodes.index(node) if node in nodes else -1
        next_node = nodes[idx + 1] if 0 <= idx < len(nodes) - 1 else None
        if status in ("completed", "failed", "abandoned"):
            next_node = None if status != "failed" else node
        event = DurableRunStateV1(
            run_id=run_id,
            workflow=spec.workflow,
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
                attended_label=spec.workflow,
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
        duplicate request for a thread that already exists is a resume,
        matched against the CHECKPOINT's own workflow (`_peek_workflow`),
        never the new request's declared one -- a request naming a
        different workflow than the thread was actually created under is
        refused outright rather than silently resumed through the wrong
        graph's node/edge schema (review finding, 2026-09-21)."""
        run_id = request.run_id
        if run_id in self._active:
            logger.info("durable_run_already_active run=%s", run_id)
            return
        existing_workflow = await self._peek_workflow(run_id)
        if existing_workflow is not None and existing_workflow != request.workflow:
            logger.error(
                "durable_run_workflow_mismatch run=%s checkpoint_workflow=%s request_workflow=%s -- refusing",
                run_id, existing_workflow, request.workflow,
            )
            return
        spec = self._spec_for(request.workflow)
        if spec is None:
            logger.error("durable_run_unknown_workflow run=%s workflow=%s", run_id, request.workflow)
            return
        snapshot = await spec.graph.aget_state(self._config(run_id))
        if snapshot and snapshot.values:
            if not snapshot.next or snapshot.values.get("admission"):
                return
            logger.info("durable_run_request_for_existing_thread run=%s -> resume", run_id)
            self._spawn(spec, run_id, None, resumed_from=snapshot.next[0] if snapshot.next else None)
            return
        initial: CuriosityRunState = {
            "run_id": run_id,
            "correlation_id": request.correlation_id,
            "workflow": spec.workflow,
            "brief": request.brief.model_dump(mode="json"),
            "attempt": 0,
        }
        self._spawn(spec, run_id, initial, resumed_from=None)

    def _spawn(
        self, spec: WorkflowSpec, run_id: str, initial: CuriosityRunState | None, *, resumed_from: str | None
    ) -> None:
        if resumed_from:
            self._resumed_from[run_id] = resumed_from
        task = asyncio.create_task(self._drive(spec, run_id, initial), name=f"durable-run-{run_id}")
        self._active[run_id] = task
        task.add_done_callback(lambda _t: self._active.pop(run_id, None))

    async def _drive(self, spec: WorkflowSpec, run_id: str, initial: CuriosityRunState | None) -> None:
        config = self._config(run_id)
        graph = spec.graph
        nodes = spec.nodes
        last_state: CuriosityRunState = initial or {}
        try:
            # astream with stream_mode="updates" yields one item per completed
            # node: {node_name: {returned keys}}. The checkpoint for that node
            # is written by the compiled graph before the next node starts.
            async for update in graph.astream(initial, config, stream_mode="updates"):
                for node, delta in (update or {}).items():
                    snap = await graph.aget_state(config)
                    last_state = dict(snap.values) if snap and snap.values else last_state
                    if node == nodes[-1]:
                        await self._emit_state(
                            last_state, spec=spec, node=node, status="completed", detail=spec.finish_detail(last_state)
                        )
                    else:
                        status = "resumed" if run_id in self._resumed_from else "running"
                        await self._emit_state(last_state, spec=spec, node=node, status=status)
        except Exception as exc:  # noqa: BLE001 -- the thread stays resumable at its next node
            snap = None
            try:
                snap = await graph.aget_state(config)
            except Exception:  # noqa: BLE001
                pass
            node = (snap.next[0] if snap and snap.next else nodes[0])
            state = dict(snap.values) if snap and snap.values else (initial or {"run_id": run_id, "correlation_id": ""})
            if node == nodes[0]:
                # A failed harness turn cannot record its own attempt (the node
                # raised before returning), so stamp it on the thread as if
                # START had written it: `next` stays harness_turn, the sweep
                # re-issues the turn as attempt+1. Verified against the saver.
                try:
                    from langgraph.graph import START

                    attempt = int(state.get("attempt") or 0) + 1
                    await graph.aupdate_state(config, {"attempt": attempt}, as_node=START)
                    state["attempt"] = attempt
                except Exception:  # noqa: BLE001
                    logger.warning("durable_run_attempt_stamp_failed run=%s", run_id, exc_info=True)
            logger.warning("durable_run_node_failed run=%s node=%s err=%s -- resumable", run_id, node, exc)
            await self._emit_state(state, spec=spec, node=node, status="failed", detail=self._failed_detail(spec, node, state, exc))

    @staticmethod
    def _failed_detail(spec: WorkflowSpec, node: str, state: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        """`{"error", "node"}` plus `turn_correlation_id` when the workflow
        can name it. A helper that itself raises must not turn a resumable
        failure into a lost state event, so it degrades to the bare shape."""
        detail: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"[:500], "node": node}
        if spec.failed_turn_correlation_id is not None:
            try:
                corr = spec.failed_turn_correlation_id(state)
            except Exception:  # noqa: BLE001
                corr = None
            if corr:
                detail["turn_correlation_id"] = corr
        return detail

    # --- resume -------------------------------------------------------------

    async def unfinished_threads(self) -> list[tuple[str, str, datetime | None, str]]:
        """(thread_id, next_node, checkpoint_ts, workflow) for every thread
        whose latest checkpoint still has a next node. Newest checkpoint per
        thread wins (`alist` yields newest first). `workflow` is read
        straight off the SAME checkpoint tuple this scan already holds (not
        a second `_peek_workflow` round trip per thread -- review finding,
        2026-09-21), still strictly BEFORE any graph-specific `aget_state`
        call; see the module docstring for why that ordering matters."""
        # MATERIALISE the listing before asking for any state. The Postgres
        # saver serialises its cursor use behind one asyncio.Lock; `alist` is
        # an async generator that holds that lock while it yields, and
        # `aget_state` needs the same lock -- calling one inside the other
        # deadlocked the runner at boot the first time a checkpoint existed
        # (live, 2026-09-07 01:16Z: "Waiting for application startup" forever,
        # zero Postgres activity). Newest checkpoint per thread wins.
        newest: dict[str, tuple[datetime | None, str]] = {}
        async for cp in self._checkpointer.alist(None):
            thread_id = str(((cp.config or {}).get("configurable") or {}).get("thread_id") or "")
            if not thread_id or thread_id in newest:
                continue
            checkpoint = cp.checkpoint or {}
            ts_raw = checkpoint.get("ts")
            ts = None
            if isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    ts = None
            values = checkpoint.get("channel_values") or {}
            workflow_raw = values.get("workflow")
            workflow = str(workflow_raw) if isinstance(workflow_raw, str) and workflow_raw else DEFAULT_WORKFLOW
            newest[thread_id] = (ts, workflow)
        out: list[tuple[str, str, datetime | None, str]] = []
        for thread_id, (ts, workflow) in newest.items():
            spec = self._spec_for(workflow)
            if spec is None:
                logger.warning("durable_run_resume_unknown_workflow thread=%s workflow=%s -- skipped", thread_id, workflow)
                continue
            snap = await spec.graph.aget_state(self._config(thread_id))
            if not snap or not snap.next:
                continue
            if snap.values.get("admission"):
                continue  # admission runtime owns these graph interrupts
            out.append((thread_id, str(snap.next[0]), ts, workflow))
        return out

    async def resume_unfinished(self) -> dict[str, int]:
        counts = {"resumed": 0, "abandoned": 0, "active": 0}
        now = datetime.now(timezone.utc)
        for thread_id, next_node, ts, workflow in await self.unfinished_threads():
            if thread_id in self._active:
                counts["active"] += 1
                continue
            # unfinished_threads() already filters to registered workflows
            # only -- an unresolvable spec here would be a real invariant
            # break, not a routine skip, so it fails loudly rather than a
            # silent `continue` (review finding, 2026-09-21).
            spec = self._spec_for(workflow)
            assert spec is not None, f"unfinished_threads returned an unregistered workflow: {workflow}"
            # An unparseable/missing checkpoint timestamp is UNKNOWN age, not
            # zero age -- treating it as brand new disabled the one guard that
            # stops a stale checkpoint from resuming into a different day's
            # material. Unknown is the unsafe case, so it fails toward
            # abandonment (bounded by max_age_hours), never toward a silent
            # immediate resume.
            age_h = ((now - ts).total_seconds() / 3600.0) if ts is not None else float("inf")
            if age_h > self._settings.max_age_hours:
                snap = await spec.graph.aget_state(self._config(thread_id))
                state = dict(snap.values) if snap and snap.values else {"run_id": thread_id, "correlation_id": ""}
                await self._emit_state(
                    state, spec=spec, node=next_node, status="abandoned", detail={"age_hours": round(age_h, 1)}
                )
                # Mark terminal so the sweep stops seeing it: drive the thread
                # to END by updating state to a terminal status and clearing next.
                await spec.graph.aupdate_state(self._config(thread_id), {"status": "abandoned"}, as_node=spec.nodes[-1])
                counts["abandoned"] += 1
                continue
            logger.info("durable_run_resume run=%s from=%s age_h=%.1f", thread_id, next_node, age_h)
            snap = await spec.graph.aget_state(self._config(thread_id))
            state = dict(snap.values) if snap and snap.values else {"run_id": thread_id, "correlation_id": ""}
            # Receipt at the moment of pickup -- a resumed harness_turn takes
            # 10-40 minutes to complete, and the table should say "resumed"
            # before then, not after.
            await self._emit_state(state, spec=spec, node=next_node, status="resumed", resumed_from=next_node)
            self._spawn(spec, thread_id, None, resumed_from=next_node)
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
