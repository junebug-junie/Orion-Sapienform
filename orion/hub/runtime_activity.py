"""What is Orion running right now -- one small in-memory reducer for Hub.

Hub is the only process that both dispatches every curiosity run (via cortex
to orion-durable-runs) and hands every harness turn to the governor, so Hub
already sees the start of everything this page shows. The rest arrives on bus
channels Hub already consumes. Nothing here is a new producer, schema, or
channel; this folds facts that already exist into one snapshot a page can
read, and pushes a version bump to whoever is listening.

Sources folded (all pre-existing):

- ``orion:durable:run:state`` (DurableRunStateV1) -- every node transition of
  a curiosity run. Fed from CuriosityInvestigation's existing subscription.
- ``_dispatch_durable_run`` in curiosity_investigation -- the only place that
  knows which *line* (investigate / self_inquiry) a run is on before it ends.
- ``orion:harness:run:step`` (HarnessRunStepV1) -- fed from HarnessStepRelay's
  existing subscription. The FIRST step for a correlation_id is the evidence
  that the governor actually started the turn; before that it is queued.
- ``execute_unified_turn`` -- Hub's own handoff to the governor: which lane
  (chat / agent, the governor's two serial dispatch loops), which mode, which
  model label, which caller (source tag), and the HarnessRunV1 outcome.
- orion-llm-gateway ``/admission`` -- per-upstream inflight/waiting gauges,
  polled by Hub. Those are the gateway's own lanes (chat / spark /
  background=metacog / agent), a different multiplexer from the governor's.

Runtime truth rule: a turn is "running" only once a step has been observed;
a run is "active" only while its last transition says running/resumed.
Absent is reported as absent (``first_step_at: null``), never guessed.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from orion.harness.fcc_motor import _extract_tool_name, summarize_harness_step
from orion.llm.routes import is_agent_route_model_label

ACTIVE_RUN_STATUSES = frozenset({"running", "resumed"})
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "abandoned"})
GOVERNOR_LANES = ("chat", "agent")

# The motor-boot marker step (orion/harness/runner.py publishes step_index=-1
# with this key) carries the whole prompt. Never keep it; only note it fired.
_COCKPIT_MARKER_KEY = "_cockpit"

_MAX_TRANSITIONS_PER_RUN = 40
_RECENT_STEP_SUMMARIES = 5


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(raw: Any, fallback: float) -> float:
    """generated_at from a DurableRunStateV1 payload (ISO string after
    model_dump) -> epoch seconds. Falls back to *fallback* on anything odd so
    a malformed timestamp can never drop the transition itself."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(raw, str) and raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return fallback
    return fallback


def lane_for_model_label(model_label: str | None) -> str:
    """Same predicate HarnessGovernorClient.run() uses to pick its dispatch
    queue, on the same field -- so the lane shown here cannot disagree with
    the queue the turn actually waits in."""
    return "agent" if is_agent_route_model_label(model_label) else "chat"


def summarize_step(step: dict[str, Any], *, index: int) -> str:
    """Short, prompt-free label for one FCC motor step."""
    if not isinstance(step, dict):
        return "step"
    if _COCKPIT_MARKER_KEY in step:
        return "motor boot"
    tool = _extract_tool_name(step)
    if tool:
        return f"tool {tool}"
    try:
        text = summarize_harness_step(step, index=max(index, 0))
    except Exception:  # noqa: BLE001 -- a label, never worth failing a fold
        text = ""
    text = " ".join(str(text or "").split())
    return text[:80] or "step"


@dataclass
class RunRecord:
    run_id: str
    correlation_id: str
    workflow: str = "curiosity.investigate"
    line: str | None = None
    status: str = "dispatched"
    node: str | None = None
    next_node: str | None = None
    resumed_from_node: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    error: str | None = None
    finish: dict[str, Any] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_RUN_STATUSES or self.status == "dispatched"


@dataclass
class TurnRecord:
    correlation_id: str
    lane: str
    mode: str
    model_label: str | None
    source: str
    requested_at: float
    first_step_at: float | None = None
    last_step_at: float | None = None
    step_count: int = 0
    recent_steps: list[str] = field(default_factory=list)
    finished_at: float | None = None
    ok: bool | None = None
    error: str | None = None
    compliance_verdict: str | None = None
    fcc_elapsed_sec: float | None = None
    served_model: str | None = None

    @property
    def phase(self) -> str:
        if self.finished_at is not None:
            return "finished"
        return "running" if self.first_step_at is not None else "queued"


class RuntimeActivity:
    """Fold + snapshot. Every fold bumps ``version`` and wakes subscribers."""

    def __init__(
        self,
        *,
        now: Callable[[], float] = time.time,
        finished_ttl_sec: float = 900.0,
        max_finished: int = 30,
        max_turns: int = 500,
        max_runs: int = 200,
    ) -> None:
        self._now = now
        self._finished_ttl_sec = max(1.0, float(finished_ttl_sec))
        self._max_finished = max(1, int(max_finished))
        self._max_turns = max(1, int(max_turns))
        self._max_runs = max(1, int(max_runs))
        self._runs: "OrderedDict[str, RunRecord]" = OrderedDict()
        self._turns: "OrderedDict[str, TurnRecord]" = OrderedDict()
        self._gateway: dict[str, Any] | None = None
        self._gateway_error: str | None = None
        self._gateway_at: float | None = None
        self.version = 0
        self._subscribers: set[asyncio.Queue] = set()

    # ------------------------------------------------------------------ pub/sub
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _bump(self) -> None:
        # Eviction rides every fold, whichever side the fold touched -- a
        # finished run must not outlive its TTL just because only turn
        # events arrived since. snapshot() sweeps too, for a quiet process.
        self._evict_runs()
        self._evict_turns()
        self.version += 1
        for q in list(self._subscribers):
            try:
                q.put_nowait(self.version)
            except asyncio.QueueFull:
                # A slow reader still gets the newest version on its next
                # drain; the queue is a wake-up, not a log.
                continue

    # ----------------------------------------------------------- curiosity runs
    def run_dispatched(self, *, run_id: str, correlation_id: str, line: str) -> None:
        now = self._now()
        rec = self._runs.get(run_id)
        if rec is None:
            rec = RunRecord(run_id=run_id, correlation_id=correlation_id, started_at=now, updated_at=now)
            self._runs[run_id] = rec
        rec.line = line or rec.line
        rec.correlation_id = correlation_id or rec.correlation_id
        rec.updated_at = now
        self._bump()

    def run_state(self, payload: dict[str, Any]) -> None:
        """One DurableRunStateV1 payload (already validated upstream or raw
        dict -- only the fields named here are read)."""
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            return
        now = self._now()
        ts = _parse_ts(payload.get("generated_at"), now)
        rec = self._runs.get(run_id)
        if rec is None:
            rec = RunRecord(
                run_id=run_id,
                correlation_id=str(payload.get("correlation_id") or ""),
                started_at=ts,
                updated_at=ts,
            )
            self._runs[run_id] = rec
        rec.workflow = str(payload.get("workflow") or rec.workflow)
        status = str(payload.get("status") or rec.status)
        rec.status = status
        rec.node = payload.get("node") or rec.node
        rec.next_node = payload.get("next_node")
        if payload.get("resumed_from_node"):
            rec.resumed_from_node = str(payload["resumed_from_node"])
        rec.updated_at = max(rec.updated_at, ts)
        detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
        if detail.get("line"):
            rec.line = str(detail["line"])
        if status in TERMINAL_RUN_STATUSES:
            rec.finished_at = ts
            if status == "failed":
                rec.error = str(detail.get("error") or "")[:300] or rec.error
            if status == "completed":
                rec.finish = {
                    k: detail[k]
                    for k in ("reach_out", "reach_out_why", "continue_line")
                    if k in detail
                }
                finding = str(detail.get("finding_text") or "")
                if finding:
                    rec.finish["finding_text"] = finding[:240]
        rec.transitions.append({"node": rec.node, "status": status, "at": _iso(ts)})
        if len(rec.transitions) > _MAX_TRANSITIONS_PER_RUN:
            del rec.transitions[: len(rec.transitions) - _MAX_TRANSITIONS_PER_RUN]
        self._bump()

    def backfill_runs(self, rows: list[dict[str, Any]]) -> int:
        """Cold-start: latest transition per run_id from
        substrate_durable_run_state. Only runs whose latest row is still
        running/resumed are worth knowing about; terminal rows are ignored.
        Returns how many were adopted. No bump per row -- one at the end."""
        adopted = 0
        for row in rows:
            status = str(row.get("status") or "")
            if status not in ACTIVE_RUN_STATUSES:
                continue
            run_id = str(row.get("run_id") or "")
            if not run_id or run_id in self._runs:
                continue
            ts = _parse_ts(row.get("generated_at"), self._now())
            first = _parse_ts(row.get("first_seen_at"), ts)
            rec = RunRecord(
                run_id=run_id,
                correlation_id=str(row.get("correlation_id") or ""),
                workflow=str(row.get("workflow") or "curiosity.investigate"),
                status=status,
                node=row.get("node"),
                next_node=row.get("next_node"),
                resumed_from_node=row.get("resumed_from_node"),
                started_at=first,
                updated_at=ts,
            )
            rec.transitions.append({"node": rec.node, "status": status, "at": _iso(ts), "backfilled": True})
            self._runs[run_id] = rec
            adopted += 1
        if adopted:
            self._bump()
        return adopted

    # ------------------------------------------------------------ harness turns
    def turn_requested(
        self,
        *,
        correlation_id: str,
        mode: str | None,
        model_label: str | None,
        source: str | None,
    ) -> None:
        cid = str(correlation_id or "")
        if not cid:
            return
        now = self._now()
        rec = TurnRecord(
            correlation_id=cid,
            lane=lane_for_model_label(model_label),
            mode=str(mode or "orion"),
            model_label=model_label or None,
            source=str(source or "chat"),
            requested_at=now,
        )
        self._turns[cid] = rec
        self._turns.move_to_end(cid)
        self._bump()

    def harness_step(self, payload: dict[str, Any]) -> None:
        cid = str(payload.get("correlation_id") or "")
        if not cid:
            return
        rec = self._turns.get(cid)
        if rec is None:
            # A step for a turn this Hub never requested (sibling replica, or
            # a turn requested before restart). Still real evidence that the
            # governor is busy on that lane -- keep it, lane unknown.
            rec = TurnRecord(
                correlation_id=cid,
                lane="unknown",
                mode="unknown",
                model_label=None,
                source="unknown",
                requested_at=self._now(),
            )
            self._turns[cid] = rec
        now = self._now()
        if rec.first_step_at is None:
            rec.first_step_at = now
        rec.last_step_at = now
        try:
            index = int(payload.get("step_index", 0))
        except (TypeError, ValueError):
            index = 0
        if index >= 0:
            rec.step_count = max(rec.step_count, index + 1)
        step = payload.get("step") if isinstance(payload.get("step"), dict) else {}
        rec.recent_steps.append(summarize_step(step, index=index))
        if len(rec.recent_steps) > _RECENT_STEP_SUMMARIES:
            del rec.recent_steps[: len(rec.recent_steps) - _RECENT_STEP_SUMMARIES]
        self._turns.move_to_end(cid)
        self._bump()

    def turn_finished(
        self,
        *,
        correlation_id: str,
        run: Any | None,
        error: str | None = None,
    ) -> None:
        cid = str(correlation_id or "")
        rec = self._turns.get(cid)
        if rec is None:
            return
        rec.finished_at = self._now()
        if run is not None:
            rec.ok = True
            rec.compliance_verdict = getattr(run, "compliance_verdict", None)
            rec.fcc_elapsed_sec = getattr(run, "fcc_elapsed_sec", None)
            rec.served_model = getattr(run, "fcc_served_model", None)
            step_count = getattr(run, "step_count", None)
            if isinstance(step_count, int):
                rec.step_count = max(rec.step_count, step_count)
        else:
            rec.ok = False
            rec.error = (error or "no_result")[:300]
        self._bump()

    # -------------------------------------------------------------- gateway
    def gateway_admission(self, snapshot: dict[str, Any] | None, *, error: str | None = None) -> None:
        self._gateway_at = self._now()
        if snapshot is None:
            self._gateway_error = error or "unavailable"
        else:
            self._gateway = snapshot
            self._gateway_error = None
        self._bump()

    # -------------------------------------------------------------- eviction
    def _evict_runs(self) -> None:
        now = self._now()
        finished = [r for r in self._runs.values() if not r.active]
        finished.sort(key=lambda r: r.finished_at or r.updated_at)
        stale = [
            r for r in finished if (now - (r.finished_at or r.updated_at)) > self._finished_ttl_sec
        ]
        overflow = finished[: max(0, len(finished) - self._max_finished)]
        for r in {id(x): x for x in stale + overflow}.values():
            self._runs.pop(r.run_id, None)
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)

    def _evict_turns(self) -> None:
        now = self._now()
        finished = [t for t in self._turns.values() if t.finished_at is not None]
        finished.sort(key=lambda t: t.finished_at or 0.0)
        stale = [t for t in finished if (now - (t.finished_at or 0.0)) > self._finished_ttl_sec]
        overflow = finished[: max(0, len(finished) - self._max_finished)]
        for t in {id(x): x for x in stale + overflow}.values():
            self._turns.pop(t.correlation_id, None)
        while len(self._turns) > self._max_turns:
            self._turns.popitem(last=False)

    # -------------------------------------------------------------- snapshot
    def _run_dict(self, r: RunRecord, now: float) -> dict[str, Any]:
        end = r.finished_at if r.finished_at is not None else now
        turn = self._turns.get(r.correlation_id)
        return {
            "run_id": r.run_id,
            "correlation_id": r.correlation_id,
            "workflow": r.workflow,
            "line": r.line,
            "status": r.status,
            "active": r.active,
            "node": r.node,
            "next_node": r.next_node,
            "resumed_from_node": r.resumed_from_node,
            "started_at": _iso(r.started_at),
            "updated_at": _iso(r.updated_at),
            "finished_at": _iso(r.finished_at),
            "duration_sec": round(max(0.0, end - r.started_at), 1),
            "error": r.error,
            "finish": r.finish,
            "transitions": list(r.transitions),
            # The harness turn this run is on, joined by correlation_id
            # (CuriosityTurnRequestV1 carries the run's own correlation_id).
            "turn": self._turn_dict(turn, now) if turn is not None else None,
        }

    def _turn_dict(self, t: TurnRecord, now: float) -> dict[str, Any]:
        end = t.finished_at if t.finished_at is not None else now
        return {
            "correlation_id": t.correlation_id,
            "lane": t.lane,
            "mode": t.mode,
            "model_label": t.model_label,
            "source": t.source,
            "phase": t.phase,
            "requested_at": _iso(t.requested_at),
            "first_step_at": _iso(t.first_step_at),
            "last_step_at": _iso(t.last_step_at),
            "finished_at": _iso(t.finished_at),
            "queued_sec": round(max(0.0, (t.first_step_at or end) - t.requested_at), 1),
            "elapsed_sec": round(max(0.0, end - t.requested_at), 1),
            "step_count": t.step_count,
            "recent_steps": list(t.recent_steps),
            "ok": t.ok,
            "error": t.error,
            "compliance_verdict": t.compliance_verdict,
            "fcc_elapsed_sec": t.fcc_elapsed_sec,
            "served_model": t.served_model,
        }

    def snapshot(self) -> dict[str, Any]:
        self._evict_runs()
        self._evict_turns()
        now = self._now()
        runs = sorted(self._runs.values(), key=lambda r: (not r.active, -r.updated_at))
        turns = list(self._turns.values())
        lanes: dict[str, Any] = {}
        for lane in GOVERNOR_LANES + ("unknown",):
            mine = [t for t in turns if t.lane == lane]
            running = [t for t in mine if t.phase == "running"]
            queued = sorted((t for t in mine if t.phase == "queued"), key=lambda t: t.requested_at)
            recent = sorted((t for t in mine if t.phase == "finished"), key=lambda t: -(t.finished_at or 0.0))
            if lane == "unknown" and not mine:
                continue
            lanes[lane] = {
                "running": [self._turn_dict(t, now) for t in running],
                "queued": [self._turn_dict(t, now) for t in queued],
                "recent": [self._turn_dict(t, now) for t in recent[:5]],
            }
        active_runs = [r for r in runs if r.active]
        busy = bool(active_runs) or any(
            lanes[l]["running"] or lanes[l]["queued"] for l in lanes
        )
        return {
            "version": self.version,
            "generated_at": _iso(now),
            "busy": busy,
            "curiosity_runs": [self._run_dict(r, now) for r in runs],
            "active_run_count": len(active_runs),
            "lanes": lanes,
            "gateway": {
                "snapshot": self._gateway,
                "error": self._gateway_error,
                "polled_at": _iso(self._gateway_at),
            },
        }


_singleton: RuntimeActivity | None = None


def get_runtime_activity() -> RuntimeActivity:
    """Process-wide instance. Hub's main.py wires bus/HTTP feeds around it;
    turn_orchestrator / curiosity_investigation / harness_step_relay call
    into it directly. One process, one answer to "what's running"."""
    global _singleton
    if _singleton is None:
        _singleton = RuntimeActivity()
    return _singleton


def reset_runtime_activity(instance: Optional[RuntimeActivity] = None) -> RuntimeActivity:
    """Tests: swap in a fresh (or fake-clock) instance."""
    global _singleton
    _singleton = instance or RuntimeActivity()
    return _singleton
