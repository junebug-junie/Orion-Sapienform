"""Stages of an admitted `reverie.visual` durable run, executed one at a time.

Design: docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md.
Contract: orion/schemas/reverie_visual_run.py.

orion-durable-runs drives the graph and sends one `ReverieVisualStepRequestV1` per
stage; each handler here runs that stage against the attempt's checkpoint
(`reverie_visual_attempt.stage_json`) and answers with a `ReverieVisualStepResultV1`.
Every stage calls the same `visual_chain.py` pieces the legacy `/visual-chain/run-once`
body runs back to back, so both paths write identical chain rows and receipts.

`stage_json` on the attempt row:

    plan                   frozen VisualPlan (prepare); never recomputed once written
    stage                  prepared | generating | generated
    generating_started_at  wall-clock start of the current/last generate (in-flight guard)
    artifact               recorded image: sha256, path, mime, bytes, width, height,
                           thermal_gate, elapsed_sec (generate)
    caption                cached re-observation so a caption retry never recaptions
    deferrals              the last _DEFERRAL_HISTORY retries {at, step, reason}
    abandoned_at, abandon_reason
    dream_hop              {carry_run_id, hop_index} of a dream.carry image hop (prepare)

Dream hop mode (`req.dream_hop`, design 2026-10-10-dream-carry-through): the same claim,
generate and caption, but prepare freezes the hop's prompt verbatim (no context, continuity,
slot rotation or interpret, read or advanced) and caption writes no chain row, production
acknowledgement or execution receipt -- a dream picture neither counts as nor continues a
waking painting. Caption done returns what was seen (`caption`); an empty caption is a retry.

Step mode never writes a deferral chain row: a row with chain_id == attempt_id would
make the eventual production row a no-op (`ON CONFLICT DO NOTHING`). Deferrals are
recorded in `stage_json.deferrals` instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, get_args

from orion.gpu_pool.client import LeaseUnavailable, durable_run_holder, validate_hold_ref
from orion.reverie.baseline import load_baseline_policy, validate_eligibility
from orion.reverie.visual_storage import StoredVisualArtifact, load_visual_artifact, store_visual_artifact
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.reverie_visual import VisualRunOutcome
from orion.schemas.reverie_visual_run import (
    NEEDS_GENERATE,
    DreamHopImageV1,
    ReverieVisualStepRequestV1,
    ReverieVisualStepResultV1,
)

from . import store
from . import visual_chain as vc
from .cortex_client import CortexExecClient
from .settings import settings

logger = logging.getLogger("orion-thought.visual_steps")

_OPEN_OUTCOMES = frozenset({"active", "unknown"})
_RUN_OUTCOMES = frozenset(get_args(VisualRunOutcome))
_DEFERRAL_HISTORY = 20
# The room cools on the order of minutes; there is no point asking again sooner.
_THERMAL_RETRY_AFTER_SEC = 300.0
# The single-flight lock is held by a run-once call or another generate (~1-2 min).
_BUSY_RETRY_AFTER_SEC = 30.0
# The legacy visual worker is switched off by an operator, not by a retry.
_LEGACY_WORKER_RETRY_AFTER_SEC = 300.0
_REASON_DETAIL_CHARS = 200
# Recording a finished render: a transient DB error must not cost the image.
_STAGE_WRITE_ATTEMPTS = 3
_STAGE_WRITE_BACKOFF_SEC = 0.5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def visual_step_generate_deadline_sec() -> float:
    """Generate's own deadline, never below lease wait + diffusion timeout + margin. The durable
    step no longer waits for a lease (its hold is the grant), but the run-once route shares
    this window (claim_visual_attempt's abandoned_in_flight_window_sec) and does wait."""
    floor = (settings.visual_chain_gpu_lease_deadline_sec
             + settings.visual_chain_diffusion_timeout_sec + 10.0)
    return max(float(settings.visual_chain_step_generate_deadline_sec), floor)


def _in_flight_window_sec() -> float:
    return 2.0 * visual_step_generate_deadline_sec()


def _detail(exc: BaseException) -> str:
    return (str(exc) or type(exc).__name__)[:_REASON_DETAIL_CHARS]


class _Step:
    """Builds results for one request; elapsed_sec defaults to wall time since start."""

    def __init__(self, req: ReverieVisualStepRequestV1):
        self.req = req
        self.started = time.monotonic()

    def result(self, status: str, *, elapsed_sec: float | None = None, **fields: Any) -> ReverieVisualStepResultV1:
        elapsed = time.monotonic() - self.started if elapsed_sec is None else elapsed_sec
        fields.setdefault("attempt_id", self.req.attempt_id)
        return ReverieVisualStepResultV1(
            run_id=self.req.run_id, correlation_id=self.req.correlation_id, step=self.req.step,
            status=status, elapsed_sec=round(max(0.0, float(elapsed)), 3), **fields,
        )

    def retry(self, reason: str, *, retry_after_sec: float | None = None, **fields: Any):
        return self.result("retry", reason=reason, retry_after_sec=retry_after_sec, **fields)

    def terminal(self, outcome: str, reason: str, **fields: Any):
        return self.result("terminal", outcome=outcome, reason=reason, **fields)


def _dream_hop_identity(hop: DreamHopImageV1 | None) -> dict | None:
    return None if hop is None else {"carry_run_id": hop.carry_run_id, "hop_index": hop.hop_index}


def _dream_plan(hop: DreamHopImageV1) -> vc.VisualPlan:
    """The hop's prompt verbatim with neutral continuity: nothing read, nothing advanced."""
    return vc.VisualPlan(
        prompt=hop.prompt, prior_description=None, prior_chain_id=None, effective_prior=None,
        continuity_fallback=None, continuity_streak=0, continuity_reset=False,
        context_slot_used=None, context_slot_rotation=0, context_slot_interpreted=None,
        context_text=None, self_study_text=None, memory_text=None, context_selection=None,
    )


def _dream_hop_mismatch(req: ReverieVisualStepRequestV1, stage: dict) -> bool:
    """True when the attempt's frozen plan was made for a different hop (or mode) than `req`."""
    plan = _frozen_plan(stage)
    if plan is None:
        return False
    if stage.get("dream_hop") != _dream_hop_identity(req.dream_hop):
        return True
    return req.dream_hop is not None and plan.prompt != req.dream_hop.prompt


def _frozen_plan(stage: dict) -> vc.VisualPlan | None:
    raw = stage.get("plan")
    if not isinstance(raw, dict):
        return None
    try:
        return vc.VisualPlan.from_json(raw)
    except (TypeError, ValueError):
        logger.warning("visual step: frozen plan unreadable keys=%s", sorted(raw))
        return None


def _with_deferral(stage: dict, step: str, reason: str, now: datetime) -> dict:
    deferrals = list(stage.get("deferrals") or [])
    deferrals.append({"at": now.isoformat(), "step": step, "reason": reason})
    stage["deferrals"] = deferrals[-_DEFERRAL_HISTORY:]
    return stage


async def _record_deferral(attempt_id: str, step: str, reason: str, now: datetime, *,
                           stage_name: str | None = None, release_abandoned: bool = False) -> None:
    def mutate(stage: dict, _row: dict) -> dict:
        if stage_name is not None:
            stage["stage"] = stage_name
        return _with_deferral(stage, step, reason, now)

    await asyncio.to_thread(store.update_visual_stage, attempt_id, mutate,
                            release_abandoned=release_abandoned)


def _recorded_artifact(record: Any) -> tuple[StoredVisualArtifact, bytes] | None:
    """The recorded image and its bytes, only if they are on disk and hash to the recorded sha."""
    if not isinstance(record, dict) or not record.get("sha256") or not record.get("path"):
        return None
    try:
        data = load_visual_artifact(record["sha256"], base_dir=settings.visual_chain_storage_dir)
    except OSError:
        return None
    if not data or hashlib.sha256(data).hexdigest() != record["sha256"]:
        return None
    stored = StoredVisualArtifact(
        sha256=record["sha256"], mime=str(record.get("mime") or ""), bytes=len(data),
        width=record.get("width"), height=record.get("height"), path=record["path"],
    )
    return stored, data


def _closed_attempt(step: _Step, row: dict) -> ReverieVisualStepResultV1:
    """A step for an attempt that is no longer open: report what it ended as."""
    result = row.get("result_json") or {}
    attempt_id = row["attempt_id"]
    if row["outcome"] == "produced":
        sha = result.get("artifact_sha256")
        return step.terminal(
            "produced", result.get("reason") or "attempt_produced", attempt_id=attempt_id,
            chain_id=_result_chain_id(row),
            artifact_sha256=sha if isinstance(sha, str) and len(sha) == 64 else None,
            execution_receipt=result.get("execution_receipt"),
            caption=_result_caption(row),
        )
    if row["outcome"] == store.VISUAL_ATTEMPT_ABANDONED:
        return step.terminal("unknown", result.get("reason") or "run_abandoned", attempt_id=attempt_id)
    outcome = row["outcome"] if row["outcome"] in _RUN_OUTCOMES else "failed"
    return step.terminal(outcome, result.get("reason") or "attempt_closed", attempt_id=attempt_id)


def _result_chain_id(row: dict) -> str | None:
    """A dream hop writes no chain row, so it has no chain id; a waking attempt's is its id."""
    result = row.get("result_json") or {}
    if result.get("dream_hop") is not None:
        return None
    return result.get("chain_id") or row["attempt_id"]


def _result_caption(row: dict) -> str | None:
    """What a produced dream hop saw; None for every waking painting."""
    result = row.get("result_json") or {}
    if result.get("dream_hop") is None:
        return None
    caption = result.get("caption")
    if not caption:
        caption = ((row.get("stage_json") or {}).get("caption") or {}).get("description")
    return caption or None


def _attempt_guard(step: _Step, row: dict | None) -> ReverieVisualStepResultV1 | None:
    """None when `row` is this request's open attempt; otherwise the result to return."""
    if row is None or row["dispatch_id"] != step.req.visual_request.dispatch_id:
        return step.terminal("failed", "attempt_mismatch")
    if _dream_hop_mismatch(step.req, row["stage_json"]):
        # Same dispatch, different hop (or a waking request on a dream attempt): never
        # report another hop's picture as this one's.
        return step.terminal("failed", "dispatch_request_mismatch")
    if row["outcome"] not in _OPEN_OUTCOMES:
        return _closed_attempt(step, row)
    return None


# ── prepare ─────────────────────────────────────────────────────────────────


async def _claim_replay(step: _Step, replay: dict, now: datetime) -> ReverieVisualStepResultV1:
    outcome, reason = replay.get("outcome"), replay.get("reason") or "claim_refused"
    if outcome == "already_satisfied":
        return step.terminal("already_satisfied", reason)
    if reason == "dispatch_request_mismatch":
        return step.terminal("failed", reason)
    if outcome == "deferred_busy":
        retry_after = None
        if reason == "retry_cooldown":
            retry_after = await asyncio.to_thread(store.load_visual_retry_after_sec, now)
        return step.retry(reason, retry_after_sec=retry_after)
    if outcome in _RUN_OUTCOMES and outcome != "unknown":
        return step.terminal(outcome, reason)
    return step.retry(reason)


async def prepare_step(bus, req: ReverieVisualStepRequestV1, *,
                       cortex_client: CortexExecClient | None = None, now_fn: Any = _now):
    """Claim the attempt for this dispatch and freeze its prompt plan. No GPU."""
    step = _Step(req)
    request = req.visual_request
    row = await asyncio.to_thread(store.load_visual_attempt_for_dispatch, request)
    if row is None:
        # Before a NEW claim only; replaying an existing claim needs no authorization.
        # Structural eligibility only: validated at the later of the baseline's own
        # observed_at and due_at (the scheduler admits a need once due_at <= its clock,
        # which may be after thought observed activity -- always so on first activation),
        # so a run that waited out retries is never ended as stale or not-due. Freshness
        # is enforced by the claim instead -- `already_satisfied` when another success landed.
        policy = load_baseline_policy()
        baseline = request.visual_baseline
        if baseline:
            if settings.visual_chain_enabled:
                # Operator state, not a property of the request: wait it out.
                return step.retry("legacy_visual_worker_enabled",
                                  retry_after_sec=_LEGACY_WORKER_RETRY_AFTER_SEC)
            reason = validate_eligibility(baseline, policy=policy,
                                          now=max(baseline.observed_at, baseline.due_at))
            if reason:
                return step.terminal("failed", reason)
        now = now_fn()
        if baseline and baseline.due_at > now:
            # Due-ness only ever becomes true, so it is checked against the real clock;
            # early (e.g. clock skew with proposal-runtime) waits rather than failing.
            return step.retry("visual_baseline_not_due",
                              retry_after_sec=(baseline.due_at - now).total_seconds())
        attempt_id, replay = await asyncio.to_thread(
            store.claim_visual_attempt, request, retry_sec=policy.retry_sec, now=now,
            abandoned_in_flight_window_sec=_in_flight_window_sec(),
            attempt_max_age_sec=settings.visual_chain_attempt_max_age_sec,
        )
        if replay is not None:
            # A concurrent prepare for this same dispatch may have claimed first.
            row = await asyncio.to_thread(store.load_visual_attempt_for_dispatch, request)
            if row is None:
                return await _claim_replay(step, replay, now)
        else:
            row = await asyncio.to_thread(store.load_visual_attempt, attempt_id)
            if row is None:
                return step.retry("attempt_missing")
    if row.get("request_mismatch"):
        return step.terminal("failed", "dispatch_request_mismatch")
    if _dream_hop_mismatch(req, row["stage_json"]):
        return step.terminal("failed", "dispatch_request_mismatch")
    if row["outcome"] not in _OPEN_OUTCOMES:
        return _closed_attempt(step, row)
    attempt_id = row["attempt_id"]
    stage = row["stage_json"]
    if _frozen_plan(stage) is not None:
        return step.result("done", attempt_id=attempt_id,
                           elapsed_sec=float(stage.get("prepare_elapsed_sec") or 0.0))

    if req.dream_hop is not None:
        # Never compute_visual_plan: a dream picture neither reads nor advances waking
        # continuity or slot rotation, and its prompt is the dream's, not interpreted.
        plan = _dream_plan(req.dream_hop)
    else:
        plan = await vc.compute_visual_plan(bus, chain_id=attempt_id, cortex_client=cortex_client)
    dream_hop = _dream_hop_identity(req.dream_hop)
    elapsed = time.monotonic() - step.started
    now = now_fn()

    def freeze(current: dict, current_row: dict) -> dict | None:
        if current_row["outcome"] not in _OPEN_OUTCOMES or _frozen_plan(current) is not None:
            return None
        current.update(plan=plan.to_json(), prepared_at=now.isoformat(),
                       prepare_elapsed_sec=round(elapsed, 3))
        if dream_hop is not None:
            current["dream_hop"] = dream_hop
        # Re-freezing an unreadable plan must not rewind a generating/generated attempt:
        # that would skip the in-flight guard or discard a recorded image.
        current.setdefault("stage", "prepared")
        return current

    frozen = await asyncio.to_thread(store.update_visual_stage, attempt_id, freeze)
    if frozen is None:
        return step.terminal("failed", "attempt_mismatch")
    if frozen["outcome"] not in _OPEN_OUTCOMES:
        return _closed_attempt(step, frozen)
    if _frozen_plan(frozen["stage_json"]) is None:
        return step.retry("plan_not_frozen")
    if _dream_hop_mismatch(req, frozen["stage_json"]):
        # A concurrent prepare for this dispatch froze a plan for a different hop.
        return step.terminal("failed", "dispatch_request_mismatch")
    return step.result("done", attempt_id=attempt_id,
                       elapsed_sec=float(frozen["stage_json"].get("prepare_elapsed_sec") or elapsed))


# ── generate ────────────────────────────────────────────────────────────────


async def _generate_and_store(prompt: str, attempt_id: str, hold: GpuLeaseRefV1, bus: Any) -> StoredVisualArtifact:
    # Attached under the run's validated diffusion hold (GPU pool stage 5.4): no second wait, and
    # the child lease outlives a hold the run gives back while diffusion is still running.
    png_bytes = await vc.generate_visual_bytes(prompt, correlation_id=attempt_id, hold=hold, bus=bus)
    return await asyncio.to_thread(
        store_visual_artifact, png_bytes, base_dir=settings.visual_chain_storage_dir
    )


# Generate work outlives the step that started it when the step's deadline passes:
# cancelling cannot stop the diffusion thread, so the task keeps the single-flight
# lock (and runs inside the run's diffusion hold) until the card is actually free, and records its own exit --
# which is what releases an attempt the run abandoned in the meantime.
_generate_tasks: set[asyncio.Task] = set()


def _generate_task_done(task: asyncio.Task) -> None:
    _generate_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error("visual step generate work failed after its step returned",
                     exc_info=task.exception())


def _mark_generating(observed: dict, started_at: datetime):
    """Compare-and-set from the stage the step decided on. A concurrent generate,
    caption reset or abandon in between leaves the row unchanged."""
    def mutate(current: dict, row: dict) -> dict | None:
        if (row["outcome"] not in _OPEN_OUTCOMES or current.get("abandoned_at")
                or current.get("stage") != observed.get("stage")
                or current.get("generating_started_at") != observed.get("generating_started_at")):
            return None
        current["stage"] = "generating"
        current["generating_started_at"] = started_at.isoformat()
        current.pop("artifact", None)
        current.pop("caption", None)
        return current

    return mutate


async def _generate_work(attempt_id: str, plan: vc.VisualPlan, observed: dict, hold: GpuLeaseRefV1, bus: Any,
                         thermal_gate: dict, now_fn: Any) -> tuple[str, Any, float | None]:
    """("generated", StoredVisualArtifact, work_sec) or ("retry", reason, work_sec|None)."""
    async with vc.visual_chain_single_flight(deadline_sec=_in_flight_window_sec()) as held:
        if not held:
            await _record_deferral(attempt_id, "generate", "deferred_busy", now_fn())
            return "retry", "deferred_busy", None
        started_at = now_fn()
        marked = await asyncio.to_thread(store.update_visual_stage, attempt_id,
                                         _mark_generating(observed, started_at))
        if (marked is None or marked["stage_json"].get("stage") != "generating"
                or marked["stage_json"].get("generating_started_at") != started_at.isoformat()):
            return "retry", "stage_changed", None
        work_started = time.monotonic()
        try:
            # Hard ceiling, the same bound past which a claim treats the attempt's GPU
            # work as gone: a diffusion hop can outlive its socket timeout, and an
            # unbounded wait here would wedge the single-flight lock until restart.
            stored = await asyncio.wait_for(_generate_and_store(plan.prompt, attempt_id, hold, bus),
                                            timeout=_in_flight_window_sec())
        except asyncio.TimeoutError:
            logger.error("visual step generate wedged attempt=%s: no diffusion exit after %.0fs",
                         attempt_id, _in_flight_window_sec())
            reason = "generate_wedged"
        except vc.DiffusionResourceDeferred as exc:
            reason = f"resource_deferred:{_detail(exc)}"
        except Exception as exc:
            logger.warning("visual step generation failed attempt=%s err=%s", attempt_id, exc)
            reason = f"generation_failed:{_detail(exc)}"
        else:
            reason = None
        work_elapsed = time.monotonic() - work_started
        if reason is not None:
            # A finished failure never blocks the next generate: back to prepared.
            await _record_deferral(attempt_id, "generate", reason, now_fn(),
                                   stage_name="prepared", release_abandoned=True)
            return "retry", reason, work_elapsed

    record = {
        "sha256": stored.sha256, "path": stored.path, "mime": stored.mime,
        "bytes": stored.bytes, "width": stored.width, "height": stored.height,
        "generated_at": now_fn().isoformat(), "thermal_gate": thermal_gate,
        "elapsed_sec": round(work_elapsed, 3),
    }
    if not await _record_generated(attempt_id, record):
        _unrecorded_renders[attempt_id] = {"started_at": started_at.isoformat(), "artifact": record}
        return "unrecorded", stored, work_elapsed
    _unrecorded_renders.pop(attempt_id, None)
    logger.info("visual step generated attempt=%s sha=%s elapsed=%.1fs",
                attempt_id, stored.sha256[:12], work_elapsed)
    return "generated", stored, work_elapsed


# Renders on disk whose stage_json write failed even after bounded retries, keyed by
# attempt_id: {"started_at", "artifact"}. The next generate for that attempt, still
# `generating` from the same start, adopts the verified file instead of re-rendering.
# In-process only: the step request carries no prior results and the stage row is the
# very thing that could not be written. After a restart the attempt waits out the
# in-flight window and re-renders; the orphan file stays content-addressed on disk.
_unrecorded_renders: dict[str, dict] = {}


def _mark_generated(record: dict, *, expect_started_at: str | None = None):
    def mutate(current: dict, row: dict) -> dict | None:
        if expect_started_at is not None and (
                row["outcome"] not in _OPEN_OUTCOMES or current.get("stage") != "generating"
                or current.get("generating_started_at") != expect_started_at):
            return None
        current["stage"] = "generated"
        current["artifact"] = dict(record)
        current.pop("caption", None)
        return current

    return mutate


async def _record_generated(attempt_id: str, record: dict) -> bool:
    for attempt in range(_STAGE_WRITE_ATTEMPTS):
        try:
            await asyncio.to_thread(store.update_visual_stage, attempt_id, _mark_generated(record),
                                    release_abandoned=True)
            return True
        except Exception as exc:
            logger.warning("visual step: recording image failed attempt=%s sha=%s try=%d/%d err=%s",
                           attempt_id, record["sha256"][:12], attempt + 1, _STAGE_WRITE_ATTEMPTS, exc)
            if attempt + 1 < _STAGE_WRITE_ATTEMPTS:
                await asyncio.sleep(_STAGE_WRITE_BACKOFF_SEC * (attempt + 1))
    logger.error("visual step: image %s on disk but unrecorded attempt=%s",
                 record["sha256"][:12], attempt_id)
    return False


async def _adopt_unrecorded(attempt_id: str, stage: dict) -> dict | None:
    """Artifact record of the unrecorded render now recorded for this attempt, or None."""
    pending = _unrecorded_renders.get(attempt_id)
    if pending is None:
        return None
    if (stage.get("stage") != "generating" or stage.get("generating_started_at") != pending["started_at"]
            or await asyncio.to_thread(_recorded_artifact, pending["artifact"]) is None):
        _unrecorded_renders.pop(attempt_id, None)
        return None
    row = await asyncio.to_thread(
        store.update_visual_stage, attempt_id,
        _mark_generated(pending["artifact"], expect_started_at=pending["started_at"]),
        release_abandoned=True,
    )
    _unrecorded_renders.pop(attempt_id, None)
    artifact = (row or {}).get("stage_json", {}).get("artifact") or {}
    if (row or {}).get("stage_json", {}).get("stage") != "generated" \
            or artifact.get("sha256") != pending["artifact"]["sha256"]:
        return None
    logger.info("visual step adopted unrecorded image attempt=%s sha=%s",
                attempt_id, artifact["sha256"][:12])
    return artifact


async def generate_step(bus, req: ReverieVisualStepRequestV1, *, now_fn: Any = _now):
    """The only GPU stage. Runs under the run's diffusion hold, which is validated first."""
    step = _Step(req)
    try:
        await validate_hold_ref(bus, req.gpu_lease, source=settings.service_name,
                                expected_holder=durable_run_holder(req.run_id))
    except LeaseUnavailable as exc:
        return step.retry(f"hold_invalid:{exc.reason}")

    attempt_id = req.attempt_id
    row = await asyncio.to_thread(store.load_visual_attempt, attempt_id)
    refused = _attempt_guard(step, row)
    if refused is not None:
        return refused
    stage = row["stage_json"]
    plan = _frozen_plan(stage)
    if plan is None:
        return step.retry("not_prepared")

    if stage.get("stage") == "generated":
        recorded = await asyncio.to_thread(_recorded_artifact, stage.get("artifact"))
        if recorded is not None:
            # Replay: the image already exists. Report the GPU time that made it once,
            # so a lost first reply still settles the run's real cost.
            return step.result("done", artifact_sha256=recorded[0].sha256,
                               elapsed_sec=float(stage["artifact"].get("elapsed_sec") or 0.0))
        logger.warning("visual step: recorded image missing attempt=%s; regenerating", attempt_id)

    adopted = await _adopt_unrecorded(attempt_id, stage)
    if adopted is not None:
        # The GPU time was spent by the earlier try; report it, not this step's wall time.
        return step.result("done", artifact_sha256=adopted["sha256"],
                           elapsed_sec=float(adopted.get("elapsed_sec") or 0.0))

    window = _in_flight_window_sec()
    if stage.get("stage") == "generating" and stage.get("generating_started_at"):
        try:
            age = (now_fn() - datetime.fromisoformat(stage["generating_started_at"])).total_seconds()
        except (TypeError, ValueError):
            age = 0.0
        if age < window:
            # An abandoned diffusion thread may still be running; a second call would
            # overlap it on the card.
            return step.retry("generate_in_flight", retry_after_sec=max(0.0, window - age))

    thermal_gate = await vc.thermal_gate_snapshot()
    if not thermal_gate["allows_gpu_work"]:
        await _record_deferral(attempt_id, "generate", "thermal_refused", now_fn())
        return step.retry("thermal_refused", retry_after_sec=_THERMAL_RETRY_AFTER_SEC)

    observed = {"stage": stage.get("stage"), "generating_started_at": stage.get("generating_started_at")}
    task = asyncio.create_task(_generate_work(attempt_id, plan, observed, req.gpu_lease, bus, thermal_gate, now_fn))
    _generate_tasks.add(task)
    task.add_done_callback(_generate_task_done)
    done, _ = await asyncio.wait({task}, timeout=visual_step_generate_deadline_sec())
    if not done:
        # The work keeps running and records its own exit; until then stage stays
        # generating and the in-flight guard keeps a retry off the card.
        reason = "generate_deadline_exceeded"
        await _record_deferral(attempt_id, "generate", reason, now_fn())
        return step.retry(reason, retry_after_sec=window)
    kind, value, work_elapsed = task.result()
    if kind == "generated":
        return step.result("done", artifact_sha256=value.sha256, elapsed_sec=work_elapsed)
    if kind == "unrecorded":
        # The image is on disk; the next generate adopts it (see _unrecorded_renders).
        return step.retry("stage_store_unavailable", artifact_sha256=value.sha256,
                          elapsed_sec=work_elapsed)
    retry_after = _BUSY_RETRY_AFTER_SEC if value == "deferred_busy" else None
    return step.retry(value, retry_after_sec=retry_after, elapsed_sec=work_elapsed)


# ── caption ─────────────────────────────────────────────────────────────────


def _produced_result(step: _Step, row: dict) -> ReverieVisualStepResultV1:
    result = row.get("result_json") or {}
    sha = result.get("artifact_sha256")
    return step.result(
        "done", outcome="produced", reason=result.get("reason") or "max_steps",
        attempt_id=row["attempt_id"], chain_id=_result_chain_id(row),
        artifact_sha256=sha if isinstance(sha, str) and len(sha) == 64 else None,
        execution_receipt=result.get("execution_receipt"), caption=_result_caption(row),
        elapsed_sec=float(row["stage_json"].get("caption_elapsed_sec") or 0.0),
    )


async def caption_step(bus, req: ReverieVisualStepRequestV1, *, now_fn: Any = _now):
    """Re-observe the recorded image and persist the run. No GPU; every write idempotent."""
    step = _Step(req)
    attempt_id = req.attempt_id
    request = req.visual_request
    row = await asyncio.to_thread(store.load_visual_attempt, attempt_id)
    if (row is not None and row["dispatch_id"] == request.dispatch_id and row["outcome"] == "produced"
            and (row.get("result_json") or {}).get("dream_hop") == _dream_hop_identity(req.dream_hop)):
        return _produced_result(step, row)
    refused = _attempt_guard(step, row)
    if refused is not None:
        return refused
    stage = row["stage_json"]
    plan = _frozen_plan(stage)
    if plan is None:
        return step.retry("not_prepared")
    if stage.get("stage") != "generated":
        return step.retry(NEEDS_GENERATE)
    recorded = await asyncio.to_thread(_recorded_artifact, stage.get("artifact"))
    if recorded is None:
        # The image is gone or corrupt: back through the hold to generate.
        now = now_fn()

        def reset(current: dict, _row: dict) -> dict:
            current["stage"] = "prepared"
            current.pop("artifact", None)
            current.pop("caption", None)
            return _with_deferral(current, "caption", "artifact_missing", now)

        await asyncio.to_thread(store.update_visual_stage, attempt_id, reset)
        return step.retry(NEEDS_GENERATE)
    stored, png_bytes = recorded

    cached = stage.get("caption")
    if isinstance(cached, dict) and "description" in cached and (
            req.dream_hop is None or _has_text(cached["description"])):
        description = cached["description"]
    else:
        try:
            upload_result: str | BaseException = await asyncio.to_thread(
                vc.upload_to_percept_store,
                png_bytes,
                base_url=settings.visual_chain_percept_store_url,
                token=settings.visual_chain_percept_store_token,
                timeout_sec=settings.visual_chain_percept_upload_timeout_sec,
            )
        except Exception as exc:
            upload_result = exc
        description = await vc.describe_uploaded_visual(
            bus, upload_result, chain_id=attempt_id, sha256=stored.sha256
        )
        captioned_at = now_fn()
        if req.dream_hop is not None and not _has_text(description):
            # The carry continues from what was seen: no caption is a retry, never a blank
            # hop, and it is not cached, so the retry captions again.
            await _record_deferral(attempt_id, "caption", "caption_empty", captioned_at)
            return step.retry("caption_empty")

        def cache(current: dict, _row: dict) -> dict:
            current["caption"] = {"description": description, "captioned_at": captioned_at.isoformat()}
            return current

        await asyncio.to_thread(store.update_visual_stage, attempt_id, cache)

    if req.dream_hop is not None:
        return await _finish_dream_hop(step, attempt_id, stored, description)

    thermal_gate = stage["artifact"].get("thermal_gate")
    chain = vc.build_production_chain(
        attempt_id, plan, artifact_sha256=stored.sha256, description=description,
        thermal_gate=thermal_gate, run_request=request.model_dump(mode="json"), now_fn=now_fn,
    )
    chain_persisted, production = await vc.persist_visual_production(chain, stored, description)
    if not chain_persisted:
        return step.retry("chain_persist_failed")
    if production is None:
        return step.retry("acknowledge_failed")

    receipt = vc.build_visual_execution_receipt(request, attempt_id, chain, "produced")
    await asyncio.to_thread(store.persist_visual_execution_receipt, attempt_id, receipt)
    elapsed = time.monotonic() - step.started
    result = {"ok": True, "ran": True, "outcome": "produced", "attempt_id": attempt_id,
              "chain_id": attempt_id, "terminal_reason": chain.terminal_reason,
              "reason": receipt.gate_reason, "refused": False, "detail": thermal_gate,
              "artifact_persisted": True, "artifact_sha256": stored.sha256,
              "produced_at": production.produced_at.isoformat(),
              "execution_receipt": receipt.model_dump(mode="json"), "durable_run_id": req.run_id}

    def record_elapsed(current: dict, _row: dict) -> dict:
        current["caption_elapsed_sec"] = round(elapsed, 3)
        return current

    await asyncio.to_thread(store.update_visual_stage, attempt_id, record_elapsed)
    await asyncio.to_thread(store.finish_visual_attempt, attempt_id, result)
    logger.info("visual step produced attempt=%s sha=%s described=%s",
                attempt_id, stored.sha256[:12], bool(description))
    return step.result("done", outcome="produced", reason=receipt.gate_reason,
                       chain_id=attempt_id, artifact_sha256=stored.sha256,
                       execution_receipt=result["execution_receipt"], elapsed_sec=elapsed)


def _has_text(description: Any) -> bool:
    return isinstance(description, str) and bool(description.strip())


async def _finish_dream_hop(step: _Step, attempt_id: str, stored: StoredVisualArtifact,
                            description: str) -> ReverieVisualStepResultV1:
    """Close a dream hop's attempt with what was painted and seen. No chain row, production
    acknowledgement or execution receipt: a dream picture is not a waking painting."""
    req = step.req
    elapsed = time.monotonic() - step.started
    result = {"ok": True, "ran": True, "outcome": "produced", "attempt_id": attempt_id,
              "chain_id": None, "reason": "dream_hop_seen", "artifact_persisted": True,
              "artifact_sha256": stored.sha256, "caption": description, "mime": stored.mime,
              "width": stored.width, "height": stored.height, "durable_run_id": req.run_id,
              "dream_hop": _dream_hop_identity(req.dream_hop)}

    def record_elapsed(current: dict, _row: dict) -> dict:
        current["caption_elapsed_sec"] = round(elapsed, 3)
        return current

    await asyncio.to_thread(store.update_visual_stage, attempt_id, record_elapsed)
    await asyncio.to_thread(store.finish_visual_attempt, attempt_id, result)
    logger.info("visual step dream hop seen attempt=%s carry=%s hop=%d sha=%s",
                attempt_id, req.dream_hop.carry_run_id, req.dream_hop.hop_index, stored.sha256[:12])
    return step.result("done", outcome="produced", reason="dream_hop_seen", chain_id=None,
                       artifact_sha256=stored.sha256, caption=description,
                       execution_receipt=None, elapsed_sec=elapsed)


# ── abandon ─────────────────────────────────────────────────────────────────


async def abandon_step(bus, req: ReverieVisualStepRequestV1, *, now_fn: Any = _now):
    """Close the attempt so it stops blocking later claims. Idempotent.

    Without attempt_id (prepare's reply was lost) the attempt is the one claimed for
    this dispatch; no such row means prepare never claimed, so nothing to close."""
    step = _Step(req)
    row = await asyncio.to_thread(
        store.abandon_visual_attempt, req.attempt_id,
        dispatch_id=req.visual_request.dispatch_id, reason="run_abandoned",
        now=now_fn(), in_flight_window_sec=_in_flight_window_sec(),
        request_json=req.visual_request.model_dump(mode="json"),
    )
    if row is None:
        return step.result("done", reason="attempt_missing")
    if row.get("dispatch_mismatch"):
        return step.terminal("failed", "attempt_mismatch")
    attempt_id = row["attempt_id"]
    if row["outcome"] == "produced":
        result = row.get("result_json") or {}
        return step.result("done", outcome="produced", reason="attempt_already_produced",
                           attempt_id=attempt_id, chain_id=result.get("chain_id") or attempt_id)
    if row["outcome"] == "unknown":
        # A generate may still be on the card: held until it records its own exit.
        return step.result("done", outcome="unknown", reason="generate_in_flight", attempt_id=attempt_id)
    return step.result("done", outcome="unknown", attempt_id=attempt_id,
                       reason=(row.get("result_json") or {}).get("reason") or "run_abandoned")


async def run_visual_step(bus, req: ReverieVisualStepRequestV1, *,
                          cortex_client: CortexExecClient | None = None,
                          now_fn: Any = _now) -> ReverieVisualStepResultV1:
    """Execute one stage. Never raises: a failure is a retry, never a failed run."""
    step = _Step(req)
    try:
        if req.step == "prepare":
            return await prepare_step(bus, req, cortex_client=cortex_client, now_fn=now_fn)
        if req.step == "generate":
            return await generate_step(bus, req, now_fn=now_fn)
        if req.step == "caption":
            return await caption_step(bus, req, now_fn=now_fn)
        return await abandon_step(bus, req, now_fn=now_fn)
    except store.VisualStageStoreUnavailable as exc:
        logger.error("visual step %s refused: stage store unavailable (%s)", req.step, exc)
        return step.retry("stage_store_unavailable")
    except Exception as exc:
        logger.exception("visual step %s failed run=%s attempt=%s", req.step, req.run_id, req.attempt_id)
        return step.retry(f"step_exception:{type(exc).__name__}")
