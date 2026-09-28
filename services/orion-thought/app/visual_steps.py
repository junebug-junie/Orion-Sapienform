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
from orion.schemas.reverie_visual import VisualRunOutcome
from orion.schemas.reverie_visual_run import (
    NEEDS_GENERATE,
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
_REASON_DETAIL_CHARS = 200


def _now() -> datetime:
    return datetime.now(timezone.utc)


def visual_step_generate_deadline_sec() -> float:
    """Generate's own deadline, never below permit wait + diffusion timeout + margin."""
    floor = (settings.visual_chain_gpu2_capacity_budget_sec
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
            chain_id=result.get("chain_id") or attempt_id,
            artifact_sha256=sha if isinstance(sha, str) and len(sha) == 64 else None,
            execution_receipt=result.get("execution_receipt"),
        )
    if row["outcome"] == store.VISUAL_ATTEMPT_ABANDONED:
        return step.terminal("unknown", result.get("reason") or "run_abandoned", attempt_id=attempt_id)
    outcome = row["outcome"] if row["outcome"] in _RUN_OUTCOMES else "failed"
    return step.terminal(outcome, result.get("reason") or "attempt_closed", attempt_id=attempt_id)


def _attempt_guard(step: _Step, row: dict | None) -> ReverieVisualStepResultV1 | None:
    """None when `row` is this request's open attempt; otherwise the result to return."""
    if row is None or row["dispatch_id"] != step.req.visual_request.dispatch_id:
        return step.terminal("failed", "attempt_mismatch")
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
        # Same authority checks as /visual-chain/run-once, before a NEW claim only:
        # replaying an existing claim never needs fresh authorization.
        policy = load_baseline_policy()
        if request.visual_baseline:
            reason = validate_eligibility(request.visual_baseline, policy=policy)
            if reason or settings.visual_chain_enabled:
                return step.terminal("failed", reason or "legacy_visual_worker_enabled")
        now = now_fn()
        attempt_id, replay = await asyncio.to_thread(
            store.claim_visual_attempt, request, retry_sec=policy.retry_sec, now=now,
            abandoned_in_flight_window_sec=_in_flight_window_sec(),
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
    if row["outcome"] not in _OPEN_OUTCOMES:
        return _closed_attempt(step, row)
    attempt_id = row["attempt_id"]
    stage = row["stage_json"]
    if _frozen_plan(stage) is not None:
        return step.result("done", attempt_id=attempt_id,
                           elapsed_sec=float(stage.get("prepare_elapsed_sec") or 0.0))

    plan = await vc.compute_visual_plan(bus, chain_id=attempt_id, cortex_client=cortex_client)
    elapsed = time.monotonic() - step.started
    now = now_fn()

    def freeze(current: dict, current_row: dict) -> dict | None:
        if current_row["outcome"] not in _OPEN_OUTCOMES or _frozen_plan(current) is not None:
            return None
        current.update(plan=plan.to_json(), stage="prepared", prepared_at=now.isoformat(),
                       prepare_elapsed_sec=round(elapsed, 3))
        return current

    frozen = await asyncio.to_thread(store.update_visual_stage, attempt_id, freeze)
    if frozen is None:
        return step.terminal("failed", "attempt_mismatch")
    if frozen["outcome"] not in _OPEN_OUTCOMES:
        return _closed_attempt(step, frozen)
    if _frozen_plan(frozen["stage_json"]) is None:
        return step.retry("plan_not_frozen")
    return step.result("done", attempt_id=attempt_id,
                       elapsed_sec=float(frozen["stage_json"].get("prepare_elapsed_sec") or elapsed))


# ── generate ────────────────────────────────────────────────────────────────


async def _generate_and_store(prompt: str, attempt_id: str) -> StoredVisualArtifact:
    png_bytes = await vc.generate_visual_bytes(prompt, correlation_id=attempt_id)
    return await asyncio.to_thread(
        store_visual_artifact, png_bytes, base_dir=settings.visual_chain_storage_dir
    )


# Generate work outlives the step that started it when the step's deadline passes:
# cancelling cannot stop the diffusion thread, so the task keeps the single-flight
# lock and GPU2 permit until the card is actually free, and records its own exit --
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


async def _generate_work(attempt_id: str, plan: vc.VisualPlan, observed: dict,
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
            stored = await asyncio.wait_for(_generate_and_store(plan.prompt, attempt_id),
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

    generated_at = now_fn()

    def mark_generated(current: dict, _row: dict) -> dict:
        current["stage"] = "generated"
        current["artifact"] = {
            "sha256": stored.sha256, "path": stored.path, "mime": stored.mime,
            "bytes": stored.bytes, "width": stored.width, "height": stored.height,
            "generated_at": generated_at.isoformat(), "thermal_gate": thermal_gate,
            "elapsed_sec": round(work_elapsed, 3),
        }
        current.pop("caption", None)
        return current

    await asyncio.to_thread(store.update_visual_stage, attempt_id, mark_generated,
                            release_abandoned=True)
    logger.info("visual step generated attempt=%s sha=%s elapsed=%.1fs",
                attempt_id, stored.sha256[:12], work_elapsed)
    return "generated", stored, work_elapsed


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
    task = asyncio.create_task(_generate_work(attempt_id, plan, observed, thermal_gate, now_fn))
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
    retry_after = _BUSY_RETRY_AFTER_SEC if value == "deferred_busy" else None
    return step.retry(value, retry_after_sec=retry_after, elapsed_sec=work_elapsed)


# ── caption ─────────────────────────────────────────────────────────────────


def _produced_result(step: _Step, row: dict) -> ReverieVisualStepResultV1:
    result = row.get("result_json") or {}
    sha = result.get("artifact_sha256")
    return step.result(
        "done", outcome="produced", reason=result.get("reason") or "max_steps",
        attempt_id=row["attempt_id"], chain_id=result.get("chain_id") or row["attempt_id"],
        artifact_sha256=sha if isinstance(sha, str) and len(sha) == 64 else None,
        execution_receipt=result.get("execution_receipt"),
        elapsed_sec=float(row["stage_json"].get("caption_elapsed_sec") or 0.0),
    )


async def caption_step(bus, req: ReverieVisualStepRequestV1, *, now_fn: Any = _now):
    """Re-observe the recorded image and persist the run. No GPU; every write idempotent."""
    step = _Step(req)
    attempt_id = req.attempt_id
    request = req.visual_request
    row = await asyncio.to_thread(store.load_visual_attempt, attempt_id)
    if row is not None and row["dispatch_id"] == request.dispatch_id and row["outcome"] == "produced":
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
    if isinstance(cached, dict) and "description" in cached:
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

        def cache(current: dict, _row: dict) -> dict:
            current["caption"] = {"description": description, "captioned_at": captioned_at.isoformat()}
            return current

        await asyncio.to_thread(store.update_visual_stage, attempt_id, cache)

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


# ── abandon ─────────────────────────────────────────────────────────────────


async def abandon_step(bus, req: ReverieVisualStepRequestV1, *, now_fn: Any = _now):
    """Close the attempt so it stops blocking later claims. Idempotent."""
    step = _Step(req)
    row = await asyncio.to_thread(
        store.abandon_visual_attempt, req.attempt_id,
        dispatch_id=req.visual_request.dispatch_id, reason="run_abandoned",
        now=now_fn(), in_flight_window_sec=_in_flight_window_sec(),
    )
    if row is None:
        return step.result("done", reason="attempt_missing")
    if row.get("dispatch_mismatch"):
        return step.terminal("failed", "attempt_mismatch")
    if row["outcome"] == "produced":
        result = row.get("result_json") or {}
        return step.result("done", outcome="produced", reason="attempt_already_produced",
                           chain_id=result.get("chain_id") or row["attempt_id"])
    if row["outcome"] == "unknown":
        # A generate may still be on the card: held until it records its own exit.
        return step.result("done", outcome="unknown", reason="generate_in_flight")
    return step.result("done", outcome="unknown", reason=(row.get("result_json") or {}).get("reason")
                       or "run_abandoned")


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
