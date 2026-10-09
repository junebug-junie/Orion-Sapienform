"""Orchestrate the Bonsai-vs-Q4 replay: plan, pin both cards per task, run both models side by side
through the recording sandbox, score, and write the results directory.

Per task: take the gpu1 hold (Q4) and the gpu2 holds (Bonsai) through the pool's operator verbs,
verify each grant is the expected role AND profile, run the task on both models concurrently,
release both holds (also on error, Ctrl-C, SIGTERM/SIGHUP), score, append to results.jsonl.
Between tasks the cards go back to normal traffic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from orion.evals.model_replay import agent_loop, scoring, write_claims
from orion.evals.model_replay.fixture import FIXTURE_PATH, ReplayTaskV1
from orion.evals.model_replay.pool_hold import (
    ACQUIRE_ORDER, SEATS, ControlTransport, Grant, HoldLedger, HoldRefused, PoolHolds,
)
from orion.evals.model_replay.sandbox import FetchCache, ToolLog, Toolbox, sandbox_env

# Rough per-kind wall time per model, from the 2026-10-07..09 window (runs_attribution / terminal
# events / stance telemetry). Only used for the dry-run estimate.
EXPECTED_SEC = {"curiosity": 3600.0, "self_sense": 400.0, "reading": 420.0, "stance_react": 180.0}
STANCE_PASS_TIMEOUT_SEC = 240.0
STEP_STALL_SEC = 420.0
# Set when the run is cancelled (Ctrl-C / SIGTERM / SIGHUP): worker threads stop before their next
# model call, so nothing keeps generating on a card whose hold was just released.
STOP = threading.Event()


@dataclass
class ReplayConfig:
    out_dir: Path
    tasks: list[ReplayTaskV1]
    repo: Path
    models: tuple[str, ...] = ACQUIRE_ORDER
    dry_run: bool = False
    grant_timeout_sec: float = 1800.0
    seed: int = 20261009
    fixture_path: Path = FIXTURE_PATH


@dataclass
class ModelRig:
    """Everything one model needs for one task. Built by RigFactory (tests inject fakes)."""

    client: agent_loop.MessagesClient
    toolbox_factory: Callable[[ReplayTaskV1, ToolLog], Toolbox]
    graphs: Any = None              # graph_scratch.ScratchGraphs or None
    cleanup: Callable[[], None] = lambda: None


RigFactory = Callable[[ReplayTaskV1, str, Grant], ModelRig]


def plan(cfg: ReplayConfig) -> dict[str, Any]:
    tasks = [{"task_id": t.task_id, "kind": t.kind, "timeout_sec": t.timeout_sec,
              "expected_sec": EXPECTED_SEC[t.kind], "fetch": t.fetch.mode, "expect": t.expect,
              "write_claim_check": t.write_claim_check} for t in cfg.tasks]
    expected = sum(EXPECTED_SEC[t.kind] for t in cfg.tasks)
    worst = sum(t.timeout_sec + (STANCE_PASS_TIMEOUT_SEC if t.kind != "stance_react" else 0) for t in cfg.tasks)
    return {
        "tasks": tasks,
        "counts": _count(t.kind for t in cfg.tasks),
        "models": {m: {"work_class": SEATS[m].work_class, "role": SEATS[m].expect_role,
                       "profile": SEATS[m].expect_profile, "extra_slot_holds": SEATS[m].extra_slot_holds}
                   for m in cfg.models},
        "cards": "gpu1 (agent, :8015) and gpu2 (agent-gpu2, :8016), held together for each task, released between tasks",
        "expected_wall_hours": round(expected / 3600.0, 1),
        "worst_case_wall_hours": round(worst / 3600.0, 1),
        "fixture_sha256": hashlib.sha256(cfg.fixture_path.read_bytes()).hexdigest() if cfg.fixture_path.exists() else None,
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def neutral_thought(task: ReplayTaskV1):
    from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1

    return ThoughtEventV1.model_validate({
        "event_id": f"replay-{task.task_id}", "correlation_id": f"replay-{task.task_id}",
        "session_id": "model-replay", "created_at": datetime.now(timezone.utc),
        "imperative": "Carry out the task in the user message.", "tone": "direct",
        "strain_refs": [], "evidence_refs": [],
        "stance_harness_slice": StanceHarnessSliceV1(task_mode="direct_response", conversation_frame="mixed",
                                                     answer_strategy="direct"),
    })


def harness_prompt(task: ReplayTaskV1, stance_text: str) -> tuple[str, bool]:
    """Production's prompt compiler over the model's own stance output (neutral thought if it did
    not parse -- recorded as stance_ok=False, the same degrade the live turn takes)."""
    from orion.harness.runner import build_harness_prompt
    from orion.schemas.harness_finalize import HarnessRepairOverlayV1
    from orion.thought.stance_react import parse_stance_react_payload

    try:
        thought = parse_stance_react_payload(stance_text, correlation_id=f"replay-{task.task_id}",
                                             session_id="model-replay")
        ok = True
    except Exception:  # noqa: BLE001
        thought, ok = neutral_thought(task), False
    prompt = build_harness_prompt(thought=thought, user_message=task.user_message,
                                  repair_overlay=HarnessRepairOverlayV1.model_validate({"mode": "default"}),
                                  reading_only=task.reading_only)
    return prompt, ok


def run_one(task: ReplayTaskV1, model: str, rig: ModelRig, out_dir: Path) -> scoring.TaskScore:
    log = ToolLog()
    score = scoring.TaskScore(task_id=task.task_id, kind=task.kind, model=model)
    transcript: dict[str, Any] = {"task_id": task.task_id, "model": model}
    try:
        toolbox = rig.toolbox_factory(task, log)
        before = None
        if rig.graphs is not None and task.write_claim_check:
            from orion.evals.model_replay.graph_scratch import prior_snapshot

            before = prior_snapshot(rig.graphs)  # a fresh copy: every task starts from the same graphs
        stance = agent_loop.run_loop(rig.client, user_message=task.stance_prompt, tools=[], system=None,
                                     call_tool=toolbox.call, timeout_sec=min(task.timeout_sec, STANCE_PASS_TIMEOUT_SEC),
                                     should_stop=STOP.is_set)
        transcript["stance"] = _loop_dict(stance)
        if task.kind == "stance_react":
            result = stance
        else:
            prompt, score.stance_ok = harness_prompt(task, stance.final_text)
            transcript["harness_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            result = agent_loop.run_loop(rig.client, user_message=prompt, tools=list(task.tools),
                                         call_tool=toolbox.call, timeout_sec=task.timeout_sec,
                                         should_stop=STOP.is_set)
            transcript["harness"] = _loop_dict(result)
        score.end = result.end
        score.empty = not result.final_text.strip()
        score.length_cut = result.end == "length_cut"
        score.steps = len(result.steps) + (len(stance.steps) if result is not stance else 0)
        score.input_tokens = result.input_tokens + (stance.input_tokens if result is not stance else 0)
        score.output_tokens = result.output_tokens + (stance.output_tokens if result is not stance else 0)
        score.elapsed_sec = round(result.elapsed_sec + (stance.elapsed_sec if result is not stance else 0), 1)
        score.final_chars = len(result.final_text)
        if result.end == "finished" and result.final_text.strip():
            score.contract_ok, score.contract_error = scoring.score_contract(task, result.final_text)
        if task.kind == "reading":
            fetches = [e for e in log.events if e.tool == "WebFetch"]
            score.fetched_ok = any(e.detail.get("ok") for e in fetches)
        if before is not None:
            from orion.evals.model_replay.graph_scratch import diff_snapshots, prior_snapshot

            score.landed = diff_snapshots(before, prior_snapshot(rig.graphs))
            wc = write_claims.check(result.final_text, score.landed)
            score.write_claims = wc.as_dict()
            score.misreported_writes = wc.misreported
        score.stubbed_writes = [e.__dict__ for e in log.writes() if e.action == "write_stubbed"]
        transcript["final_text"] = result.final_text
    except Exception as exc:  # noqa: BLE001 -- one task's crash is a scored failure, not a stopped run
        score.end = score.end or "harness_error"
        score.error = f"{type(exc).__name__}: {exc}"[:800]
    finally:
        try:
            rig.cleanup()
        except Exception:  # noqa: BLE001
            pass
        transcript["tool_log"] = log.as_dicts()
        transcript["score"] = scoring.finish(score).as_dict()
        tdir = out_dir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{task.task_id}.{model}.json").write_text(json.dumps(transcript, default=str, indent=1),
                                                         encoding="utf-8")
    return score


def _loop_dict(r: agent_loop.LoopResult) -> dict[str, Any]:
    return {"end": r.end, "stop_reason": r.stop_reason, "elapsed_sec": r.elapsed_sec,
            "steps": [s.__dict__ for s in r.steps], "final_text": r.final_text,
            "messages": r.messages[1:]}  # the first user message is the prompt, recorded elsewhere


def done_task_ids(out_dir: Path) -> set[str]:
    path = out_dir / "results.jsonl"
    if not path.exists():
        return set()
    by_task: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_task.setdefault(row["task_id"], set()).add(row["model"])
    return {t for t, ms in by_task.items() if {"q4", "bonsai"} <= ms}


def load_scores(out_dir: Path) -> list[scoring.TaskScore]:
    path = out_dir / "results.jsonl"
    if not path.exists():
        return []
    return [scoring.TaskScore(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


async def run_replay(cfg: ReplayConfig, transport: ControlTransport, rig_factory: RigFactory,
                     *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "plan.json").write_text(json.dumps(plan(cfg), indent=1), encoding="utf-8")
    ledger = HoldLedger(cfg.out_dir / "holds.jsonl")
    done = done_task_ids(cfg.out_dir)
    for i, task in enumerate(cfg.tasks, start=1):
        if task.task_id in done:
            log(f"[{i}/{len(cfg.tasks)}] {task.task_id}: already scored, skipping")
            continue
        log(f"[{i}/{len(cfg.tasks)}] {task.task_id} ({task.kind}): taking holds")
        t0 = time.time()
        try:
            async with PoolHolds(transport=transport, ledger=ledger, models=cfg.models,
                                 grant_timeout_sec=cfg.grant_timeout_sec) as seats:
                log("    holds: " + ", ".join(f"{m}={g.role}/{g.profile_name}@{g.url}" for m, g in seats.items()))
                rigs: dict[str, ModelRig] = {}
                try:
                    for m in cfg.models:
                        rigs[m] = rig_factory(task, m, seats[m])
                except BaseException:
                    for rig in rigs.values():
                        rig.cleanup()
                    raise
                try:
                    scores = await asyncio.gather(*(asyncio.to_thread(run_one, task, m, rigs[m], cfg.out_dir)
                                                    for m in cfg.models))
                except BaseException:
                    STOP.set()
                    raise
        except HoldRefused as exc:
            # Every hold is already released. Stop rather than run half a comparison; resume with --out.
            log(f"    STOPPED: {exc}. Holds released. Resume later with --out {cfg.out_dir}")
            break
        with (cfg.out_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
            for s in scores:
                fh.write(json.dumps(s.as_dict(), default=str) + "\n")
        log(f"    released after {time.time() - t0:.0f}s: " +
            ", ".join(f"{s.model}={'ok' if s.finished else s.end}" +
                      (f" misreported={s.misreported_writes}" if s.misreported_writes else "") for s in scores))
    return finalize(cfg)


def finalize(cfg: ReplayConfig) -> dict[str, Any]:
    from orion.evals.model_replay import report

    scores = load_scores(cfg.out_dir)
    summary = scoring.summarize(scores)
    scored = {s.task_id for s in scores}
    missing = [t.task_id for t in cfg.tasks if t.task_id not in scored]
    if missing:
        d = summary["decision"]
        d["eligible"] = False
        d["verdict"] = (f"PARTIAL: {len(missing)} of {len(cfg.tasks)} tasks not scored yet, so no verdict "
                        f"(so far: {d['verdict']})")
        d["missing_tasks"] = missing
    summary["plan"] = plan(cfg)
    summary["repo_head"] = _git_head(cfg.repo)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["tasks"] = [s.as_dict() for s in scores]
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    report.write_report(cfg.out_dir, summary, scores, cfg.tasks, random.Random(cfg.seed))
    return summary


def _git_head(repo: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def install_signal_unwind(loop: asyncio.AbstractEventLoop, main: "asyncio.Task[Any]") -> None:
    """SIGTERM/SIGHUP cancel the main task, so every `async with PoolHolds` releases on the way out."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, main.cancel)
        except (NotImplementedError, RuntimeError):
            pass


@dataclass
class LiveRigFactory:
    """Real rigs: a worker client on the grant's URL, a sandbox container, and a fresh scratch
    FalkorDB loaded with the start-of-replay graph copies (tasks that have shell tools)."""

    repo: Path
    run_tag: str
    sandbox_image: str
    graph_image: str
    graph_dumps: dict[str, bytes]
    sql: Any
    http: Any
    docker_ro: Any
    fetch_cache: FetchCache

    def __call__(self, task: ReplayTaskV1, model: str, grant: Grant) -> ModelRig:
        from orion.evals.model_replay.graph_scratch import ScratchGraphs
        from orion.evals.model_replay.sandbox import ShellSandbox, sandbox_name

        holder: dict[str, Any] = {}
        graphs = None
        if "Bash" in task.tools:
            graphs = ScratchGraphs.fresh(self.graph_image, self.graph_dumps,
                                         name=sandbox_name(self.run_tag, task.task_id, f"{model}-graph"))
            holder["graphs"] = graphs

        def toolbox_factory(t: ReplayTaskV1, log: ToolLog) -> Toolbox:
            env = sandbox_env(deadline_epoch=time.time() + t.timeout_sec + STANCE_PASS_TIMEOUT_SEC,
                              budget_sec=t.timeout_sec, stall_sec=STEP_STALL_SEC)
            shell = ShellSandbox(repo=self.repo, image=self.sandbox_image,
                                 name=sandbox_name(self.run_tag, t.task_id, model), env=env)
            if t.tools and not t.reading_only:
                shell.start()
            holder["shell"] = shell
            return Toolbox(log=log, shell=shell, graph=graphs, sql=self.sql,
                           http=self.http, docker_ro=self.docker_ro, fetch_cache=self.fetch_cache,
                           allowed=tuple(t.tools), fetch_mode=t.fetch.mode, fetch_fail_error=t.fetch.fail_error)

        def cleanup() -> None:
            for part in ("shell", "graphs"):
                if part in holder:
                    holder[part].stop()

        return ModelRig(client=agent_loop.HttpMessagesClient(grant.url), toolbox_factory=toolbox_factory,
                        graphs=graphs, cleanup=cleanup)
