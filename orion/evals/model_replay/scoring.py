"""Automated scoring of one (task, model) replay, the summary across tasks, and the decision rule.

Decision rule (Juniper, 2026-10-09): Bonsai is eligible to take gpu1 iff
    finish rate >= 90%  AND  zero misreported writes  AND  finish rate within 5 points of Q4.

"Finished" for a task means the turn ended on its own with a non-empty answer, not cut by the
token limit, and -- where the task has a contract -- the answer parses into it with production's
own validators: WorldPulseReadHandoffV1 for reading turns (the pipeline's exact steps), and
ThoughtEventV1 via orion.thought.stance_react.parse_stance_react_payload for stance passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from orion.evals.model_replay.fixture import ReplayTaskV1

FINISH_RATE_MIN = 0.90
MAX_GAP_POINTS = 5.0
MODELS = ("q4", "bonsai")
# Ends that say the replay's own machinery failed (sandbox/scratch graph/pool/worker unreachable,
# run cancelled), not the model. A task with one is void: left out of BOTH models' denominators and
# re-run on resume, so a recalled gpu2 hold cannot fail Bonsai alone.
VOID_ENDS = frozenset({"harness_error", "infra_error", "cancelled"})


@dataclass
class TaskScore:
    task_id: str
    kind: str
    model: str
    end: str = ""
    finished: bool = False
    empty: bool = False
    length_cut: bool = False
    contract_ok: Optional[bool] = None    # None = task has no machine contract
    contract_error: str = ""
    stance_ok: Optional[bool] = None      # harness tasks: did the model's own stance pass parse
    misreported_writes: int = 0
    write_claims: dict[str, Any] = field(default_factory=dict)
    landed: dict[str, Any] = field(default_factory=dict)
    stubbed_writes: list[dict[str, Any]] = field(default_factory=list)
    fetched_ok: Optional[bool] = None
    input_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0
    elapsed_sec: float = 0.0
    final_chars: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_reading(text: str, task: ReplayTaskV1) -> tuple[bool, str]:
    """world_pulse_read_pipeline._stage1_read's parse+validate, minus the server-side evidence gate."""
    from orion.core.llm_json import parse_json_object
    from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1

    try:
        parsed = parse_json_object(text)
        parsed["trace_id"] = parsed.get("trace_id") or "replay"
        if task.reading_seed is not None:
            parsed["seed_ref"] = task.reading_seed.model_dump(mode="json")
        parsed.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        parsed["producer_hint"] = "world_pulse_read_pipeline"
        parsed["read_evidence"] = []
        WorldPulseReadHandoffV1.model_validate(parsed)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:300]


def validate_thought(text: str) -> tuple[bool, str]:
    from orion.thought.stance_react import parse_stance_react_payload

    try:
        parse_stance_react_payload(text, correlation_id="replay", session_id="replay")
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:300]


def score_contract(task: ReplayTaskV1, text: str) -> tuple[Optional[bool], str]:
    if task.expect == "reading_handoff_json":
        return validate_reading(text, task)
    if task.expect == "thought_json":
        return validate_thought(text)
    return None, ""


def finish(score: TaskScore) -> TaskScore:
    score.finished = score.end == "finished" and not score.empty and score.contract_ok is not False
    return score


def summarize(scores: list[TaskScore]) -> dict[str, Any]:
    void = sorted({s.task_id for s in scores if s.end in VOID_ENDS})
    per_model: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        rows = [s for s in scores if s.model == m and s.task_id not in void]
        n = len(rows)
        fin = sum(s.finished for s in rows)
        by_kind: dict[str, dict[str, int]] = {}
        for s in rows:
            k = by_kind.setdefault(s.kind, {"n": 0, "finished": 0})
            k["n"] += 1
            k["finished"] += int(s.finished)
        per_model[m] = {
            "tasks": n,
            "finished": fin,
            "finish_rate": (fin / n) if n else None,
            "misreported_writes": sum(s.misreported_writes for s in rows),
            "tasks_with_misreported_writes": [s.task_id for s in rows if s.misreported_writes],
            "unmentioned_writes": sum(len(s.write_claims.get("unmentioned") or []) for s in rows),
            "empty": sum(s.empty for s in rows),
            "length_cut": sum(s.length_cut for s in rows),
            "contract_failures": [s.task_id for s in rows if s.contract_ok is False],
            "ends": _count(s.end for s in rows),
            "output_tokens": sum(s.output_tokens for s in rows),
            "elapsed_sec": round(sum(s.elapsed_sec for s in rows), 1),
            "by_kind": by_kind,
        }
    decision = decide(per_model)
    if void:
        decision["eligible"] = False
        decision["verdict"] = (f"PARTIAL: {len(void)} task(s) void (replay machinery failed, not the model): "
                               f"re-run with --out; so far: {decision['verdict']}")
    return {"per_model": per_model, "void_tasks": void, "decision": decision}


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def decide(per_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    q4, bon = per_model.get("q4") or {}, per_model.get("bonsai") or {}
    fr_b, fr_q = bon.get("finish_rate"), q4.get("finish_rate")
    if fr_b is None or fr_q is None:
        return {"eligible": False, "verdict": "INCOMPLETE: a model has no scored tasks", "checks": {}}
    gap = (fr_q - fr_b) * 100.0
    checks = {
        "bonsai_finish_rate_ge_90": fr_b >= FINISH_RATE_MIN,
        "bonsai_zero_misreported_writes": bon.get("misreported_writes", 0) == 0,
        "bonsai_within_5_points_of_q4": gap <= MAX_GAP_POINTS,
    }
    eligible = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    verdict = (f"ELIGIBLE: Bonsai may replace Q4 on gpu1 (finish {fr_b:.0%} vs Q4 {fr_q:.0%}, "
               f"0 misreported writes)") if eligible else \
        (f"NOT ELIGIBLE: failed {', '.join(failed)} (Bonsai finish {fr_b:.0%}, Q4 {fr_q:.0%}, "
         f"gap {gap:+.1f} pts, Bonsai misreported writes {bon.get('misreported_writes', 0)})")
    return {"eligible": eligible, "verdict": verdict, "checks": checks, "gap_points": round(gap, 1),
            "bonsai_finish_rate": fr_b, "q4_finish_rate": fr_q}
