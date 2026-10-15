"""Resume a killed FCC turn where it left off, instead of from zero.

A durable run retries its harness turn when the governor cancels it, the
service restarts, or the lane lease changes. Until now every retry re-issued
the whole prompt in a fresh `claude -p` process, so a 112-step investigation
that was killed at step 111 started again at step 1.

The `claude` CLI already persists every turn as a transcript on the governor's
named volume (`harness-claude-config:/root/.claude`), and `claude -p --resume
<session-id>` reloads it in a new process. Proved live 2026-10-14: a fresh
process recalled a codeword from the earlier one, and a real transcript that
ended on an unanswered tool call resumed and correctly said where it had been
cut off. This module decides, per turn, whether to start a session or resume
one. It owns no process handling.

`resume_key` is the durable run's own id (stable across attempts). The turn
correlation id is NOT used: it is deliberately fenced per lease generation.

Resume is refused (a fresh session starts instead) when:
  * there is no transcript to resume,
  * the last attempt finished (`completed`) -- a new request is a new turn,
  * the last attempt died on the context ceiling -- resuming a full context
    just dies again; the cut-short finalize owns that case,
  * the transcript would not fit the lane's window now (a new lease can land
    on a smaller one),
  * the run already resumed `max_resumes` times (a poisoned session must not
    loop forever).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# uuid5 namespace: fixed, so a key maps to the same session id in every process.
_NAMESPACE = uuid.UUID("5b0a6f7e-0f3c-4b1d-9a2e-6f1c9d0e7a41")

DEFAULT_MAX_RESUMES = 3

STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_CEILING = "ceiling"
STATE_INTERRUPTED = "interrupted"

_REGISTRY_DIRNAME = "orion-resume"


@dataclass(frozen=True)
class ResumePlan:
    """What `build_claude_argv` should do for this attempt."""

    session_id: str
    resume: bool
    resumes: int
    reason: str
    generation: int = 0

    def metadata(self) -> dict[str, Any]:
        """Trace evidence stamped on the turn's final frame."""
        return {
            "session_id": self.session_id,
            "resumed": self.resume,
            "resumes": self.resumes,
            "generation": self.generation,
            "reason": self.reason,
        }


def claude_home() -> Path:
    from orion.fcc.context_budget import orion_fcc_claude_config_dir

    override = orion_fcc_claude_config_dir()
    return Path(override) if override else Path.home() / ".claude"


def session_id_for(resume_key: str, generation: int = 0) -> str:
    name = resume_key if generation == 0 else f"{resume_key}#{generation}"
    return str(uuid.uuid5(_NAMESPACE, name))


def find_transcript(home: Path, session_id: str) -> Optional[Path]:
    """The CLI files transcripts under `projects/<cwd-slug>/<session-id>.jsonl`."""
    for hit in (home / "projects").glob(f"*/{session_id}.jsonl"):
        return hit
    return None


def _registry_path(home: Path, resume_key: str) -> Path:
    digest = hashlib.sha256(resume_key.encode("utf-8")).hexdigest()[:24]
    return home / _REGISTRY_DIRNAME / f"{digest}.json"


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, path)


def max_resumes() -> int:
    raw = str(os.environ.get("HARNESS_FCC_MAX_SESSION_RESUMES") or "").strip()
    try:
        return int(raw) if raw else DEFAULT_MAX_RESUMES
    except ValueError:
        return DEFAULT_MAX_RESUMES


def plan_attempt(
    resume_key: str,
    *,
    home: Optional[Path] = None,
    window_chars: Optional[int] = None,
    limit: Optional[int] = None,
) -> ResumePlan:
    """Decide start-vs-resume for this attempt and record that it is running.

    The record is written BEFORE the process starts, so a kill at any point
    leaves state `running` -- which is resumable.
    """
    home = home or claude_home()
    limit = max_resumes() if limit is None else limit
    path = _registry_path(home, resume_key)
    record = _read(path)
    generation = int(record.get("generation") or 0)
    resumes = int(record.get("resumes") or 0)
    state = str(record.get("state") or "")
    session_id = session_id_for(resume_key, generation)

    def fresh(reason: str, *, bump: bool) -> ResumePlan:
        gen = generation + 1 if bump else generation
        sid = session_id_for(resume_key, gen)
        _write(path, {"key": resume_key, "generation": gen, "resumes": 0, "state": STATE_RUNNING})
        return ResumePlan(session_id=sid, resume=False, resumes=0, reason=reason, generation=gen)

    if not record:
        return fresh("first_attempt", bump=False)
    if state == STATE_COMPLETED:
        return fresh("previous_attempt_completed", bump=True)
    if state == STATE_CEILING:
        return fresh("previous_attempt_hit_context_ceiling", bump=True)
    if resumes >= limit:
        return fresh("resume_limit_reached", bump=True)
    transcript = find_transcript(home, session_id)
    if transcript is None:
        return fresh("transcript_missing", bump=True)
    if window_chars is not None and transcript.stat().st_size >= window_chars:
        return fresh("transcript_exceeds_lane_window", bump=True)
    _write(
        path,
        {"key": resume_key, "generation": generation, "resumes": resumes + 1, "state": STATE_RUNNING},
    )
    return ResumePlan(
        session_id=session_id,
        resume=True,
        resumes=resumes + 1,
        reason="resumed_after_" + (state or STATE_RUNNING),
        generation=generation,
    )


def mark_end(resume_key: str, state: str, *, home: Optional[Path] = None) -> None:
    """Record how the attempt ended. A process that is killed never reaches
    this, which is exactly what leaves it `running` (resumable)."""
    home = home or claude_home()
    path = _registry_path(home, resume_key)
    record = _read(path)
    if not record:
        return
    record["state"] = state
    _write(path, record)


def continuation_prompt(plan: ResumePlan) -> str:
    """What the resumed session is told. The task itself is already in the
    transcript; repeating it would invite redoing finished steps."""
    return (
        f"[Resumed after an interruption - attempt {plan.resumes + 1}.] Your previous process "
        "was stopped from outside (a restart, a lease change or a cancel), not by you. Your "
        "earlier work and tool results are above. Pick up from your last step: do not redo "
        "steps that already completed, and if a tool call has no result, run it again. "
        "Finish the original task and give your conclusion."
    )
