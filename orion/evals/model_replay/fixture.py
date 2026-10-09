"""The committed replay task set: what each task is, where it came from, and how it is scored.

A task is frozen at extraction time (`scripts/extract_model_replay_fixture.py`) from real history,
so a replay months later asks both models the exact same thing. The fixture holds only what the
task needs: the prompt the model saw (or the inputs to rebuild it with production code), the
historical outcome for orientation, and nothing else from the source rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tasks.v1.jsonl"

TaskKind = Literal["curiosity", "self_sense", "reading", "stance_react"]

# Tool surfaces, mirroring orion/harness/fcc_motor.py build_claude_argv: reading turns get
# WebFetch/WebSearch only (`--tools WebFetch,WebSearch`); every other harness turn gets the
# Claude Code shell/file tools. stance_react is a single tool-less completion (cortex-exec).
HARNESS_TOOLS = ("Bash", "Read", "Grep", "Glob", "WebFetch", "WebSearch")
READING_TOOLS = ("WebFetch", "WebSearch")


class ReadingSeed(BaseModel):
    """Enough of WorldPulseReadSeedV1 to validate the handoff exactly as the pipeline does."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str
    kind: Literal["finding", "digest_item", "reading"]
    run_id: str
    url: str
    title: str = ""
    section: str = ""


class FetchPlan(BaseModel):
    """How WebFetch behaves for this task. ``fail`` replays the historical fetch failure verbatim
    for every URL, so both models face the same broken source; ``live`` fetches once per URL per
    replay (HTTP GET) and serves the same cached text to both models."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["live", "fail"] = "live"
    fail_error: str = ""


class HistoricalOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = ""            # agent (Q4) | agent-gpu2 (Bonsai) | mixed | unknown
    outcome: str = ""         # completed | failed | ...
    note: str = ""


class ReplayTaskV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["model_replay.task.v1"] = "model_replay.task.v1"
    task_id: str
    kind: TaskKind
    source: str                         # e.g. durable_admission_runs:d4db8c2bacb4
    # The user message of the turn: the frozen curiosity brief, the self-sense question, the
    # reading stage-1 prompt, or (stance_react tasks) the user_message the stance pass assesses.
    user_message: str
    # stance_react tasks: the full rendered stance prompt (orion/cognition/prompts/stance_react.j2).
    # Harness tasks: the stance prompt rendered for this user_message; the replay runs it first,
    # exactly as execute_unified_turn does, and builds the harness prompt from the model's own
    # thought with orion.harness.runner.build_harness_prompt.
    stance_prompt: str
    tools: list[str] = Field(default_factory=list)
    reading_only: bool = False
    reading_seed: Optional[ReadingSeed] = None
    fetch: FetchPlan = Field(default_factory=FetchPlan)
    # Wall-clock budget per model for this task, seconds (the harness sees it as ORION_TURN_*).
    timeout_sec: float
    expect: Literal["prose", "reading_handoff_json", "thought_json"] = "prose"
    # Curiosity runs are told to write priors into their own graph; the write-claim check runs.
    write_claim_check: bool = False
    snapshot: str = ""                  # self_sense: which lived-answer snapshot
    historical: HistoricalOutcome = Field(default_factory=HistoricalOutcome)


def load_tasks(path: Path = FIXTURE_PATH) -> list[ReplayTaskV1]:
    tasks = [ReplayTaskV1.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task_id in fixture")
    return tasks


def dump_tasks(tasks: list[ReplayTaskV1], path: Path = FIXTURE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
                            for t in tasks), encoding="utf-8")
