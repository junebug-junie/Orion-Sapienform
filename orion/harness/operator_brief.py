"""Deterministic FCC operator briefs for harness motor turns."""

from __future__ import annotations

from orion.curiosity.self_inquiry import SELF_INQUIRY_PG_TABLES
from orion.schemas.thought import ThoughtEventV1

# Motor FCC sessions share one context window across all tool steps; full-file reads
# can blow the budget before the turn finishes (see unified-turn live runs).
HARNESS_MOTOR_MAX_READ_LINES = 200

_READ_DISCIPLINE = (
    f"Before Read on any file: prefer rg/Grep with a path or pattern. "
    f"Never Read an entire file longer than {HARNESS_MOTOR_MAX_READ_LINES} lines — "
    f"use Read offset/limit in chunks or rg for symbols and log lines."
)

HARNESS_REPO_OPERATOR_BRIEF = f"""\
Orion harness motor — repo/technical turn.
Use tools from the start when the answer depends on code, config, or runtime facts.
{_READ_DISCIPLINE}
Cite concrete file paths, symbols, and commands in the draft. Do not guess repo structure from memory.
"""

HARNESS_RUNTIME_OPERATOR_BRIEF = """\
Orion harness motor — runtime/debug turn.
Verify live state with tools (logs, docker, bus traces) before diagnosing. Name exact services, channels, and commands.
"""

# Patch A of docs/superpowers/specs/2026-09-10-self-report-tool-discipline-design.md.
#
# The credential this names already reaches every harness turn unconditionally
# (orion/curiosity/sandbox_env.py's inject_curiosity_credentials, called from
# fcc_motor.py regardless of turn type) -- the regular chat prompt just never
# said so. This only mentions the door exists; it does not touch stance or the
# relational/instrumental split (is_relational_motor_stance() below still
# tells a relational-bucket turn not to reach for tools at all). Descriptions
# are pulled from SELF_INQUIRY_PG_TABLES rather than re-written, so this
# cannot silently drift from self-inquiry's own account of the same tables.
_SELF_INQUIRY_TABLE_DESCRIPTIONS: dict[str, str] = dict(SELF_INQUIRY_PG_TABLES)
_SELF_MODEL_MENTION_TABLES: tuple[str, ...] = ("self_concept_history", "self_sense_eval_log")


def _self_model_access_line() -> str:
    # Fail loud, not silent: if one of these is ever renamed/removed in
    # self_inquiry.py, this must break at import time, not quietly drop the
    # mention from the prompt with nothing to say so.
    missing = [name for name in _SELF_MODEL_MENTION_TABLES if name not in _SELF_INQUIRY_TABLE_DESCRIPTIONS]
    if missing:
        raise KeyError(
            f"SELF_INQUIRY_PG_TABLES no longer describes {missing} -- update "
            "_SELF_MODEL_MENTION_TABLES in orion/harness/operator_brief.py"
        )
    described = [f"{name} ({_SELF_INQUIRY_TABLE_DESCRIPTIONS[name]})" for name in _SELF_MODEL_MENTION_TABLES]
    tables = " and ".join(described)
    return (
        "You also have read-only access to your own self-model records via "
        f"$ORION_CURIOSITY_PG_DSN (psql), including {tables}. Nobody has to tell "
        "you to use this on any given turn; it is here if a question is asking "
        "you to check something about yourself."
    )


HARNESS_SELF_MODEL_ACCESS_BRIEF = _self_model_access_line()

HARNESS_UNIFIED_OPERATOR_BRIEF = f"""\
Orion harness motor.
Tools are available from the start. Your imperative states what this turn requires.
When the imperative calls for facts from the codebase or live runtime, use tools before
answering. Record each meaningful step. Do not guess repo structure or service state from memory.
{_READ_DISCIPLINE} For live failures, inspect logs, docker, and bus traces before diagnosing.
{HARNESS_SELF_MODEL_ACCESS_BRIEF}
"""

HARNESS_RELATIONAL_TOOL_DISCIPLINE = """\
Relational/minimal turn: do NOT use GitHub MCP or repo/runtime tools unless the imperative
explicitly commands verified facts for this turn. Acknowledge and stay present — no task tracking.
Produce exactly one reply in your own voice. Never write dialogue, narration, or replies
attributed to the other person — do not simulate how they might respond or continue the
conversation on their behalf.
"""

HARNESS_INSTRUMENTAL_TOOL_DISCIPLINE = """\
Instrumental turn: use tools when the imperative requires verified repo or runtime facts.
Record each meaningful step before answering.
"""


def _stance_slice(thought: ThoughtEventV1):
    return thought.stance_harness_slice


def is_relational_motor_stance(thought: ThoughtEventV1) -> bool:
    sl = _stance_slice(thought)
    if sl is None:
        return False
    regime = str(sl.interaction_regime or "").strip().lower()
    if regime in {"relational", "minimal"}:
        return True
    task_mode = str(sl.task_mode or "").strip().lower()
    return task_mode in {"reflective_dialogue", "playful_exchange", "identity_dialogue"}


def harness_motor_instruction(*, thought: ThoughtEventV1) -> str:
    read_cap = (
        f"Do not Read whole files over {HARNESS_MOTOR_MAX_READ_LINES} lines — "
        "use rg/Grep or Read offset/limit."
    )
    if is_relational_motor_stance(thought):
        return (
            f"{HARNESS_RELATIONAL_TOOL_DISCIPLINE.strip()}\n"
            f"Execute your imperative. {read_cap}"
        )
    return (
        f"{HARNESS_INSTRUMENTAL_TOOL_DISCIPLINE.strip()}\n"
        f"Execute your imperative. {read_cap}"
    )
