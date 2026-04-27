from __future__ import annotations

import re

from .types import ChatTurnRecord, TurnBlockRecord

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
LOG_LINE_RE = re.compile(r"(^|\n)\s*(traceback|error|warn|info|\d+\s+rows?|candidate_count=0)[^\n]*", re.IGNORECASE)


def extract_blocks(turns: list[ChatTurnRecord], include_reasoning: bool = False) -> list[TurnBlockRecord]:
    blocks: list[TurnBlockRecord] = []
    for turn in turns:
        joined = f"{turn.prompt}\n{turn.response}"
        code_or_command = _extract_code_or_command(joined)
        logs_or_errors = _extract_logs(joined)
        blocks.append(
            TurnBlockRecord(
                turn_id=turn.turn_id,
                created_at=turn.created_at,
                user_problem_block=_clean_block(turn.prompt),
                assistant_answer_block=_clean_block(turn.response),
                command_or_code_block=code_or_command,
                log_or_error_block=logs_or_errors,
                optional_reasoning_summary_block=_reasoning_summary(turn.thought_process) if include_reasoning else "",
            )
        )
    return blocks


def _extract_code_or_command(text: str) -> str:
    matches = [m.group(0).strip() for m in CODE_BLOCK_RE.finditer(text)]
    if matches:
        return "\n\n".join(matches[:2]).strip()
    shell_like = [line.strip() for line in text.splitlines() if line.strip().startswith(("$ ", "docker ", "pytest ", "python ", "psql "))]
    return "\n".join(shell_like[:8]).strip()


def _extract_logs(text: str) -> str:
    lines = [m.group(0).strip() for m in LOG_LINE_RE.finditer(text)]
    return "\n".join(lines[:10]).strip()


def _reasoning_summary(thought_process: str) -> str:
    if not thought_process.strip():
        return ""
    trimmed = " ".join(thought_process.split())
    return trimmed[:600]


def _clean_block(text: str) -> str:
    return text.strip()[:4000]
