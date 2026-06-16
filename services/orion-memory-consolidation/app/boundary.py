from __future__ import annotations

from typing import Any

from orion.memory.consolidation_classify import binary_score_from_top_logprobs, parse_classify_lines
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


def should_close_window(turn: MemoryTurnPersistedV1, scores: dict, settings) -> bool:
    phase = (
        (turn.spark_meta.get("conversation_phase") or {}).get("phase_change") or "unknown"
    )
    bnd = float(scores.get("conversation_boundary_score") or 0.0)
    if phase in {"long_gap", "next_day", "stale_thread"} and bnd >= settings.MEMORY_BOUNDARY_SCORE_THRESHOLD:
        return True
    if phase == "unknown" and bnd >= settings.MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD:
        return True
    if phase in {"same_breath", "short_pause", "resumed_thread"} and bnd >= settings.MEMORY_BOUNDARY_OVERRIDE_THRESHOLD:
        return True
    return False


def scores_from_llm_result(content: str, raw: dict[str, Any]) -> tuple[float | None, float | None]:
    logprobs = (((raw.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or [])
    mem_score = bnd_score = None
    line = "memory"
    for entry in logprobs:
        tok = str(entry.get("token") or "").strip().upper()
        if tok == "MEMORY:":
            line = "memory"
            continue
        if tok == "BOUNDARY:":
            line = "boundary"
            continue
        if tok in ("YES", "NO"):
            tops = entry.get("top_logprobs") or [{"token": tok, "logprob": entry.get("logprob")}]
            score = binary_score_from_top_logprobs(tops)
            if line == "memory":
                mem_score = score
            elif line == "boundary":
                bnd_score = score
    mem_yes, bnd_yes = parse_classify_lines(content)
    if mem_score is None and mem_yes:
        mem_score = 0.85 if mem_yes == "YES" else 0.15
    if bnd_score is None and bnd_yes:
        bnd_score = 0.85 if bnd_yes == "YES" else 0.15
    return mem_score, bnd_score
