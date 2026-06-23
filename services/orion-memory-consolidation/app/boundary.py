from __future__ import annotations

from typing import Any

from orion.memory.consolidation_classify import binary_score_from_top_logprobs, parse_classify_lines
from orion.memory.turn_change_classify import (
    SHIFT_KINDS,
    appraisal_confidence,
    enum_scores_from_top_logprobs,
    parse_novel_shift_lines,
)
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


def scores_from_llm_result(content: str, raw: dict[str, Any]) -> dict[str, Any]:
    logprobs = (((raw.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or [])
    mem_score = bnd_score = novelty_score = None
    shift_scores: dict[str, float] | None = None
    shift_kind: str | None = None
    line = "novel"
    novelty_from_logprobs = False

    for entry in logprobs:
        tok = str(entry.get("token") or "").strip().upper()
        if tok == "NOVEL:":
            line = "novel"
            continue
        if tok == "SHIFT:":
            line = "shift"
            continue
        if tok == "MEMORY:":
            line = "memory"
            continue
        if tok == "BOUNDARY:":
            line = "boundary"
            continue

        tops = entry.get("top_logprobs") or [{"token": tok, "logprob": entry.get("logprob")}]

        if line == "novel" and tok in ("YES", "NO"):
            novelty_score = binary_score_from_top_logprobs(tops)
            if novelty_score is not None:
                novelty_from_logprobs = True
        elif line == "shift" and tok in SHIFT_KINDS:
            shift_scores = enum_scores_from_top_logprobs(tops, SHIFT_KINDS)
            if shift_scores:
                shift_kind = max(shift_scores, key=shift_scores.get)
        elif line == "memory" and tok in ("YES", "NO"):
            mem_score = binary_score_from_top_logprobs(tops)
        elif line == "boundary" and tok in ("YES", "NO"):
            bnd_score = binary_score_from_top_logprobs(tops)

    novel_yes, shift_txt = parse_novel_shift_lines(content)
    mem_yes, bnd_yes = parse_classify_lines(content)

    if novelty_score is None and novel_yes:
        novelty_score = 0.85 if novel_yes == "YES" else 0.15
    if shift_kind is None and shift_txt:
        shift_kind = shift_txt
    if mem_score is None and mem_yes:
        mem_score = 0.85 if mem_yes == "YES" else 0.15
    if bnd_score is None and bnd_yes:
        bnd_score = 0.85 if bnd_yes == "YES" else 0.15

    shift_conf_score = None
    if shift_scores and shift_kind:
        shift_conf_score = shift_scores.get(shift_kind)
    confidence = appraisal_confidence(novelty_score, shift_conf_score)
    scoring_source = "logprobs" if novelty_from_logprobs else ("text" if novelty_score is not None else "missing")

    return {
        "novelty_score": novelty_score,
        "shift_kind": shift_kind,
        "shift_scores": shift_scores or {},
        "confidence": confidence,
        "scoring_source": scoring_source,
        "memory_significance_score": mem_score,
        "conversation_boundary_score": bnd_score,
    }
