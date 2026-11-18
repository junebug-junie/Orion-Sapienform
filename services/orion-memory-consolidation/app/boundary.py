from __future__ import annotations

import re
from typing import Any

from orion.memory.consolidation_classify import binary_score_from_top_logprobs, parse_classify_lines
from orion.memory.turn_change_classify import (
    SHIFT_KINDS,
    appraisal_confidence,
    enum_scores_from_top_logprobs,
    parse_novel_shift_lines,
)
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1

_LABEL_SUFFIX = {
    "novel": re.compile(r"NOVEL:\s*$", re.IGNORECASE),
    "shift": re.compile(r"SHIFT:\s*$", re.IGNORECASE),
    "memory": re.compile(r"MEMORY:\s*$", re.IGNORECASE),
    "boundary": re.compile(r"BOUNDARY:\s*$", re.IGNORECASE),
}


_REORIENT_PHASES = frozenset({"long_gap", "next_day", "stale_thread"})
_ACTIVE_PHASES = frozenset({"same_breath", "short_pause"})


def legacy_view(turn: MemoryTurnPersistedV1, settings) -> MemoryTurnPersistedV1:
    """The turn as the live legacy path saw it before boundary Fix 1.

    With MEMORY_LEGACY_BOUNDARY_USE_PHASE off (default), the conversation_phase
    stamp is hidden from the legacy window rule and from the classify prompt,
    so live window closing is unchanged while Rule 3 runs in shadow.
    """
    if getattr(settings, "MEMORY_LEGACY_BOUNDARY_USE_PHASE", False):
        return turn
    if "conversation_phase" not in turn.spark_meta:
        return turn
    meta = {k: v for k, v in turn.spark_meta.items() if k != "conversation_phase"}
    return turn.model_copy(update={"spark_meta": meta})


def turn_phase(turn: MemoryTurnPersistedV1) -> str | None:
    """The wall-clock phase stamped on the turn (Fix 1), or None if absent."""
    phase = (turn.spark_meta.get("conversation_phase") or {}).get("phase_change")
    return str(phase) if phase else None


def legacy_close_reason(turn: MemoryTurnPersistedV1, scores: dict, settings) -> str | None:
    """Why the legacy rule closes the window on this turn, or None.

    Unchanged behavior; this only names the branch so the window can record it.
    """
    phase = turn_phase(turn) or "unknown"
    bnd = float(scores.get("conversation_boundary_score") or 0.0)
    if phase in _REORIENT_PHASES and bnd >= settings.MEMORY_BOUNDARY_SCORE_THRESHOLD:
        return f"legacy:phase_{phase}+llm"
    if phase == "unknown" and bnd >= settings.MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD:
        return "legacy:unknown_phase+llm"
    # Active chat phases: quick-lane BOUNDARY:YES is too noisy to close windows.
    # Rely on long_gap/stale_thread/unknown signals and time-gap fallback instead.
    return None


def should_close_window(turn: MemoryTurnPersistedV1, scores: dict, settings) -> bool:
    return legacy_close_reason(turn, scores, settings) is not None


def rule3_boundary(
    *,
    phase: str | None,
    boundary_score: float | None,
    gap_sec: float | None,
    settings,
) -> tuple[bool, str]:
    """Episode boundary Rule 3 (Juniper, 2026-10-01), evaluated when a turn arrives.

    - long_gap / next_day / stale_thread -> boundary (the wall clock alone;
      its meaning for these phases is "reorient").
    - resumed_thread -> boundary only if the LLM boundary score is
      >= MEMORY_BOUNDARY_OVERRIDE_THRESHOLD (0.92).
    - same_breath / short_pause -> never a boundary.
    - no phase (absent or "unknown") -> the existing fallback: a gap of
      >= MEMORY_WINDOW_FALLBACK_GAP_SEC between this turn and the previous one.

    Returns (is_boundary, reason). The reason names the branch taken whether
    or not it closes, so a shadow record can show why a turn did NOT split.
    """
    if phase in _REORIENT_PHASES:
        return True, f"v2:phase_{phase}"
    if phase == "resumed_thread":
        threshold = float(settings.MEMORY_BOUNDARY_OVERRIDE_THRESHOLD)
        if boundary_score is not None and float(boundary_score) >= threshold:
            return True, "v2:resumed_thread+llm"
        return False, "v2:resumed_thread_below_llm_threshold"
    if phase in _ACTIVE_PHASES:
        return False, f"v2:phase_{phase}"
    fallback = float(settings.MEMORY_WINDOW_FALLBACK_GAP_SEC)
    if gap_sec is not None and gap_sec >= fallback:
        return True, "v2:no_phase_time_gap"
    return False, "v2:no_phase_no_gap"


def _normalize_token(token: str) -> str:
    return str(token or "").strip().upper()


def _label_line_reached(buf: str, line: str) -> bool:
    pat = _LABEL_SUFFIX.get(line)
    return bool(pat and pat.search(buf))


def _normalize_binary_tops(top_logprobs: list[dict]) -> list[dict]:
    best: dict[str, float] = {}
    for entry in top_logprobs or []:
        tok = _normalize_token(entry.get("token"))
        lp = entry.get("logprob")
        if tok in ("YES", "NO") and isinstance(lp, (int, float)):
            lp = float(lp)
            if tok not in best or lp > best[tok]:
                best[tok] = lp
    return [{"token": k, "logprob": v} for k, v in best.items()]


def _normalize_shift_tops(top_logprobs: list[dict]) -> list[dict]:
    """Map BPE-split shift tokens (e.g. ' TOP') onto SHIFT_KINDS for softmax."""
    best: dict[str, float] = {}
    for entry in top_logprobs or []:
        raw = str(entry.get("token") or "")
        tok = _normalize_token(raw)
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)):
            continue
        lp = float(lp)
        candidates: list[str] = []
        if tok in SHIFT_KINDS:
            candidates = [tok]
        else:
            candidates = [k for k in SHIFT_KINDS if k.startswith(tok) or tok in k]
        for kind in candidates:
            if kind not in best or lp > best[kind]:
                best[kind] = lp
    return [{"token": k, "logprob": v} for k, v in best.items()]


def _resolve_shift_kind(fragment: str) -> str | None:
    accum = _normalize_token(fragment)
    if not accum:
        return None
    if accum in SHIFT_KINDS:
        return accum
    matches = [k for k in SHIFT_KINDS if k.startswith(accum) or accum in k]
    if len(matches) == 1:
        return matches[0]
    return None


def scores_from_llm_result(content: str, raw: dict[str, Any]) -> dict[str, Any]:
    logprobs = (((raw.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or [])
    mem_score = bnd_score = novelty_score = None
    shift_scores: dict[str, float] | None = None
    shift_kind: str | None = None
    novelty_from_logprobs = False

    buf = ""
    line: str | None = None
    shift_parts: list[str] = []
    novelty_scored = shift_scored = mem_scored = bnd_scored = False

    for entry in logprobs:
        piece = str(entry.get("token") or "")
        buf += piece

        if not novelty_scored and _label_line_reached(buf, "novel"):
            line = "novel"
            continue
        if not shift_scored and _label_line_reached(buf, "shift"):
            line = "shift"
            shift_parts = []
            continue
        if not mem_scored and _label_line_reached(buf, "memory"):
            line = "memory"
            continue
        if not bnd_scored and _label_line_reached(buf, "boundary"):
            line = "boundary"
            continue

        tops = entry.get("top_logprobs") or [{"token": piece, "logprob": entry.get("logprob")}]
        norm = _normalize_token(piece)

        if line == "novel" and not novelty_scored and norm in ("YES", "NO"):
            scored = binary_score_from_top_logprobs(_normalize_binary_tops(tops))
            if scored is not None:
                novelty_score = scored
                novelty_from_logprobs = True
                novelty_scored = True
                line = None
        elif line == "shift" and not shift_scored:
            shift_parts.append(piece)
            accum = "".join(shift_parts)
            normed_tops = _normalize_shift_tops(tops)
            if normed_tops:
                candidate = enum_scores_from_top_logprobs(normed_tops, SHIFT_KINDS)
                if candidate:
                    shift_scores = candidate
                    shift_kind = max(candidate, key=candidate.get)
            resolved = _resolve_shift_kind(accum)
            if resolved:
                shift_kind = resolved
                if not shift_scores:
                    shift_scores = {resolved: 1.0}
                shift_scored = True
                line = None
            elif "\n" in piece:
                shift_scored = True
                line = None
        elif line == "memory" and not mem_scored and norm in ("YES", "NO"):
            scored = binary_score_from_top_logprobs(_normalize_binary_tops(tops))
            if scored is not None:
                mem_score = scored
                mem_scored = True
                line = None
        elif line == "boundary" and not bnd_scored and norm in ("YES", "NO"):
            scored = binary_score_from_top_logprobs(_normalize_binary_tops(tops))
            if scored is not None:
                bnd_score = scored
                bnd_scored = True
                line = None

    novel_yes, shift_txt = parse_novel_shift_lines(content)
    mem_yes, bnd_yes = parse_classify_lines(content)

    if novelty_score is None and novel_yes:
        novelty_score = 0.85 if novel_yes == "YES" else 0.15
    if shift_kind is None and shift_txt:
        shift_kind = shift_txt
    if shift_txt and not shift_scores:
        shift_scores = {shift_txt: 1.0}
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
