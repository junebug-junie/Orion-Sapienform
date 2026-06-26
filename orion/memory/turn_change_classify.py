from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

_NOVEL_LINE = re.compile(r"^NOVEL:\s*(YES|NO)\s*$", re.I | re.M)
_SHIFT_LINE = re.compile(r"^SHIFT:\s*(NONE|TOPIC|STANCE|REPAIR)\s*$", re.I | re.M)

SHIFT_KINDS: tuple[str, ...] = ("NONE", "TOPIC", "STANCE", "REPAIR")


def enum_scores_from_top_logprobs(
    top_logprobs: list[dict],
    tokens: tuple[str, ...],
) -> dict[str, float] | None:
    token_set = {t.upper() for t in tokens}
    lps: dict[str, float] = {}
    for entry in top_logprobs or []:
        tok = str(entry.get("token") or "").strip().upper()
        lp = entry.get("logprob")
        if tok in token_set and isinstance(lp, (int, float)):
            lps[tok] = float(lp)
    if not lps:
        return None
    exps = {k: math.exp(v) for k, v in lps.items()}
    total = sum(exps.values())
    if total <= 0:
        return None
    return {k: exps[k] / total for k in tokens if k in exps}


def binary_margin(score: float | None) -> float | None:
    if score is None:
        return None
    return abs(float(score) - 0.5)


def appraisal_confidence(*scores: float | None) -> float | None:
    margins = [2.0 * abs(float(s) - 0.5) for s in scores if s is not None]
    return min(margins) if margins else None


def novel_margin_below_threshold(novelty_score: float | None, *, margin: float) -> bool:
    m = binary_margin(novelty_score)
    return m is not None and m < float(margin)


def shift_novelty_mismatch(
    shift_kind: str | None,
    novelty_score: float | None,
    shift_scores: dict[str, float] | None,
    *,
    novelty_ceiling: float = 0.35,
    shift_floor: float = 0.5,
) -> bool:
    if shift_kind not in ("TOPIC", "STANCE", "REPAIR"):
        return False
    if not isinstance(novelty_score, (int, float)) or float(novelty_score) >= novelty_ceiling:
        return False
    scores = shift_scores if isinstance(shift_scores, dict) else {}
    sk = scores.get(shift_kind)
    return isinstance(sk, (int, float)) and float(sk) >= shift_floor


def reconcile_novelty_with_shift(
    scores: dict[str, Any],
    *,
    blend: float = 0.65,
) -> dict[str, Any]:
    """Lift novelty when shift mass is strong but the NOVEL line says NO."""
    shift_kind = scores.get("shift_kind")
    novelty = scores.get("novelty_score")
    shift_scores = scores.get("shift_scores") if isinstance(scores.get("shift_scores"), dict) else {}
    if not shift_novelty_mismatch(shift_kind, novelty, shift_scores):
        return scores
    sk_score = float(shift_scores[str(shift_kind)])
    lifted = max(float(novelty or 0.0), sk_score * blend)
    out = dict(scores)
    out["novelty_score"] = lifted
    out["confidence"] = appraisal_confidence(lifted, sk_score)
    out["novelty_adjusted_from_shift"] = True
    return out


def should_session_reappraise(
    scores: dict[str, Any],
    *,
    margin: float,
    prior_turn_count: int,
) -> bool:
    if prior_turn_count < 1:
        return False
    novelty = scores.get("novelty_score")
    shift_kind = scores.get("shift_kind")
    if novel_margin_below_threshold(novelty, margin=margin):
        return True
    return shift_novelty_mismatch(shift_kind, novelty, scores.get("shift_scores"))


def parse_novel_shift_lines(text: str) -> tuple[str | None, str | None]:
    novel = _NOVEL_LINE.search(text or "")
    shift = _SHIFT_LINE.search(text or "")
    return (
        novel.group(1).upper() if novel else None,
        shift.group(1).upper() if shift else None,
    )


def _clip_pair(prompt: str, response: str, *, limit: int = 300) -> tuple[str, str]:
    def _c(s: str) -> str:
        s = (s or "").strip()
        return s if len(s) <= limit else s[: limit - 3] + "..."

    return _c(prompt), _c(response)


def build_turn_change_prompt(
    *,
    prompt: str,
    response: str,
    baseline_mode: str,
    baseline_text: str,
    phase: str = "unknown",
) -> str:
    p, r = _clip_pair(prompt, response)
    return (
        "Compare CURRENT vs BASELINE. Output exactly four lines.\n"
        "NOVEL: YES when the user adds facts, events, feelings, goals, identity claims, "
        "or reframes the conversation vs baseline.\n"
        "NOVEL: NO only when the exchange is pure continuation with no new substance.\n"
        "SHIFT: NONE | TOPIC (subject) | STANCE (identity/beliefs/relationship framing) | "
        "REPAIR (correction/recovery).\n"
        "If SHIFT is TOPIC, STANCE, or REPAIR, NOVEL should be YES unless it is trivial repetition.\n"
        "NOVEL: YES or NO\n"
        "SHIFT: NONE or TOPIC or STANCE or REPAIR\n"
        "MEMORY: YES or NO\n"
        "BOUNDARY: YES or NO\n\n"
        f"baseline_mode={baseline_mode}\n"
        f"phase={phase}\n"
        f"--- BASELINE ---\n{baseline_text.strip()}\n"
        f"--- CURRENT ---\nUser: {p!r}\nOrion: {r!r}\n"
    )


def build_change_only_prompt(
    *,
    prompt: str,
    response: str,
    baseline_text: str,
    phase: str = "unknown",
) -> str:
    p, r = _clip_pair(prompt, response)
    return (
        "Re-appraise CURRENT vs the session window baseline. Output exactly two lines.\n"
        "NOVEL: YES for new facts, disclosures, identity/autonomy claims, or reframing.\n"
        "NOVEL: NO only for pure continuation with no new substance.\n"
        "If SHIFT is TOPIC, STANCE, or REPAIR, NOVEL should be YES unless trivial repetition.\n"
        "NOVEL: YES or NO\n"
        "SHIFT: NONE or TOPIC or STANCE or REPAIR\n\n"
        f"phase={phase}\n"
        f"--- SESSION WINDOW BASELINE ---\n{baseline_text.strip()}\n"
        f"--- CURRENT ---\nUser: {p!r}\nOrion: {r!r}\n"
    )


def build_turn_change_appraisal(
    *,
    baseline_mode: str,
    prior_correlation_id: str | None,
    novelty_score: float | None,
    shift_kind: str | None,
    shift_scores: dict[str, float] | None,
    confidence: float | None,
    status: str,
) -> dict[str, Any]:
    return {
        "baseline_mode": baseline_mode,
        "prior_correlation_id": prior_correlation_id,
        "novelty_score": novelty_score,
        "shift_kind": shift_kind,
        "shift_scores": shift_scores or {},
        "confidence": confidence,
        "turn_change_status": status,
        "turn_change_ts": datetime.now(timezone.utc).isoformat(),
    }
