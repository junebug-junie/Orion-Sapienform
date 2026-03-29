from __future__ import annotations

import re
from typing import Any, Dict


_HYPOTHESIS_PATTERNS = (r"\bhypothesis\b", r"\bmaybe\b", r"\bif\b", r"\bpossible\b")
_SELF_CORRECTION_PATTERNS = (r"\bi was wrong\b", r"\bcorrection\b", r"\bupdate\b", r"\bon second thought\b")
_EPISTEMIC_MARKERS = (r"\bseems\b", r"\blikely\b", r"\buncertain\b", r"\bprobably\b", r"\bmight\b")
_PLANNING_MARKERS = (r"\bplan\b", r"\bnext\b", r"\bfirst\b", r"\bthen\b", r"\bstep\b")
_UNCERTAINTY_MARKERS = (r"\bunknown\b", r"\bunsure\b", r"\buncertain\b", r"\bnot sure\b")


def extract_reasoning_features(text: str) -> Dict[str, Any]:
    body = str(text or "")
    lowered = body.lower()
    lines = [line for line in body.splitlines() if line.strip()]
    sentence_like = [seg for seg in re.split(r"[.!?\n]+", body) if seg.strip()]

    def _count(patterns: tuple[str, ...]) -> int:
        return sum(len(re.findall(pat, lowered)) for pat in patterns)

    reasoning_depth = max(1, min(5, len(lines) // 3 + len(sentence_like) // 4))
    hypothesis_count = _count(_HYPOTHESIS_PATTERNS)
    self_correction_hits = _count(_SELF_CORRECTION_PATTERNS)
    epistemic_hits = _count(_EPISTEMIC_MARKERS)
    planning_hits = _count(_PLANNING_MARKERS)
    uncertainty_hits = _count(_UNCERTAINTY_MARKERS)

    total_tokens = max(1, len(lowered.split()))
    epistemic_score = round(epistemic_hits / total_tokens, 4)

    return {
        "reasoning_depth": reasoning_depth,
        "hypothesis_count": hypothesis_count,
        "self_correction_detected": self_correction_hits > 0,
        "epistemic_language_score": epistemic_score,
        "planning_intent_detected": planning_hits > 0,
        "uncertainty_markers": uncertainty_hits,
    }
