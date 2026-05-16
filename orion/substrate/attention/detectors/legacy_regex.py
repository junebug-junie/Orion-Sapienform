from __future__ import annotations

import re
from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import STOP_PHRASES, compact, stable_id

_PLACE_RE = re.compile(r"\b(in|at|near|from)\s+([A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,}){0,3})")
_PROPER_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,}){0,3})\b")
_ACTIVITY_RE = re.compile(
    r"\b(?:working on|building|debugging|planning|trying|running|designing|implementing|testing)\s+([^.!?\n]{3,90})",
    flags=re.IGNORECASE,
)
_NAMED_RE = re.compile(r"\b(?:called|named|about|around|with|for)\s+([^.!?\n]{3,80})", flags=re.IGNORECASE)


class LegacyRegexSignalDetector:
    """Deterministic v1 fallback for current-turn phrase candidates."""

    detector_id = "legacy_regex_v1"

    def detect(
        self,
        ctx: dict[str, Any],
        inputs: dict[str, Any],  # noqa: ARG002
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        user_text = compact(ctx.get("user_message") or ctx.get("raw_user_text") or "", 600)
        candidates: list[tuple[str, str, str]] = []
        for match in _ACTIVITY_RE.finditer(user_text):
            candidates.append((compact(match.group(1), 90), "activity", "activity_phrase"))
        for match in _NAMED_RE.finditer(user_text):
            candidates.append((compact(match.group(1), 80), "other", "named_phrase"))
        for match in _PLACE_RE.finditer(user_text):
            candidates.append((compact(match.group(2), 80), "place", "place_phrase"))
        for match in _PROPER_RE.finditer(user_text):
            phrase = compact(match.group(1), 80)
            if phrase.lower() not in STOP_PHRASES:
                candidates.append((phrase, "concept", "proper_phrase"))

        out: list[AttentionSignalV1] = []
        seen: set[str] = set()
        for phrase, hint, kind in candidates:
            cleaned = phrase.strip(" ,:;()[]{}")
            key = cleaned.lower()
            if len(cleaned) < 3 or key in seen or key in STOP_PHRASES:
                continue
            seen.add(key)
            out.append(
                AttentionSignalV1(
                    signal_id=stable_id("attention-signal", f"{self.detector_id}:{key}"),
                    source=self.detector_id,
                    target_text=cleaned,
                    target_type_hint=hint,
                    signal_kind=kind,
                    salience=0.72,
                    confidence=0.68,
                    evidence_refs=["ctx.user_message"],
                    provenance={"detector": self.detector_id, "belief_lineage": list(belief_lineage or [])[:8]},
                )
            )
        return out
