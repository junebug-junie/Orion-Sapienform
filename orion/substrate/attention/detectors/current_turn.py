from __future__ import annotations

from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import STOP_PHRASES, bounded, compact, stable_id


class CurrentTurnSignalDetector:
    """Current user-turn detector.

    Reads LLM-judged novelty/salience candidates precomputed into
    `ctx["current_turn_llm_signals"]` by
    `services/orion-cortex-exec/app/current_turn_llm_signals.py`'s
    `populate_current_turn_llm_signals()` -- a real same-turn LLM RPC call
    made BEFORE `build_attention_frame()` runs, from
    `chat_stance.py::build_chat_stance_inputs`. `detect()` itself is a pure,
    synchronous reader of that precomputed list; it makes no network call
    (this repo's `AttentionSignalDetector.detect()` Protocol is synchronous
    -- see `orion/substrate/attention/detectors/base.py`).

    Replaces the deleted `LegacyRegexSignalDetector`
    (`orion/substrate/attention/detectors/legacy_regex.py`), whose
    `_PROPER_RE` regex matched ANY capitalized word as a "proper noun"
    candidate -- including sentence-initial interjections like "Heck" in
    "Heck yeah!" -- filtered only by a 9-word `STOP_PHRASES` allowlist with
    no interjection/filler coverage. Confirmed live: those garbage
    candidates could be selected as an "ask" action and threaded into the
    live chat-turn stance brief, so Orion could literally ask the user a
    clarifying question about a garbage interjection in a real reply.

    If `ctx["current_turn_llm_signals"]` is absent or empty (LLM call
    skipped because the frame is disabled, failed, timed out, or genuinely
    found nothing), this returns `[]` -- same fail-open shape the old
    detector already had for empty `user_text`.
    """

    detector_id = "current_turn_v1"

    def detect(
        self,
        ctx: dict[str, Any],
        inputs: dict[str, Any],  # noqa: ARG002
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        candidates = ctx.get("current_turn_llm_signals")
        if not candidates:
            return []

        out: list[AttentionSignalV1] = []
        seen: set[str] = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            phrase = compact(item.get("phrase"), 80).strip(" ,:;()[]{}")
            key = phrase.lower()
            if len(phrase) < 2 or key in seen or key in STOP_PHRASES:
                continue
            seen.add(key)
            hint = str(item.get("type") or "other").strip().lower() or "other"
            confidence_raw = item.get("confidence")
            confidence = bounded(confidence_raw) if confidence_raw is not None else 0.68
            out.append(
                AttentionSignalV1(
                    signal_id=stable_id("attention-signal", f"{self.detector_id}:{key}"),
                    source=self.detector_id,
                    target_text=phrase,
                    target_type_hint=hint,
                    signal_kind="llm_novelty_v1",
                    salience=0.72,
                    confidence=confidence,
                    evidence_refs=["ctx.user_message"],
                    provenance={"detector": self.detector_id, "belief_lineage": list(belief_lineage or [])[:8]},
                )
            )
        return out
