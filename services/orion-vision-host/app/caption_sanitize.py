from __future__ import annotations

from collections import Counter

from orion.vision.caption_echo import CAPTION_PROMPT, is_caption_prompt_echo

_STOPLIST = frozenset({"youtube", "google", "video", "watching", "webcam", "com"})
# Below this many real tokens, a single-token-dominance ratio is meaningless
# (a genuine 2-word answer is 100% "dominated" by its rarer token) -- only
# flag repetition once there's enough length for repeating the *same* token
# this much to be a real signal, not an artifact of being short.
_MIN_TOKENS_FOR_REPETITION_CHECK = 4


def _tokenize(text: str) -> list[str]:
    """Punctuation-stripped, non-empty tokens. Filters on the *stripped*
    token, not just whitespace -- confirmed live 2026-08-20: a
    punctuation-only response ("?") passed an earlier version of this that
    filtered on `t.strip()` (whitespace only) and survived every downstream
    check as a 1-token, 0%-stoplist "answer", exactly the schema-valid-but-
    meaningless-content this repo's own metric quality gate names as a
    failure mode."""
    return [t.strip(".,!?|").lower() for t in text.split() if t.strip(".,!?|")]


def _degenerate_reason(tokens: list[str]) -> str | None:
    """Shared quality checks both sanitizers below run against their own
    (already echo-checked, already tokenized) text. Returns a rejection
    reason, or None if the tokens look like real content."""
    if not tokens:
        return "empty"
    stop_hits = sum(1 for t in tokens if t in _STOPLIST)
    if stop_hits / len(tokens) > 0.4:
        return "stoplist_ratio"
    # Confirmed live 2026-08-20, the same VQA smoke session that found the
    # punctuation-only-response bug above: BLIP-base answered "what color
    # is the door?" with "| by person | cci | cci | cci | cci | cci | cci |"
    # -- a real, non-empty, non-stoplisted, non-echo string that is still
    # obvious repetition garbage, not a real answer. None of those tokens
    # are on the (topic-specific) stoplist, so nothing above caught it.
    # This is a general degeneracy signal, not VQA-specific -- a captioner
    # stuck in a repetition loop is exactly as meaningless as one that
    # echoed the prompt.
    if len(tokens) >= _MIN_TOKENS_FOR_REPETITION_CHECK:
        most_common_count = Counter(tokens).most_common(1)[0][1]
        if most_common_count / len(tokens) > 0.4:
            return "repetition_degenerate"
    return None


def sanitize_caption(raw: str) -> tuple[str | None, bool, str | None]:
    text = (raw or "").strip()
    if is_caption_prompt_echo(text):
        return None, False, "prompt_echo"
    if len(text) < 12:
        return None, False, "too_short"
    reason = _degenerate_reason(_tokenize(text))
    if reason:
        return None, False, reason
    return text, True, None


def sanitize_answer(raw: str, question: str) -> tuple[str | None, bool, str | None]:
    """VQA counterpart to ``sanitize_caption`` -- same echo/degeneracy
    checks, but the echo check is against the caller's actual ``question``,
    not the fixed ``CAPTION_PROMPT``: a VQA echo means the model handed the
    question back verbatim, and checking against the wrong fixed string
    would never catch that. A short, direct answer ("yes", "the door") is a
    normal, correct VQA response, not degenerate the way a 2-word caption
    would be -- so this has no length floor, deliberately not copy-pasted
    from ``sanitize_caption``'s 12-char one.
    """
    text = (raw or "").strip()
    if is_caption_prompt_echo(text, prompt=question):
        return None, False, "prompt_echo"
    reason = _degenerate_reason(_tokenize(text))
    if reason:
        return None, False, reason
    return text, True, None
