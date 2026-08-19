"""Summary-only language-surface stability metrics from OpenAI logprobs."""
from __future__ import annotations

import math
from typing import Any

from app.settings import settings

SCHEMA_VERSION = "v1"
CONFIDENCE_SEMANTICS = "language_surface_stability_not_truth"


def _token_logprob(entry: dict[str, Any]) -> float | None:
    lp = entry.get("logprob")
    if isinstance(lp, (int, float)):
        return float(lp)
    return None


def _top1_margin(entry: dict[str, Any]) -> float | None:
    tops = entry.get("top_logprobs")
    if not isinstance(tops, list) or len(tops) < 2:
        return None
    lps = [_token_logprob(t) for t in tops if isinstance(t, dict)]
    lps = [x for x in lps if x is not None]
    if len(lps) < 2:
        return None
    lps.sort(reverse=True)
    return lps[0] - lps[1]


def _entropy_proxy(entry: dict[str, Any]) -> float | None:
    tops = entry.get("top_logprobs")
    if not isinstance(tops, list) or not tops:
        return None
    lps = [_token_logprob(t) for t in tops if isinstance(t, dict)]
    lps = [x for x in lps if x is not None]
    if not lps:
        return None
    max_lp = max(lps)
    weights = [math.exp(lp - max_lp) for lp in lps]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return ent


def _count_unstable_spans(margins: list[float | None], *, min_len: int) -> int:
    run = 0
    count = 0
    threshold = float(getattr(settings, "llm_logprob_low_margin_threshold", 0.5))
    for m in margins:
        if m is not None and m < threshold:
            run += 1
            if run == min_len:
                count += 1
        else:
            run = 0
    return count


def summarize_logprob_content(content: list[dict[str, Any]]) -> dict[str, Any]:
    logprobs: list[float] = []
    margins: list[float | None] = []
    entropies: list[float] = []
    low_margin = 0
    low_logprob = 0
    low_margin_threshold = float(getattr(settings, "llm_logprob_low_margin_threshold", 0.5))
    low_logprob_threshold = float(getattr(settings, "llm_logprob_low_logprob_threshold", -2.0))

    for entry in content:
        if not isinstance(entry, dict):
            continue
        lp = _token_logprob(entry)
        if lp is not None:
            logprobs.append(lp)
            if lp < low_logprob_threshold:
                low_logprob += 1
        margin = _top1_margin(entry)
        margins.append(margin)
        if margin is not None and margin < low_margin_threshold:
            low_margin += 1
        ent = _entropy_proxy(entry)
        if ent is not None:
            entropies.append(ent)

    if not logprobs:
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "diagnostic_only": True,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "token_count_observed": 0,
        }

    span_min = int(getattr(settings, "llm_logprob_unstable_span_min_len", 3))
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "diagnostic_only": True,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "token_count_observed": len(logprobs),
        "mean_logprob": sum(logprobs) / len(logprobs),
        "min_logprob": min(logprobs),
        "mean_top1_margin": (sum(m for m in margins if m is not None) / max(1, sum(1 for m in margins if m is not None))),
        "low_margin_token_count": low_margin,
        "low_logprob_token_count": low_logprob,
        "entropy_proxy_mean": (sum(entropies) / len(entropies)) if entropies else None,
        "unstable_span_count": _count_unstable_spans(margins, min_len=span_min),
    }


def _legacy_completion_entry_to_logprob_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize llama.cpp's older per-token /completion shape into this module's
    canonical {"token", "logprob", "top_logprobs"} entry.

    That older shape is `{"content": tok_str, "probs": [{"tok_str": ..., "prob": ...}, ...]}`
    per generated token -- a real, historical llama.cpp response format, distinct from
    both the live-verified 2026-08-19 shape (see below) and the wrapper shape a prior
    version of this function speculatively supported (which never matched any real
    server response and, worse, silently absorbed *this* real shape by mistaking its
    per-token `probs` alternative list for an all-tokens wrapper -- see PR history).
    `prob` here is a plain probability, not a logprob; convert via math.log.
    """
    token = entry.get("content")
    alts = entry.get("probs")
    if not isinstance(token, str) or not isinstance(alts, list):
        return None
    tops: list[dict[str, Any]] = []
    own_logprob: float | None = None
    for alt in alts:
        if not isinstance(alt, dict):
            continue
        prob = alt.get("prob")
        if not isinstance(prob, (int, float)) or prob <= 0:
            continue
        lp = math.log(prob)
        tops.append({"token": alt.get("tok_str"), "logprob": lp})
        if alt.get("tok_str") == token:
            own_logprob = lp
    if own_logprob is None and tops:
        # The sampled token wasn't among its own listed alternatives (can happen
        # under aggressive top-k truncation) -- use the best alternative rather
        # than dropping a real generated token entirely.
        own_logprob = max(t["logprob"] for t in tops)
    if own_logprob is None:
        return None
    return {"token": token, "logprob": own_logprob, "top_logprobs": tops}


def native_completion_probs_to_logprob_content(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize llama.cpp /completion prob shapes into OpenAI logprobs.content entries."""
    if not isinstance(raw, dict):
        return []

    probs: list[Any] = []
    if isinstance(raw.get("probs"), list):
        probs = raw["probs"]
    elif isinstance(raw.get("completion_probabilities"), list) and raw["completion_probabilities"]:
        items = raw["completion_probabilities"]
        first = items[0]
        if isinstance(first, dict) and "token" in first:
            # Live-verified 2026-08-19 (atlas-worker-fast-1): a flat list of
            # per-token entries, each already carrying logprob/top_logprobs directly.
            probs = items
        elif isinstance(first, dict) and "content" in first and isinstance(first.get("probs"), list):
            # Older llama.cpp per-token shape (content/probs with tok_str/prob).
            probs = [
                normalized
                for e in items
                if isinstance(e, dict)
                and (normalized := _legacy_completion_entry_to_logprob_entry(e)) is not None
            ]

    content: list[dict[str, Any]] = []
    for entry in probs:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        if token is None:
            continue
        lp = entry.get("logprob")
        if lp is None and entry.get("prob") is not None:
            continue
        tops = entry.get("top_logprobs")
        if tops is None:
            tops = entry.get("top_probs")
        normalized_tops: list[dict[str, Any]] = []
        if isinstance(tops, list):
            for t in tops:
                if not isinstance(t, dict):
                    continue
                t_lp = t.get("logprob")
                if t_lp is None and t.get("prob") is not None:
                    continue
                normalized_tops.append({"token": t.get("token"), "logprob": t_lp})
        content.append({"token": token, "logprob": lp, "top_logprobs": normalized_tops})
    return content


def extract_llm_uncertainty_from_native_completion(
    raw_data: dict[str, Any],
    *,
    source: str = "llamacpp_native_completion",
) -> dict[str, Any] | None:
    if not isinstance(raw_data, dict):
        return None
    content = native_completion_probs_to_logprob_content(raw_data)
    if not content:
        return None
    summary = summarize_logprob_content(content)
    summary["source"] = source
    return summary


def extract_llm_uncertainty_from_openai_response(
    raw_data: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any] | None:
    if not isinstance(raw_data, dict):
        return None
    choices = raw_data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    logprobs = first.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    summary = summarize_logprob_content(content)
    summary["source"] = source
    return summary
