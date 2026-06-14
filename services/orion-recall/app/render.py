from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple

from orion.core.contracts.recall import MemoryItemV1

try:
    from .snippet_dedupe import OrionDigestDeduper, transcript_snippet_user_lean
except ImportError:  # pragma: no cover
    from snippet_dedupe import OrionDigestDeduper, transcript_snippet_user_lean  # type: ignore

CHARS_PER_RENDER_TOKEN = 5


def _estimate_tokens_strict(text: str) -> int:
    """Conservative token estimate for strict_prompt_budget profiles."""
    stripped = (text or "").strip()
    if not stripped:
        return 0
    word_est = len(stripped.split())
    char_est = (len(stripped) + 3) // 4
    return max(word_est, char_est, 1)


def _estimate_tokens_default(text: str) -> int:
    """Legacy word-count budget estimator (scheduler journal and default lanes)."""
    return max(len((text or "").split()), 1)


def _clamp_snippet(snippet: str, max_chars: int) -> str:
    text = (snippet or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def render_items(
    items: Iterable[MemoryItemV1],
    budget_tokens: int,
    profile_name: Optional[str] = None,
    *,
    diagnostic: bool = False,
    budget_indicator: bool = True,
    render_transcript_user_only: bool = False,
    strict_prompt_budget: bool = False,
    render_char_budget: Optional[int] = None,
    max_snippet_chars: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Optional[str]]]]:
    """
    Render memory items into a compact bullet list suitable for prompts.

    Default lanes use legacy word-count budgeting with no char cap or snippet clamp.
    Opt-in strict_prompt_budget profiles also enforce render_char_budget and
    max_render_snippet_chars (configured explicitly on the recall profile YAML).
    """
    lines: List[str] = []
    tokens_used = 0
    chars_used = 0
    estimate_tokens = _estimate_tokens_strict if strict_prompt_budget else _estimate_tokens_default
    budget_chars: Optional[int] = None
    if strict_prompt_budget:
        budget_chars = int(render_char_budget or 0) or max(int(budget_tokens) * CHARS_PER_RENDER_TOKEN, 1)
    snippet_cap = int(max_snippet_chars) if strict_prompt_budget and max_snippet_chars else None
    budget_dropped: List[Dict[str, Optional[str]]] = []

    items_list = list(items)
    claims = [item for item in items_list if "claim" in (item.tags or [])]
    others = [item for item in items_list if item not in claims]
    orion_deduper = OrionDigestDeduper()
    is_graphtri = bool(profile_name) and (
        str(profile_name) == "graphtri.v1" or str(profile_name).startswith("graphtri")
    )
    emitted_ids: Set[str] = set()

    def _append_line(line: str) -> bool:
        nonlocal tokens_used, chars_used
        est_tokens = estimate_tokens(line)
        if tokens_used + est_tokens > budget_tokens:
            return False
        if budget_chars is not None:
            line_chars = len(line) + (1 if lines else 0)
            if chars_used + line_chars > budget_chars:
                return False
            chars_used += line_chars
        tokens_used += est_tokens
        lines.append(line)
        return True

    def _prepare_snippet(raw: str) -> str:
        snippet = (raw or "").strip().replace("\n", " ")
        if snippet_cap:
            snippet = _clamp_snippet(snippet, snippet_cap)
        return snippet

    def _render_group(label: str, group: List[MemoryItemV1]) -> None:
        if not group:
            return
        if not _append_line(f"{label}:"):
            return
        for item in group:
            if item.id in emitted_ids:
                continue
            snippet = _prepare_snippet(item.snippet or "")
            if not snippet:
                continue
            if render_transcript_user_only:
                lean = transcript_snippet_user_lean(snippet)
                if lean:
                    snippet = _prepare_snippet(lean)
            if not orion_deduper.should_emit_snippet(snippet):
                continue
            prefix = f"[{item.source}"
            if item.source_ref:
                prefix += f":{item.source_ref}"
            prefix += "]"
            line = f"- {prefix} {snippet}"
            if not _append_line(line):
                budget_dropped.append(
                    {"id": item.id, "source": str(item.source or ""), "snippet": snippet[:160]}
                )
                return
            emitted_ids.add(item.id)

    if is_graphtri:
        def _score(it: MemoryItemV1) -> float:
            try:
                return float(it.score or 0.0)
            except Exception:
                return 0.0

        non_claims = list(others)
        non_claims.sort(
            key=lambda it: (
                1 if str(it.source or "") == "vector" else 0,
                _score(it),
            ),
            reverse=True,
        )

        def _is_refusal_snippet(text: str) -> bool:
            t = (text or "").lower()
            return any(
                p in t
                for p in [
                    "i don't have a specific memory",
                    "i don't have a memory",
                    "i don't remember",
                    "i cant recall",
                    "i can't recall",
                    "i do not recall",
                    "not in my recent context",
                    "if you could provide more details",
                    "if you could share more details",
                ]
            )

        skipped_refusal = 0
        top_vector = [
            it
            for it in non_claims
            if str(it.source or "") == "vector" and not _is_refusal_snippet(it.snippet or "")
        ][:5]
        top_other = [
            it
            for it in non_claims
            if str(it.source or "") != "vector" and not _is_refusal_snippet(it.snippet or "")
        ][:3]
        top_items = top_vector + top_other
        if top_items:
            _append_line("=== TOP RELEVANT SNIPPETS ===")
            for item in top_items:
                if item.id in emitted_ids:
                    continue
                snippet = _prepare_snippet(item.snippet or "")
                if render_transcript_user_only:
                    lean = transcript_snippet_user_lean(snippet)
                    if lean:
                        snippet = _prepare_snippet(lean)
                if _is_refusal_snippet(snippet):
                    skipped_refusal += 1
                    continue
                if not snippet:
                    continue
                if not orion_deduper.should_emit_snippet(snippet):
                    continue
                prefix = f"[{item.source}"
                if item.source_ref:
                    prefix += f":{item.source_ref}"
                prefix += "]"
                line = f"- {prefix} {snippet}"
                if not _append_line(line):
                    budget_dropped.append(
                        {"id": item.id, "source": str(item.source or ""), "snippet": snippet[:160]}
                    )
                    break
                emitted_ids.add(item.id)
        logger = logging.getLogger("orion-recall.render")
        logger.info(
            "graphtri_top_snippets_summary top_selected_count=%s top_skipped_refusal_count=%s",
            len(emitted_ids),
            skipped_refusal,
        )

        _render_group("High-salience claims", claims)
        remaining = [item for item in others if item.id not in emitted_ids]
        _render_group("Recent context", remaining)

        first_line = next((line for line in lines if line.startswith("- ")), "")
        digest_has_ai_ml_architect = ("AI/ML" in "\n".join(lines)) or ("Architect" in "\n".join(lines))
        logger.info(
            "graphtri_render_guardrails top_snippets_first_line=%r digest_has_ai_ml_architect=%s digest_has_refusal_phrase=%s",
            first_line,
            digest_has_ai_ml_architect,
            _is_refusal_snippet("\n".join(lines)),
        )
    else:
        _render_group("High-salience claims", claims)
        _render_group("Recent context", others)

    rendered = "\n".join(lines)
    n_drop = len(budget_dropped)
    if budget_indicator and n_drop > 0 and not diagnostic:
        indicator = f"\n[+{n_drop} more items dropped due to budget]"
        est = estimate_tokens(indicator)
        indicator_chars = len(indicator)
        fits_tokens = tokens_used + est <= budget_tokens
        fits_chars = budget_chars is None or chars_used + indicator_chars <= budget_chars
        if fits_tokens and fits_chars:
            rendered = f"{rendered}{indicator}"

    return rendered, budget_dropped
