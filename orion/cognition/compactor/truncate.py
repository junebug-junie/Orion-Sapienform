from __future__ import annotations


def truncate_at_word_boundary(text: str, max_chars: int) -> tuple[str, bool]:
    """Trim `text` to at most `max_chars`, breaking on the last whitespace
    before the cutoff instead of slicing mid-word/mid-sentence, so long
    compactor input (a pasted chat transcript, a PR body) degrades to a
    coherent prefix instead of a fragment the digest LLM has to guess the
    rest of. Considers space, newline, and tab as break points -- real
    long-form text (bullet lists, numbered steps, code blocks, markdown PR
    bodies) is newline-delimited, not space-delimited, and a space-only
    check misses those boundaries entirely. Falls back to a hard cut only
    if no whitespace exists in range (e.g. one long unbroken token).

    Returns (trimmed_text, was_truncated).
    """
    if len(text) <= max_chars:
        return text, False
    window = text[:max_chars]
    boundary = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
    # Don't collapse to a near-empty prefix if the first "word" is huge.
    if boundary > max_chars * 0.5:
        window = window[:boundary]
    return window.rstrip() + "…", True
