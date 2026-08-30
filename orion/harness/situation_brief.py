"""Canonical, persistent explainer for how to read the harness prefix's
Situation block.

orion/situational/context.py::_build_prompt_fragment renders each section's
usage caveat inline, per-section, and only when there's something to caveat
(e.g. curiosity/reverie's self-generated framing, the affect-read hedge, the
per-section "do not infer" lines). Those inline caveats also compete for the
fragment's own char budget -- _build_prompt_fragment appends its "cautions"
list most-important-first and silently drops whatever doesn't fit off the
tail (see that function's own comments on the 2026-08-26 truncation
incident, where a caution was cut off mid-sentence). This brief is the
class-level, budget-immune backstop: rendered once per compiled prefix, it
explains the *conventions* the Situation block follows as a whole --
confidence/absence handling, self-generated vs external content -- so that
convention survives even on a turn where a per-section caveat got truncated
or was never emitted (curiosity/reverie are omitted entirely, not rendered
as a placeholder, when there is nothing to show -- see the comment at
context.py ~L1639).

Mirrors orion/fcc/self_index_brief.py's shape: a lines() function plus an
append_*_harness_brief(parts) helper, wired into compile_harness_prefix near
the other capability briefs.
"""

from __future__ import annotations


def situation_block_brief_lines() -> list[str]:
    return [
        (
            "How to read the Situation block above: it is live, per-turn context "
            "assembled just before this turn -- not memory, and not guaranteed "
            "complete. A section renders only when its underlying data is present "
            "and fresh; when a line instead says unavailable, or ends with 'do not "
            "infer', that means exactly what it says -- do not guess, infer, or "
            "fabricate a value to fill that gap, and do not treat an absent or "
            "truncated section as though it were a stated fact."
        ),
        (
            "Your own open world-priors (curiosity) and reverie/dream threads, when "
            "shown, are self-generated content -- from your own worldview graph and "
            "your own past reverie readings -- not externally verified fact, not "
            "something Juniper told you, and not durable memory. Treat them as "
            "tentative interior color you may draw on, never as established truth "
            "to assert to Juniper."
        ),
        (
            "General posture: use situation context only when it materially serves "
            "this turn's imperative. A section being present is not an instruction "
            "to mention, narrate, or perform it."
        ),
    ]


def append_situation_block_harness_brief(
    parts: list[str],
    *,
    situation_prompt_fragment: str | None,
) -> None:
    """Append the canonical Situation-block explainer once, when a situation
    fragment was actually rendered this turn.

    Gated the same way compile_harness_prefix already gates the fragment
    itself (~L186): no situation fragment this turn means there is no
    Situation block for this explainer to describe, so it stays silent --
    matching the byte-identical-when-absent guarantee that parameter's own
    test covers.
    """
    if not situation_prompt_fragment:
        return
    parts.extend(situation_block_brief_lines())
