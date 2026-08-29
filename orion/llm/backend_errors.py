"""One owner for the LLM gateway's `[Error: ...]` sentinel text.

`orion-llm-gateway`'s backend callers return a normal result dict on failure with
an error string in `text` rather than raising (see
`services/orion-llm-gateway/app/llm_backend.py`). That is a deliberate, load-bearing
design decision -- every caller of `run_llm_chat()` consumes a dict, and making the
failure path raise would change behavior across the whole chat path.

The cost is that a failure is indistinguishable from an answer *by shape*: the only
thing marking it is the leading sentinel, which was an unowned literal repeated at
~10 sites. This module gives it one definition and one predicate, so a consumer that
needs to know "is this real model output or a backend failure?" can ask instead of
pattern-matching on its own.

Confirmed live 2026-08-29: during a ~45 minute circe outage, 663 gateway timeouts
each returned `[Error: llamacpp timed out after waiting]` in `text`, and
`orion-cortex-exec` persisted them as `trace_role="reasoning"` -- 936 rows of
`orion_metacognitive_trace` back to 2026-08-16 in which Orion's recorded reasoning
is a transport error message.

This is deliberately a narrow sensor over a known, single-producer format, not a
general-purpose text classifier. If the gateway ever gains a structured failure
channel, that becomes the right signal and this module should be retired.
"""

from __future__ import annotations

# The exact prefix llm_backend.py emits. Byte-identical to the literal it replaced.
BACKEND_ERROR_PREFIX = "[Error: "


def is_backend_error_text(text: str | None) -> bool:
    """True when `text` is the gateway's failure sentinel rather than model output.

    Leading whitespace is tolerated because callers routinely `.strip()` at
    different points in the chain; anything else is a real answer, including an
    answer that merely *mentions* an error, since the sentinel is a prefix.
    """
    if not text:
        return False
    return text.lstrip().startswith(BACKEND_ERROR_PREFIX)
