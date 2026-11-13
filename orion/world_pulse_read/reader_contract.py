"""The reader's JSON output contract, fenced so the stance pass does not see it.

A reading turn is one unified turn. Its prompt goes to two LLM passes:

1. the stance pass (``stance_react``), which must answer in ThoughtEventV1 JSON
   (``imperative``, ``tone``, ...), and
2. the reader (the FCC motor), which must answer in the reading JSON
   (``summary``, ``article_summary``, ... or ``what_i_learned``, ...).

Both contracts used to reach the stance pass. Live 2026-10-26..28, 13 of 102
failed ``reading.turn`` runs were the stance pass mixing them up (raw stance
text in ``cognition_traces``, verb ``stance_react``):

- 6 of 6 reading stances served by Qwen3.6-35B wrote the reader's keys:
  ``"summary"`` in the ``imperative`` slot (4) or the whole reading answer (2).
  The same model gave a valid stance on 5 of 5 non-reading turns.
- 3 Bonsai-2-27B stances spent the whole 8000-token budget reasoning ("There
  is tension ... Which schema?") and returned empty content
  (``stance_react exec result missing thought payload``).

The reader prompt builders wrap the contract in these fences.
``stance_view_of_reading_prompt`` swaps the fenced block for one line, so the
stance pass still knows a strict output shape exists but sees only its own.
The reader still receives the full prompt.
"""

from __future__ import annotations

READER_CONTRACT_BEGIN = "<<<READER OUTPUT CONTRACT>>>"
READER_CONTRACT_END = "<<<END READER OUTPUT CONTRACT>>>"

STANCE_CONTRACT_PLACEHOLDER = (
    "[Here the reader gets its required JSON output shape. That shape is for the "
    "downstream reader only, so this stance pass does not see it.]"
)


def defuse_fence_markers(untrusted: str) -> str:
    """Untrusted text (a page, a document, a Stage 1 echo) must not carry a fence
    marker: one would make the stance view fail closed (full two-contract prompt) or
    hide that text from the stance. Swaps the marker's ``<<<`` for look-alike
    guillemets; nothing else changes."""
    text = str(untrusted or "")
    for marker in (READER_CONTRACT_BEGIN, READER_CONTRACT_END):
        text = text.replace(marker, marker.replace("<<<", "\u2039\u2039\u2039"))
    return text


def fence_reader_contract(contract: str) -> str:
    """Wrap the reader's output contract in the fences. Ends with a newline."""
    body = str(contract or "").strip("\n")
    return f"{READER_CONTRACT_BEGIN}\n{body}\n{READER_CONTRACT_END}\n"


def stance_view_of_reading_prompt(prompt: str) -> str | None:
    """The prompt with each fenced reader-contract block replaced by one line.

    None when the prompt has no fence, or the fences do not pair up in order
    (BEGIN, END, BEGIN, END, ...): a brief written before the fences existed,
    or a damaged prompt. The caller then gives the stance pass the full
    prompt, the same as before this change.
    """
    text = str(prompt or "")
    if READER_CONTRACT_BEGIN not in text:
        return None
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(READER_CONTRACT_BEGIN, pos)
        next_end = text.find(READER_CONTRACT_END, pos)
        if start < 0:
            if next_end >= 0:
                return None  # an END with no BEGIN
            out.append(text[pos:])
            return "".join(out)
        end = text.find(READER_CONTRACT_END, start)
        if end < 0 or next_end < start:
            return None  # BEGIN with no END, or an END before this BEGIN
        if text.find(READER_CONTRACT_BEGIN, start + len(READER_CONTRACT_BEGIN), end) >= 0:
            return None  # nested BEGIN
        out.append(text[pos:start])
        out.append(STANCE_CONTRACT_PLACEHOLDER)
        pos = end + len(READER_CONTRACT_END)
