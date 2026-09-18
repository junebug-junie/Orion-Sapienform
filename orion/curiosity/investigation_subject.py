"""Short investigation subject for Mind appraisal.

Curiosity's kickoff prompt is a long harness brief (schema, Cypher teach
blocks, HelpRequest instructions). Mind should appraise the *subject of
the work* — the claim under investigation and any continuation note —
not the operator manual. The motor still receives the full kickoff.
"""

from __future__ import annotations


def build_investigation_subject(
    *,
    claim: str | None,
    continue_note: str | None,
    max_chars: int = 1200,
) -> str:
    claim_s = (claim or "").strip()
    note_s = (continue_note or "").strip()
    claim_line = claim_s if claim_s else "Investigation claim: not yet chosen."
    note_line = f"Continue note: {note_s}" if note_s else "Continue note: (none)."
    text = (
        "Orion investigation subject (self-authored).\n"
        f"{claim_line}\n"
        f"{note_line}\n"
    )
    return text.strip()[: max(64, int(max_chars))]
