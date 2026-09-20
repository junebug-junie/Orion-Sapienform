"""Format Mind work-shape (+ optional progress) for role-teach extra_lines.

Pure helpers: allow-listed advisory lines only. Orion still authors
:InvestigationRole / HelpRequest; Python never MERGEs hire choice.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_WORK_SHAPE_KEYS = ("expected_depth", "cross_cutting")
_FORESIGHT_KEY = "foresight_note"
_USER_INTENT_KEY = "user_intent"
_MAX_FORESIGHT_CHARS = 240
_DISCLOSURE_HEADER = "Mind work-shape for this sitting (advisory):"
_CONTRACTOR_HELP_MARKER = "ASKING FOR CONTRACTOR HELP"
_ROLE_SITTING_MARKER = "YOUR ROLE FOR THIS SITTING"

_LABELS = {
    "expected_depth": "expected depth",
    "cross_cutting": "cross-cutting",
}
_DEEP_HIRE_NUDGE = (
    "Mind reads this sitting as deep work. Strongly prefer hire_cursor for the "
    "archaeology; keep a short local look only so tried_summary is grounded. "
    "You still author priors and findings."
)


def format_access_refusal_progress(count: int) -> list[str]:
    if count < 2:
        return []
    return [
        "Access refused at least twice this sitting. Hand off to Cursor now "
        "(write hire_cursor + HelpRequest with what you already tried)."
    ]


def format_budget_spent_progress(*, status: str, next_hop_n: int | None = None) -> list[str]:
    if status != "refused_budget":
        return []
    hop = f"hop {next_hop_n}" if next_hop_n is not None else "your last hop notes"
    return [
        f"Cursor budget is spent. Do not open another HelpRequest until budget is clear. "
        f"Resume from {hop} / continue local crawl from what you already wrote."
    ]


def _as_short_str(value: Any, *, limit: int | None = None) -> str | None:
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if limit is not None:
        text = text[:limit]
    return text or None


def _is_missing_or_unknown(value: Any) -> bool:
    text = _as_short_str(value)
    return text is None or text.lower() == "unknown"


def format_role_teach_disclosure(
    mind_work_shape: Mapping[str, Any] | None,
    *,
    progress_lines: Sequence[str] = (),
) -> list[str]:
    """Build short advisory lines for role-teach ``extra_lines``.

    Allow-list: ``expected_depth``, ``cross_cutting``, ``foresight_note``, and
    optionally ``user_intent`` when present as a short string. Returns ``[]``
    when there is nothing useful to disclose (None/empty shape, or all present
    work-shape values missing/``unknown`` with empty foresight/intent and no
    progress). Literal ``unknown`` foresight is treated as empty.
    """
    cleaned_progress = [str(line).strip() for line in progress_lines if str(line).strip()]

    if not mind_work_shape:
        if cleaned_progress:
            return [_DISCLOSURE_HEADER, *cleaned_progress]
        return []

    depth = _as_short_str(mind_work_shape.get("expected_depth"))
    cross = _as_short_str(mind_work_shape.get("cross_cutting"))
    foresight = _as_short_str(
        mind_work_shape.get(_FORESIGHT_KEY), limit=_MAX_FORESIGHT_CHARS
    )
    if foresight and foresight.lower() == "unknown":
        foresight = None
    user_intent = _as_short_str(
        mind_work_shape.get(_USER_INTENT_KEY), limit=_MAX_FORESIGHT_CHARS
    )

    work_values_empty = all(
        _is_missing_or_unknown(mind_work_shape.get(key)) for key in _WORK_SHAPE_KEYS
    )
    if (
        work_values_empty
        and not foresight
        and not user_intent
        and not cleaned_progress
    ):
        return []

    lines: list[str] = [_DISCLOSURE_HEADER]
    if depth and depth.lower() != "unknown":
        lines.append(f"- {_LABELS['expected_depth']}: {depth}")
    if cross and cross.lower() != "unknown":
        lines.append(f"- {_LABELS['cross_cutting']}: {cross}")
    if foresight:
        lines.append(f"- foresight: {foresight}")
    if user_intent:
        lines.append(f"- intent: {user_intent}")
    if depth and depth.lower() == "deep":
        lines.append(_DEEP_HIRE_NUDGE)
    lines.extend(cleaned_progress)
    return lines


def splice_role_teach_disclosure(prompt: str, extra_lines: Sequence[str]) -> str:
    """Insert disclosure lines into a role-teach / harness prompt.

    Prefer insert immediately before the ``ASKING FOR CONTRACTOR HELP`` line.
    If neither that marker nor ``YOUR ROLE FOR THIS SITTING`` is present,
    return ``prompt`` unchanged (never prepend into non-role-teach text).
    Idempotent: if already spliced, return ``prompt`` unchanged.
    """
    cleaned = [str(line) for line in extra_lines if str(line).strip()]
    if not cleaned:
        return prompt

    block = "\n".join(cleaned)
    # Idempotent: header (or whole block) already present.
    if _DISCLOSURE_HEADER in prompt or block in prompt:
        return prompt

    # Only splice into real role-teach prompts — never prepend elsewhere.
    if (
        _CONTRACTOR_HELP_MARKER not in prompt
        and _ROLE_SITTING_MARKER not in prompt
    ):
        return prompt

    lines = prompt.splitlines(keepends=True)
    insert_at: int | None = None
    for idx, line in enumerate(lines):
        if _CONTRACTOR_HELP_MARKER in line:
            insert_at = idx
            break

    if insert_at is None:
        # Role marker present but no contractor-help line: leave prompt alone
        # rather than inventing a prepend site.
        return prompt

    block_with_gap = block + "\n\n"
    before = "".join(lines[:insert_at])
    after = "".join(lines[insert_at:])
    if before and not before.endswith("\n"):
        before += "\n"
    return before + block_with_gap + after
