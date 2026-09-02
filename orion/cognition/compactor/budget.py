from __future__ import annotations

from orion.cognition.compactor.truncate import truncate_at_word_boundary


def fit_fields_within_budget(fields: dict[str, tuple[str, int]]) -> tuple[dict[str, str], list[str]]:
    """Trim over-budget compactor output at a word boundary instead of rejecting it.

    This replaces the older ``assert_fields_within_budget``, which raised
    ``compactor_output_over_budget:<field>`` and turned an otherwise-complete
    digest into a hard workflow failure. That was the wrong trade: by the time
    these fields are checked the payload has already parsed as JSON and passed
    the digest model's own validation, so an over-long ``card_summary`` is a
    formatting miss on structurally sound content -- and the full narrative is
    preserved in ``journal_body`` regardless. Failing instead fed the scheduler's
    retry path, which re-ran the whole digest (LLM call included) until the model
    happened to land under the cap. Confirmed live 2026-08-27 and 2026-08-30:
    5 `compactor_output_over_budget:card_summary` failures on github_compactor_pass,
    each one a complete digest thrown away.

    Every returned value is guaranteed ``len(value) <= max_chars`` for any input:
    the ellipsis ``truncate_at_word_boundary`` appends is paid for out of the
    budget, not added on top of it, so the caller's cap still holds exactly.
    ``len`` is the right measure here -- the caps are prompt/display conventions
    on ``MemoryCardCreateV1.summary``, a plain ``str`` with no length constraint
    and no byte-bounded column behind it, so UTF-8 byte length is not the bound.

    Returns ``(fitted_values, trimmed_field_names)``. The names are returned rather
    than logged here so callers can surface the repair as run evidence instead of
    silently shortening Orion's own summary of its work.
    """
    fitted: dict[str, str] = {}
    trimmed: list[str] = []
    for name, (value, max_chars) in fields.items():
        text = value or ""
        limit = int(max_chars)
        if len(text) <= limit:
            fitted[name] = text
            continue
        if limit < 2:
            # No room to spend a character on an ellipsis and still stay under the
            # cap, so take a hard slice. (`truncate_at_word_boundary` always
            # appends one, which would land at limit + 1.)
            fitted[name] = text[:max(0, limit)]
        else:
            # Reserve one character for the ellipsis so the result lands at or
            # under `limit`, not `limit + 1`.
            fitted[name], _ = truncate_at_word_boundary(text, limit - 1)
        trimmed.append(name)
    return fitted, sorted(trimmed)
