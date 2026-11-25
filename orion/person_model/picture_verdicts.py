"""Juniper's newest answer on a picture wins (2026-11-25, issue #3854).

A picture verdict (``vision_ask_picture_verdict``) says a picture is ("same")
or is not ("different") the person a card asked about. She can answer the same
picture twice, e.g. "someone else" and then, 22 seconds later, "yes, that's
her". The newer firm answer replaces the older one for that picture and that
person; "unsure" never replaces a firm answer.

"That picture": the same crop (``crop_id``) or the same shown image
(``image_ref``). "That person": the same asked group, or a group carrying the
same name now (case-insensitive).

One SQL fragment for every reader that acts on a picture's identity: the
person model's album, the split on "someone else" pictures, and the People
panel's pictures.
"""

from __future__ import annotations

FIRM = ("same", "different")


def overridden_sql(alias: str) -> str:
    """SQL boolean: a newer opposite firm answer exists for the verdict row
    ``alias`` (a ``vision_ask_picture_verdict`` alias, verdict 'same' or
    'different') on the same picture for the same person. Readers keep a row
    only when this is false; the newest answer is never overridden. Two opposite
    answers with the same ``answered_at`` both stand (separate clicks, so this
    does not happen in practice). No bind parameters: safe for SQLAlchemy and asyncpg."""
    a = alias
    return (
        "EXISTS (SELECT 1 FROM vision_ask_picture_verdict o "
        "LEFT JOIN vision_individual oi ON oi.individual_id = o.individual_id "
        f"LEFT JOIN vision_individual ai ON ai.individual_id = {a}.individual_id "
        f"WHERE o.verdict IN ('same', 'different') AND o.verdict <> {a}.verdict "
        f"AND (o.crop_id = {a}.crop_id OR o.image_ref = {a}.image_ref) "
        f"AND o.answered_at > {a}.answered_at "
        f"AND (o.individual_id = {a}.individual_id "
        "OR (btrim(ai.label) <> '' AND lower(btrim(oi.label)) = lower(btrim(ai.label)))))"
    )
