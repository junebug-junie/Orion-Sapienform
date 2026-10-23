"""Pictures the interlocutor tied to a person (2026-10-23).

A camera cluster can hold more than one person, so "the newest pictures of the
person's clusters" can show three different people on a "Who is X to you?"
card. These are only pictures the interlocutor named or picked:

- the first picture of a "who is this?" card she answered with the cluster's
  current name (the card asks about that picture);
- a picture she marked "the same person" on such a card;
- a face she named on a "same person?" card;
- a picture she picked as that name on an "Is this <name>?" card
  (vision_person_exemplar, source ``confirm_pick_yes``).

One SQL string for both readers: orion-sql-writer (SQLAlchemy, ``:ids``) and the
Hub (asyncpg, ``$1::text[]``). Rows: person_id, image_ref, at -- newest first.
"""

from __future__ import annotations

CONFIRM_PICK_YES = "confirm_pick_yes"
CONFIRM_PICK_NO = "confirm_pick_no"

_CONFIRMED_PICTURES_SQL = """
SELECT person_id, ref AS image_ref, at FROM (
  SELECT l.person_id, coalesce(a.image_refs->>0, a.image_ref) AS ref, a.answered_at AS at
  FROM person_link l
  JOIN vision_individual i ON i.individual_id = l.source_ref
  JOIN orion_ask a ON a.source_kind = 'vision_individual' AND a.source_ref = l.source_ref
   AND a.status = 'answered' AND lower(btrim(a.answer)) = lower(btrim(i.label))
  WHERE l.source_kind = 'vision_individual' AND l.revoked_at IS NULL AND l.person_id = ANY(@IDS@)
  UNION ALL
  SELECT l.person_id, v.image_ref, v.answered_at
  FROM person_link l
  JOIN vision_individual i ON i.individual_id = l.source_ref
  JOIN vision_ask_picture_verdict v ON v.individual_id = l.source_ref AND v.verdict = 'same'
   AND lower(btrim(v.name)) = lower(btrim(i.label))
  WHERE l.source_kind = 'vision_individual' AND l.revoked_at IS NULL AND l.person_id = ANY(@IDS@)
  UNION ALL
  SELECT l.person_id,
         a.image_refs->>(CASE WHEN split_part(a.source_ref, '|', 1) = l.source_ref THEN 0 ELSE 1 END),
         a.answered_at
  FROM person_link l
  JOIN vision_individual i ON i.individual_id = l.source_ref
  JOIN orion_ask a ON a.source_kind = 'vision_same_individual' AND a.status = 'answered'
   AND l.source_ref IN (split_part(a.source_ref, '|', 1), split_part(a.source_ref, '|', 2))
   AND lower(btrim(a.answer_detail->'names'->>l.source_ref)) = lower(btrim(i.label))
  WHERE l.source_kind = 'vision_individual' AND l.revoked_at IS NULL AND l.person_id = ANY(@IDS@)
  UNION ALL
  SELECT p.person_id, coalesce(c.face_thumb_ref, c.thumb_ref), e.added_at
  FROM vision_person_exemplar e
  JOIN vision_crop_observation c ON c.crop_id = e.crop_id
  JOIN person p ON lower(btrim(p.display_name)) = lower(btrim(e.person_label))
  WHERE e.source = 'confirm_pick_yes' AND e.polarity = 1 AND e.revoked_at IS NULL
    AND p.person_id = ANY(@IDS@)
) x
WHERE ref LIKE 'thumb:%'
ORDER BY person_id, at DESC NULLS LAST, ref
"""


def confirmed_pictures_sql(ids_param: str) -> str:
    """The query with its one parameter, e.g. ``":ids"`` or ``"$1::text[]"``."""
    return _CONFIRMED_PICTURES_SQL.replace("@IDS@", ids_param)


def newest_distinct(rows, limit: int) -> dict:
    """person_id -> up to ``limit`` refs, newest first, no repeats (rows as the query orders them)."""
    out: dict = {}
    for person_id, ref, _at in rows:
        refs = out.setdefault(person_id, [])
        if ref not in refs and len(refs) < limit:
            refs.append(ref)
    return out
