"""Walkway stories spot-check card (motion stories slice 1, 2026-10-16).

docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md.
``orion-sql-writer`` (app/vision_motion_loop.py) writes one row per passing
thing into ``vision_motion_story``: a plain sentence plus the boxes it rests
on. Juniper decided (2026-10-16) that stories stay HUB ONLY until she has
spot-checked them: at least ``GATE_MIN_CHECKED`` checked, at least
``GATE_MIN_RIGHT`` fully right, and zero wrong names. This module is that
spot-check:

- ``GET /api/vision/motion-stories`` lists recent published stories (clean
  frames by default) with their evidence boxes and thumbnails, and the
  running tally against the gate.
- ``POST /api/vision/motion-stories/{story_id}/verdict`` records Right /
  Wrong name / Wrong motion / Wrong other in ``vision_motion_story_verdict``.
  One verdict per story; a new click replaces the old one. "Can't tell"
  (2026-10-17: no picture, or too dark) is stored as ``verdict NULL,
  cant_tell true``: the story leaves the unchecked queue but never counts as
  right or wrong.
- ``GET /api/vision/motion-stories/{story_id}/strip/{name}`` serves the
  story's pictures of the moment (``crop_0..3.jpg``, ``frame.jpg``) that
  orion-sql-writer saved while the frames still existed
  (app/vision_story_strip.py). ``checkable_only`` (default on) lists only
  stories with a saved frame that is not dark, so the queue is all judgeable.

The tally counts one rules version at a time (the newest one with stories):
a verdict on old rules never vouches for new ones. No bus event: nothing
consumes a verdict yet, and the table is the only truth.

**Who the story is about** (2026-10-18, Juniper: "it gives no option to name
them or correct the name. Make sure it connects to the broader detection
recognition."). Each listed story carries ``people``: one entry per distinct
``vision_individual`` in its evidence crops (``role='walker'``) and in the
crops of the stories it walked together with (``role='companion'``), with the
individual's current name. ``POST /api/vision/motion-stories/{story_id}/people/{individual_id}``
records what Juniper says about one of them. No parallel identity store: every
action is an ``orion_ask`` row, written already answered, that the existing
consumers apply:

- ``name``: a ``vision_individual`` ask (a "who is this?" answer) with
  ``answer`` = the name and ``answer_detail.story_id``. orion-sql-writer sets it
  as a TYPED label (``label_ask_id`` = this ask, last typed name wins,
  ``[VISION_LABEL_REPLACED]``), copies it to everyone confirmed the same person,
  closes their open "who is this?" cards, and links the individual to an
  existing one she typed the same name for. ``OrionAskAnsweredV1`` is published
  so orion-substrate-runtime writes the substrate entity.
- ``confirm`` ("Yes, that's <name>", 2026-11-24; only on a row that has a name,
  and only the name the card showed: 409 ``label_changed`` otherwise): the same
  ``vision_individual`` answer as a typed name (so a copied name she said yes to
  becomes her typed label), with ``answer_detail.pictures`` = one "same" picture
  verdict per picture of that individual the card SHOWED.
- ``someone_else`` (optionally with a name for who it really is) / ``unsure``:
  a ``vision_story_person`` ask whose ``answer_detail.pictures`` holds one
  picture verdict ("different" / "unsure") per picture of that
  individual the card SHOWED. orion-sql-writer stores them in
  ``vision_ask_picture_verdict``; "someone else" ones are split off by the
  split-on-verdict pass (app/vision_split.py).

**Each person's own pictures** (2026-11-24, Juniper: "how the f would i know
which is which"). Every person carries ``pictures`` (the strip crops that are
theirs, by crop_id), ``boxes`` (their numbered boxes on the story's frame) and
``pictured``. A person with no picture of their own is shown but not asked: the
person route refuses them (``no_picture_of_person``).

A correction of the name the story was written with (a different name, or
"someone else" on the named walker) also records the story's "Wrong name"
verdict, so the gate tally sees it. The story sentence is re-derived when it is
listed (``display_sentence``): a direct answer on THIS story wins for it;
otherwise the same strict rule sql-writer uses (orion.vision.story_naming) runs
on the names individuals carry now. ``sentence`` stays what was written.

**How she could tell** (2026-11-24, Juniper: "someone walking with back turned --
orion asks if it is jason? I can say no because he is pushing a stroller and I
know jason doesnt have young kids..."). Every person action takes an optional
free-text ``why``, stored as ``answer_detail.why`` on the same ask row and shown
back on the card. Nothing reasons over it yet: it is the raw context-clue record
a later primitive (who someone is from more than their face/build) will read.

**One person per linked group** (2026-11-24, who-card redesign Patch A,
docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md). Each listed story
also carries ``persons``: camera groups joined by 'same' links shown as one person
(orion/vision/story_persons.py). A person answer with ``group: true`` is written for
each of its camera groups on that group's shown pictures (at most four rows); a
repeat of a group's latest answer on the story is skipped. ``POST .../persons/{person_key}/not-same``
records "these are not the same person" for one direct link.

Same asyncpg pool as the ask card (``app.state.memory_pg_pool``); all DB calls
are awaited (``scripts/check_async_routes_not_blocking.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from orion.vision import story_persons
from orion.vision.story_naming import evidence_crops, name_from_crops, rename_subject
from scripts import ask_routes
from scripts.ask_routes import WHY_MAX_CHARS, _iso, _pool, _raise_store_http, _read_thumb, clean_name, clean_why


def _raise_motion_http(exc: BaseException) -> None:
    """The ask card's store-error mapping (schema missing / transport -> 503), with this
    card's own detail names so a 503 here is not mistaken for the ask table."""
    try:
        _raise_store_http(exc)
    except HTTPException as http:
        detail = str(http.detail).replace("ask_", "motion_story_", 1)
        raise HTTPException(status_code=http.status_code, detail=detail) from exc

logger = logging.getLogger("orion-hub.motion_stories")

router = APIRouter(tags=["motion-stories"])

GATE_MIN_CHECKED = 50
GATE_MIN_RIGHT = 0.80
GATE_MAX_WRONG_NAMES = 0
VERDICTS = ("right", "wrong_name", "wrong_motion", "wrong_other")
CANT_TELL = "cant_tell"
MAX_LIST = 50
_STORY_ID_RE = re.compile(r"[0-9a-f]{32}")
# The only file names sql-writer writes into a strip directory.
_STRIP_NAME_RE = re.compile(r"(?:crop_[0-3]|frame)\.jpg")
DEFAULT_STRIP_DIR = "/mnt/telemetry/vision/story_strips"

_LIST_SQL = """
SELECT s.story_id, s.stream_id, s.kind, s.started_at, s.ended_at, s.detection_count, s.used_detections,
       s.speed_class, s.zone_path, s.dir_lateral, s.dir_depth, s.individual_id, s.name_label,
       s.companion_kinds, s.clean, s.sentence, s.evidence, s.rules_version,
       CASE WHEN v.cant_tell THEN 'cant_tell' ELSE v.verdict END AS verdict,
       s.evidence_strip, s.frame_luminance, s.scene_luminance, s.scene_contrast, s.dark,
       s.together_story_ids
FROM vision_motion_story s
LEFT JOIN vision_motion_story_verdict v ON v.story_id = s.story_id
WHERE s.published AND ($1::boolean IS FALSE OR s.clean)
  AND ($2::boolean IS FALSE OR v.story_id IS NULL)
  AND ($4::integer IS NULL OR s.rules_version = $4)
  AND ($5::boolean IS FALSE OR coalesce(s.dark IS FALSE AND s.evidence_strip->'frame'->>'file' IS NOT NULL
                                         AND NOT (s.evidence_strip ? 'pruned_at'), false))
ORDER BY s.started_at DESC
LIMIT $3
"""

_TALLY_SQL = """
SELECT CASE WHEN v.cant_tell THEN 'cant_tell' ELSE v.verdict END AS verdict, count(*) AS n,
       count(*) FILTER (WHERE s.name_label IS NOT NULL) AS named
FROM vision_motion_story_verdict v
JOIN vision_motion_story s ON s.story_id = v.story_id
WHERE s.published AND s.clean AND s.rules_version = $1
GROUP BY 1
"""


_CHECKABLE = ("s.dark IS FALSE AND s.evidence_strip->'frame'->>'file' IS NOT NULL "
              "AND NOT (s.evidence_strip ? 'pruned_at')")
_NOT_CHECKABLE_SQL = (
    "SELECT count(*) FROM vision_motion_story s WHERE s.published AND s.clean AND s.rules_version = $1 "
    f"AND NOT coalesce({_CHECKABLE}, false)"
)


_UPSERT_VERDICT = """
INSERT INTO vision_motion_story_verdict (story_id, verdict, rules_version, cant_tell)
VALUES ($1, $2, $3, $4)
ON CONFLICT (story_id) DO UPDATE
  SET verdict = EXCLUDED.verdict, rules_version = EXCLUDED.rules_version,
      cant_tell = EXCLUDED.cant_tell, updated_at = now()
"""

# The automatic "Wrong name" from a name correction: only on a story with no
# verdict yet, "Right", or "Can't tell" (verdict NULL). Returns the story id when written.
_WRONG_NAME_VERDICT = """
INSERT INTO vision_motion_story_verdict AS v (story_id, verdict, rules_version, cant_tell)
VALUES ($1, 'wrong_name', $2, false)
ON CONFLICT (story_id) DO UPDATE
  SET verdict = 'wrong_name', rules_version = EXCLUDED.rules_version, cant_tell = false, updated_at = now()
  WHERE v.verdict IS NULL OR v.verdict IN ('right', 'wrong_name')
RETURNING story_id
"""

# One story, the list's columns (the person route returns it re-derived).
_ONE_STORY_SQL = _LIST_SQL.split("FROM vision_motion_story s")[0] + (
    "FROM vision_motion_story s LEFT JOIN vision_motion_story_verdict v ON v.story_id = s.story_id "
    "WHERE s.story_id = $1")


class VerdictBody(BaseModel):
    verdict: Literal["right", "wrong_name", "wrong_motion", "wrong_other", "cant_tell"]


WHO_IS_THIS_KIND = ask_routes.WHO_IS_THIS_KIND        # "vision_individual"
STORY_PERSON_KIND = "vision_story_person"
JUNIPER_STORY_CARD_REF = "rule:juniper_story_naming"   # == orion.grammar.camera_ask_acts (a test pins it)
PERSON_ACTIONS = ("name", "confirm", "someone_else", "unsure")
_INDIVIDUAL_ID_RE = re.compile(r"[0-9A-Za-z_.:-]{1,128}")
_THUMB_HEX_RE = re.compile(r"[0-9a-f]{64}")


class PersonBody(BaseModel):
    # "confirm" (2026-11-24): "Yes, that's <name>" on a row that already has a name.
    action: Literal["name", "confirm", "someone_else", "unsure"]
    # For "name": who this is. For "someone_else": optionally who it really is. Ignored for "confirm".
    name: str = Field(default="", max_length=ask_routes.MAX_ANSWER_CHARS)
    # How Juniper could tell, in her words (optional, any action): "pushing a stroller,
    # Jason has no young kids, not his build". Stored as answer_detail.why on the ask row.
    why: str = Field(default="", max_length=WHY_MAX_CHARS)
    # Who-card Patch A (2026-11-24): the answer is for the whole linked person (every camera group
    # the card showed a picture of), not only ``individual_id``'s group.
    group: bool = False
    # The crop ids of the pictures the card SHOWED for this person (group answers). When the
    # server's pictures for the person differ (a link landed since the card rendered), 409
    # ``pictures_changed``: never a verdict on a picture she did not see.
    crop_ids: Optional[list[str]] = Field(default=None, max_length=16)


def tally_from_counts(counts: dict[str, int], named_checked: int, rules_version: int | None) -> dict[str, Any]:
    checked = sum(counts.get(v, 0) for v in VERDICTS)
    right = counts.get("right", 0)
    wrong_names = counts.get("wrong_name", 0)
    pct = right / checked if checked else None
    passed = (checked >= GATE_MIN_CHECKED and pct is not None and pct >= GATE_MIN_RIGHT
              and wrong_names <= GATE_MAX_WRONG_NAMES)
    return {
        "rules_version": rules_version,
        "checked": checked,
        "right": right,
        "wrong_name": wrong_names,
        "wrong_motion": counts.get("wrong_motion", 0),
        "wrong_other": counts.get("wrong_other", 0),
        "named_checked": named_checked,
        # Juniper could not judge these (no picture, too dark): never right, never wrong.
        "cant_tell": counts.get(CANT_TELL, 0),
        "pct_right": round(pct, 4) if pct is not None else None,
        "gate": {"min_checked": GATE_MIN_CHECKED, "min_right": GATE_MIN_RIGHT,
                 "max_wrong_names": GATE_MAX_WRONG_NAMES, "passed": passed},
    }


async def _tally(conn: Any) -> dict[str, Any]:
    version = await conn.fetchval(
        "SELECT max(rules_version) FROM vision_motion_story WHERE published AND clean")
    if version is None:
        return tally_from_counts({}, 0, None)
    rows = await conn.fetch(_TALLY_SQL, version)
    counts = {r["verdict"]: int(r["n"]) for r in rows if r["verdict"]}
    named = sum(int(r["named"]) for r in rows if r["verdict"] in VERDICTS)
    out = tally_from_counts(counts, named, int(version))
    # The default queue is checkable stories only, so the gate speaks for those.
    # Say how many published stories it does not cover (dark, or no picture).
    out["not_checkable"] = int(await conn.fetchval(_NOT_CHECKABLE_SQL, version) or 0)
    return out


def _story_dict(row: Any) -> dict[str, Any]:
    out = dict(row)
    ev = out.get("evidence")
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except ValueError:
            ev = []
    out["evidence"] = list(ev or [])
    strip = out.get("evidence_strip")
    if isinstance(strip, str):
        try:
            strip = json.loads(strip)
        except ValueError:
            strip = None
    out["evidence_strip"] = strip if isinstance(strip, dict) else None
    for key in ("zone_path", "companion_kinds", "together_story_ids"):
        out[key] = list(out.get(key) or [])
    for key in ("started_at", "ended_at"):
        out[key] = _iso(out.get(key))
    return out


@router.get("/api/vision/motion-stories")
async def list_motion_stories(
    request: Request,
    limit: int = Query(20, ge=1, le=MAX_LIST),
    clean_only: bool = Query(True),
    unchecked_only: bool = Query(False),
    checkable_only: bool = Query(True),
) -> dict[str, Any]:
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            # Only stories under the rules the tally counts: a verdict on an
            # older rules version would never count toward the gate.
            tally = await _tally(conn)
            rows = await conn.fetch(_LIST_SQL, clean_only, unchecked_only, limit, tally["rules_version"],
                                    checkable_only)
            stories = [_story_dict(r) for r in rows]
            await _enrich(conn, stories)
    except Exception as exc:
        _raise_motion_http(exc)
    return {"ok": True, "stories": stories, "tally": tally}


@router.post("/api/vision/motion-stories/{story_id}/verdict")
async def record_motion_story_verdict(story_id: str, body: VerdictBody, request: Request) -> dict[str, Any]:
    if not _STORY_ID_RE.fullmatch(story_id or ""):
        raise HTTPException(status_code=400, detail="bad_story_id")
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                version = await conn.fetchval(
                    "SELECT rules_version FROM vision_motion_story WHERE story_id = $1 AND published", story_id)
                if version is None:
                    raise HTTPException(status_code=404, detail="story_not_found")
                await conn.execute(
                    _UPSERT_VERDICT,
                    story_id, None if body.verdict == CANT_TELL else body.verdict, version,
                    body.verdict == CANT_TELL,
                )
            tally = await _tally(conn)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_motion_http(exc)
    logger.info("motion_story_verdict story_id=%s verdict=%s", story_id, body.verdict)
    return {"ok": True, "story_id": story_id, "verdict": body.verdict, "tally": tally}


# --- who the story is about -----------------------------------------------------------------------

_LABELS_SQL = ("SELECT individual_id, kind, label, label_copied FROM vision_individual "
               "WHERE individual_id = ANY($1::text[])")

# Every answer Juniper gave on these stories, from the ask rows this card writes
# (a handful per story; the latest one per person is picked in Python).
_STORY_ACTIONS_SQL = """
SELECT answer_detail->>'story_id' AS story_id, source_ref AS individual_id,
       answer_detail->>'action' AS action, answer_detail->>'name' AS name,
       answer_detail->>'why' AS why, ask_id, answered_at
FROM orion_ask
WHERE source_kind = ANY($2::text[]) AND status = 'answered'
  AND answer_detail->>'story_id' = ANY($1::text[])
ORDER BY answered_at, created_at, ask_id
"""


def _evidence_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = []
    return [e for e in (raw or []) if isinstance(e, dict)]


# A story's evidence is frozen when the story closes, often seconds before the
# individuals reducer assigns its crops (2026-10-20: 101 of 147 person stories
# over 5 days said "no individual" for crops that were linked). orion-sql-writer
# now fills those nulls in the story row (vision_motion_loop.relink_stories);
# this read covers the seconds before that and stories it has not reached,
# while the crop rows still exist (VISION_CROP_RETENTION_DAYS, 7; person crops are kept).
_LATE_LINKS_SQL = """
SELECT crop_id, individual_id FROM vision_crop_observation
WHERE crop_id = ANY($1::text[]) AND individual_id IS NOT NULL
"""


def unlinked_crop_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [str(e["crop_id"]) for e in evidence if e.get("crop_id") and e.get("individual_id") is None]


def fill_late_links(evidence: list[dict[str, Any]], links: dict[str, str]) -> list[dict[str, Any]]:
    """``evidence`` with null individual_ids filled from ``links`` (crop_id -> id).
    An id already in the evidence is never changed."""
    return [{**e, "individual_id": links[str(e.get("crop_id"))]}
            if e.get("individual_id") is None and links.get(str(e.get("crop_id"))) else e
            for e in evidence]


async def _late_links(conn: Any, evidences: list[list[dict[str, Any]]]) -> dict[str, str]:
    """crop_id -> individual_id for the unlinked crops in ``evidences``. Fail-open
    ({}), inside a savepoint so a failure never aborts the caller's transaction
    (e.g. vision_crop_observation.individual_id missing: manual_migration_vision_split_v1.sql)."""
    ids = sorted({cid for ev in evidences for cid in unlinked_crop_ids(ev)})
    if not ids:
        return {}
    try:
        async with conn.transaction():
            rows = await conn.fetch(_LATE_LINKS_SQL, ids)
    except Exception as exc:  # noqa: BLE001 - the stored evidence still renders
        logger.warning("motion_story_late_links_failed error=%s", exc)
        return {}
    return {r["crop_id"]: r["individual_id"] for r in rows}


def people_in(evidence: list[dict[str, Any]], role: str, kind: str) -> list[dict[str, Any]]:
    """One entry per distinct individual id in ``evidence``, in first-seen order."""
    out: dict[str, dict[str, Any]] = {}
    for e in evidence:
        iid = e.get("individual_id")
        if not iid or not isinstance(iid, str):
            continue
        p = out.setdefault(iid, {"individual_id": iid, "short_id": iid[:8], "role": role, "kind": kind,
                                 "crop_ids": [], "thumb": None, "thumb_crop_id": None})
        if e.get("crop_id"):
            p["crop_ids"].append(str(e["crop_id"]))
        if p["thumb"] is None and isinstance(e.get("thumb"), str) and _THUMB_HEX_RE.fullmatch(e["thumb"]):
            p["thumb"] = e["thumb"]
            p["thumb_crop_id"] = str(e["crop_id"]) if e.get("crop_id") else None
    return list(out.values())


# --- each person's own pictures (2026-11-24) ---------------------------------------------------------
# Juniper, on a story with six "who is this?" rows and one unlabelled picture: "how the f
# would i know which is which". Each person row now carries the pictures of THAT
# individual: the strip crops whose crop_id is one of theirs (sql-writer writes crop_id
# on every strip crop, app/vision_story_strip.py), else vision-host's own thumbnail of
# one of their crops. A walker's boxes on the story's frame are numbered the way
# sql-writer drew them: 1..n over the evidence entries that have a crop_id and a
# 4-number box, first green, last red, the rest blue. A person with no picture of their
# own is not asked: the person route refuses it, so no answer is blind.

def strip_files(strip: Any) -> dict[str, dict[str, Any]]:
    """crop_id -> {"file", "t"} for the saved crops of one story strip ({} when pruned)."""
    if not isinstance(strip, dict) or strip.get("pruned_at"):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for c in strip.get("crops") or []:
        if (isinstance(c, dict) and c.get("crop_id") and isinstance(c.get("file"), str)
                and _STRIP_NAME_RE.fullmatch(c["file"])):
            out.setdefault(str(c["crop_id"]), {"file": c["file"], "t": c.get("t")})
    return out


def box_numbers(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """crop_id -> {"n", "colour"}: the number and colour sql-writer drew for that box
    on the story's frame (app/vision_story_strip.assemble_strip)."""
    ev = [e for e in evidence if isinstance(e, dict) and e.get("crop_id")]
    boxed = [e for e in ev if isinstance(e.get("box"), list) and len(e["box"]) == 4]
    n = len(boxed)
    out: dict[str, dict[str, Any]] = {}
    for j, e in enumerate(boxed):
        colour = "green" if j == 0 else "red" if j == n - 1 else "blue"
        out.setdefault(str(e["crop_id"]), {"n": j + 1, "colour": colour})
    return out


def attach_pictures(person: dict[str, Any], strip_story_id: str, strip: Any,
                    boxes: Optional[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Set ``pictures``, ``boxes``, ``picture_crop_ids`` and ``pictured`` on one person.
    ``strip_story_id`` / ``strip``: the story whose strip holds this person's crops (a
    companion's are in the story it walked in). ``boxes``: the box numbers on the shown
    frame, or None when this person is not boxed on it (a companion, or no frame)."""
    files = strip_files(strip)
    pics = [{"story_id": strip_story_id, "file": files[cid]["file"], "t": files[cid]["t"], "crop_id": cid}
            for cid in dict.fromkeys(person.get("crop_ids") or []) if cid in files]
    person["pictures"] = pics
    person["boxes"] = [boxes[cid] for cid in dict.fromkeys(person.get("crop_ids") or []) if boxes and cid in boxes]
    shown = [p["crop_id"] for p in pics]
    if not shown and person.get("thumb") and person.get("thumb_crop_id"):
        shown = [person["thumb_crop_id"]]
    person["picture_crop_ids"] = shown
    person["pictured"] = bool(shown)
    return person


def _strip_of(raw: Any) -> Optional[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def story_display(story: dict[str, Any], labels: dict[str, str],
                  answers: list[dict[str, Any]]) -> tuple[Optional[str], str, str]:
    """(display name, sentence to show, why) for one story. ``answers``: every
    answer Juniper gave on this story, oldest first, each with its individual_id.

    Her latest name / "yes, that's <name>" / "someone else" about any of the story's
    walker ids wins for this story: a name or a yes -> that name; "someone else" -> who she said it really is,
    or no name. A later "not sure" does not undo it. With none, the strict naming
    rule on the names the individuals carry now (``why='rule'``)."""
    walkers = {p["individual_id"] for p in story.get("people") or [] if p.get("role") == "walker"}
    direct = [a for a in answers if a.get("individual_id") in walkers
              and a.get("action") in ("name", "confirm", "someone_else")]
    if direct:
        name = (direct[-1].get("name") or "").strip() or None
        why = "juniper"
    else:
        _, name, _ = name_from_crops(evidence_crops(story.get("evidence") or []), labels)
        why = "rule"
    sentence = rename_subject(story.get("sentence") or "", story.get("kind") or "person",
                              story.get("name_label"), name)
    return name, sentence, why


# Street continuity (orion-sql-writer app/vision_handoff_loop.py, 2026-10-20): the
# other camera's sighting of the same passage. met / early / late = a predicted handoff
# arrived; paired = both cameras saw them at the same moment on the central walk.
_HANDOFF_SQL = """
SELECT source_story_id, target_story_id, source_stream, target_stream, transit_sec, status
FROM vision_handoff
WHERE status IN ('met', 'early', 'late', 'paired')
  AND (source_story_id = ANY($1::text[]) OR target_story_id = ANY($1::text[]))
ORDER BY scored_at DESC NULLS LAST
"""
CAMERA_WORDS = {"doorbell": "door camera", "walkway": "walkway camera"}


def handoff_info(story_id: str, row: Any) -> Optional[dict[str, Any]]:
    """{"kind": "from" | "next" | "same", "camera", "seconds"} for one story,
    from one vision_handoff row it is part of; None if it is not."""
    if row is None or row["status"] not in ("met", "early", "late", "paired"):
        return None
    if story_id == row["target_story_id"]:
        other = row["source_stream"]
        kind = "same" if row["status"] == "paired" else "from"
    elif story_id == row["source_story_id"]:
        other = row["target_stream"]
        kind = "same" if row["status"] == "paired" else "next"
    else:
        return None
    secs = row["transit_sec"]
    return {"kind": kind, "camera": CAMERA_WORDS.get(other or "", f"{other} camera"),
            "seconds": None if secs is None else int(round(abs(float(secs))))}


async def _handoffs(conn: Any, stories: list[dict[str, Any]]) -> None:
    """``handoff`` on each story (None when no other camera saw this passage).
    Fail-open: before manual_migration_vision_handoff_v1.sql the cards show no line."""
    for s in stories:
        s["handoff"] = None
    if not stories:
        return
    try:
        async with conn.transaction():   # a savepoint: a missing table never aborts a caller's transaction
            rows = await conn.fetch(_HANDOFF_SQL, [s["story_id"] for s in stories])
    except Exception as exc:  # noqa: BLE001 - the card still shows the stored story
        logger.debug("motion_story_handoff_lookup_failed error=%s", exc)
        return
    for s in stories:
        for r in rows:
            info = handoff_info(s["story_id"], r)
            if info is not None:
                s["handoff"] = info
                break


async def _enrich(conn: Any, stories: list[dict[str, Any]]) -> None:
    """Add ``people``, ``display_name``, ``display_sentence``, ``name_source`` and ``handoff``.
    Fail-open: a missing individuals table leaves the stored sentence and no people."""
    await _handoffs(conn, stories)
    if stories:
        links = await _late_links(conn, [s.get("evidence") or [] for s in stories])
        for s in stories:
            s["evidence"] = fill_late_links(s.get("evidence") or [], links)
    for s in stories:
        s["people"] = people_in(s.get("evidence") or [], "walker", s.get("kind") or "person")
        strip = s.get("evidence_strip")
        framed = isinstance(strip, dict) and not strip.get("pruned_at") and isinstance(strip.get("frame"), dict)
        boxes = box_numbers(s.get("evidence") or []) if framed else None
        for p in s["people"]:
            attach_pictures(p, s["story_id"], strip, boxes)
        s["display_name"], s["display_sentence"], s["name_source"] = s.get("name_label"), s.get("sentence"), "stored"
        # Each camera group alone until the links are read (fail-open: a failed read keeps this).
        s["persons"] = story_persons.group_people(s["people"], {})
    if not stories:
        return
    try:
        # A savepoint: a failed read never aborts a caller's transaction (fail-open there too).
        async with conn.transaction():
            together = sorted({t for s in stories for t in s.get("together_story_ids") or []})
            partners: dict[str, Any] = {}
            if together:
                for r in await conn.fetch(
                        "SELECT story_id, kind, evidence, evidence_strip FROM vision_motion_story "
                        "WHERE story_id = ANY($1::text[])", together):
                    partners[r["story_id"]] = {"kind": r["kind"], "evidence": _evidence_list(r["evidence"]),
                                               "strip": _strip_of(r["evidence_strip"])}
                plinks = await _late_links(conn, [r["evidence"] for r in partners.values()])
                for r in partners.values():
                    r["evidence"] = fill_late_links(r["evidence"], plinks)
            for s in stories:
                seen = {p["individual_id"] for p in s["people"]}
                s["_areas"] = story_persons.crop_areas(
                    [s.get("evidence") or []] + [partners[t]["evidence"] for t in s.get("together_story_ids") or []
                                                 if t in partners])
                for t in s.get("together_story_ids") or []:
                    r = partners.get(t)
                    if r is None:
                        continue
                    for p in people_in(r["evidence"], "companion", r["kind"] or "person"):
                        if p["individual_id"] not in seen:
                            seen.add(p["individual_id"])
                            p["story_id"] = t
                            # Their pictures are in the story they walked in; not boxed on this frame.
                            attach_pictures(p, t, r["strip"], None)
                            s["people"].append(p)
            ids = sorted({p["individual_id"] for s in stories for p in s["people"]}
                         | {e.get("individual_id") for s in stories for e in s.get("evidence") or []
                            if isinstance(e.get("individual_id"), str)})
            rows = {r["individual_id"]: r for r in await conn.fetch(_LABELS_SQL, ids)} if ids else {}
            acts: dict[str, list[dict[str, Any]]] = {}
            for r in await conn.fetch(_STORY_ACTIONS_SQL, [s["story_id"] for s in stories],
                                      [WHO_IS_THIS_KIND, STORY_PERSON_KIND]):
                acts.setdefault(r["story_id"], []).append({
                    "individual_id": r["individual_id"], "action": r["action"], "name": r["name"],
                    "why": r["why"] or None, "ask_id": r["ask_id"], "answered_at": _iso(r["answered_at"])})
    except Exception as exc:  # noqa: BLE001 - the card still shows the stored story
        logger.warning("motion_story_people_lookup_failed error=%s", exc)
        for s in stories:
            s.pop("_areas", None)
        return
    labels = {iid: (r["label"] or "").strip() for iid, r in rows.items() if (r["label"] or "").strip()}
    for s in stories:
        mine = acts.get(s["story_id"], [])
        latest = {a["individual_id"]: a for a in mine}   # oldest first: the last one wins
        for p in s["people"]:
            r = rows.get(p["individual_id"])
            label = (r["label"] or "").strip() if r is not None else ""
            p["label"] = label or None
            p["label_copied"] = bool(label and r["label_copied"])
            p["known"] = r is not None
            p["story_action"] = latest.get(p["individual_id"])
        s["display_name"], s["display_sentence"], s["name_source"] = story_display(s, labels, mine)
    await _persons(conn, stories)


# --- one person per linked group (who-card redesign Patch A, 2026-11-24) -----------------------------
# docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md. 304 of 313 live "same person" links
# join two groups with one typed name, and 148 rows on 97 stories repeated an already-linked person
# ("Sasha", "Sasha"). ``persons`` shows each linked group once (orion/vision/story_persons.py); an
# answer with ``group: true`` is written for each camera group in it.

# A 'same' row joins two groups unless a newer "same person? no" about that pair is answered and
# still waiting for sql-writer (which writes it over the row on its next cycle): the card shows
# the two apart at once, and nothing fans out over a link she just said is wrong.
_LIVE_SAME = (
    "v.verdict = 'same' AND NOT EXISTS (SELECT 1 FROM orion_ask q WHERE q.source_kind = 'vision_same_individual' "
    "AND q.status = 'answered' AND q.source_ref = v.individual_a || '|' || v.individual_b "
    "AND lower(btrim(q.answer)) IN ('different', 'no') AND q.answered_at > v.decided_at)")
# Every id reachable from each story person over 'same' links (also through groups not in the story).
_COMPONENT_SQL = """
WITH RECURSIVE g(start, id) AS (
  SELECT x, x FROM unnest($1::text[]) x
  UNION
  SELECT g.start, CASE WHEN v.individual_a = g.id THEN v.individual_b ELSE v.individual_a END
  FROM vision_individual_pair_verdict v JOIN g ON g.id IN (v.individual_a, v.individual_b)
  WHERE """ + _LIVE_SAME + """)
SELECT start, id FROM g WHERE start <> id
"""
# The same walk with one pair left out: would these two still be joined without that row?
_COMPONENT_WITHOUT_SQL = """
WITH RECURSIVE g(start, id) AS (
  SELECT x, x FROM unnest($1::text[]) x
  UNION
  SELECT g.start, CASE WHEN v.individual_a = g.id THEN v.individual_b ELSE v.individual_a END
  FROM vision_individual_pair_verdict v JOIN g ON g.id IN (v.individual_a, v.individual_b)
  WHERE """ + _LIVE_SAME + """ AND NOT (v.individual_a = $2 AND v.individual_b = $3))
SELECT start, id FROM g WHERE start <> id
"""
# The direct 'same' rows between two of these people (one "not the same person?" button each).
_DIRECT_LINKS_SQL = """
SELECT v.individual_a, v.individual_b FROM vision_individual_pair_verdict v
WHERE """ + _LIVE_SAME + """ AND v.individual_a = ANY($1::text[]) AND v.individual_b = ANY($1::text[])
"""


async def _link_rows(conn: Any, ids: list[str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """(id -> 'same' component root, direct links among ``ids``). Inside a savepoint, so a
    failure never aborts the caller's transaction; the caller decides what a failure means."""
    async with conn.transaction():
        reach = await conn.fetch(_COMPONENT_SQL, ids)
        direct = await conn.fetch(_DIRECT_LINKS_SQL, ids)
    roots = story_persons.link_roots([(r["start"], r["id"]) for r in reach])
    return roots, [(r["individual_a"], r["individual_b"]) for r in direct]


async def _persons(conn: Any, stories: list[dict[str, Any]]) -> None:
    """``persons`` on each story: its people grouped by 'same' links. Fail-open: one camera
    group per person (today's card) when the links cannot be read."""
    ids = sorted({p["individual_id"] for s in stories for p in s.get("people") or []})
    roots: dict[str, str] = {}
    links: list[tuple[str, str]] = []
    if ids:
        try:
            roots, links = await _link_rows(conn, ids)
        except Exception as exc:  # noqa: BLE001 - the card still asks each camera group
            logger.warning("motion_story_links_lookup_failed error=%s", exc)
    for s in stories:
        s["persons"] = story_persons.group_people(s.get("people") or [], roots, s.pop("_areas", None), links)


def wrong_name_correction(story_name: Optional[str], role: str, action: str, name: str,
                          current_label: Optional[str] = None) -> bool:
    """Is this answer a correction of the name the story was written with?
    Only about the walker: "someone else", or a name that is neither the written
    one nor the name the individual already carries (re-saving a rename made
    elsewhere is not a verdict on the rule)."""
    if not story_name or role != "walker":
        return False
    if action == "someone_else":
        return True
    typed = name.casefold()
    return (action == "name" and typed != story_name.strip().casefold()
            and typed != (current_label or "").strip().casefold())


_INSERT_STORY_ASK = """
INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, image_ref, image_refs, status, answer,
                       answered_at, created_at, source_kind, source_ref, answer_detail)
VALUES ($1, 'juniper', $2, $3::jsonb, $4, $5::jsonb, 'answered', $6, now(), now(), $7, $8, $9::jsonb)
RETURNING ask_id, status, answer, answered_at, source_kind, source_ref
"""


async def _single_target(conn: Any, story: Any, story_id: str, individual_id: str) -> Optional[dict[str, Any]]:
    """One camera group the card listed (walker, else companion), with the pictures it showed."""
    # The same late links the list shows (_enrich), so a person the card offers can be named.
    ev = _evidence_list(story["evidence"])
    ev = fill_late_links(ev, await _late_links(conn, [ev]))
    person = next((p for p in people_in(ev, "walker", story["kind"])
                   if p["individual_id"] == individual_id), None)
    if person is not None:
        attach_pictures(person, story_id, _strip_of(story["evidence_strip"]), None)
    if person is None and story["together_story_ids"]:
        order = {t: i for i, t in enumerate(story["together_story_ids"])}
        partners = await conn.fetch(
            "SELECT story_id, kind, evidence, evidence_strip FROM vision_motion_story "
            "WHERE story_id = ANY($1::text[])", list(story["together_story_ids"]))
        # The list's order (together_story_ids), so the pictures answered are the ones it showed.
        for r in sorted(partners, key=lambda r: order.get(r["story_id"], len(order))):
            pev = _evidence_list(r["evidence"])
            pev = fill_late_links(pev, await _late_links(conn, [pev]))
            person = next((p for p in people_in(pev, "companion", r["kind"])
                           if p["individual_id"] == individual_id), None)
            if person is not None:
                attach_pictures(person, r["story_id"], _strip_of(r["evidence_strip"]), None)
                break
    return person


async def _group_targets(conn: Any, story_id: str, individual_id: str) -> tuple[Optional[dict[str, Any]],
                                                                                list[dict[str, Any]]]:
    """(the person holding ``individual_id``, its camera groups the card SHOWED a picture of),
    exactly as the list builds it (``_enrich``), each group with only its shown pictures."""
    fresh = [_story_dict(r) for r in await conn.fetch(_ONE_STORY_SQL, story_id)]
    if not fresh:
        return None, []
    async with conn.transaction():       # a savepoint: _enrich's fail-open reads never abort the write
        await _enrich(conn, fresh)
    person = next((p for p in fresh[0].get("persons") or [] if individual_id in p["individual_ids"]), None)
    if person is None:
        return None, []
    by_id = {p["individual_id"]: p for p in fresh[0].get("people") or []}
    targets = [dict(by_id[g["individual_id"]], picture_crop_ids=g["picture_crop_ids"])
               for g in person["groups"] if g["asked"]]
    return person, targets


@router.post("/api/vision/motion-stories/{story_id}/people/{individual_id}")
async def record_story_person(story_id: str, individual_id: str, body: PersonBody,
                              request: Request) -> dict[str, Any]:
    """Name / yes / "someone else" / "not sure" for one person on a story (see the module doc).
    ``group: true``: for every camera group of the linked person that holds ``individual_id``."""
    if not _STORY_ID_RE.fullmatch(story_id or ""):
        raise HTTPException(status_code=400, detail="bad_story_id")
    if not _INDIVIDUAL_ID_RE.fullmatch(individual_id or ""):
        raise HTTPException(status_code=400, detail="bad_individual_id")
    name = clean_name(body.name)
    if body.action == "name" and not name:
        raise HTTPException(status_code=422, detail="name_empty")
    if body.action == "unsure":
        name = ""
    pool = _pool(request)
    rows: list[Any] = []
    skipped: list[str] = []
    wrong_name = False
    person_key: Optional[str] = None
    why = clean_why(body.why)
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                story = await conn.fetchrow(
                    "SELECT story_id, kind, name_label, rules_version, evidence, together_story_ids, evidence_strip "
                    "FROM vision_motion_story WHERE story_id = $1 AND published", story_id)
                if story is None:
                    raise HTTPException(status_code=404, detail="story_not_found")
                if body.group:
                    person, targets = await _group_targets(conn, story_id, individual_id)
                    if person is None:
                        raise HTTPException(status_code=422, detail="individual_not_in_story")
                    person_key = person["person_key"]
                    if body.crop_ids is not None and set(body.crop_ids) != {c["crop_id"] for c in person["pictures"]}:
                        raise HTTPException(status_code=409, detail="pictures_changed")
                else:
                    one = await _single_target(conn, story, story_id, individual_id)
                    if one is None:
                        raise HTTPException(status_code=422, detail="individual_not_in_story")
                    targets = [one] if one["pictured"] else []
                if not targets:
                    # The card shows no picture of them and does not ask: never a blind answer.
                    raise HTTPException(status_code=422, detail="no_picture_of_person")
                ids = [t["individual_id"] for t in targets]
                current = {r["individual_id"]: r["label"] for r in await conn.fetch(
                    "SELECT individual_id, label FROM vision_individual WHERE individual_id = ANY($1::text[])", ids)}
                if any(i not in current for i in ids):
                    raise HTTPException(status_code=404, detail="individual_not_found")
                labels = {i: clean_name(current[i]) for i in ids}
                if (body.action == "someone_else" and name
                        and any(lb and name.casefold() == lb.casefold() for lb in labels.values())):
                    # "Someone else, called <their own name>": contradictory (the split refuses it too).
                    raise HTTPException(status_code=422, detail="someone_else_takes_another_name")
                if body.action == "confirm":
                    # "Yes, that's <name>": only for a person that has a name, and only the name the card
                    # showed her (a rename or relabel since the card rendered must not get her yes). In a
                    # linked person a camera group with no name yet takes the name; another name refuses.
                    named = {lb.casefold(): lb for lb in labels.values() if lb}
                    if not named:
                        raise HTTPException(status_code=422, detail="nothing_to_confirm")
                    if len(named) != 1 or name.casefold() not in named:
                        raise HTTPException(status_code=409, detail="label_changed")
                    name = named[name.casefold()]
                latest = {a["individual_id"]: {"action": a["action"], "name": a["name"]}
                          for a in await conn.fetch(_STORY_ACTIONS_SQL, [story_id],
                                                    [WHO_IS_THIS_KIND, STORY_PERSON_KIND])}
                for t in targets:
                    iid = t["individual_id"]
                    if story_persons.already_said(latest.get(iid), body.action, name):
                        # Idempotent (Orion's question 4): the same answer again writes no second row.
                        skipped.append(iid)
                        continue
                    spec = story_persons.person_answer(
                        action=body.action, name=name, story_id=story_id, picture_crop_ids=t["picture_crop_ids"],
                        why=why, person_key=person_key, group_size=len(targets))
                    thumb = f"thumb:{t['thumb']}" if t.get("thumb") else None
                    refs = [f"motion_story:{story_id}", f"individual:{iid}"]
                    if spec["kind"] == WHO_IS_THIS_KIND:
                        # Juniper's own card, not an ask Orion made: it opens no vision.ask act and counts in no
                        # ask statistic (attention patch 4a, orion/grammar/camera_ask_acts.py).
                        refs.append(JUNIPER_STORY_CARD_REF)
                    rows.append(await conn.fetchrow(
                        _INSERT_STORY_ASK, f"ask-{uuid.uuid4().hex}", spec["question"], json.dumps(refs),
                        thumb, json.dumps([thumb] if thumb else []), spec["answer"], spec["kind"], iid,
                        json.dumps(spec["detail"])))
                    wrong_name = wrong_name or wrong_name_correction(
                        story["name_label"], t["role"], body.action, name, current[iid])
                if wrong_name:
                    # Never over a verdict about something else (wrong motion / wrong other).
                    wrong_name = (await conn.fetchval(
                        _WRONG_NAME_VERDICT, story_id, story["rules_version"])) is not None
    except HTTPException:
        raise
    except Exception as exc:
        _raise_motion_http(exc)
    published = False
    for row in rows:
        if row["source_kind"] == WHO_IS_THIS_KIND:
            # The substrate entity, exactly as a Hub "who is this?" answer writes it. Right after
            # the commit: a failed read-back below must never lose it.
            published = await ask_routes._publish_answered(ask_routes._event_from_row(row)) or published
    stories, tally = await _readback(pool, story_id)
    ask_ids = [r["ask_id"] for r in rows]
    logger.info("motion_story_person story_id=%s individual_id=%s action=%s group=%s ask_ids=%s skipped=%s "
                "wrong_name=%s published=%s why_chars=%d",
                story_id, individual_id, body.action, body.group, ",".join(ask_ids) or "-",
                ",".join(skipped) or "-", wrong_name, published, len(why))
    return {"ok": True, "story_id": story_id, "individual_id": individual_id, "action": body.action,
            "ask_id": ask_ids[0] if ask_ids else None, "ask_ids": ask_ids, "skipped_individual_ids": skipped,
            "person_key": person_key, "skipped_all": not ask_ids, "wrong_name_recorded": wrong_name,
            "published": published,
            "story": stories[0] if stories else None, "tally": tally}


async def _readback(pool: Any, story_id: str) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """The story as the list shows it now, and the tally. ([], None) on failure: the answer
    is saved already, and the card reloads on its next poll."""
    try:
        async with pool.acquire() as conn:
            fresh = [_story_dict(r) for r in await conn.fetch(_ONE_STORY_SQL, story_id)]
            await _enrich(conn, fresh)
            return fresh, await _tally(conn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("motion_story_person_readback_failed story_id=%s error=%s", story_id, exc)
        return [], None


class NotSameBody(BaseModel):
    # One direct 'same' link between two camera groups of the person (the card's button names it).
    a: str = Field(min_length=1, max_length=128)
    b: str = Field(min_length=1, max_length=128)


@router.post("/api/vision/motion-stories/{story_id}/persons/{person_key}/not-same")
async def record_not_same_person(story_id: str, person_key: str, body: NotSameBody,
                                 request: Request) -> dict[str, Any]:
    """"These camera groups are not the same person" on a grouped row (Orion's question 1).

    A wrong 'same' link (most are made by a typed name, ``_link_same_name``) hides once linked
    groups show as one person. This writes the existing "same person?" answer, ``different``,
    for one DIRECT link of the person; sql-writer's pair path writes it over the 'same' row,
    and the two show apart again. Refused (409 ``still_linked``) when the two would stay joined
    through other 'same' links: one answer must not look like it split them when it did not."""
    if not _STORY_ID_RE.fullmatch(story_id or ""):
        raise HTTPException(status_code=400, detail="bad_story_id")
    for v in (person_key, body.a, body.b):
        if not _INDIVIDUAL_ID_RE.fullmatch(v or ""):
            raise HTTPException(status_code=400, detail="bad_individual_id")
    pool = _pool(request)
    row: Any = None
    already = False
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                if await conn.fetchval("SELECT 1 FROM vision_motion_story WHERE story_id = $1 AND published",
                                       story_id) is None:
                    raise HTTPException(status_code=404, detail="story_not_found")
                a, b = sorted((body.a, body.b))
                row = await conn.fetchrow(_PENDING_NOT_SAME_SQL, story_persons.pair_ref(a, b))
                # A second click before sql-writer applied the first: the same answer, not a new row.
                already = row is not None
                if not already:
                    fresh = [_story_dict(r) for r in await conn.fetch(_ONE_STORY_SQL, story_id)]
                    if fresh:
                        async with conn.transaction():
                            await _enrich(conn, fresh)
                    person = next((p for p in (fresh[0].get("persons") if fresh else None) or []
                                   if p["person_key"] == person_key), None)
                    if person is None:
                        raise HTTPException(status_code=422, detail="person_not_in_story")
                    if {"a": a, "b": b} not in person["links"]:
                        raise HTTPException(status_code=422, detail="not_a_direct_link_of_this_person")
                    # Would they still be joined without this row? Then the pair answer cannot split them.
                    rest = [(r["start"], r["id"]) for r in await conn.fetch(_COMPONENT_WITHOUT_SQL, [a], a, b)]
                    if story_persons.connected(rest, a, b):
                        raise HTTPException(status_code=409, detail="still_linked")
                    spec = story_persons.not_same_answer(a=a, b=b, story_id=story_id, person_key=person_key)
                    refs = [f"motion_story:{story_id}", f"individual:{a}", f"individual:{b}", JUNIPER_STORY_CARD_REF]
                    row = await conn.fetchrow(
                        _INSERT_STORY_ASK, f"ask-{uuid.uuid4().hex}", spec["question"], json.dumps(refs),
                        None, json.dumps([]), spec["answer"], spec["kind"], spec["source_ref"],
                        json.dumps(spec["detail"]))
    except HTTPException:
        raise
    except Exception as exc:
        _raise_motion_http(exc)
    stories, tally = await _readback(pool, story_id)
    logger.info("motion_story_not_same story_id=%s person_key=%s pair=%s ask_id=%s already=%s",
                story_id, person_key, row["source_ref"], row["ask_id"], already)
    return {"ok": True, "story_id": story_id, "person_key": person_key, "ask_id": row["ask_id"],
            "pair": row["source_ref"], "already_recorded": already,
            "story": stories[0] if stories else None, "tally": tally}


# Her "not the same person" for this pair, answered and not yet applied by sql-writer.
_PENDING_NOT_SAME_SQL = """
SELECT ask_id, source_ref FROM orion_ask
WHERE source_kind = 'vision_same_individual' AND status = 'answered' AND applied_at IS NULL
  AND source_ref = $1 AND lower(btrim(answer)) IN ('different', 'no')
ORDER BY answered_at DESC LIMIT 1
"""



def _strip_dir() -> Path:
    try:
        from scripts.settings import settings

        raw = str(getattr(settings, "HUB_VISION_STORY_STRIP_DIR", "") or "")
    except Exception:  # settings env not loaded (bare test process)
        raw = os.getenv("HUB_VISION_STORY_STRIP_DIR", "")
    return Path(raw.strip() or DEFAULT_STRIP_DIR)


def _read_strip(root: Path, story_id: str, name: str) -> bytes | None:
    """The story directory must be a real directory, not a link out of the root."""
    story_dir = root / story_id
    if story_dir.is_symlink():
        return None
    return _read_thumb(story_dir / name)


@router.get("/api/vision/motion-stories/{story_id}/strip/{name}")
async def get_motion_story_strip(story_id: str, name: str) -> Response:
    """One picture of a story's moment. Both path parts are whitelisted by pattern
    (32-hex story id; crop_0..3.jpg or frame.jpg), so no traversal is possible;
    the read refuses links, FIFOs and oversize files (same reader as the crop thumbs).
    404 when absent (not saved, frame was gone, or deleted after 30 days)."""
    if not _STORY_ID_RE.fullmatch(story_id or "") or not _STRIP_NAME_RE.fullmatch(name or ""):
        raise HTTPException(status_code=400, detail="bad_strip_path")
    data = await asyncio.to_thread(_read_strip, _strip_dir(), story_id, name)
    if data is None:
        raise HTTPException(status_code=404, detail="strip_not_found")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )
