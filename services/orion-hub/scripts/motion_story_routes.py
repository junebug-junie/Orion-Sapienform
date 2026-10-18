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
- ``someone_else`` (optionally with a name for who it really is) / ``unsure``:
  a ``vision_story_person`` ask whose ``answer_detail.pictures`` holds one
  picture verdict per crop of that individual in the story. orion-sql-writer
  stores them in ``vision_ask_picture_verdict``; "someone else" ones are split
  off by the split-on-verdict pass (app/vision_split.py).

A correction of the name the story was written with (a different name, or
"someone else" on the named walker) also records the story's "Wrong name"
verdict, so the gate tally sees it. The story sentence is re-derived when it is
listed (``display_sentence``): a direct answer on THIS story wins for it;
otherwise the same strict rule sql-writer uses (orion.vision.story_naming) runs
on the names individuals carry now. ``sentence`` stays what was written.

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

from orion.vision.story_naming import evidence_crops, name_from_crops, rename_subject
from scripts import ask_routes
from scripts.ask_routes import _iso, _pool, _raise_store_http, _read_thumb, clean_name


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

# One story, the list's columns (the person route returns it re-derived).
_ONE_STORY_SQL = _LIST_SQL.split("FROM vision_motion_story s")[0] + (
    "FROM vision_motion_story s LEFT JOIN vision_motion_story_verdict v ON v.story_id = s.story_id "
    "WHERE s.story_id = $1")


class VerdictBody(BaseModel):
    verdict: Literal["right", "wrong_name", "wrong_motion", "wrong_other", "cant_tell"]


WHO_IS_THIS_KIND = ask_routes.WHO_IS_THIS_KIND        # "vision_individual"
STORY_PERSON_KIND = "vision_story_person"
PERSON_ACTIONS = ("name", "someone_else", "unsure")
_INDIVIDUAL_ID_RE = re.compile(r"[0-9A-Za-z_.:-]{1,128}")
_THUMB_HEX_RE = re.compile(r"[0-9a-f]{64}")


class PersonBody(BaseModel):
    action: Literal["name", "someone_else", "unsure"]
    # For "name": who this is. For "someone_else": optionally who it really is.
    name: str = Field(default="", max_length=ask_routes.MAX_ANSWER_CHARS)


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

# Juniper's latest answer per (story, individual), from the ask rows this card writes.
_STORY_ACTIONS_SQL = """
SELECT DISTINCT ON (answer_detail->>'story_id', source_ref)
       answer_detail->>'story_id' AS story_id, source_ref AS individual_id,
       answer_detail->>'action' AS action, answer_detail->>'name' AS name, ask_id, answered_at
FROM orion_ask
WHERE source_kind = ANY($2::text[]) AND status = 'answered'
  AND answer_detail->>'story_id' = ANY($1::text[])
ORDER BY answer_detail->>'story_id', source_ref, answered_at DESC, created_at DESC, ask_id DESC
"""


def _evidence_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = []
    return [e for e in (raw or []) if isinstance(e, dict)]


def people_in(evidence: list[dict[str, Any]], role: str, kind: str) -> list[dict[str, Any]]:
    """One entry per distinct individual id in ``evidence``, in first-seen order."""
    out: dict[str, dict[str, Any]] = {}
    for e in evidence:
        iid = e.get("individual_id")
        if not iid or not isinstance(iid, str):
            continue
        p = out.setdefault(iid, {"individual_id": iid, "short_id": iid[:8], "role": role, "kind": kind,
                                 "crop_ids": [], "thumb": None})
        if e.get("crop_id"):
            p["crop_ids"].append(str(e["crop_id"]))
        if p["thumb"] is None and isinstance(e.get("thumb"), str) and _THUMB_HEX_RE.fullmatch(e["thumb"]):
            p["thumb"] = e["thumb"]
    return list(out.values())


def story_display(story: dict[str, Any], labels: dict[str, str],
                  actions: dict[str, dict[str, Any]]) -> tuple[Optional[str], str, str]:
    """(display name, sentence to show, why) for one story.

    A direct answer by Juniper on this story about its walker wins for this
    story (the latest one): a name -> that name; "someone else" -> who she said
    it really is, or no name. Otherwise the strict naming rule on the names the
    individuals carry now (``why='rule'``). "Not sure" changes nothing."""
    walkers = {p["individual_id"] for p in story.get("people") or [] if p.get("role") == "walker"}
    direct = [a for iid, a in actions.items() if iid in walkers and a.get("action") in ("name", "someone_else")]
    if direct:
        latest = max(direct, key=lambda a: (str(a.get("answered_at") or ""), str(a.get("ask_id") or "")))
        name = (latest.get("name") or "").strip() or None
        why = "juniper"
    else:
        _, name, _ = name_from_crops(evidence_crops(story.get("evidence") or []), labels)
        why = "rule"
    sentence = rename_subject(story.get("sentence") or "", story.get("kind") or "person",
                              story.get("name_label"), name)
    return name, sentence, why


async def _enrich(conn: Any, stories: list[dict[str, Any]]) -> None:
    """Add ``people``, ``display_name``, ``display_sentence`` and ``name_source``.
    Fail-open: a missing individuals table leaves the stored sentence and no people."""
    for s in stories:
        s["people"] = people_in(s.get("evidence") or [], "walker", s.get("kind") or "person")
        s["display_name"], s["display_sentence"], s["name_source"] = s.get("name_label"), s.get("sentence"), "stored"
    if not stories:
        return
    try:
        together = sorted({t for s in stories for t in s.get("together_story_ids") or []})
        partners: dict[str, Any] = {}
        if together:
            for r in await conn.fetch(
                    "SELECT story_id, kind, evidence FROM vision_motion_story WHERE story_id = ANY($1::text[])",
                    together):
                partners[r["story_id"]] = r
        for s in stories:
            seen = {p["individual_id"] for p in s["people"]}
            for t in s.get("together_story_ids") or []:
                r = partners.get(t)
                if r is None:
                    continue
                for p in people_in(_evidence_list(r["evidence"]), "companion", r["kind"] or "person"):
                    if p["individual_id"] not in seen:
                        seen.add(p["individual_id"])
                        p["story_id"] = t
                        s["people"].append(p)
        ids = sorted({p["individual_id"] for s in stories for p in s["people"]}
                     | {e.get("individual_id") for s in stories for e in s.get("evidence") or []
                        if isinstance(e.get("individual_id"), str)})
        rows = {r["individual_id"]: r for r in await conn.fetch(_LABELS_SQL, ids)} if ids else {}
        acts: dict[str, dict[str, dict[str, Any]]] = {}
        for r in await conn.fetch(_STORY_ACTIONS_SQL, [s["story_id"] for s in stories],
                                  [WHO_IS_THIS_KIND, STORY_PERSON_KIND]):
            acts.setdefault(r["story_id"], {})[r["individual_id"]] = {
                "action": r["action"], "name": r["name"], "ask_id": r["ask_id"], "answered_at": _iso(r["answered_at"])}
    except Exception as exc:  # noqa: BLE001 - the card still shows the stored story
        logger.warning("motion_story_people_lookup_failed error=%s", exc)
        return
    labels = {iid: (r["label"] or "").strip() for iid, r in rows.items() if (r["label"] or "").strip()}
    for s in stories:
        mine = acts.get(s["story_id"], {})
        for p in s["people"]:
            r = rows.get(p["individual_id"])
            label = (r["label"] or "").strip() if r is not None else ""
            p["label"] = label or None
            p["label_copied"] = bool(label and r["label_copied"])
            p["known"] = r is not None
            p["story_action"] = mine.get(p["individual_id"])
        s["display_name"], s["display_sentence"], s["name_source"] = story_display(s, labels, mine)


def wrong_name_correction(story_name: Optional[str], role: str, action: str, name: str) -> bool:
    """Is this answer a correction of the name the story was written with?
    Only about the walker: a different name, or "someone else"."""
    if not story_name or role != "walker":
        return False
    if action == "someone_else":
        return True
    return action == "name" and name.casefold() != story_name.strip().casefold()


_INSERT_STORY_ASK = """
INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, image_ref, image_refs, status, answer,
                       answered_at, created_at, source_kind, source_ref, answer_detail)
VALUES ($1, 'juniper', $2, $3::jsonb, $4, $5::jsonb, 'answered', $6, now(), now(), $7, $8, $9::jsonb)
RETURNING ask_id, status, answer, answered_at, source_kind, source_ref
"""


@router.post("/api/vision/motion-stories/{story_id}/people/{individual_id}")
async def record_story_person(story_id: str, individual_id: str, body: PersonBody,
                              request: Request) -> dict[str, Any]:
    """Name / "someone else" / "not sure" for one person on a story (see the module doc)."""
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
    row: Any = None
    wrong_name = False
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                story = await conn.fetchrow(
                    "SELECT story_id, kind, name_label, rules_version, evidence, together_story_ids "
                    "FROM vision_motion_story WHERE story_id = $1 AND published", story_id)
                if story is None:
                    raise HTTPException(status_code=404, detail="story_not_found")
                person = next((p for p in people_in(_evidence_list(story["evidence"]), "walker", story["kind"])
                               if p["individual_id"] == individual_id), None)
                if person is None and story["together_story_ids"]:
                    for r in await conn.fetch(
                            "SELECT story_id, kind, evidence FROM vision_motion_story WHERE story_id = ANY($1::text[]) "
                            "ORDER BY story_id", list(story["together_story_ids"])):
                        person = next((p for p in people_in(_evidence_list(r["evidence"]), "companion", r["kind"])
                                       if p["individual_id"] == individual_id), None)
                        if person is not None:
                            break
                if person is None:
                    raise HTTPException(status_code=422, detail="individual_not_in_story")
                current = await conn.fetchrow("SELECT label FROM vision_individual WHERE individual_id = $1",
                                              individual_id)
                if current is None:
                    raise HTTPException(status_code=404, detail="individual_not_found")
                if (body.action == "someone_else" and name and current["label"]
                        and name.casefold() == current["label"].strip().casefold()):
                    # "Someone else, called <their own name>": contradictory (the split refuses it too).
                    raise HTTPException(status_code=422, detail="someone_else_takes_another_name")
                thumb = f"thumb:{person['thumb']}" if person["thumb"] else None
                detail: dict[str, Any] = {"story_id": story_id, "action": body.action, "via": "motion_story"}
                if name:
                    detail["name"] = name
                if body.action == "name":
                    kind, answer = WHO_IS_THIS_KIND, name
                    question = "Who is this? (named on a walkway story)"
                else:
                    kind, answer = STORY_PERSON_KIND, None
                    question = ("Is this someone else? (said on a walkway story)" if body.action == "someone_else"
                                else "Not sure who this is (said on a walkway story)")
                    verdict = "different" if body.action == "someone_else" else "unsure"
                    detail["pictures"] = [
                        {"image_ref": f"crop:{cid}", "index": i, "verdict": verdict, "crop_id": cid,
                         "name": name or None}
                        for i, cid in enumerate(person["crop_ids"])]
                row = await conn.fetchrow(
                    _INSERT_STORY_ASK, f"ask-{uuid.uuid4().hex}", question,
                    json.dumps([f"motion_story:{story_id}", f"individual:{individual_id}"]),
                    thumb, json.dumps([thumb] if thumb else []), answer, kind, individual_id, json.dumps(detail))
                wrong_name = wrong_name_correction(story["name_label"], person["role"], body.action, name)
                if wrong_name:
                    await conn.execute(_UPSERT_VERDICT, story_id, "wrong_name", story["rules_version"], False)
            listed = await conn.fetch(_ONE_STORY_SQL, story_id)
            stories = [_story_dict(r) for r in listed]
            await _enrich(conn, stories)
            tally = await _tally(conn)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_motion_http(exc)
    published = False
    if body.action == "name":
        # The substrate entity, exactly as a Hub "who is this?" answer writes it.
        published = await ask_routes._publish_answered(ask_routes._event_from_row(row))
    logger.info("motion_story_person story_id=%s individual_id=%s action=%s ask_id=%s wrong_name=%s published=%s",
                story_id, individual_id, body.action, row["ask_id"], wrong_name, published)
    return {"ok": True, "story_id": story_id, "individual_id": individual_id, "action": body.action,
            "ask_id": row["ask_id"], "wrong_name_recorded": wrong_name, "published": published,
            "story": stories[0] if stories else None, "tally": tally}


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
