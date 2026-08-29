"""Read-only Hub API for the Reverie tab -- a historical, human-visible view
of both reverie chains (CLAUDE.md section 0A: "if Orion says it reasoned,
remembered, perceived, reflected, or decided, there must be inspectable
evidence for that claim" -- until this tab, neither chain had one).

Two independent chains, no shared code between them (see
docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md §1-2):

1. **Text chain** (`services/orion-thought/app/chain.py`, live since
   mid-July): a self-driven tick narrates the current attention coalition.
   A settled run closes into one `substrate_reverie_chain` row referencing
   its `substrate_reverie_thought` rows (via `chain_json.thought_ids`, a
   plain JSON list -- no FK). Downstream: a settled chain queues a
   `dream_compaction_request_queue` row (`origin_chain_id`); a re-igniting
   theme fires a `substrate_reverie_resonance_alert` (`theme_key`).

2. **Visual chain** (`services/orion-thought/app/visual_chain.py`, live
   since 2026-08-25): generate -> store -> caption. One `reverie_visual_chain`
   row per run, `reverie_visual_artifact` rows via a REAL FK on `chain_id`.
   `prior_description` is the continuity thread; `chain_json.context_text`
   (Patch 3) is the reverie context-seed; `chain_json.continuity_streak`/
   `continuity_reset` (Patch 4) record whether THIS run's own prompt was
   forced to drop continuity; `chain_json.self_study_text` (Patch 5) is a
   second, richer context-seed from the self-study analysis system;
   `chain_json.memory_text` (Patch 6) is a third, from the Recall system's
   `memory_crystallizations` table; `chain_json.context_slot_used` (Patch
   7) records WHICH of those three actually entered the prompt this run --
   see that patch's note below, all three are recorded but only one is
   ever actually rendered -- nothing else downstream of any of these
   exists yet.

Everything here is a read. No writes, no bus publishes. Blocking SQLAlchemy
calls are offloaded via `asyncio.to_thread` -- this is a UI-facing endpoint
on Hub's shared event loop, and `text_recent` alone makes up to 4 sequential
round trips; running them inline would stall unrelated concurrent async work
(e.g. chat streaming) for the duration.

Images are served from the same content-addressed disk path
`orion.reverie.visual_storage.store_visual_artifact` writes to (bind-mounted
read-only into this container -- see docker-compose.yml), following
`chat_attachments.py`'s exact discipline: the declared DB `mime` is
re-verified against the actual bytes (never trusted blindly -- a stale or
corrupted DB value falls back to a fresh sniff, then to
`application/octet-stream`, exactly like `chat_attachments.load_bytes`), and
the read bytes' own sha256 is checked against the requested id before
serving (chain-of-custody, same pattern
`api_vision_carbon_latest_frame_image` uses for a cross-host proxy).

**Cross-service path coupling**: this file's `REVERIE_VISUAL_STORAGE_DIR`
must point at the exact same filesystem path as `orion-thought`'s
`ORION_VISUAL_CHAIN_STORAGE_DIR` (see both `.env_example`s) -- nothing
enforces this at startup. If they drift, images 404 with "artifact file
missing on disk" even though the DB rows are fine.

**Privacy note** (design doc §7, revisited for Patch 3 as this comment
required): the prompt now also includes Orion's own reverie-thought
interpretation (`orion-thought`'s `store.load_latest_reverie_interpretation`)
as a context-seed. This crosses no NEW privacy boundary: interpretation text
is the text chain's own narration, already gated by the coalition-grounding +
hollow guard (`orion/schemas/reverie.py`) before a row is ever written, and
already surfaced verbatim by this same tab's sibling `text_recent` endpoint
below -- a second consumer of an already-exposed field, not a new exposure.
No raw chat/dream content reaches the prompt (still a deliberately narrow
first slice of design doc §1's full "recent activity, chats, dreams" list) --
widening the source set further is a separate, later change that must redo
this same check.

**Privacy note, Patch 5** (design doc §16): a second context-seed,
`self_study_text`, was added from the self-study analysis system's four
deterministic window-contrast producers (concept induction, vision events,
affective state, co-creation signals) -- pure numeric prose, no chat quotes,
confirmed by reading real bodies before writing the reader
(`orion-thought`'s `store.load_latest_self_study_reflection`,
`store._SAFE_SELF_STUDY_SOURCE_PREFIXES`). `memory_crystallizations`
("actual memory") was declined at the time -- see Patch 6 note below, which
reverses that call. The `chat_history_compactor` digest ("recent chat")
remains declined: a daily-schedule producer with no evidence it has ever
fired in production.

**Privacy note, Patch 6** (design doc §17): a third context-seed,
`memory_text`, from `memory_crystallizations` (`orion-thought`'s
`store.load_latest_memory_crystallization`) -- reverses Patch 5's declined
call on the same table, on new evidence about audience rather than a
change in content filtering. This route (and the whole Reverie tab it
backs) is not published outside this host -- no `ports:` mapping in this
service's `docker-compose.yml` -- and has no per-user auth; there is one
possible viewer, and that viewer is also the original source of everything
`memory_crystallizations` holds. `memory_text` is verbatim `summary` text
from that table, filtered only to `status='active'` (a pipeline-lifecycle
filter, not a content one) -- unlike `self_study_text`, this is NOT
restricted to a safe-content allowlist, by design.

**Patch 7** (design doc §18): not a privacy change -- `context_text`,
`self_study_text`, and `memory_text` are all still computed and recorded
on every run exactly as before. What changed is that only ONE of them
(`chain_json.context_slot_used` names which) actually enters the
diffusion prompt each run -- the diffusion model's real 77-token text-
encoder budget meant concatenating all three (Patches 3/5/6's original
design) silently discarded most of them anyway. This tab now shows the
honest distinction CLAUDE.md §0A calls for: which context-seeds were
*available* this run (still all three, still all real) versus which one
*actually reached the image* (`context_slot_used`) -- a real gap this tab
previously had no way to show, since `chain_json.prompt` alone doesn't
reveal that everything past token 77 was invisible to the model.

**Privacy/governance correction, same day** (design doc §20): the Patch 6
note above called `status='active'` "a pipeline-lifecycle filter" implying
the crystallization pipeline's governor had reviewed the content. Verified
live that this is false for most of the table -- `formation_policy.py`'s
`AUTO_ACTIVE_KINDS` sets `status='active'` on creation with zero
governor review; only 21 of 652 real `active` rows have ever been touched
by an actual decision. `store.load_latest_memory_crystallization` now also
requires a real `memory_crystallization_history` row with `op='approve'`
-- the pipeline's actual audit trail, not its near-universal default
status.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import create_engine, text

from orion.reverie.visual_storage import SUPPORTED_MIMES, load_visual_artifact, sniff_image

from app.settings import settings

logger = logging.getLogger("orion-hub.reverie")

router = APIRouter(prefix="/api/reverie", tags=["reverie"])

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_LIMIT = 20
MAX_LIMIT = 100

_engine_instance: Any = None


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
        _engine_instance = create_engine(uri, pool_pre_ping=True)
    return _engine_instance


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        v = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return v.isoformat()
    return None


def _thought_ids_of(chain_json: Any) -> list[str]:
    """Normalize chain_json.thought_ids to a list[str] -- the one place both
    the batch-fetch and the per-chain lookup read this field from, so they
    can never disagree on key type (review finding: they previously did --
    the batch query str()-cast, the per-chain lookup didn't, so a
    non-string id would silently vanish from every chain's thought list with
    no error)."""
    cj = chain_json if isinstance(chain_json, dict) else {}
    ids = cj.get("thought_ids") or []
    if not isinstance(ids, list):
        return []
    return [str(i) for i in ids]


# ───────────────────────────────────────────────────────────────
# Visual chain
# ───────────────────────────────────────────────────────────────


def _fetch_visual_recent(
    limit: int, before: datetime | None
) -> tuple[list[dict], dict[str, list[dict]], bool]:
    """Cursor-paginated on ``created_at`` rather than OFFSET (review finding):
    ``reverie_visual_chain`` gets a new row every ~600s from the live worker,
    so OFFSET's "skip N rows in the current ORDER BY" has no stable meaning
    across two requests a real operator's Prev/Next clicks are seconds to
    minutes apart -- a concurrent insert shifts every row's offset by one,
    silently re-showing or skipping a row. A ``created_at <`` cursor has no
    such window: a row already fetched keeps its position relative to the
    cursor no matter what else gets inserted.

    Also avoids a second COUNT(*) round trip (former ``total``, itself racy
    against this same SELECT on two separate queries) -- fetches one extra
    row past ``limit`` and reports ``has_more`` from whether it showed up,
    the standard cursor-pagination trick.
    """
    fetch_limit = limit + 1
    with _engine().connect() as conn:
        if before is not None:
            chain_rows = (
                conn.execute(
                    text(
                        "SELECT chain_id, created_at, theme_key, terminal_reason, "
                        "ema_salience, prior_description, chain_json "
                        "FROM reverie_visual_chain "
                        "WHERE created_at < :before "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"limit": fetch_limit, "before": before},
                )
                .mappings()
                .all()
            )
        else:
            chain_rows = (
                conn.execute(
                    text(
                        "SELECT chain_id, created_at, theme_key, terminal_reason, "
                        "ema_salience, prior_description, chain_json "
                        "FROM reverie_visual_chain "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"limit": fetch_limit},
                )
                .mappings()
                .all()
            )
        has_more = len(chain_rows) > limit
        chain_rows = list(chain_rows)[:limit]
        chain_ids = [r["chain_id"] for r in chain_rows]
        artifacts_by_chain: dict[str, list[dict[str, Any]]] = {}
        if chain_ids:
            artifact_rows = (
                conn.execute(
                    text(
                        "SELECT sha256, chain_id, step_index, mime, bytes, "
                        "width, height, description, created_at "
                        "FROM reverie_visual_artifact "
                        "WHERE chain_id = ANY(:chain_ids) "
                        "ORDER BY step_index ASC"
                    ),
                    {"chain_ids": chain_ids},
                )
                .mappings()
                .all()
            )
            for a in artifact_rows:
                artifacts_by_chain.setdefault(a["chain_id"], []).append(dict(a))
    return chain_rows, artifacts_by_chain, has_more


@router.get("/visual/recent")
async def visual_recent(
    limit: int = Query(DEFAULT_LIMIT, ge=1), before: datetime | None = Query(None)
) -> dict[str, Any]:
    limit = _clamp_limit(limit)
    try:
        chain_rows, artifacts_by_chain, has_more = await asyncio.to_thread(
            _fetch_visual_recent, limit, before
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("reverie visual_recent query failed: %s", exc)
        raise HTTPException(status_code=503, detail="reverie_visual_query_failed") from exc

    chains = []
    for c in chain_rows:
        cj = c["chain_json"] if isinstance(c["chain_json"], dict) else {}
        artifacts = [
            {
                "sha256": a["sha256"],
                "step_index": a["step_index"],
                "mime": a["mime"],
                "bytes": a["bytes"],
                "width": a["width"],
                "height": a["height"],
                "description": a["description"],
                "created_at": _iso(a["created_at"]),
                "image_url": f"/api/reverie/visual/image/{a['sha256']}",
            }
            for a in artifacts_by_chain.get(c["chain_id"], [])
        ]
        chains.append(
            {
                "chain_id": c["chain_id"],
                "created_at": _iso(c["created_at"]),
                "theme_key": c["theme_key"],
                "terminal_reason": c["terminal_reason"],
                "ema_salience": c["ema_salience"],
                "prior_description": c["prior_description"],
                "prompt": cj.get("prompt"),
                # Patch 3: surfaced as its own field, not just prose baked into
                # `prompt` above -- lets the UI show "what Orion was narrating"
                # distinctly rather than requiring a reader to eyeball it out
                # of a full sentence (CLAUDE.md §0A: inspectable evidence, not
                # schema presence).
                "context_text": cj.get("context_text"),
                # Patch 5: a second, richer context-seed -- real quantified
                # self-observation, not a bare narration sentence. Same
                # "own field, not just prose" reasoning as context_text
                # above.
                "self_study_text": cj.get("self_study_text"),
                # Patch 6: a third context-seed -- real shared-life memory
                # crystallization content. Same "own field" reasoning as
                # context_text/self_study_text above.
                "memory_text": cj.get("memory_text"),
                # Patch 7: which ONE of context_text/self_study_text/
                # memory_text actually entered THIS run's prompt --
                # "context", "self_study", "memory", or null (nothing had
                # content). All three fields above are still recorded
                # regardless; this is the honest "which one actually
                # reached the model" signal the diffusion model's 77-token
                # budget made necessary (see module docstring).
                "context_slot_used": cj.get("context_slot_used"),
                # Patch 8: the concrete visual metaphor the metacog interpretation step
                # actually invented for context_slot_used's raw text this run -- null when
                # nothing was selected, interpretation is disabled, or the metacog call
                # failed/timed out (build_visual_prompt then fell back to the raw slot text
                # unchanged, exactly Patch 7's behavior). Surfaced as its own field so the tab
                # can show the raw clause NEXT TO what it was actually turned into -- the
                # concrete answer to "how does this translate into fluffy cloud??" is this
                # field being visibly non-null and visibly not a cloud.
                "context_slot_interpreted": cj.get("context_slot_interpreted"),
                # Patch 4: whether THIS run's own prompt was forced to drop
                # prior_description continuity (visual_chain.py::
                # resolve_visual_chain_continuity) -- surfaced so the tab can
                # show "this is a fresh seed point", not just a normal
                # continuity step.
                "continuity_streak": cj.get("continuity_streak"),
                "continuity_reset": cj.get("continuity_reset"),
                # A failed generation's chain_json carries "error" instead of
                # "artifact_sha256"/"description" (visual_chain.py's
                # _generation_failed) -- surfaced so the cockpit can show
                # *why* a run produced no image instead of just an empty card.
                "error": cj.get("error"),
                "artifacts": artifacts,
            }
        )
    return {
        "ok": True,
        "chains": chains,
        "has_more": has_more,
        "limit": limit,
        # Cursor for the *next* ("older") page -- the last row's own
        # created_at, echoed back so the client doesn't need to parse
        # timestamps itself, just round-trip this value as `before`.
        "next_before": chains[-1]["created_at"] if chains else None,
    }


def _fetch_artifact_mime(sha256: str) -> dict | None:
    with _engine().connect() as conn:
        row = (
            conn.execute(
                text("SELECT mime FROM reverie_visual_artifact WHERE sha256 = :sha256"),
                {"sha256": sha256},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


@router.get("/visual/image/{sha256}")
async def visual_image(sha256: str) -> Response:
    if not _SHA256_RE.match(sha256):
        raise HTTPException(status_code=400, detail="invalid artifact id")
    try:
        row = await asyncio.to_thread(_fetch_artifact_mime, sha256)
    except Exception as exc:
        logger.warning("reverie visual_image lookup failed sha=%s err=%s", sha256[:12], exc)
        raise HTTPException(status_code=503, detail="reverie_visual_query_failed") from exc
    if not row:
        raise HTTPException(status_code=404, detail="artifact not found")

    try:
        data = await asyncio.to_thread(
            load_visual_artifact, sha256, base_dir=settings.REVERIE_VISUAL_STORAGE_DIR
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact file missing on disk") from exc

    # Chain-of-custody: the sha256 is the caller-supplied filename, so verify
    # the bytes actually hash to it rather than trusting the filesystem
    # lookup alone (same discipline orion-percept-store's own docstring
    # describes: "a content-addressed store returning the wrong content is
    # worse than one returning nothing").
    if hashlib.sha256(data).hexdigest() != sha256:
        logger.error("reverie visual_image hash mismatch sha=%s", sha256[:12])
        raise HTTPException(status_code=500, detail="artifact integrity check failed")

    # Never trust the stored DB mime blindly (review finding -- this
    # module's own docstring claimed chat_attachments.py's discipline but
    # didn't apply it here yet): re-sniff and fall back exactly like
    # chat_attachments.load_bytes does for a missing/edited sidecar.
    mime = row["mime"]
    if mime not in SUPPORTED_MIMES:
        sniffed = sniff_image(data)
        mime = sniffed[0] if sniffed else "application/octet-stream"

    return Response(
        content=data,
        media_type=mime,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ───────────────────────────────────────────────────────────────
# Text chain
# ───────────────────────────────────────────────────────────────


def _fetch_text_recent(limit: int) -> dict[str, Any]:
    with _engine().connect() as conn:
        chain_rows = (
            conn.execute(
                text(
                    "SELECT chain_id, created_at, theme_key, terminal_reason, "
                    "ema_salience, committed_proposal_id, chain_json "
                    "FROM substrate_reverie_chain "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        chain_ids = [r["chain_id"] for r in chain_rows]
        theme_keys = sorted(
            {r["theme_key"] for r in chain_rows if r["theme_key"] and r["theme_key"] != "unknown"}
        )
        all_thought_ids = sorted({tid for r in chain_rows for tid in _thought_ids_of(r["chain_json"])})

        thoughts_by_id: dict[str, dict[str, Any]] = {}
        if all_thought_ids:
            thought_rows = (
                conn.execute(
                    text(
                        "SELECT thought_id, created_at, salience, interpretation "
                        "FROM substrate_reverie_thought "
                        "WHERE thought_id = ANY(:ids)"
                    ),
                    {"ids": all_thought_ids},
                )
                .mappings()
                .all()
            )
            for t in thought_rows:
                thoughts_by_id[t["thought_id"]] = dict(t)

        compaction_chain_ids: set[str] = set()
        if chain_ids:
            rows = (
                conn.execute(
                    text(
                        "SELECT DISTINCT origin_chain_id FROM dream_compaction_request_queue "
                        "WHERE origin_chain_id = ANY(:chain_ids)"
                    ),
                    {"chain_ids": chain_ids},
                )
                .mappings()
                .all()
            )
            compaction_chain_ids = {r["origin_chain_id"] for r in rows}

        resonance_counts: dict[str, int] = {}
        if theme_keys:
            rows = (
                conn.execute(
                    text(
                        "SELECT theme_key, count(*) AS n FROM substrate_reverie_resonance_alert "
                        "WHERE theme_key = ANY(:themes) GROUP BY theme_key"
                    ),
                    {"themes": theme_keys},
                )
                .mappings()
                .all()
            )
            resonance_counts = {r["theme_key"]: int(r["n"]) for r in rows}

    return {
        "chain_rows": list(chain_rows),
        "thoughts_by_id": thoughts_by_id,
        "compaction_chain_ids": compaction_chain_ids,
        "resonance_counts": resonance_counts,
    }


@router.get("/text/recent")
async def text_recent(limit: int = Query(DEFAULT_LIMIT, ge=1)) -> dict[str, Any]:
    limit = _clamp_limit(limit)
    try:
        fetched = await asyncio.to_thread(_fetch_text_recent, limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("reverie text_recent query failed: %s", exc)
        raise HTTPException(status_code=503, detail="reverie_text_query_failed") from exc

    thoughts_by_id = fetched["thoughts_by_id"]
    compaction_chain_ids = fetched["compaction_chain_ids"]
    resonance_counts = fetched["resonance_counts"]

    chains = []
    for c in fetched["chain_rows"]:
        ids = _thought_ids_of(c["chain_json"])
        thoughts = [
            {
                "thought_id": tid,
                "created_at": _iso(thoughts_by_id[tid]["created_at"]),
                "salience": thoughts_by_id[tid]["salience"],
                "interpretation": thoughts_by_id[tid]["interpretation"],
            }
            for tid in ids
            if tid in thoughts_by_id
        ]
        chains.append(
            {
                "chain_id": c["chain_id"],
                "created_at": _iso(c["created_at"]),
                "theme_key": c["theme_key"],
                "terminal_reason": c["terminal_reason"],
                "ema_salience": c["ema_salience"],
                "thoughts": thoughts,
                "downstream": {
                    # dream_compaction_request_queue.consumed_at is never set
                    # anywhere in the codebase today (confirmed live,
                    # 2026-08-26) -- REM re-folds the same backlog every
                    # pass. Reported honestly as "queued", not "applied":
                    # nothing downstream of this queue ever mutates memory
                    # (Phase G's applier is separate and gated).
                    "compaction_queued": c["chain_id"] in compaction_chain_ids,
                    # A count over the whole theme, not this specific chain
                    # -- the resonance detector operates over a window of
                    # chains sharing a theme, not one chain in isolation, so
                    # attributing a single alert to a single chain would
                    # overclaim causality.
                    "theme_resonance_alert_count": resonance_counts.get(c["theme_key"], 0),
                },
            }
        )
    return {"ok": True, "chains": chains}
