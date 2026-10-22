"""One record per person, kept current from the camera names (2026-10-22).

Spec: docs/superpowers/specs/2026-10-22-people-one-record-design.md.

Runs inside the individuals cycle (``run_one_individuals_cycle``), under its
lock and after Juniper's answers are applied, so a name she just typed is
linked the same tick. Each call:

1. ``person`` / ``person_link`` follow the named clusters (pure plan:
   ``orion/person_model/record.py``). Conflicts are logged, never merged.
2. Stated pronouns from ``config/person_model/people.yaml`` become facts.
3. Answered "person_fact" asks become ``person_fact`` rows, her words verbatim.
4. New "person_fact" asks open, under their own daily cap: one open question
   per person, people asked about least so far first, then the most-seen.
   Each question shows the person's best camera picture when there is one.

Trace: ``[PERSON_RECORD] persons=.. links_added=.. links_revoked=.. conflicts=..
facts_applied=.. asks_opened=..`` in the individuals tick summary.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from orion.person_model import facts as pf
from orion.person_model import record as pr

logger = logging.getLogger("sql-writer.person_record")

PERSON_TABLES = ("person", "person_link", "person_fact")


@dataclass(frozen=True)
class PersonRecordConfig:
    ask_daily_cap: int = 3
    ask_cooldown_days: float = 7.0
    ask_expiry_days: float = 7.0
    local_tz: str = "America/Denver"


def tables_present(conn) -> bool:
    from sqlalchemy import text

    return all(conn.execute(text("SELECT to_regclass(:t)"), {"t": t}).scalar() is not None for t in PERSON_TABLES)


def _people():
    from orion.person_model.person import DEFAULT_PEOPLE_PATH, load_people

    try:
        return load_people(DEFAULT_PEOPLE_PATH)
    except FileNotFoundError:
        logger.warning("person_record people.yaml missing at %s; camera names only", DEFAULT_PEOPLE_PATH)
        return []


def sync_links(conn, people) -> Tuple[pr.Plan, Dict[str, int]]:
    from sqlalchemy import text

    clusters = pr.clusters_from_rows(conn.execute(text(
        "SELECT individual_id, kind, label, label_ask_id FROM vision_individual "
        "WHERE label IS NOT NULL AND btrim(label) <> ''")).fetchall())
    different = [(r[0], r[1]) for r in conn.execute(text(
        "SELECT individual_a, individual_b FROM vision_individual_pair_verdict WHERE verdict = 'different'"))]
    persons = [pr.Person(r[0], r[1], r[2], r[3]) for r in conn.execute(text(
        "SELECT person_id, display_name, role, status FROM person"))]
    links = {(r[0], r[1]): r[2] for r in conn.execute(text(
        "SELECT source_kind, source_ref, person_id FROM person_link WHERE revoked_at IS NULL"))}
    plan = pr.plan(clusters, different, people, persons, links)
    for p in plan.new_persons:
        conn.execute(text(
            "INSERT INTO person (person_id, display_name, role) VALUES (:i, :n, :r) ON CONFLICT DO NOTHING"),
            {"i": p.person_id, "n": p.display_name, "r": p.role})
    for kind, ref in plan.revoke:
        conn.execute(text(
            "UPDATE person_link SET revoked_at = now() WHERE source_kind = :k AND source_ref = :r "
            "AND revoked_at IS NULL"), {"k": kind, "r": ref})
    for link in plan.add_links:
        conn.execute(text(
            "INSERT INTO person_link (link_id, person_id, source_kind, source_ref, decided_by, evidence_ref) "
            "VALUES (:id, :p, :k, :r, :d, :e) "
            "ON CONFLICT (link_id) DO UPDATE SET revoked_at = NULL, created_at = now()"),
            {"id": link.link_id, "p": link.person_id, "k": link.source_kind, "r": link.source_ref,
             "d": link.decided_by, "e": link.evidence_ref})
    for c in plan.conflicts:
        logger.warning("person_link_conflict individual_id=%s reason=%s persons=%s",
                       c.individual_id, c.reason, ",".join(c.person_ids))
    return plan, {"persons_added": len(plan.new_persons), "links_added": len(plan.add_links),
                  "links_revoked": len(plan.revoke), "link_conflicts": len(plan.conflicts)}


def seed_stated_facts(conn, people) -> int:
    from sqlalchemy import text

    n = 0
    for f in pf.stated_pronoun_facts(people):
        n += conn.execute(text(
            "INSERT INTO person_fact (fact_id, person_id, category, value_text, voice, source_ref) "
            "SELECT :fact_id, :person_id, :category, :value_text, :voice, :source_ref "
            "WHERE EXISTS (SELECT 1 FROM person WHERE person_id = :person_id) "
            "ON CONFLICT DO NOTHING"), f).rowcount or 0
    return n


def apply_fact_answers(conn, now: datetime) -> int:
    """Answered person_fact asks -> person_fact rows (fact_id = ask_id). Idempotent via applied_at."""
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT ask_id, source_ref, answer FROM orion_ask WHERE source_kind = :k AND status = 'answered' "
        "AND applied_at IS NULL ORDER BY answered_at, ask_id"), {"k": pf.SOURCE_KIND_PERSON_FACT}).fetchall()
    applied = 0
    for ask_id, ref, answer in rows:
        parsed = pf.parse_ask_ref(ref)
        value = pf.answer_value(answer)
        if parsed and value:
            person_id, category = parsed
            applied += conn.execute(text(
                "INSERT INTO person_fact (fact_id, person_id, category, value_text, voice, source_ref) "
                "SELECT :f, :p, :c, :v, :voice, :src WHERE EXISTS (SELECT 1 FROM person WHERE person_id = :p) "
                "ON CONFLICT DO NOTHING"),
                {"f": ask_id, "p": person_id, "c": category, "v": value, "voice": pf.VOICE_ASK_ANSWER,
                 "src": f"ask:{ask_id}"}).rowcount or 0
        else:
            logger.warning("person_fact_answer_unusable ask_id=%s ref=%s", ask_id, ref)
        conn.execute(text("UPDATE orion_ask SET applied_at = :t WHERE ask_id = :a"), {"t": now, "a": ask_id})
    return applied


# People Orion may ask about, least asked-about first, then the most-seen on camera.
_ASK_CANDIDATES_SQL = """
SELECT p.person_id, p.display_name, p.role,
       coalesce((SELECT sum(i.distinct_days) FROM person_link l JOIN vision_individual i
                 ON i.individual_id = l.source_ref
                 WHERE l.person_id = p.person_id AND l.source_kind = 'vision_individual'
                   AND l.revoked_at IS NULL), 0) AS days,
       (SELECT count(*) FROM orion_ask a WHERE a.source_kind = :k
          AND split_part(a.source_ref, '|', 1) = p.person_id) AS asked
FROM person p
WHERE p.status = 'active' AND p.role <> 'self'
  AND NOT EXISTS (SELECT 1 FROM orion_ask o WHERE o.source_kind = :k AND o.status = 'open'
                  AND split_part(o.source_ref, '|', 1) = p.person_id)
ORDER BY asked, days DESC, p.person_id
"""

# A category is done for a person when she answered it (a live fact), or a question
# about it is open, answered, dismissed, or expired within the cooldown.
_DONE_SQL = """
SELECT category FROM person_fact WHERE person_id = :p AND removed_at IS NULL
UNION
SELECT split_part(a.source_ref, '|', 2) FROM orion_ask a
WHERE a.source_kind = :k AND split_part(a.source_ref, '|', 1) = :p
  AND (a.status IN ('open', 'answered', 'dismissed') OR (a.status = 'expired' AND a.created_at > :cool))
"""


def _pictures(conn, person_id: str, now: datetime, individuals_cfg) -> List[str]:
    """Up to 3 pictures from the person's most-seen linked cluster, or []."""
    from sqlalchemy import text

    from app.vision_individuals import _images, _sighting_rows

    if individuals_cfg is None:
        return []
    ids = [r[0] for r in conn.execute(text(
        "SELECT l.source_ref FROM person_link l JOIN vision_individual i ON i.individual_id = l.source_ref "
        "WHERE l.person_id = :p AND l.source_kind = 'vision_individual' AND l.revoked_at IS NULL "
        "ORDER BY i.sighting_count DESC, l.source_ref"), {"p": person_id})]
    for iid in ids:
        images = _images(_sighting_rows(conn, iid), now, individuals_cfg)
        if images:
            return images
    return []


def open_fact_asks(conn, now: datetime, cfg: PersonRecordConfig, individuals_cfg=None) -> List[dict]:
    from sqlalchemy import text

    from app.vision_individuals import local_day_start, remaining_ask_budget
    from orion.schemas.ask import OrionAskV1

    if cfg.ask_daily_cap <= 0:
        return []
    asked_today = conn.execute(text(
        "SELECT count(*) FROM orion_ask WHERE source_kind = :k AND created_at >= :d"),
        {"k": pf.SOURCE_KIND_PERSON_FACT, "d": local_day_start(now, ZoneInfo(cfg.local_tz))}).scalar() or 0
    budget = remaining_ask_budget(daily_cap=cfg.ask_daily_cap, asked_today=int(asked_today))
    out: List[dict] = []
    if budget <= 0:
        return out
    cool = now - timedelta(days=cfg.ask_cooldown_days)
    for person_id, name, role, _days, _asked in conn.execute(
            text(_ASK_CANDIDATES_SQL), {"k": pf.SOURCE_KIND_PERSON_FACT}).fetchall():
        if len(out) >= budget:
            break
        done = {r[0] for r in conn.execute(text(_DONE_SQL),
                                           {"p": person_id, "k": pf.SOURCE_KIND_PERSON_FACT, "cool": cool})}
        nxt = pf.next_question(name=name, role=role, done=done)
        if nxt is None:
            continue
        category, question = nxt
        images = _pictures(conn, person_id, now, individuals_cfg)
        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}", question=question,
            evidence_refs=[f"person:{person_id}", f"category:{category}"],
            image_ref=images[0] if images else None, image_refs=images, created_at=now,
            expires_at=now + timedelta(days=cfg.ask_expiry_days),
            source_kind=pf.SOURCE_KIND_PERSON_FACT, source_ref=pf.ask_ref(person_id, category),
        )
        inserted = conn.execute(text(
            "INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, image_ref, image_refs, status, "
            "created_at, expires_at, source_kind, source_ref) "
            "VALUES (:id, :of, :q, CAST(:ev AS jsonb), :img, CAST(:imgs AS jsonb), 'open', :c, :x, :sk, :sr) "
            "ON CONFLICT (source_kind, source_ref) WHERE status = 'open' DO NOTHING RETURNING ask_id"),
            {"id": ask.ask_id, "of": ask.asked_of, "q": ask.question, "ev": json.dumps(ask.evidence_refs),
             "img": ask.image_ref, "imgs": json.dumps(ask.image_refs), "c": ask.created_at,
             "x": ask.expires_at, "sk": ask.source_kind, "sr": ask.source_ref}).fetchone()
        if inserted:
            out.append(ask.model_dump(mode="json"))
    return out


def run_person_record(conn, now: datetime, cfg: PersonRecordConfig,
                      individuals_cfg=None) -> Tuple[Dict[str, Any], List[dict]]:
    """One pass. Returns (summary, opened asks). Skips quietly while the migration is not applied."""
    if not tables_present(conn):
        return {"person_record": "migration_missing"}, []
    people = _people()
    _plan, summary = sync_links(conn, people)
    summary["facts_seeded"] = seed_stated_facts(conn, people)
    summary["facts_applied"] = apply_fact_answers(conn, now)
    opened = open_fact_asks(conn, now, cfg, individuals_cfg)
    summary["person_asks_opened"] = len(opened)
    return summary, opened
