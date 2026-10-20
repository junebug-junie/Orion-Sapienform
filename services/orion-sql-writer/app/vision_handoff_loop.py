"""Clocked reducer for street continuity (rules: app/vision_handoff.py).

Every ``VISION_HANDOFF_INTERVAL_SEC`` (default 15 s), under advisory lock
``HANDOFF_LOCK_KEY``:

1. **Open.** Motion stories that landed since the cursor
   (``vision_handoff_cursor`` name ``stories``, by ``created_at`` with a 10 s
   re-read; ``handoff_id`` is uuid5 of portal + story, insert ``ON CONFLICT DO
   NOTHING``) open gap / approach expectations. While a window is still ahead,
   Redis steering is set: ``orion:vision:expect:<walkway>`` (extend-only,
   rhythm's ``expectation_ids`` kept; ``handoff_until`` makes the rhythm loop
   hold the key) and, for doorbell targets, ``orion:vision:look:doorbell``
   (value = handoff id, TTL = window; the retina opens a capped live view).
   Motion stories land ~45-75 s after the walker left (settle + link gap +
   tick), so a short window is often already past: ``steered`` stays empty
   and the summary counts ``steer_too_late``.
2. **Grade** open rows whose late margin has passed and whose target stream's
   motion cursor has caught up (``vision_motion_story_cursor``); stuck for
   ``stuck_after_sec`` -> ``unscorable``. Writes ``handoff_met`` /
   ``handoff_late`` / ``handoff_missed`` vision_events rows; a miss opens its
   hypotheses.
3. **Overlap pairs** on the central walk, one decision per doorbell story once
   the walkway cursor has passed it.
4. **Rings nobody was seen approaching** (``unannounced`` rows + two
   hypotheses: walkway ``detector_miss`` or ``out_of_view``).
5. **Hypotheses** settle from data by their deadlines; an ``unexplained``
   miss writes a ``vision_unresolved`` row (reason ``handoff_missed``) and the
   ``handoff_missed`` street event.
6. **Asks**: strong met / paired pairs open capped ``vision_same_individual``
   asks. Never a label, never a merge.
7. **Prune** with the motion-story retention.

The ``StreetEventV1`` outbox (``vision_handoff.street_event`` with
``street_published_at`` NULL) is published by the async loop on
``orion:vision:street:events`` and marked; a failed publish is retried next
tick.

Missing migration -> ``MigrationMissing`` -> the loop backs off with a clear
log line (apply services/orion-sql-db/manual_migration_vision_handoff_v1.sql).
Missing edges migration (PR #2822) -> edges are derived from story evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from app.settings import Settings
from app.vision_handoff import (
    ARRIVED, HANDOFF_LOCK_KEY, HANDOFF_RULE_REF, HYPOTHESES, LOOK_KEY, MODEL_VERSION, RING_HYPOTHESES,
    SOURCE_KIND_SAME, STREET_CHANNEL, Band, Candidate, HandoffConfig, StoryFacts, StreetFacts, appearance,
    approach_departure, arrival_consistent, attention_gate, choose_band, decide_match, decide_overlap, early_from,
    facts_from_row, gap_departure, grade_unmatched, handoff_id, hypothesis_deadlines, hypothesis_priors,
    merged_expect_value, narrative, observed_g, overlap_seconds, p_value_arrival, p_value_missed,
    salience, same_person_question, settle_final, street_event, strong_appearance, ttl_seconds, turned_back,
    unusual_for_hour, window_for,
)
from app.vision_individuals import MigrationMissing, is_missing_relation
from app.vision_individuals_loop import next_backoff
from app.vision_rhythm import EXPECT_KEY_PREFIX, coverage_fraction

logger = logging.getLogger("sql-writer.vision_handoff_loop")

HANDOFF_MIGRATION = "services/orion-sql-db/manual_migration_vision_handoff_v1.sql"
REQUIRED_TABLES = ("vision_handoff", "vision_handoff_hypothesis", "vision_handoff_cursor", "vision_motion_story",
                   "vision_motion_story_cursor", "vision_events", "vision_scene_inventory", "vision_crop_observation")
CURSOR_OVERLAP_SEC = 10.0
# Overlap pairs and rings are decided once both cameras' stories have landed;
# this bounds how far back an undecided one is looked for.
PENDING_LOOKBACK_SEC = 6 * 3600.0
# A ring is "announced" by an approach departure this long before it.
RING_APPROACH_LOOKBACK_SEC = 120.0
STORIES_PER_TICK = 500
CROPS_PER_SIDE = 12
HOUR_HISTORY_DAYS = 28
# A street event is a bid for attention NOW: after a restart or a backlog,
# facts older than this are recorded in vision_handoff but not published.
# approach_door is a heads-up, so it has a much shorter shelf life.
STREET_MAX_AGE_SEC = 3600.0
APPROACH_MAX_AGE_SEC = 300.0


@dataclass(frozen=True)
class HandoffLoopConfig:
    rules: HandoffConfig = field(default_factory=HandoffConfig)
    frame_default: Tuple[float, float] = (2560.0, 1920.0)
    frame_by_stream: Tuple[Tuple[str, Tuple[float, float]], ...] = ()
    initial_lookback_sec: float = 3600.0
    retention_days: float = 90.0
    steer_enabled: bool = True
    self_look_enabled: bool = True
    street_events_enabled: bool = True
    same_asks_enabled: bool = True
    unresolved_enabled: bool = True

    def frame_for(self, stream_id: str) -> Tuple[float, float]:
        return dict(self.frame_by_stream).get(stream_id, self.frame_default)


def _csv(raw: str) -> Tuple[str, ...]:
    return tuple(p.strip() for p in (raw or "").split(",") if p.strip())


def loop_config(settings: Settings) -> HandoffLoopConfig:
    from app.vision_motion_loop import parse_frame_sizes

    rules = HandoffConfig(
        kinds=_csv(settings.vision_handoff_kinds),
        look_kinds=_csv(settings.vision_handoff_look_kinds),
        min_met=int(settings.vision_handoff_min_met),
        min_coverage=float(settings.vision_handoff_min_coverage),
        appearance_bars=(("face", float(settings.vision_handoff_min_face_cosine)),
                         ("body", float(settings.vision_handoff_min_body_cosine)),
                         ("osnet", float(settings.vision_handoff_min_osnet_cosine))),
        attention_p=float(settings.vision_handoff_attention_p),
        same_ask_daily_cap=int(settings.vision_handoff_same_ask_daily_cap),
        ask_expiry_days=float(settings.vision_ask_expiry_days),
        ask_cooldown_days=float(settings.vision_ask_cooldown_days),
        thumb_retention_days=float(settings.vision_crop_thumb_retention_days),
        local_tz=str(settings.vision_local_tz),
    )
    return HandoffLoopConfig(
        rules=rules,
        frame_default=(float(settings.vision_motion_frame_width), float(settings.vision_motion_frame_height)),
        frame_by_stream=parse_frame_sizes(str(settings.vision_motion_frame_size_by_stream)),
        initial_lookback_sec=float(settings.vision_handoff_initial_lookback_sec),
        retention_days=float(settings.vision_motion_story_retention_days),
        steer_enabled=bool(settings.vision_handoff_steer_enabled),
        self_look_enabled=bool(settings.vision_handoff_self_look_enabled),
        street_events_enabled=bool(settings.vision_handoff_street_events_enabled),
        same_asks_enabled=int(settings.vision_handoff_same_ask_daily_cap) > 0,
    )


# ---------------------------------------------------------------------------
# Reads (also used by the replay eval, read-only)
# ---------------------------------------------------------------------------

STORY_COLS = ("story_id, stream_id, kind, started_at, ended_at, dir_lateral, dir_depth, zone_path, individual_id, "
              "crop_ids, evidence, speed_class, drop_reason, created_at")
EDGE_COLS = ("entry_edge", "exit_edge", "exit_foot_x", "exit_speed_fw_per_sec", "frame_w", "frame_h")


def story_select(edges: bool) -> str:
    extra = ", ".join(EDGE_COLS) if edges else ", ".join(f"NULL AS {c}" for c in EDGE_COLS)
    return f"SELECT {STORY_COLS}, {extra} FROM vision_motion_story s"


def edges_present(conn) -> bool:
    from app.vision_motion_loop import columns_missing

    return not columns_missing(conn, "vision_motion_story", EDGE_COLS)


def read_stories(conn, where: str, params: Mapping[str, Any], *, edges: bool, cfg: HandoffLoopConfig,
                 limit: int = STORIES_PER_TICK) -> List[StoryFacts]:
    from sqlalchemy import text

    rows = conn.execute(text(f"{story_select(edges)} WHERE {where} ORDER BY s.created_at, s.story_id LIMIT {int(limit)}"),
                        dict(params)).mappings().fetchall()
    out = []
    for r in rows:
        try:
            out.append(facts_from_row(r, cfg.frame_for(r["stream_id"])))
        except Exception as exc:  # one malformed story must not stop the stream
            logger.warning("handoff_story_unreadable story=%s error=%s", r.get("story_id"), exc)
    return out


def coverage(conn, stream_id: str, a: datetime, b: datetime, gap_sec: float) -> float:
    """Share of [a, b] the camera delivered census frames for (the doorbell:
    only during Ring bursts and self-looks)."""
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT COALESCE(window_start_ts, extract(epoch FROM observed_at)), "
        "COALESCE(window_end_ts, extract(epoch FROM observed_at)) FROM vision_scene_inventory "
        "WHERE stream_id=:s AND frame_count > 0 AND observed_at >= :a AND observed_at <= :b"),
        {"s": stream_id, "a": a - timedelta(seconds=gap_sec + 10), "b": b + timedelta(seconds=gap_sec + 10)}).fetchall()
    return coverage_fraction([(float(r[0]), float(r[1])) for r in rows], a, b, gap_tolerance_sec=gap_sec)


def stories_caught_up(conn, stream_id: str, through: datetime) -> bool:
    from sqlalchemy import text

    row = conn.execute(text("SELECT processed_until FROM vision_motion_story_cursor WHERE stream_id=:s"),
                       {"s": stream_id}).fetchone()
    return bool(row and row[0] and row[0] >= through)


def person_guess(conn, individual_ids: Sequence[Optional[str]], have_table: bool) -> Optional[str]:
    """The person model's ACCEPT guess for any of these individuals (known person), else None."""
    from sqlalchemy import text

    ids = [i for i in individual_ids if i]
    if not ids or not have_table:
        return None
    row = conn.execute(text(
        "SELECT person_label FROM vision_person_attribution WHERE individual_id = ANY(:i) AND revoked_at IS NULL "
        "AND band = 'accept' ORDER BY score DESC LIMIT 1"), {"i": ids}).fetchone()
    return row[0] if row else None


def learned_gaps(conn, portal_id: str, *, exclude: Optional[str] = None
                 ) -> Tuple[Dict[str, List[float]], Dict[str, int], Dict[str, int]]:
    """(g values of met+early+late rows per kind, met counts per kind, missed counts per kind)."""
    from sqlalchemy import text

    learned: Dict[str, List[float]] = {}
    met: Dict[str, int] = {}
    missed: Dict[str, int] = {}
    for kind, status, g in conn.execute(text(
            "SELECT subject_kind, status, g FROM vision_handoff WHERE portal_id=:p AND status IN ('met','early','late','missed') "
            "AND handoff_id <> :x ORDER BY scored_at DESC LIMIT 2000"), {"p": portal_id, "x": exclude or ""}).fetchall():
        if status == "missed":
            missed[kind] = missed.get(kind, 0) + 1
            continue
        if g is not None:
            learned.setdefault(kind, []).append(float(g))
        if status == "met":
            met[kind] = met.get(kind, 0) + 1
    return learned, met, missed


def vectors(conn, crop_ids: Sequence[str], tz: ZoneInfo) -> List[Dict[str, Any]]:
    from app.vision_person_model import load_crops_by_id

    ids = list(crop_ids)[-CROPS_PER_SIDE:]
    return [c.vecs for c in load_crops_by_id(conn, ids, tz).values()]


def _table_present(conn, name: str) -> bool:
    from sqlalchemy import text

    return bool(conn.execute(text("SELECT to_regclass(:t) IS NOT NULL"), {"t": name}).scalar())


def _check_tables(conn) -> None:
    from sqlalchemy import text

    missing = conn.execute(text("SELECT t FROM unnest(CAST(:t AS text[])) AS t WHERE to_regclass(t) IS NULL"),
                           {"t": list(REQUIRED_TABLES)}).fetchall()
    if missing:
        raise MigrationMissing("missing tables " + ",".join(r[0] for r in missing) + f" -- apply {HANDOFF_MIGRATION}")


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _insert_handoff(conn, row: Mapping[str, Any]) -> bool:
    from sqlalchemy import text

    cols = list(row)
    vals = ", ".join(f"CAST(:{c} AS jsonb)" if c == "street_event" else f":{c}" for c in cols)
    params = {c: (json.dumps(row[c]) if c == "street_event" and row[c] is not None else row[c]) for c in cols}
    res = conn.execute(text(f"INSERT INTO vision_handoff ({', '.join(cols)}) VALUES ({vals}) "
                            "ON CONFLICT (handoff_id) DO NOTHING"), params)
    return bool(res.rowcount)


def _update_handoff(conn, hid: str, **fields: Any) -> None:
    from sqlalchemy import text

    sets = ", ".join(f"{c} = CAST(:{c} AS jsonb)" if c == "street_event" else f"{c} = :{c}" for c in fields)
    params = {c: (json.dumps(v) if c == "street_event" and v is not None else v) for c, v in fields.items()}
    params["hid"] = hid
    conn.execute(text(f"UPDATE vision_handoff SET {sets} WHERE handoff_id = :hid"), params)


def _vision_event(conn, *, event_id: str, event_type: str, narrative_text: str, stream_id: Optional[str],
                  salience_v: float, refs: Sequence[str], tags: Sequence[str], confidence: float = 1.0) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_events (event_id, event_type, narrative, entities, tags, confidence, salience, "
        "evidence_refs, stream_id, created_at) VALUES (:id, :et, :n, CAST(:e AS jsonb), CAST(:tg AS jsonb), :c, "
        ":sal, CAST(:ev AS jsonb), :sid, now()) ON CONFLICT (event_id) DO NOTHING"),
        {"id": event_id, "et": event_type, "n": narrative_text, "e": json.dumps([]), "tg": json.dumps(list(tags)),
         "c": confidence, "sal": salience_v, "ev": json.dumps(list(refs)), "sid": stream_id})


# ---------------------------------------------------------------------------
# Redis steering
# ---------------------------------------------------------------------------


class Steering:
    """One Redis client per cycle on ORION_BUS_URL (the same Redis the rhythm
    loop and the retina use). Every failure is logged and counted, never raised."""

    def __init__(self, redis_url: Optional[str]):
        self.client = None
        self.failures = 0
        if redis_url:
            try:
                import redis

                self.client = redis.Redis.from_url(redis_url, socket_timeout=3, socket_connect_timeout=3)
            except Exception as exc:
                logger.warning("handoff_redis_unavailable error=%s", exc)

    def expect(self, stream_id: str, hid: str, until: datetime, now: datetime) -> bool:
        """Extend-only: rhythm's ids are kept and its TTL is never shortened."""
        ttl = ttl_seconds(until, now)
        if self.client is None or ttl is None:
            return False
        key = f"{EXPECT_KEY_PREFIX}{stream_id}"
        try:
            current = self.client.ttl(key)
            self.client.set(key, merged_expect_value(self.client.get(key), hid, until),
                            ex=max(ttl, int(current) if current and int(current) > 0 else 0))
            return True
        except Exception as exc:
            self.failures += 1
            logger.warning("handoff_expect_key_failed stream=%s error=%s", stream_id, exc)
            return False

    def look(self, hid: str, until: datetime, now: datetime) -> bool:
        """orion:vision:look:doorbell = handoff id. An open look (someone else
        approaching) is kept, only its TTL extended: one live view is enough."""
        ttl = ttl_seconds(until, now)
        if self.client is None or ttl is None:
            return False
        try:
            if self.client.set(LOOK_KEY, hid, ex=ttl, nx=True):
                return True
            current = self.client.ttl(LOOK_KEY)
            if current is not None and 0 < int(current) < ttl:
                self.client.expire(LOOK_KEY, ttl)
            return False
        except Exception as exc:
            self.failures += 1
            logger.warning("handoff_look_key_failed error=%s", exc)
            return False

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    conn: Any
    cfg: HandoffLoopConfig
    topology: Any
    now: datetime
    tz: ZoneInfo
    edges: bool
    have_attr: bool
    steer: Steering
    summary: Dict[str, Any]
    _learned: Dict[str, Tuple[Dict[str, List[float]], Dict[str, int], Dict[str, int]]] = field(default_factory=dict)

    def learned(self, portal_id: str):
        if portal_id not in self._learned:
            self._learned[portal_id] = learned_gaps(self.conn, portal_id)
        return self._learned[portal_id]

    def bump(self, key: str, n: int = 1) -> None:
        self.summary[key] = self.summary.get(key, 0) + n


def _street(ctx: Ctx, f: StreetFacts) -> Optional[dict]:
    if not ctx.cfg.street_events_enabled:
        return None
    max_age = APPROACH_MAX_AGE_SEC if f.kind == "approach_door" else STREET_MAX_AGE_SEC
    if (ctx.now - f.occurred_at).total_seconds() > max_age:
        ctx.bump("street_stale_skipped")
        return None
    try:
        return street_event(f)
    except Exception as exc:  # a schema reject is a bug: log it, keep the handoff
        logger.error("handoff_street_event_invalid kind=%s handoff=%s error=%s", f.kind, f.hid, exc)
        return None


def step_open(ctx: Ctx, stories: Sequence[StoryFacts]) -> None:
    rules = ctx.cfg.rules
    for f in stories:
        for portal in ctx.topology.of_kind("gap") + ctx.topology.of_kind("approach"):
            if portal.kind == "gap":
                dep = gap_departure(f, portal, rules)
                if dep is None:
                    continue
                dist, speed = dep
            elif approach_departure(f, portal, rules):
                dist, speed = 0.0, 1.0
            else:
                continue
            learned, met, _ = ctx.learned(portal.id)
            band = choose_band(f.kind, learned, met, portal, rules)
            ws, we, lu = window_for(f.ended_at, dist, speed, band, rules)
            hid = handoff_id(portal.id, f.story_id)
            guess = person_guess(ctx.conn, [f.individual_id], ctx.have_attr)
            row = {"handoff_id": hid, "portal_id": portal.id, "portal_kind": portal.kind,
                   "source_stream": f.stream_id, "target_stream": portal.to_stream, "source_story_id": f.story_id,
                   "subject_kind": f.kind, "individual_id": f.individual_id, "person_guess": guess,
                   "departed_at": f.ended_at, "opened_at": ctx.now, "window_start": ws, "window_end": we,
                   "late_until": lu, "exit_speed": None if portal.kind == "approach" else speed,
                   "exit_dist": None if portal.kind == "approach" else dist, "band_source": band.source,
                   "band_lo": band.lo, "band_hi": band.hi, "model_version": MODEL_VERSION, "status": "open"}
            if portal.kind == "approach":
                row["street_event"] = _street(ctx, StreetFacts(
                    kind="approach_door", hid=hid, portal_id=portal.id, stream_ids=[f.stream_id, portal.to_stream],
                    occurred_at=f.ended_at, subject=f.kind, individual_id=f.individual_id, person_guess=guess,
                    refs=[f"story:{f.story_id}"]))
            if not _insert_handoff(ctx.conn, row):
                continue
            ctx.bump(f"opened_{portal.kind}")
            steered: List[str] = []
            if we <= ctx.now:
                ctx.bump("steer_too_late")
            else:
                if ctx.cfg.steer_enabled and portal.to_stream != "doorbell" and \
                        ctx.steer.expect(portal.to_stream, hid, we, ctx.now):
                    steered.append(f"{EXPECT_KEY_PREFIX}{portal.to_stream}")
                if ctx.cfg.self_look_enabled and portal.to_stream == "doorbell" and f.kind in rules.look_kinds \
                        and ctx.steer.look(hid, we, ctx.now):
                    steered.append(LOOK_KEY)
            if steered:
                ctx.bump("steered")
                _update_handoff(ctx.conn, hid, steered=steered)


def _hour_unusual(ctx: Ctx, portal_id: str, t: datetime) -> bool:
    from sqlalchemy import text

    row = ctx.conn.execute(text(
        "SELECT count(*) FILTER (WHERE extract(hour FROM departed_at AT TIME ZONE :tz) = :h), min(departed_at) "
        "FROM vision_handoff WHERE portal_id = :p AND departed_at > :since"),
        {"tz": str(ctx.tz.key), "h": t.astimezone(ctx.tz).hour, "p": portal_id,
         "since": ctx.now - timedelta(days=HOUR_HISTORY_DAYS)}).fetchone()
    first = row[1] if row else None
    days = min(float(HOUR_HISTORY_DAYS), max(1.0, (ctx.now - first).total_seconds() / 86400.0)) if first else 1.0
    return unusual_for_hour(int(row[0] or 0), days, ctx.cfg.rules.unusual_hour_max_per_day)


def _approach_hits(ctx: Ctx, portal, a: datetime, b: datetime) -> List[Tuple[datetime, Optional[str], Optional[str]]]:
    """(time, event_id, story_id): Ring motion / ring events and doorbell person stories in [a, b]."""
    from sqlalchemy import text

    hits = [(r[1], r[0], None) for r in ctx.conn.execute(text(
        "SELECT event_id, created_at FROM vision_events WHERE stream_id=:s AND event_type = ANY(:ev) "
        "AND created_at >= :a AND created_at <= :b ORDER BY created_at"),
        {"s": portal.to_stream, "ev": list(portal.to_events), "a": a, "b": b}).fetchall()]
    for s in read_stories(ctx.conn, "s.stream_id = :s AND s.kind = ANY(:k) AND s.started_at >= :a AND s.started_at <= :b",
                          {"s": portal.to_stream, "k": list(ctx.cfg.rules.look_kinds), "a": a, "b": b},
                          edges=ctx.edges, cfg=ctx.cfg, limit=50):
        hits.append((s.started_at, None, s.story_id))
    return sorted(hits, key=lambda h: h[0])


def _gap_candidates(ctx: Ctx, portal, kind: str, a: datetime, b: datetime, src_vecs) -> List[Candidate]:
    out = []
    for s in read_stories(ctx.conn, "s.stream_id = :s AND s.kind = :k AND s.started_at >= :a AND s.started_at <= :b",
                          {"s": portal.to_stream, "k": kind, "a": a, "b": b}, edges=ctx.edges, cfg=ctx.cfg, limit=50):
        if s.drop_reason == "prop" or not arrival_consistent(s, portal, ctx.cfg.rules):
            continue
        score, ch = appearance(src_vecs, vectors(ctx.conn, s.crop_ids, ctx.tz)) if src_vecs else (None, None)
        out.append(Candidate(s, score, ch))
    return out


def _source(ctx: Ctx, story_id: Optional[str]) -> Optional[StoryFacts]:
    if not story_id:
        return None
    got = read_stories(ctx.conn, "s.story_id = :i", {"i": story_id}, edges=ctx.edges, cfg=ctx.cfg, limit=1)
    return got[0] if got else None


def step_grade(ctx: Ctx) -> None:
    from sqlalchemy import text

    rules = ctx.cfg.rules
    due = ctx.conn.execute(text(
        "SELECT * FROM vision_handoff WHERE status='open' AND late_until <= :t ORDER BY late_until LIMIT 200"),
        {"t": ctx.now - timedelta(seconds=rules.grade_lag_sec)}).mappings().fetchall()
    for r in due:
        portal = ctx.topology.by_id(r["portal_id"])
        if portal is None:
            _update_handoff(ctx.conn, r["handoff_id"], status="unscorable", scored_at=ctx.now)
            ctx.bump("unscorable")
            continue
        if not stories_caught_up(ctx.conn, r["target_stream"], r["late_until"]):
            if ctx.now - r["late_until"] < timedelta(seconds=rules.stuck_after_sec):
                ctx.bump("grade_deferred")
                continue
            _update_handoff(ctx.conn, r["handoff_id"], status="unscorable", scored_at=ctx.now)
            ctx.bump("unscorable")
            logger.warning("[HANDOFF] %s target stories never caught up -> unscorable", r["handoff_id"])
            continue
        src = _source(ctx, r["source_story_id"])
        upd: Dict[str, Any] = {"scored_at": ctx.now}
        target_t: Optional[datetime] = None
        status: str
        early = early_from(r["departed_at"], r["window_start"], r["window_end"])
        if portal.kind == "approach":
            hits = _approach_hits(ctx, portal, early, r["late_until"])
            if hits:
                target_t, ev_id, story_id = hits[0]
                status = ("early" if target_t < r["window_start"] else "met" if target_t <= r["window_end"]
                          else "late")
                upd.update(target_event_id=ev_id, target_story_id=story_id)
            else:
                status = "none"
        else:
            src_vecs = vectors(ctx.conn, src.crop_ids, ctx.tz) if src else []
            cands = _gap_candidates(ctx, portal, r["subject_kind"], early, r["late_until"], src_vecs)
            d = decide_match(cands, window_start=r["window_start"], window_end=r["window_end"],
                             late_until=r["late_until"], cfg=rules, early=early)
            status = d.status
            if d.chosen is not None:
                c = d.chosen
                target_t = c.story.started_at
                upd.update(target_story_id=c.story.story_id, target_individual_id=c.story.individual_id,
                           appearance_score=c.score, appearance_channel=c.channel)
        if status in ARRIVED or status == "ambiguous":
            transit = (target_t - r["departed_at"]).total_seconds()
            upd["transit_sec"] = transit
        if status in ARRIVED:
            g = observed_g(upd["transit_sec"], r["exit_dist"] or 0.0, r["exit_speed"] or 1.0, portal.kind)
            learned, _met, _ = ctx.learned(portal.id)
            calib_kind = "person" if r["band_source"] == "learned_person_widened" else r["subject_kind"]
            band = Band(r["band_lo"], r["band_hi"], r["band_source"], tuple(learned.get(calib_kind) or ()))
            p = p_value_arrival(g, band)
            guess = r["person_guess"] or person_guess(ctx.conn, [r["individual_id"], upd.get("target_individual_id")],
                                                      ctx.have_attr)
            gate = attention_gate(p, known_person=bool(guess), unusual_hour=_hour_unusual(ctx, portal.id, r["departed_at"]),
                                  cfg=rules)
            upd.update(g=g, p_value=p, attention_gate=gate, person_guess=guess, status=status)
            et = f"handoff_{status}"
            upd["salience"] = salience(et, r["subject_kind"], known_person=bool(guess), p=p)
            _vision_event(ctx.conn, event_id=f"{et}-{r['handoff_id']}", event_type=et,
                          narrative_text=narrative(status, kind=r["subject_kind"], source_stream=r["source_stream"],
                                                   target_stream=r["target_stream"], transit_sec=upd["transit_sec"]),
                          stream_id=r["target_stream"], salience_v=upd["salience"], tags=[portal.id, "handoff"],
                          refs=[f"handoff:{r['handoff_id']}", f"story:{r['source_story_id']}"]
                          + ([f"story:{upd['target_story_id']}"] if upd.get("target_story_id") else [])
                          + ([f"event:{upd['target_event_id']}"] if upd.get("target_event_id") else []))
            if status == "met" and guess:
                upd["street_event"] = _street(ctx, StreetFacts(
                    kind="handoff_met", hid=r["handoff_id"], portal_id=portal.id,
                    stream_ids=[r["source_stream"], r["target_stream"]], occurred_at=target_t, subject=r["subject_kind"],
                    individual_id=r["individual_id"], person_guess=guess, p_value=p, gate=gate,
                    refs=[f"story:{r['source_story_id']}"]
                    + ([f"story:{upd['target_story_id']}"] if upd.get("target_story_id") else [])))
            ctx._learned.pop(portal.id, None)
        elif status == "ambiguous":
            upd["status"] = "ambiguous"
        else:
            cov = coverage(ctx.conn, r["target_stream"], r["window_start"], r["window_end"], rules.census_gap_sec)
            status = grade_unmatched(cov, rules)
            upd.update(status=status, coverage=cov)
            if status == "missed":
                _learned, met_c, missed_c = ctx.learned(portal.id)
                arrived = len(_learned.get(r["subject_kind"]) or [])  # met + late of this kind
                p = p_value_missed(missed_c.get(r["subject_kind"], 0), arrived)
                upd.update(p_value=p, salience=salience("handoff_missed", r["subject_kind"], known_person=bool(
                    r["person_guess"]), p=p))
                _vision_event(ctx.conn, event_id=f"handoff_missed-{r['handoff_id']}", event_type="handoff_missed",
                              narrative_text=narrative("missed", kind=r["subject_kind"],
                                                       source_stream=r["source_stream"],
                                                       target_stream=r["target_stream"]),
                              stream_id=r["target_stream"], salience_v=upd["salience"], tags=[portal.id, "handoff"],
                              refs=[f"handoff:{r['handoff_id']}", f"story:{r['source_story_id']}",
                                    f"coverage:{cov:.2f}"])
                _open_hypotheses(ctx, r, portal)
                ctx._learned.pop(portal.id, None)
        ctx.bump(upd["status"])
        _update_handoff(ctx.conn, r["handoff_id"], **upd)
        logger.info("[HANDOFF] graded id=%s portal=%s kind=%s status=%s transit=%s g=%s p=%s coverage=%s",
                    r["handoff_id"], portal.id, r["subject_kind"], upd["status"], upd.get("transit_sec"),
                    upd.get("g"), upd.get("p_value"), upd.get("coverage"))


def _true_counts(ctx: Ctx, portal_id: str) -> Dict[str, int]:
    from sqlalchemy import text

    return {k: int(n) for k, n in ctx.conn.execute(text(
        "SELECT h.kind, count(*) FROM vision_handoff_hypothesis h JOIN vision_handoff o ON o.handoff_id = h.handoff_id "
        "WHERE o.portal_id = :p AND h.status = 'true' GROUP BY h.kind"), {"p": portal_id}).fetchall()}


def _write_hypothesis(ctx: Ctx, hid: str, kind: str, prior: float, deadline: datetime, status: str = "open",
                      evidence: Optional[dict] = None) -> None:
    from sqlalchemy import text

    ctx.conn.execute(text(
        "INSERT INTO vision_handoff_hypothesis (handoff_id, kind, prior, status, settled_by, evidence, deadline, "
        "settled_at) VALUES (:h, :k, :p, :s, :by, CAST(:e AS jsonb), :d, :at) ON CONFLICT (handoff_id, kind) DO NOTHING"),
        {"h": hid, "k": kind, "p": prior, "s": status, "by": None if status == "open" else "data",
         "e": json.dumps(evidence or {}), "d": deadline, "at": None if status == "open" else ctx.now})


def _set_hypothesis(ctx: Ctx, hid: str, kind: str, status: str, evidence: Optional[dict] = None) -> None:
    from sqlalchemy import text

    ctx.conn.execute(text(
        "UPDATE vision_handoff_hypothesis SET status=:s, settled_by='data', settled_at=:t, "
        "evidence = evidence || CAST(:e AS jsonb) WHERE handoff_id=:h AND kind=:k AND status='open'"),
        {"s": status, "t": ctx.now, "e": json.dumps(evidence or {}), "h": hid, "k": kind})
    ctx.bump(f"hyp_{kind}_{status}")


def _detector_crops(ctx: Ctx, stream_id: str, kind: str, a: datetime, b: datetime) -> int:
    from sqlalchemy import text

    return int(ctx.conn.execute(text(
        "SELECT count(*) FROM vision_crop_observation WHERE stream_id=:s AND label=:l AND observed_at >= :a "
        "AND observed_at <= :b"), {"s": stream_id, "l": kind, "a": a, "b": b}).scalar() or 0)


def _open_hypotheses(ctx: Ctx, r: Mapping[str, Any], portal) -> None:
    priors = hypothesis_priors(_true_counts(ctx, portal.id))
    deadlines = hypothesis_deadlines(graded_at=ctx.now, departed_at=r["departed_at"], late_until=r["late_until"],
                                     cfg=ctx.cfg.rules)
    hid = r["handoff_id"]
    kind = "person" if portal.kind == "approach" else r["subject_kind"]
    n = _detector_crops(ctx, r["target_stream"], kind, r["window_start"], r["late_until"])
    for k in HYPOTHESES:
        if k == "detector_miss":
            _write_hypothesis(ctx, hid, k, priors[k], deadlines[k], "true" if n else "false",
                              {"target_crops": n, "label": kind})
        else:
            _write_hypothesis(ctx, hid, k, priors[k], deadlines[k])


def step_hypotheses(ctx: Ctx) -> None:
    from sqlalchemy import text

    rows = ctx.conn.execute(text(
        "SELECT DISTINCT o.* FROM vision_handoff o JOIN vision_handoff_hypothesis h ON h.handoff_id = o.handoff_id "
        "WHERE h.status = 'open' AND o.status = 'missed' ORDER BY o.late_until LIMIT 200")).mappings().fetchall()
    for r in rows:
        portal = ctx.topology.by_id(r["portal_id"])
        if portal is None:
            continue
        hid = r["handoff_id"]
        hyps = {k: (s, d) for k, s, d in ctx.conn.execute(text(
            "SELECT kind, status, deadline FROM vision_handoff_hypothesis WHERE handoff_id=:h"), {"h": hid}).fetchall()}
        src = _source(ctx, r["source_story_id"])
        late_dl = hyps.get("late", ("", ctx.now))[1]
        if hyps.get("late", ("",))[0] == "open":
            if portal.kind == "approach":
                hit = next((h for h in _approach_hits(ctx, portal, r["late_until"], min(ctx.now, late_dl))
                            if h[0] > r["late_until"]), None)
                if hit:
                    _set_hypothesis(ctx, hid, "late", "true", {"at": hit[0].isoformat(), "event_id": hit[1],
                                                               "story_id": hit[2]})
            else:
                c = next((c for c in _gap_candidates(ctx, portal, r["subject_kind"], r["late_until"],
                                                     min(ctx.now, late_dl), [])
                          if c.story.started_at > r["late_until"]), None)
                if c:
                    _set_hypothesis(ctx, hid, "late", "true", {"story_id": c.story.story_id,
                                                               "at": c.story.started_at.isoformat()})
        tb_dl = hyps.get("turned_back", ("", ctx.now))[1]
        if hyps.get("turned_back", ("",))[0] == "open" and src is not None:
            later = read_stories(ctx.conn, "s.stream_id = :s AND s.kind = :k AND s.started_at > :a AND s.started_at <= :b",
                                 {"s": src.stream_id, "k": src.kind, "a": src.ended_at, "b": min(ctx.now, tb_dl)},
                                 edges=ctx.edges, cfg=ctx.cfg, limit=50)
            back = next((s for s in later if turned_back(src, s, portal)), None)
            if back:
                _set_hypothesis(ctx, hid, "turned_back", "true", {"story_id": back.story_id})
        deadline = max(d for _s, d in hyps.values())
        if ctx.now < deadline:
            continue
        if not (stories_caught_up(ctx.conn, r["target_stream"], deadline)
                and stories_caught_up(ctx.conn, r["source_stream"], deadline)) \
                and ctx.now - deadline < timedelta(seconds=ctx.cfg.rules.stuck_after_sec):
            continue
        current = {k: s for k, s, _d in ctx.conn.execute(text(
            "SELECT kind, status, deadline FROM vision_handoff_hypothesis WHERE handoff_id=:h"), {"h": hid}).fetchall()}
        last_zone = src.zone_path[-1] if src and src.zone_path else None
        final = settle_final(current, last_zone=last_zone, place_zones=ctx.topology.place_zones.get(r["source_stream"], ()))
        for k, s in final.items():
            if current.get(k) == "open":
                _set_hypothesis(ctx, hid, k, s, {"last_zone": last_zone} if k == "entered_place" else None)
        if final.get("unexplained") == "true":
            _unexplained(ctx, r, portal)


def _unexplained(ctx: Ctx, r: Mapping[str, Any], portal) -> None:
    """The existing curiosity seam (vision_unresolved, the council's table and
    ORM model) plus the handoff_missed street event."""
    hid = r["handoff_id"]
    if ctx.cfg.unresolved_enabled:
        from sqlalchemy.dialects.postgresql import insert

        from app.models.vision_walkway import VisionUnresolvedSQL
        from orion.schemas.vision import VisionUnresolvedV1

        item = VisionUnresolvedV1(
            unresolved_id=f"handoff-{hid}", stream_id=r["target_stream"], observed_at=r["late_until"],
            reason="handoff_missed",
            description=narrative("missed", kind=r["subject_kind"], source_stream=r["source_stream"],
                                  target_stream=r["target_stream"])
            + " Not late, did not turn back, not a detector miss, no known place to go.",
            what_was_tried=["detector_miss: false", "late: false", "turned_back: false", "entered_place: false"],
            evidence_refs=[f"handoff:{hid}", f"story:{r['source_story_id']}", f"portal:{portal.id}"])
        ctx.conn.execute(insert(VisionUnresolvedSQL.__table__).values(
            **item.model_dump(exclude={"schema_version"})).on_conflict_do_nothing())
        ctx.bump("unresolved_written")
    if r["street_event"] is None:
        ev = _street(ctx, StreetFacts(
            kind="handoff_missed", hid=hid, portal_id=portal.id, stream_ids=[r["source_stream"], r["target_stream"]],
            occurred_at=r["late_until"], subject=r["subject_kind"], individual_id=r["individual_id"],
            person_guess=r["person_guess"], p_value=r["p_value"],
            gate=attention_gate(r["p_value"], known_person=bool(r["person_guess"]),
                                unusual_hour=_hour_unusual(ctx, portal.id, r["departed_at"]), cfg=ctx.cfg.rules),
            refs=[f"story:{r['source_story_id']}", "hypothesis:unexplained"]))
        if ev is not None:
            _update_handoff(ctx.conn, hid, street_event=ev)


def step_overlap(ctx: Ctx) -> None:
    from sqlalchemy import text

    rules = ctx.cfg.rules
    for portal in ctx.topology.of_kind("overlap"):
        a_stream, b_stream = portal.streams
        pending = read_stories(
            ctx.conn,
            "s.stream_id = :s AND s.kind = ANY(:k) AND s.ended_at > :since AND coalesce(s.drop_reason, '') <> 'prop' "
            "AND NOT EXISTS (SELECT 1 FROM vision_handoff h WHERE h.portal_id = :p AND h.source_story_id = s.story_id)",
            {"s": a_stream, "k": list(rules.kinds), "since": ctx.now - timedelta(seconds=PENDING_LOOKBACK_SEC),
             "p": portal.id}, edges=ctx.edges, cfg=ctx.cfg, limit=100)
        for d in pending:
            skew = timedelta(seconds=portal.max_skew_sec)
            if not stories_caught_up(ctx.conn, b_stream, d.ended_at + skew):
                ctx.bump("overlap_deferred")
                continue
            others = read_stories(
                ctx.conn, "s.stream_id = :s AND s.kind = :k AND s.started_at <= :b AND s.ended_at >= :a "
                "AND coalesce(s.drop_reason, '') <> 'prop'",
                {"s": b_stream, "k": d.kind, "a": d.started_at - skew, "b": d.ended_at + skew},
                edges=ctx.edges, cfg=ctx.cfg, limit=20)
            others.sort(key=lambda o: -overlap_seconds(d, o))
            src_vecs = vectors(ctx.conn, d.crop_ids, ctx.tz)
            cands = []
            for o in others:
                score, ch = appearance(src_vecs, vectors(ctx.conn, o.crop_ids, ctx.tz)) if src_vecs else (None, None)
                cands.append(Candidate(o, score, ch))
            dec = decide_overlap(cands, rules)
            c = dec.chosen
            hid = handoff_id(portal.id, d.story_id)
            row = {"handoff_id": hid, "portal_id": portal.id, "portal_kind": "overlap", "source_stream": a_stream,
                   "target_stream": b_stream, "source_story_id": d.story_id, "subject_kind": d.kind,
                   "individual_id": d.individual_id, "departed_at": d.ended_at, "opened_at": ctx.now,
                   "window_start": d.started_at - skew, "window_end": d.ended_at + skew,
                   "model_version": MODEL_VERSION, "status": dec.status, "scored_at": ctx.now}
            if c is not None:
                guess = person_guess(ctx.conn, [d.individual_id, c.story.individual_id], ctx.have_attr)
                row.update(target_story_id=c.story.story_id, target_individual_id=c.story.individual_id,
                           appearance_score=c.score, appearance_channel=c.channel, person_guess=guess,
                           transit_sec=(c.story.started_at - d.started_at).total_seconds())
                if dec.status == "paired":
                    row["salience"] = salience("overlap_pair", d.kind, known_person=bool(guess), p=None)
                    row["street_event"] = _street(ctx, StreetFacts(
                        kind="overlap_pair", hid=hid, portal_id=portal.id, stream_ids=[a_stream, b_stream],
                        occurred_at=max(d.started_at, c.story.started_at), subject=d.kind,
                        individual_id=c.story.individual_id or d.individual_id, person_guess=guess,
                        refs=[f"story:{d.story_id}", f"story:{c.story.story_id}"]))
            if not _insert_handoff(ctx.conn, row):
                continue
            ctx.bump(f"overlap_{dec.status}")
            if dec.status == "paired":
                _vision_event(ctx.conn, event_id=f"overlap_pair-{hid}", event_type="overlap_pair",
                              narrative_text=narrative("paired", kind=d.kind, source_stream=a_stream,
                                                       target_stream=b_stream),
                              stream_id=b_stream, salience_v=row.get("salience") or 0.0, tags=[portal.id, "handoff"],
                              refs=[f"handoff:{hid}", f"story:{d.story_id}", f"story:{c.story.story_id}"])


def step_rings(ctx: Ctx) -> None:
    """A ring with no approach seen first (spec section 2)."""
    from sqlalchemy import text

    rules = ctx.cfg.rules
    for portal in ctx.topology.of_kind("approach"):
        if "doorbell_ring" not in portal.to_events:
            continue
        rings = ctx.conn.execute(text(
            "SELECT e.event_id, e.created_at FROM vision_events e WHERE e.stream_id = :s AND e.event_type = 'doorbell_ring' "
            "AND e.created_at > :since AND NOT EXISTS (SELECT 1 FROM vision_handoff h WHERE h.target_event_id = e.event_id) "
            "ORDER BY e.created_at LIMIT 50"),
            {"s": portal.to_stream, "since": ctx.now - timedelta(seconds=PENDING_LOOKBACK_SEC)}).fetchall()
        for ev_id, t in rings:
            look_from = t - timedelta(seconds=RING_APPROACH_LOOKBACK_SEC)
            if not stories_caught_up(ctx.conn, portal.from_stream, t + timedelta(seconds=30)):
                ctx.bump("ring_deferred")
                continue
            announced = ctx.conn.execute(text(
                "SELECT 1 FROM vision_handoff WHERE portal_id = :p AND source_story_id IS NOT NULL "
                "AND departed_at >= :a AND departed_at <= :b LIMIT 1"), {"p": portal.id, "a": look_from, "b": t}).fetchone()
            if announced:
                continue
            cov = coverage(ctx.conn, portal.from_stream, look_from, t, rules.census_gap_sec)
            hid = handoff_id(portal.id, "ring", ev_id)
            if not _insert_handoff(ctx.conn, {
                    "handoff_id": hid, "portal_id": portal.id, "portal_kind": "approach",
                    "source_stream": portal.from_stream, "target_stream": portal.to_stream, "target_event_id": ev_id,
                    "subject_kind": "person", "opened_at": ctx.now, "window_start": look_from, "window_end": t,
                    "model_version": MODEL_VERSION, "status": "unannounced", "coverage": cov, "scored_at": ctx.now}):
                continue
            ctx.bump("ring_unannounced")
            if cov >= rules.min_coverage:
                n = _detector_crops(ctx, portal.from_stream, "person", look_from, t)
                priors = hypothesis_priors(_true_counts(ctx, portal.id), RING_HYPOTHESES)
                _write_hypothesis(ctx, hid, "detector_miss", priors["detector_miss"], ctx.now,
                                  "true" if n else "false", {"walkway_person_crops": n})
                _write_hypothesis(ctx, hid, "out_of_view", priors["out_of_view"], ctx.now,
                                  "false" if n else "true", {"walkway_coverage": round(cov, 3)})


def step_asks(ctx: Ctx) -> int:
    """Strong met / paired pairs -> capped "same person?" asks. Never a label or a merge."""
    from sqlalchemy import text

    from app.vision_individuals import _insert_ask, ask_image_refs, local_day_start, pair_ref, same_groups
    from orion.schemas.ask import OrionAskV1

    rules = ctx.cfg.rules
    if not ctx.cfg.same_asks_enabled or rules.same_ask_daily_cap <= 0:
        return 0
    asked = int(ctx.conn.execute(text(
        "SELECT count(*) FROM orion_ask WHERE source_kind = :k AND created_at >= :d AND evidence_refs @> CAST(:m AS jsonb)"),
        {"k": SOURCE_KIND_SAME, "d": local_day_start(ctx.now, ctx.tz), "m": json.dumps([HANDOFF_RULE_REF])}).scalar() or 0)
    budget = rules.same_ask_daily_cap - asked
    if budget <= 0:
        return 0
    rows = ctx.conn.execute(text(
        "SELECT handoff_id, source_stream, target_stream, individual_id, target_individual_id, appearance_score, "
        "appearance_channel, transit_sec FROM vision_handoff WHERE status IN ('met', 'paired') AND ask_id IS NULL "
        "AND individual_id IS NOT NULL AND target_individual_id IS NOT NULL AND individual_id <> target_individual_id "
        "AND scored_at > :since ORDER BY appearance_score DESC NULLS LAST LIMIT 50"),
        {"since": ctx.now - timedelta(days=1)}).mappings().fetchall()
    rows = [r for r in rows if strong_appearance(r["appearance_score"], r["appearance_channel"], rules)]
    if not rows:
        return 0
    cool = ctx.now - timedelta(days=rules.ask_cooldown_days)
    skip = {r[0] for r in ctx.conn.execute(text(
        "SELECT source_ref FROM orion_ask WHERE source_kind=:k AND (status IN ('open','answered') "
        "OR (status IN ('expired','dismissed') AND created_at > :cool))"), {"k": SOURCE_KIND_SAME, "cool": cool}).fetchall()}
    verdicts = ctx.conn.execute(text(
        "SELECT individual_a, individual_b, verdict FROM vision_individual_pair_verdict")).fetchall()
    skip |= {pair_ref(v[0], v[1]) for v in verdicts}
    groups = same_groups((v[0], v[1]) for v in verdicts if v[2] == "same")
    n = 0
    for r in rows:
        if n >= budget:
            break
        a, b = r["individual_id"], r["target_individual_id"]
        ref = pair_ref(a, b)
        if ref in skip or groups.get(a, a) == groups.get(b, b):
            continue
        imgs = []
        for iid in (a, b):
            srows = ctx.conn.execute(text(
                "SELECT thumb_ref, ended_at, zone_no_embed, face_thumb_ref FROM vision_individual_sighting "
                "WHERE individual_id=:i AND NOT stationary ORDER BY started_at DESC LIMIT 20"), {"i": iid}).fetchall()
            refs = ask_image_refs([tuple(s) for s in srows], now=ctx.now, thumb_retention_days=rules.thumb_retention_days,
                                  ask_expiry_days=rules.ask_expiry_days, limit=1)
            if refs:
                imgs.append(refs[0])
        if len(imgs) != 2 or imgs[0] == imgs[1]:
            ctx.bump("ask_no_pictures")
            continue
        import uuid

        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}",
            question=same_person_question(r["source_stream"], r["target_stream"], r["transit_sec"]),
            evidence_refs=[f"individual:{a}", f"individual:{b}", f"handoff:{r['handoff_id']}", HANDOFF_RULE_REF,
                           f"{r['appearance_channel']}_cosine:{float(r['appearance_score']):.3f}"],
            image_ref=imgs[0], image_refs=imgs, created_at=ctx.now,
            expires_at=ctx.now + timedelta(days=rules.ask_expiry_days),
            source_kind=SOURCE_KIND_SAME, source_ref=ref)
        if _insert_ask(ctx.conn, ask):
            _update_handoff(ctx.conn, r["handoff_id"], ask_id=ask.ask_id)
            skip.add(ref)
            n += 1
    ctx.bump("asks_opened", n)
    return n


def step_prune(ctx: Ctx) -> None:
    from sqlalchemy import text

    if ctx.cfg.retention_days <= 0:
        return
    cut = ctx.now - timedelta(days=ctx.cfg.retention_days)
    n = ctx.conn.execute(text("DELETE FROM vision_handoff WHERE created_at < :c"), {"c": cut}).rowcount or 0
    if n:
        ctx.conn.execute(text("DELETE FROM vision_handoff_hypothesis h WHERE NOT EXISTS "
                              "(SELECT 1 FROM vision_handoff o WHERE o.handoff_id = h.handoff_id)"))
    ctx.summary["pruned"] = n


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------


def run_one_handoff_cycle(*, postgres_uri: str, cfg: HandoffLoopConfig, topology, now: Optional[datetime] = None,
                          redis_url: Optional[str] = None, engine=None) -> Dict[str, Any]:
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    summary: Dict[str, Any] = {"stories": 0}
    own = engine is None
    engine = engine or create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=1)
    lock_conn = None
    steer = Steering(redis_url if (cfg.steer_enabled or cfg.self_look_enabled) else None)
    try:
        lock_conn = engine.connect()
        if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": HANDOFF_LOCK_KEY}).scalar():
            summary["skipped"] = "another instance holds the handoff lock"
            return summary
        lock_conn.commit()
        with engine.begin() as conn:
            _check_tables(conn)
            edges = edges_present(conn)
            have_attr = _table_present(conn, "vision_person_attribution")
            if not edges:
                summary["edges"] = "derived from evidence (edges migration not applied)"
        tz = ZoneInfo(cfg.rules.local_tz)

        def ctx_for(conn) -> Ctx:
            return Ctx(conn=conn, cfg=cfg, topology=topology, now=ts, tz=tz, edges=edges, have_attr=have_attr,
                       steer=steer, summary=summary)

        # 1. Open, in the same transaction as the cursor move.
        with engine.begin() as conn:
            row = conn.execute(text("SELECT last_created_at FROM vision_handoff_cursor WHERE name='stories' FOR UPDATE")
                               ).fetchone()
            since = row[0] if row and row[0] else ts - timedelta(seconds=cfg.initial_lookback_sec)
            stories = read_stories(conn, "s.created_at > :a AND s.created_at <= :b",
                                   {"a": since - timedelta(seconds=CURSOR_OVERLAP_SEC),
                                    "b": ts - timedelta(seconds=1)}, edges=edges, cfg=cfg)
            summary["stories"] = len(stories)
            step_open(ctx_for(conn), stories)
            new_cursor = max([s.created_at for s in stories if s.created_at] + [since])
            conn.execute(text(
                "INSERT INTO vision_handoff_cursor (name, last_created_at, updated_at) VALUES ('stories', :t, now()) "
                "ON CONFLICT (name) DO UPDATE SET last_created_at = EXCLUDED.last_created_at, updated_at = now()"),
                {"t": new_cursor})
        for name, step in (("grade", step_grade), ("overlap", step_overlap), ("rings", step_rings),
                           ("hypotheses", step_hypotheses), ("asks", step_asks), ("prune", step_prune)):
            try:
                with engine.begin() as conn:
                    step(ctx_for(conn))
            except Exception as exc:
                if is_missing_relation(exc):
                    raise MigrationMissing(str(exc)) from exc
                logger.warning("handoff_step_failed step=%s error=%s", name, exc)
                summary[f"{name}_failed"] = str(exc)[:200]
        if steer.failures:
            summary["redis_failures"] = steer.failures
    except Exception as exc:
        if not isinstance(exc, MigrationMissing) and is_missing_relation(exc):
            raise MigrationMissing(str(exc)) from exc
        raise
    finally:
        steer.close()
        if lock_conn is not None:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": HANDOFF_LOCK_KEY})
                lock_conn.close()
            except Exception:
                pass
        if own:
            engine.dispose()
    return summary


def fetch_outbox(engine, limit: int = 50) -> List[Tuple[str, dict]]:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT handoff_id, street_event FROM vision_handoff WHERE street_event IS NOT NULL "
            "AND street_published_at IS NULL ORDER BY created_at LIMIT :n"), {"n": limit}).fetchall()
    return [(r[0], r[1] if isinstance(r[1], dict) else json.loads(r[1])) for r in rows]


def mark_published(engine, hids: Sequence[str], at: datetime) -> None:
    from sqlalchemy import text

    if hids:
        with engine.begin() as conn:
            conn.execute(text("UPDATE vision_handoff SET street_published_at = :t WHERE handoff_id = ANY(:h)"),
                         {"t": at, "h": list(hids)})


async def publish_outbox(bus, engine, *, service_name: str, service_version: str) -> int:
    """Publish pending StreetEventV1 rows; mark each one only after its publish succeeded."""
    import uuid

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    items = await asyncio.to_thread(fetch_outbox, engine)
    done: List[str] = []
    for hid, ev in items:
        try:
            env = BaseEnvelope(kind="vision.street.event.v1", source=ServiceRef(name=service_name, version=service_version),
                               correlation_id=uuid.UUID(ev["correlation_id"]), payload=ev)
            await bus.publish(STREET_CHANNEL, env)
            done.append(hid)
            logger.info("[HANDOFF] street_event kind=%s handoff=%s salience=%s corr=%s", ev.get("kind"), hid,
                        ev.get("salience"), ev.get("correlation_id"))
        except Exception as exc:
            logger.warning("handoff_street_publish_failed handoff=%s error=%s", hid, exc)
            break
    await asyncio.to_thread(mark_published, engine, done, datetime.now(timezone.utc))
    return len(done)


async def vision_handoff_loop(settings: Settings) -> None:
    enabled = bool(getattr(settings, "vision_handoff_enabled", False))
    interval = float(getattr(settings, "vision_handoff_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if not enabled or interval <= 0 or not postgres_uri:
        logger.info("vision_handoff_loop DISABLED (VISION_HANDOFF_ENABLED=%s interval=%s)", enabled, interval)
        return
    from sqlalchemy import create_engine

    from orion.vision.topology import load_topology

    try:
        topology = load_topology()
    except Exception as exc:
        logger.error("vision_handoff_loop DISABLED: topology invalid (%s)", exc)
        return
    cfg = loop_config(settings)
    redis_url = str(settings.orion_bus_url or "") or None
    bus = None
    if cfg.street_events_enabled and bool(settings.orion_bus_enabled) and redis_url:
        from orion.core.bus.async_service import OrionBusAsync

        bus = OrionBusAsync(url=redis_url)
        try:
            await bus.connect()
        except Exception as exc:
            logger.error("vision_handoff street-event bus connect failed: %s -- events wait in the outbox", exc)
    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=2, max_overflow=1)
    logger.info("vision_handoff_loop starting interval_sec=%.0f portals=%s cfg=%s", interval,
                [p.id for p in topology.portals], cfg)
    backoff: Optional[float] = None
    try:
        while True:
            await asyncio.sleep(backoff or interval)
            try:
                summary = await asyncio.to_thread(run_one_handoff_cycle, postgres_uri=postgres_uri, cfg=cfg,
                                                  topology=topology, redis_url=redis_url, engine=engine)
                backoff = None
                if bus is not None:
                    try:
                        summary["street_published"] = await publish_outbox(
                            bus, engine, service_name=settings.service_name, service_version=settings.service_version)
                    except Exception as exc:
                        logger.warning("handoff_outbox_failed error=%s", exc)
                busy = {k: v for k, v in summary.items() if v and k != "stories"}
                (logger.info if busy else logger.debug)("[HANDOFF] tick %s", summary)
            except asyncio.CancelledError:
                raise
            except MigrationMissing as exc:
                backoff = next_backoff(interval, backoff)
                logger.error("[HANDOFF] migration not applied: %s -- retrying in %.0fs", exc, backoff)
            except Exception as exc:
                logger.warning("vision_handoff_cycle_failed error=%s", exc)
    finally:
        engine.dispose()
        if bus is not None:
            try:
                await bus.close()
            except Exception:
                pass
