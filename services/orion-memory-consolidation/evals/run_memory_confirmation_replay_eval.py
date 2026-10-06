"""Replay shadow memories through the REAL confirmation-card producer and report which cards it opens.

Label-free. Counts and categories only: no statement or question text is printed or written.

Two inputs, both replayed through ``orion.memory.episode.confirmation.run_tick`` in a throwaway
Postgres (``--scratch-dsn``, an admin DSN; a database is created and dropped per scenario):

* **live**: the rows in the live ``episode_memory`` table (``--live-dsn``, read in a read-only
  transaction), copied as they are. This is what would happen if the loop deployed today.
* **v3_relabel**: the stakes the v3 distiller prompt gave the same kind of episodes, from the
  committed counts-only summary ``results/2026-10-06-distill-stakes-before-after-summary.json``
  (one synthetic memory per high-stakes count, placeholder text). The live rows predate v3, so
  this is the closest honest preview of what the loop will be asked to carry.

Each input runs two scenarios: Juniper never answers (cards expire every 7 days, the next five
open), and Juniper confirms every card within a day. Both check the 5-open-card cap on every tick.

    python services/orion-memory-consolidation/evals/run_memory_confirmation_replay_eval.py \\
        --scratch-dsn postgresql://postgres:postgres@127.0.0.1:55499/postgres \\
        --live-dsn postgresql://postgres:postgres@127.0.0.1:55432/conjourney \\
        --summary-json services/orion-memory-consolidation/evals/results/2026-10-06-memory-confirmation-replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.memory.episode import confirmation as c  # noqa: E402

SQL_DIR = REPO_ROOT / "services" / "orion-sql-db"
MIGRATIONS = (
    "manual_migration_episode_memory_v1.sql",
    "manual_migration_walkway_camera_v1.sql",
    "manual_migration_attention_loop_outcome.sql",
    "manual_migration_memory_confirmation_v1.sql",
)
V3_SUMMARY = Path(__file__).resolve().parent / "results" / "2026-10-06-distill-stakes-before-after-summary.json"
T0 = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
MEMORY_COLUMNS = ("memory_id", "episode_id", "purpose", "voice", "channel", "statement", "occurred_at", "stakes",
                  "stakes_reason", "confirmation_state", "strength", "half_life_days", "last_reinforced_at",
                  "status", "created_at")


async def load_live(live_dsn: str) -> list[dict[str, Any]]:
    import asyncpg

    conn = await asyncpg.connect(live_dsn, server_settings={"default_transaction_read_only": "on"})
    try:
        rows = await conn.fetch(f"SELECT {', '.join(MEMORY_COLUMNS)} FROM episode_memory ORDER BY created_at")
    finally:
        await conn.close()
    return [dict(r) for r in rows]


# The summary records stakes only, not voice. ASSUMED voice per category: Orion's conclusions about
# itself are its own view from chat; everything else is something Juniper said. Counts of cards
# "framed as Juniper said" for this input therefore restate the assumption, not a measurement.
ORION_SELF_REASONS = frozenset({"orion_machinery", "orion_asks_direction", "orion_relationship"})


def synth_v3(summary_path: Path = V3_SUMMARY) -> list[dict[str, Any]]:
    """One synthetic memory per v3 'high:<category>' count, in episode order. Placeholder text."""
    data = json.loads(summary_path.read_text())
    out: list[dict[str, Any]] = []
    for i, ep in enumerate(data["episodes"]):
        for key, n in sorted((ep.get("v3_final") or {}).items()):
            stakes, _, reason = key.partition(":")
            own = reason in ORION_SELF_REASONS
            for j in range(int(n)):
                out.append({
                    "memory_id": uuid.uuid4(), "episode_id": ep["name"],
                    "purpose": "orion_view" if own else "about_juniper",
                    "voice": "orion_thought" if own else "juniper_said", "channel": "chat",
                    "statement": f"synthetic memory {i}.{j} ({key})", "occurred_at": None, "stakes": stakes,
                    "stakes_reason": reason if reason != "-" else None,
                    "confirmation_state": "pending_confirmation" if stakes == "high" else "auto",
                    "strength": 0.9, "half_life_days": 180.0, "last_reinforced_at": T0, "status": "active",
                    "created_at": T0 - timedelta(days=30) + timedelta(days=i, minutes=j),
                })
    return out


async def _scratch(admin_dsn: str):
    import asyncpg

    name = f"memconf_eval_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(admin_dsn)
    await admin.execute(f'CREATE DATABASE "{name}"')
    await admin.close()
    pool = await asyncpg.create_pool(dsn=admin_dsn.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=2)
    for f in MIGRATIONS:
        await pool.execute((SQL_DIR / f).read_text())
    return name, pool


async def _drop(admin_dsn: str, name: str, pool) -> None:
    import asyncpg

    await pool.close()
    admin = await asyncpg.connect(admin_dsn)
    await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    await admin.close()


async def _insert(pool, rows: list[dict[str, Any]]) -> None:
    cols = ", ".join(MEMORY_COLUMNS)
    params = ", ".join(f"${i}" for i in range(1, len(MEMORY_COLUMNS) + 1))
    for r in rows:
        await pool.execute(f"INSERT INTO episode_memory ({cols}) VALUES ({params})",
                           *[r[k] for k in MEMORY_COLUMNS])


async def _answer_all_open(pool, now: datetime) -> int:
    """Juniper confirms every open card (what the Hub's resolve route writes)."""
    rows = await pool.fetch("SELECT ask_id, source_ref FROM orion_ask WHERE status='open' AND source_kind=$1",
                            c.SOURCE_KIND)
    for r in rows:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("UPDATE orion_ask SET status='answered', answer='confirmed', answered_at=$2 "
                                   "WHERE ask_id=$1", r["ask_id"], now)
                await conn.execute(
                    "INSERT INTO attention_loop_outcome (outcome_id, loop_id, theme_key, verdict, actor, note, "
                    "features_at_close, created_at) VALUES ($1, $2, $2, 'resolved', 'juniper', '', $3::jsonb, $4)",
                    c.outcome_id_for(r["ask_id"]), r["source_ref"],
                    json.dumps(c.outcome_features(resolution="confirmed", ask_id=r["ask_id"],
                                                  memory_id=c.memory_id_from_loop(r["source_ref"]))), now)
    return len(rows)


async def run_scenario(admin_dsn: str, rows: list[dict[str, Any]], *, answers: bool, days: int = 35) -> dict:
    name, pool = await _scratch(admin_dsn)
    try:
        await _insert(pool, rows)
        ticks: list[dict[str, Any]] = []
        max_open = 0
        step = timedelta(days=1)
        for d in range(days + 1):
            now = T0 + d * step
            summary = await c.run_tick(pool, now=now)
            open_n = await pool.fetchval("SELECT count(*) FROM orion_ask WHERE status='open'")
            max_open = max(max_open, int(open_n))
            answered = await _answer_all_open(pool, now + timedelta(hours=6)) if answers else 0
            if any(summary.values()) or answered:
                ticks.append({"day": d, **summary, "open_after": int(open_n), "answered": answered})
        cards = await pool.fetch(
            "SELECT m.stakes_reason, m.voice, m.channel, a.question FROM orion_ask a "
            "JOIN episode_memory m ON a.source_ref = m.confirmation_loop_id")
        states = Counter(r["confirmation_state"] for r in await pool.fetch(
            "SELECT confirmation_state FROM episode_memory WHERE stakes='high'"))
        return {
            "cards_opened": len(cards),
            "cards_by_category": dict(Counter(r["stakes_reason"] or "uncategorized" for r in cards)),
            # Source monitoring, counted on the real rendered text (the text itself is not kept).
            "cards_framed_as_juniper_said": sum(1 for r in cards if r["question"].startswith("You told me")),
            "cards_framed_as_orions_own_take": sum(1 for r in cards if "my own take" in r["question"]),
            "cards_framed_as_internal_channel": sum(1 for r in cards if "not from anything you told me" in r["question"]),
            "max_open_at_once": max_open,
            "cap_held": max_open <= c.MAX_OPEN_CARDS,
            "high_stakes_end_states": dict(states),
            "ticks_with_activity": ticks,
        }
    finally:
        await _drop(admin_dsn, name, pool)


def describe(rows: list[dict[str, Any]]) -> dict:
    return {
        "memories": len(rows),
        "by_stakes": dict(Counter(f"{r['stakes']}:{r['stakes_reason'] or '-'}" for r in rows)),
        "by_state": dict(Counter(r["confirmation_state"] for r in rows)),
        "by_voice_channel": dict(Counter(f"{r['voice']}/{r['channel']}" for r in rows)),
    }


async def main_async(args) -> dict:
    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Counts only. The real producer (orion.memory.episode.confirmation.run_tick) replayed in a "
                "throwaway Postgres, one tick per simulated day. 'never_answered': no answers, cards expire "
                "after 7 days. 'all_confirmed': every open card confirmed the day it opens. v3_relabel voices are "
                "ASSUMED from the category (Orion-self categories = orion_thought), so its framing counts restate "
                "that assumption.",
        "cap": c.MAX_OPEN_CARDS,
        "ask_ttl_days": c.ASK_TTL.days,
        "inputs": {},
    }
    inputs: dict[str, list[dict[str, Any]]] = {}
    if args.live_dsn:
        inputs["live"] = await load_live(args.live_dsn)
    inputs["v3_relabel"] = synth_v3()
    for name, rows in inputs.items():
        out["inputs"][name] = {
            "rows": describe(rows),
            "never_answered": await run_scenario(args.scratch_dsn, rows, answers=False),
            "all_confirmed": await run_scenario(args.scratch_dsn, rows, answers=True),
        }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch-dsn", required=True, help="admin DSN of a THROWAWAY Postgres (never production)")
    ap.add_argument("--live-dsn", default=None, help="read-only source of the live episode_memory rows")
    ap.add_argument("--summary-json", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.live_dsn and args.scratch_dsn.rsplit("/", 1)[0] == args.live_dsn.rsplit("/", 1)[0]:
        ap.error("--scratch-dsn must not be the live server: the eval creates and drops databases there")
    result = asyncio.run(main_async(args))
    text = json.dumps(result, indent=1, default=str)
    print(text)
    if args.summary_json:
        args.summary_json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
