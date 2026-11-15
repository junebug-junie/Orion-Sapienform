"""Replay real chat queries through both PCR chat phases and count, per source,
what would land in the chat prompt's DURABLE MEMORY (issue #3461).

Read-only by construction: concept_region runs without its reinforcement
write, and the (pre-#3461) active_packet collector's retrieval-event insert and
crystallization recall boost are replaced with no-ops when that module exists.
Prints counts only, never recalled text (the queries are private chat).

Run it inside the recall container so it reaches the live Postgres/Falkor with
the service's own .env (cwd /app), pointing --code-root at the tree to test:

    docker cp <tree> orion-athena-recall:/tmp/r3461/<label>
    docker exec -w /app orion-athena-recall \\
        python /tmp/r3461/<label>/evals/run_chat_durable_memory_noise_eval.py \\
        --code-root /tmp/r3461/<label> --queries /tmp/r3461/queries.json --label <label>

--code-root must hold app/ (services/orion-recall/app) and orion/ (repo orion/).
--queries is a JSON list of query strings. --continuity-minutes widens the
continuity window (live default 120) so a replay run hours after the last chat
still sees recent Hub and AI Town turns; use the same value for both labels.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
from typing import Any, Dict, List

NOISE_KINDS = (
    "sql_chat:aitown_chat_history_log",
    "bus_synaptic_anomaly",
    "active_packet",
    "concept_region:topicfoundry",
)

_LINE = re.compile(r"^- \[([a-z_]+(?::[a-z_]+)?)\]\s*(.*)$")


def classify(rendered: str) -> collections.Counter:
    """Count rendered DURABLE MEMORY lines by source tag; topic-foundry
    concept_region lines are split out as their own kind."""
    out: collections.Counter = collections.Counter()
    for line in (rendered or "").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        if tag.startswith("bus_synaptic_anomaly"):
            tag = "bus_synaptic_anomaly"
        if tag == "concept_region" and "topicfoundry" in rest:
            tag = "concept_region:topicfoundry"
        out[tag] += 1
    return out


def _make_read_only(worker: Any) -> None:
    from app.collectors import concept_region as cr

    worker.fetch_concept_region_fragment_and_reinforce = (
        lambda q, *, store, abandoned=None, **_kw: cr.fetch_concept_region_fragment(q, store=store)
    )
    try:
        import app.collectors.active_packet as ap  # only present before #3461

        async def _no_insert(*_a, **_k):
            return None

        ap.insert_retrieval_event = _no_insert
        import orion.memory.crystallization.retriever as retriever

        async def _no_boost(**_k):
            return None

        retriever._apply_recall_boost = _no_boost
    except ImportError:
        pass


async def _run(queries: List[str], label: str, continuity_minutes: int | None = None) -> Dict[str, Any]:
    import asyncpg  # type: ignore

    from app import worker
    from orion.core.contracts.recall import RecallQueryV1

    _make_read_only(worker)
    if continuity_minutes:
        # Both the fetch (profile sql_since_minutes) and the render window
        # (settings) must widen. Patch get_profile to hand back a copy: the
        # pre-#3461 worker mutated the cached continuity profile in place,
        # which would leak the widened window into later calls unevenly.
        worker.settings.RECALL_CONTINUITY_SQL_MINUTES = int(continuity_minutes)
        _orig_get_profile = worker.get_profile

        def _get_profile(name: str) -> Dict[str, Any]:
            prof = dict(_orig_get_profile(name))
            if prof.get("profile") == "chat.continuity.v1":
                prof["sql_since_minutes"] = int(continuity_minutes)
            return prof

        worker.get_profile = _get_profile
    pool = await asyncpg.create_pool(dsn=worker.settings.RECALL_PG_DSN, min_size=1, max_size=2)
    worker.set_recall_pg_pool(pool)
    totals: Dict[str, collections.Counter] = {"continuity": collections.Counter(), "purposeful": collections.Counter()}
    turns_with: Dict[str, collections.Counter] = {"all": collections.Counter()}
    empty_digests = 0
    try:
        for i, text in enumerate(queries):
            per_turn: collections.Counter = collections.Counter()
            for phase, profile, intent in (
                ("continuity", "chat.continuity.v1", "continuity"),
                ("purposeful", "chat.belief.open_loop.v1", "open_loop"),
            ):
                q = RecallQueryV1(
                    fragment=text,
                    profile=profile,
                    recall_phase=phase,
                    retrieval_intent=intent,
                    session_id=f"eval-3461-{i}",
                )
                bundle, _decision = await worker.process_recall(q, corr_id=f"eval-3461-{label}-{i}-{phase}")
                counts = classify(bundle.rendered)
                totals[phase].update(counts)
                per_turn.update(counts)
            if not per_turn:
                empty_digests += 1
            for kind in per_turn:
                turns_with["all"][kind] += 1
    finally:
        await pool.close()
    noise_lines = sum(totals[p][k] for p in totals for k in NOISE_KINDS)
    return {
        "label": label,
        "continuity_minutes": int(worker.settings.RECALL_CONTINUITY_SQL_MINUTES),
        "turns": len(queries),
        "lines_by_phase": {p: dict(c) for p, c in totals.items()},
        "turns_containing": dict(turns_with["all"]),
        "noise_lines_total": noise_lines,
        "turns_with_empty_durable_memory": empty_digests,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--continuity-minutes", type=int, default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.code_root)
    with open(args.queries, encoding="utf-8") as fh:
        queries = [str(q) for q in json.load(fh) if str(q).strip()]
    print(json.dumps(asyncio.run(_run(queries, args.label, args.continuity_minutes)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
