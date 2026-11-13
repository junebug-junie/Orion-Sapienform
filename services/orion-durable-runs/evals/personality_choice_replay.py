#!/usr/bin/env python3
"""Personality rev 4 replay: fold the live curiosity choices through the choice-only reducer. READ-ONLY.

Reads ``curiosity_offer_decisions`` JOIN ``curiosity_run_outcomes`` (the same query the curiosity sync
runs), builds the ``PersonalityChoiceV1`` rows, embeds every option text at vector-host (compute only,
nothing stored), folds them with ``orion.personality.graph.fold`` and prints what the graph would hold.
Writes nothing anywhere.

    python3 services/orion-durable-runs/evals/personality_choice_replay.py \\
        --dsn postgresql://postgres:postgres@localhost:55432/conjourney --embed-url http://localhost:8320/embedding

Checks (exit 1 when one fails; the metric quality gate's "live-data sanity" step, run on real rows):
  * choices exist, and at least one departs from the market's order (rank != 1): revealed preference
    needs departures to read anything (spec rev 4 M1: >= 25% departures);
  * directions are not degenerate: the median pairwise cosine of unrelated choices sits near 0
    (|median| < 0.2), so ``match_threshold`` separates real clusters from noise;
  * every node label is computed ("X over Y, n of m"), never a prose excerpt.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[3]
SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(SERVICE)]

from orion.personality.graph import DEFAULT_PARAMS, _dot, fold  # noqa: E402
from orion.schemas.personality import PersonalityChoiceV1  # noqa: E402

from app.personality_driver import all_texts, choice_evidence  # noqa: E402
from app.personality_sources import CURIOSITY_SQL, curiosity_run_choices  # noqa: E402

LABEL_SHAPE = re.compile(r"^.+ over .+, \d+ of \d+$|^.+, \d+ of \d+$")


def summarize(choices: Sequence[PersonalityChoiceV1], vectors: dict[str, list[float]], now: datetime) -> dict:
    """Pure: the replay's numbers and checks for a set of choices and their option vectors."""
    evs = [choice_evidence(c, vectors) for c in choices]
    dirs = [e.embedding for e in evs if e.embedding]
    sims = [_dot(a, b) for a, b in itertools.combinations(dirs, 2)]
    res = fold([], evs, now=now)
    nodes = sorted(res.nodes, key=lambda n: -n.weight)
    ranks = [c.chosen_market_rank for c in choices]
    departed = sum(1 for r in ranks if r != 1)
    median = statistics.median(sims) if sims else None
    checks = {
        "has_choices": bool(choices),
        "has_departures": departed > 0,
        "directions_not_degenerate": median is not None and abs(median) < 0.2,
        "labels_computed": all(LABEL_SHAPE.match(n.label or "") for n in nodes),
    }
    return {
        "choices": len(choices), "decisions": len({c.act_id for c in choices}),
        "rank1": sum(1 for r in ranks if r == 1), "departed": departed,
        "departure_share": departed / len(choices) if choices else 0.0,
        "directions": len(dirs), "pairwise_cos_median": median,
        "pairwise_cos_p99": sorted(sims)[int(0.99 * (len(sims) - 1))] if sims else None,
        "nodes": len(nodes), "active": sum(1 for n in nodes if n.status == "active"),
        "opposed_nodes": sum(1 for n in nodes if n.opposed_count),
        "top": [(round(n.weight, 3), n.status, n.label) for n in nodes[:10]],
        "match_threshold": DEFAULT_PARAMS.match_threshold, "checks": checks,
    }


def _rows(dsn: str) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        conn.execute("SET default_transaction_read_only = on")
        return list(conn.execute(CURIOSITY_SQL, {"at": epoch, "ref": "", "hi": datetime.now(timezone.utc),
                                                 "limit": 100000}).fetchall())


def _embedder(url: str) -> Callable[[str], list[float]]:
    import httpx

    client = httpx.Client(timeout=30)

    def embed(text: str) -> list[float]:
        doc = "personality-replay-" + hashlib.sha1(text.encode()).hexdigest()[:16]
        resp = client.post(url, json={"doc_id": doc, "text": text})
        resp.raise_for_status()
        return list(resp.json()["embedding"])

    return embed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--embed-url", default="http://localhost:8320/embedding")
    args = ap.parse_args(argv)
    choices = [c for r in _rows(args.dsn) for c in curiosity_run_choices(r)]
    embed = _embedder(args.embed_url)
    vectors: dict[str, Any] = {}
    for t in sorted({t for c in choices for t in all_texts(c)}):
        vectors[t] = embed(t)
    out = summarize(choices, vectors, datetime.now(timezone.utc))
    for k, v in out.items():
        if k == "top":
            print("top nodes:")
            for w, st, label in v:
                print(f"  {w:7.3f} {st:9s} {label}")
        else:
            print(f"{k}: {v}")
    return 0 if all(out["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
