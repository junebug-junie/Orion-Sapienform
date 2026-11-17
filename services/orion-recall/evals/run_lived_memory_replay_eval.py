"""Replay eval for lived-memory recall (issue #3533).

Replays real Juniper chat messages through ``app/lived_memory.py`` and reports, per quality bar
(``--min-fit``):

- **hit rate**: share of referring messages whose target record id lands in the injected lines;
- **noise**: lines injected on her last N real messages, and the share of small-talk messages
  (``is_low_info_social`` or 8 words or fewer) that inject anything;
- **latency**: per-call timings (snapshot, dense, rerank, total).

Prints counts and ids only, never chat or memory text: the bank and the messages are private.
The bank lives host-side (``~/.orion/evals/lived_memory_recall_bank.json``, mode 600) and holds
chat_history_log ids plus target doc ids; message text is read from Postgres at run time. Each
replayed message only sees records stored before it (``as_of``), so a message cannot be answered
by the memory distilled from it.

Run from the repo root on the host (reaches Postgres, Chroma and the vector host on localhost):

    python services/orion-recall/evals/run_lived_memory_replay_eval.py \\
        --bank ~/.orion/evals/lived_memory_recall_bank.json \\
        --dsn postgresql://...@127.0.0.1:55432/conjourney \\
        --rerank-url http://127.0.0.1:8320/rerank --min-fit 0.02,0.05,0.1

The committed fixture eval (``evals/test_lived_memory_replay_eval.py``) runs the same scoring on
made-up records with a fake embedder and reranker.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve()
SERVICE = HERE.parents[1]
REPO = HERE.parents[3]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.lived_memory import LivedMemoryConfig, LivedMemoryRecall, rank  # noqa: E402
from orion.introspect.semantic_index import SearchConfig  # noqa: E402
from orion.memory.lived_memory import COLLECTION  # noqa: E402
from orion.memory.low_info_social import is_low_info_social  # noqa: E402

MAX_LINES = 4
SMALL_TALK_MAX_WORDS = 8


def is_small_talk(text: str) -> bool:
    return is_low_info_social(text) or len(str(text or "").split()) <= SMALL_TALK_MAX_WORDS


def score_bar(results: Sequence[Dict[str, Any]], noise: Sequence[Dict[str, Any]], min_fit: float,
              now: datetime) -> Dict[str, Any]:
    """Pure scoring of already-reranked candidates at one bar (testable without any service)."""
    hits, per_item = 0, []
    for r in results:
        chosen = rank(r["reranked"], min_fit=min_fit, max_lines=MAX_LINES, now=r.get("now") or now)
        ids = [s.rec.doc_id for s in chosen]
        hit = any(t in ids for t in r["targets"])
        hits += int(hit)
        per_item.append({"label": r["label"], "hit": hit, "lines": len(ids),
                         "target_rank_before_bar": r.get("target_rank")})
    lines_all, small, small_inject = [], 0, 0
    for n in noise:
        chosen = rank(n["reranked"], min_fit=min_fit, max_lines=MAX_LINES, now=n.get("now") or now)
        lines_all.append(len(chosen))
        if n["small_talk"]:
            small += 1
            small_inject += int(bool(chosen))
    return {
        "min_fit": min_fit,
        "hit_rate": round(hits / len(results), 3) if results else None,
        "hits": hits,
        "referring": len(results),
        "items": per_item,
        "noise_messages": len(noise),
        "noise_mean_lines": round(statistics.mean(lines_all), 2) if lines_all else None,
        "noise_share_any_line": round(sum(1 for x in lines_all if x) / len(lines_all), 3) if lines_all else None,
        "small_talk_messages": small,
        "small_talk_share_injecting": round(small_inject / small, 3) if small else None,
    }


async def _replay(args: argparse.Namespace) -> Dict[str, Any]:
    import asyncpg  # type: ignore
    import httpx

    bank = json.loads(Path(args.bank).expanduser().read_text(encoding="utf-8"))
    pool = await asyncpg.create_pool(args.dsn, min_size=1, max_size=2)
    cfg = LivedMemoryConfig(
        search=SearchConfig(chroma_url=args.chroma_url, embed_url=args.embed_url, collection=args.collection,
                            min_similarity=0.0),
        rerank_url=args.rerank_url, min_fit=0.0, max_lines=64, timeout_sec=30.0,
        min_query_tokens=args.min_query_tokens)
    lm = LivedMemoryRecall(pool, cfg)
    timings: List[Dict[str, int]] = []
    results: List[Dict[str, Any]] = []
    noise: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        await lm.snapshot(force=True)
        for item in bank["referring"]:
            row = await pool.fetchrow("SELECT prompt, created_at FROM chat_history_log WHERE id = $1", item["chat_id"])
            if row is None:
                continue
            at = row["created_at"].replace(tzinfo=timezone.utc)
            out = await lm.recall(row["prompt"], client=client, as_of=at)
            timings.append(out.debug.get("timings_ms") or {})
            ordered = sorted(out.reranked, key=lambda x: -x[1])
            rank_of = next((i + 1 for i, (rec, _l) in enumerate(ordered) if rec.doc_id in item["targets"]), None)
            results.append({"label": item["label"], "targets": item["targets"], "reranked": out.reranked,
                            "now": at, "target_rank": rank_of,
                            "target_in_pool": rank_of is not None, "errors": out.debug.get("error") or out.debug.get("dense_error")})
        rows = await pool.fetch(
            "SELECT id, prompt, created_at FROM chat_history_log WHERE source = $1 AND length(trim(prompt)) > 0 "
            "AND prompt NOT LIKE $2 ORDER BY created_at DESC LIMIT $3",
            bank.get("noise_source", "hub_orion"), bank.get("noise_exclude_prefix", "Orion — ") + "%",
            int(bank.get("noise_limit", 150)))
        for row in rows:
            at = row["created_at"].replace(tzinfo=timezone.utc)
            out = await lm.recall(row["prompt"], client=client, as_of=at)
            timings.append(out.debug.get("timings_ms") or {})
            noise.append({"reranked": out.reranked, "now": at, "small_talk": is_small_talk(row["prompt"])})
    await pool.close()
    now = datetime.now(timezone.utc)
    bars = [score_bar(results, noise, float(x), now) for x in str(args.min_fit).split(",") if x.strip()]

    def pct(key: str, q: float) -> int | None:
        vals = sorted(t[key] for t in timings if key in t)
        return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None

    return {
        "min_query_tokens": args.min_query_tokens,
        "referring": len(results),
        "targets_in_candidate_pool": sum(1 for r in results if r["target_in_pool"]),
        "errors": [r["label"] for r in results if r["errors"]],
        "noise_messages": len(noise),
        "latency_ms": {k: {"p50": pct(k, 0.5), "p90": pct(k, 0.9), "max": pct(k, 0.999)}
                       for k in ("snapshot", "dense", "rerank", "recheck", "total")},
        "bars": bars,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="~/.orion/evals/lived_memory_recall_bank.json")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--chroma-url", default="http://127.0.0.1:8500")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8320/embedding")
    ap.add_argument("--rerank-url", default="http://127.0.0.1:8320/rerank")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--min-fit", default="0.002,0.004,0.006,0.01,0.02")
    ap.add_argument("--min-query-tokens", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(asyncio.run(_replay(args)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
