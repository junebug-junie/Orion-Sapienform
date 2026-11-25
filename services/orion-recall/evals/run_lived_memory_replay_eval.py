"""Replay eval for lived-memory recall (issues #3533, #3534, #3573).

Replays real Juniper chat messages through ``app/lived_memory.py`` and reports, per variant
(``--variants``: step1 = no rewrite, no links; rewrite; links; step2 = both; degraded = step 2
with the reranker pointed at a dead endpoint) and per quality bar (``--min-fit``):

- **hit rate**, split by bank ``kind`` (``explicit`` names the thing; ``implicit`` leans on the
  shared past without naming or asking): share of referring messages whose target record id
  lands in the injected lines;
- **noise**: lines injected on her last N real messages, and the share of small-talk messages
  (``is_low_info_social`` or 8 words or fewer) that inject anything;
- **latency**: per-call timings (snapshot, rewrite, dense, links, rerank, total), per variant.
  The first rewrite variant calls the fast lane for real; later ones reuse its answers (cached
  per message), so only that first variant's rewrite and total times are live.

The rewrite reads the same conversation's earlier ``chat_history_log`` turns (as of the replayed
message) and calls the LLM gateway over the bus (``--bus-url``, route ``--rewrite-route``).

Prints counts and ids only, never chat or memory text: the bank and the messages are private.
The bank lives host-side (``~/.orion/evals/lived_memory_recall_bank.json``, mode 600) and holds
chat_history_log ids plus target doc ids; message text is read from Postgres at run time. Each
replayed message only sees records stored before it (``as_of``), so a message cannot be answered
by the memory distilled from it.

Run from the repo root on the host (reaches Postgres, Chroma and the vector host on localhost):

    python services/orion-recall/evals/run_lived_memory_replay_eval.py \\
        --bank ~/.orion/evals/lived_memory_recall_bank.json \\
        --dsn postgresql://...@127.0.0.1:55432/conjourney \\
        --rerank-url http://127.0.0.1:8320/rerank --bus-url redis://<bus-host>:6379/0 \\
        --variants step1,rewrite,links,step2,degraded --min-fit 0.004

The committed fixture eval (``evals/test_lived_memory_replay_eval.py``) runs the same scoring on
made-up records with a fake embedder and reranker.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
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
    by_kind: Dict[str, List[int]] = {}
    for r in results:
        if r.get("chosen_ids") is not None:  # degraded runs choose without a fit score
            ids = list(r["chosen_ids"])[:MAX_LINES]
        else:
            chosen = rank(r["reranked"], min_fit=min_fit, max_lines=MAX_LINES, now=r.get("now") or now)
            ids = [s.rec.doc_id for s in chosen]
        hit = any(t in ids for t in r["targets"])
        hits += int(hit)
        k = by_kind.setdefault(r.get("kind") or "explicit", [0, 0])
        k[0] += int(hit)
        k[1] += 1
        per_item.append({"label": r["label"], "kind": r.get("kind"), "hit": hit, "lines": len(ids), "gate": r.get("gate"),
                         "target_rank_before_bar": r.get("target_rank")})
    lines_all, small, small_inject = [], 0, 0
    # Issue #3613: small-talk injection split by the short-message gate that let the search run,
    # and the share among pure social messages (is_low_info_social) alone.
    by_gate: Dict[str, List[int]] = {}
    social, social_inject = 0, 0
    for n in noise:
        if n.get("chosen_ids") is not None:
            chosen = list(n["chosen_ids"])[:MAX_LINES]
        else:
            chosen = rank(n["reranked"], min_fit=min_fit, max_lines=MAX_LINES, now=n.get("now") or now)
        lines_all.append(len(chosen))
        if n["small_talk"]:
            small += 1
            small_inject += int(bool(chosen))
            g = by_gate.setdefault(n.get("gate") or "none", [0, 0])
            g[0] += int(bool(chosen))
            g[1] += 1
        if n.get("social"):
            social += 1
            social_inject += int(bool(chosen))
    return {
        "min_fit": min_fit,
        "hit_rate": round(hits / len(results), 3) if results else None,
        "hits": hits,
        "referring": len(results),
        "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())},
        "items": per_item,
        "noise_messages": len(noise),
        "noise_mean_lines": round(statistics.mean(lines_all), 2) if lines_all else None,
        "noise_share_any_line": round(sum(1 for x in lines_all if x) / len(lines_all), 3) if lines_all else None,
        "small_talk_messages": small,
        "small_talk_share_injecting": round(small_inject / small, 3) if small else None,
        "small_talk_injecting_by_gate": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_gate.items())},
        "social_messages": social,
        "social_share_injecting": round(social_inject / social, 3) if social else None,
    }


VARIANTS = {
    # name: (rewrite, links, dead reranker)
    "step1": (False, False, False),
    "rewrite": (True, False, False),
    "links": (False, True, False),
    "step2": (True, True, False),
    "degraded": (True, True, True),
}


class _CachedRewriter:
    """Calls the real rewriter once per message; later variants reuse the answer."""

    def __init__(self, inner: Any, seed: Dict[str, List[str]] | None = None) -> None:
        self.inner = inner
        self.cache: Dict[str, List[str]] = dict(seed or {})
        self.errors = 0

    @staticmethod
    def key(message: str, short: bool = False) -> str:
        # The short-message prompt (issue #3613) is a different question: its own cache entry.
        return ("short:" if short else "") + hashlib.sha256(message.encode("utf-8")).hexdigest()

    async def __call__(self, message: str, context: Any, *, short: bool = False) -> List[str]:
        key = self.key(message, short)
        if key in self.cache:
            return self.cache[key]
        try:
            out = await (self.inner(message, context, short=True) if short else self.inner(message, context))
        except Exception:
            self.errors += 1
            raise
        self.cache[key] = list(out)
        return out


async def _replay(args: argparse.Namespace) -> Dict[str, Any]:
    import asyncpg  # type: ignore
    import httpx

    from app.lived_rewrite import BusRewriter

    bank = json.loads(Path(args.bank).expanduser().read_text(encoding="utf-8"))
    pool = await asyncpg.create_pool(args.dsn, min_size=1, max_size=2)
    bus = None
    rewriter = None
    if args.bus_url:
        from orion.core.bus.async_service import OrionBusAsync
        from orion.core.bus.bus_schemas import ServiceRef

        bus = OrionBusAsync(url=args.bus_url)
        await bus.connect()
        seed = None
        if args.rewrite_cache and Path(args.rewrite_cache).expanduser().exists():
            seed = json.loads(Path(args.rewrite_cache).expanduser().read_text(encoding="utf-8"))
        rewriter = _CachedRewriter(BusRewriter(
            bus, channel=args.llm_channel, route=args.rewrite_route, timeout_sec=args.rewrite_timeout,
            source=ServiceRef(name="orion-recall", version="replay-eval", node="eval")), seed)

    referring_rows = []
    # Budget mode (issue #3662) measures the chat-turn deadline, not hit rate: only the last real
    # messages are replayed, under the live end-to-end budget.
    for item in ([] if args.budget_sec else bank["referring"]):
        row = await pool.fetchrow("SELECT prompt, created_at, session_id FROM chat_history_log WHERE id = $1",
                                  item["chat_id"])
        if row is not None:
            referring_rows.append((item, row))
    noise_rows = await pool.fetch(
        "SELECT id, prompt, created_at, session_id FROM chat_history_log WHERE source = $1 "
        "AND length(trim(prompt)) > 0 AND prompt NOT LIKE $2 ORDER BY created_at DESC LIMIT $3",
        bank.get("noise_source", "hub_orion"), bank.get("noise_exclude_prefix", "Orion — ") + "%",
        int(args.noise_limit or bank.get("noise_limit", 150)))

    report: Dict[str, Any] = {"min_query_tokens": args.min_query_tokens, "referring": len(referring_rows),
                              "noise_messages": len(noise_rows), "variants": {}}
    now = datetime.now(timezone.utc)
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        use_rewrite, use_links, dead = VARIANTS[name]
        if use_rewrite and rewriter is None:
            report["variants"][name] = {"skipped": "no --bus-url"}
            continue
        cfg = LivedMemoryConfig(
            search=SearchConfig(chroma_url=args.chroma_url, embed_url=args.embed_url, collection=args.collection,
                                min_similarity=0.0),
            rerank_url=args.dead_rerank_url if dead else args.rerank_url, min_fit=0.0 if not dead else args.degraded_min_fit,
            max_lines=64 if not dead and not args.budget_sec else MAX_LINES,
            timeout_sec=float(args.budget_sec or 30.0),
            min_query_tokens=args.min_query_tokens, rewrite_timeout_sec=args.rewrite_timeout,
            link_top_k=args.link_top_k if use_links else 0)
        # Budget mode times the live rewrite in every variant: a cached answer would cost 0 ms.
        variant_rewriter = (_CachedRewriter(rewriter.inner) if args.budget_sec and rewriter is not None
                            else rewriter)
        lm = LivedMemoryRecall(pool, cfg, rewriter=variant_rewriter if use_rewrite else None)
        timings: List[Dict[str, int]] = []
        results: List[Dict[str, Any]] = []
        noise: List[Dict[str, Any]] = []
        outcomes: Dict[str, int] = {}
        gates: Dict[str, int] = {}
        backends: Dict[str, int] = {}
        degraded_reasons: Dict[str, int] = {}
        partial = 0
        rerank_errors: Dict[str, int] = {}

        def note(out: Any) -> None:
            nonlocal partial
            for reason in out.debug.get("degraded") or []:
                degraded_reasons[reason] = degraded_reasons.get(reason, 0) + 1
            partial += int(bool(out.debug.get("rerank_partial")))
            if out.debug.get("rerank_error"):  # exception type only, never text
                err = str(out.debug["rerank_error"]).split(":", 1)[0]
                rerank_errors[err] = rerank_errors.get(err, 0) + 1
            timings.append(out.debug.get("timings_ms") or {})
            g = out.debug.get("gate") or "none"
            gates[g] = gates.get(g, 0) + 1
            o = (out.debug.get("rewrite") or {}).get("outcome") or "none"
            outcomes[o] = outcomes.get(o, 0) + 1
            b = out.backend or "no_search"
            backends[b] = backends.get(b, 0) + 1

        async with httpx.AsyncClient(timeout=30.0) as client:
            await lm.snapshot(force=True)
            for item, row in referring_rows:
                at = row["created_at"].replace(tzinfo=timezone.utc)
                out = await lm.recall(row["prompt"], client=client, as_of=at, session_id=row["session_id"])
                note(out)
                ordered = sorted(out.reranked, key=lambda x: -x[1])
                rank_of = next((i + 1 for i, (rec, _l) in enumerate(ordered) if rec.doc_id in item["targets"]), None)
                results.append({"label": item["label"], "kind": item.get("kind"), "targets": item["targets"],
                                "gate": out.debug.get("gate"),
                                "reranked": out.reranked, "now": at, "target_rank": rank_of,
                                "chosen_ids": [c.rec.doc_id for c in out.chosen] if dead else None,
                                "target_in_pool": rank_of is not None or any(
                                    c.rec.doc_id in item["targets"] for c in out.chosen),
                                "errors": out.debug.get("error")})
            for row in noise_rows:
                at = row["created_at"].replace(tzinfo=timezone.utc)
                out = await lm.recall(row["prompt"], client=client, as_of=at, session_id=row["session_id"])
                note(out)
                noise.append({"reranked": out.reranked, "now": at, "small_talk": is_small_talk(row["prompt"]),
                              "social": is_low_info_social(row["prompt"]), "gate": out.debug.get("gate"),
                              "chosen_ids": [c.rec.doc_id for c in out.chosen] if dead else None})

        print(f"variant {name} done: {len(results)} referring, {len(noise)} noise", file=sys.stderr, flush=True)

        def pct(key: str, q: float) -> int | None:
            vals = sorted(t[key] for t in timings if key in t)
            return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None

        searched = backends.get("degraded", 0) + backends.get("hybrid", 0)
        bars = [score_bar(results, noise, float(x), now) for x in str(args.min_fit).split(",") if x.strip()]
        report["variants"][name] = {
            "targets_in_candidate_pool": {k: f"{sum(1 for r in results if r['kind'] == k and r['target_in_pool'])}/"
                                             f"{sum(1 for r in results if r['kind'] == k)}"
                                          for k in sorted({r['kind'] for r in results})},
            "errors": [r["label"] for r in results if r["errors"]],
            "rewrite_outcomes": outcomes,
            "gates": gates,
            "backends": backends,
            # Share of searched messages that fell back to exact words (issue #3662 bar: < 2%).
            "degraded_rate": round(backends.get("degraded", 0) / searched, 3) if searched else None,
            "degraded_reasons": degraded_reasons,
            "rerank_partial": partial,
            "rerank_errors": rerank_errors,
            "latency_ms": {k: {"p50": pct(k, 0.5), "p90": pct(k, 0.9), "max": pct(k, 0.999)}
                           for k in ("snapshot", "rewrite", "dense", "links", "rerank", "recheck", "total")},
            "bars": [{k: v for k, v in b.items() if k != "items" or args.items} for b in bars],
        }
    if rewriter is not None and args.rewrite_cache:
        # Same rewrites on the next run, so a before/after compares the code, not the LLM's
        # run-to-run variance. Private (Orion's searches over Juniper's messages): host-side, 600.
        path = Path(args.rewrite_cache).expanduser()
        path.write_text(json.dumps(rewriter.cache), encoding="utf-8")
        os.chmod(path, 0o600)
    if rewriter is not None:
        report["rewrite_calls"] = len(rewriter.cache)
        report["rewrite_errors"] = rewriter.errors
    if bus is not None:
        await bus.close()
    await pool.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="~/.orion/evals/lived_memory_recall_bank.json")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--chroma-url", default="http://127.0.0.1:8500")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8320/embedding")
    ap.add_argument("--rerank-url", default="http://127.0.0.1:8320/rerank")
    ap.add_argument("--dead-rerank-url", default="http://127.0.0.1:9/rerank",
                    help="an endpoint nothing listens on, for the degraded variant")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--min-fit", default="0.004")
    ap.add_argument("--degraded-min-fit", type=float, default=0.004)
    ap.add_argument("--min-query-tokens", type=int, default=5)
    ap.add_argument("--variants", default="step1,step2")
    ap.add_argument("--bus-url", default="")
    ap.add_argument("--llm-channel", default="orion:exec:request:LLMGatewayService")
    ap.add_argument("--rewrite-route", default="quick_turn")
    ap.add_argument("--rewrite-timeout", type=float, default=1.8)
    ap.add_argument("--link-top-k", type=int, default=6)
    ap.add_argument("--budget-sec", type=float, default=0.0,
                    help="replay under this end-to-end budget (live: RECALL_LIVED_MEMORY_TIMEOUT_SEC) and "
                         "skip the referring bank; reports degraded_rate (issue #3662). Calls run one at a "
                         "time on whatever else the GPU is doing, so on an idle P4 this is a lower bound: "
                         "the live rate needs recall_telemetry, or a run beside GPU load")
    ap.add_argument("--rewrite-cache", default="",
                    help="host-side JSON of rewrites keyed by message hash: read if present, written after "
                         "the run, so a before/after pair sees the same searches (not with --budget-sec)")
    ap.add_argument("--noise-limit", type=int, default=0, help="last N real messages (default: the bank's)")
    ap.add_argument("--items", action="store_true", help="include per-item hits (labels are private)")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(_replay(args)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
