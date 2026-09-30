"""Bounded-retrieval eval over real recall queries (no live backends).

Design: docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md
Fixture: evals/fixtures/recall_telemetry_queries_2026-09-29.json -- every distinct
(query, verb, profile) in live recall_telemetry on 2026-09-29.

For each real query this runs the real process_recall with every backend
replaced by an in-memory stub that (a) counts calls and (b) sleeps a fixed
per-call latency, and reports:

  intake       which text recall searches (caller / condensed / fragment) and its length
  expansion    sub-query count and whether any entity sub-query is a stopword
  work         backend calls, bounded pipeline vs the uncapped rollback
               (RECALL_MAX_SUB_QUERIES=0, RECALL_MAX_QUERY_CHARS=0) on the same stubs,
               plus the old sequential design's projected calls (signals x units)
  wall time    measured with the stubs (so: orchestration + fusion cost, not I/O)
  stability    top-8 selected ids, bounded vs uncapped, for queries <= 500 chars,
               with a lexical stub retriever over a corpus built from the fixture

What it CAN measure: that intake/expansion are bounded on real inputs, that
feeds run once, that the work per recall no longer scales with prompt length,
and how much the cap/stopword filter moves the top-8 when retrievers are
lexical. What it CANNOT measure: real backend latency, real recall quality
(the stub retriever is not Falkor/Postgres), or anything about the live
deploy. Acceptance check 6 in the design (live p99 < 10s) stays UNVERIFIED
until the service is deployed and recall_telemetry is re-read.

Run:  python services/orion-recall/evals/run_recall_bounded_retrieval_eval.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(SERVICE_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import worker  # noqa: E402
from orion.core.contracts.recall import MemoryBundleV1, RecallQueryV1  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "recall_telemetry_queries_2026-09-29.json"
K = 4
MAX_QUERY_CHARS = 600
STUB_LATENCY_S = 0.005
TOP_N = 8
FEED_NAMES = ("bus_synaptic_anomaly", "falkor_chat", "sql_chat_pairs", "sql_chat_msgs", "sql_timeline_recent", "sql_timeline_related")
# Units per signal in the pre-2026-09-29 sequential loop with this stub profile:
# falkor_neighborhood, bus_synaptic_anomaly, falkor_chat, sql_chat (2 calls),
# sql_timeline (2 calls) = 7 backend calls per signal.
OLD_CALLS_PER_SIGNAL = 7


def load_rows() -> List[Dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]


def _corpus(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lexical 'memories': sentences from the fixture itself, stable ids."""
    seen: Dict[str, str] = {}
    for r in rows:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", r["query"] or ""):
            sent = sent.strip()
            if 30 <= len(sent) <= 400 and sent not in seen:
                seen[sent] = f"mem-{len(seen):04d}"
    now = time.time()
    return [
        {"id": mid, "source": "falkor_neighborhood", "text": text, "ts": now - 3600, "score": 0.6, "tags": []}
        for text, mid in seen.items()
    ]


class _Counter:
    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}

    def spy(self, name: str, result_fn):
        async def _fn(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            await asyncio.sleep(STUB_LATENCY_S)
            return result_fn(*args, **kwargs)

        return _fn


def _lexical(corpus: List[Dict[str, Any]]):
    def _match(*_a, query_text: str = "", max_items: int = 4, **_k):
        toks = {t for t in re.findall(r"[a-z0-9]{4,}", (query_text or "").lower())}
        if not toks:
            return []
        scored = []
        for c in corpus:
            ctoks = set(re.findall(r"[a-z0-9]{4,}", c["text"].lower()))
            ov = len(toks & ctoks)
            if ov:
                scored.append((-ov, c["id"], c))
        scored.sort()
        return [dict(c) for _o, _i, c in scored[:max_items]]

    return _match


def _profile() -> Dict[str, Any]:
    return {
        "profile": "reflect.v1",
        "rdf_top_k": 4,
        "max_total_items": TOP_N,
        "max_per_source": TOP_N,
        "render_budget_tokens": 512,
        "enable_anchor_candidates": True,
        "enable_query_expansion": True,
    }


def _run_one(row: Dict[str, Any], corpus: List[Dict[str, Any]], *, bounded: bool) -> Dict[str, Any]:
    counter = _Counter()
    s = worker.settings
    feed_item = {"id": "feed-recent-1", "source": "bus_synaptic_anomaly", "text": "recent bus anomaly", "ts": time.time(), "score": 0.5, "tags": []}
    import app.recall_v2 as recall_v2

    async def _shadow(q, *, profile=None):
        return MemoryBundleV1(), {}

    async def _no_ts(*_a, **_k):
        return {}

    with ExitStack() as st:
        patch = lambda obj, name, val: st.enter_context(mock.patch.object(obj, name, val))  # noqa: E731
        patch(s, "RECALL_BUS_SYNAPTIC_ANOMALY_IN_CHAT", True)
        patch(s, "RECALL_FALKOR_IN_CHAT", True)
        patch(s, "RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT", True)
        patch(s, "RECALL_ENABLE_SQL_CHAT", True)
        patch(s, "RECALL_ENABLE_SQL_TIMELINE", True)
        patch(s, "RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED", False)
        patch(s, "RECALL_INTENT_ROUTING_ENABLED", False)
        patch(s, "RECALL_PCR_ENABLED", False)
        patch(s, "RECALL_DEADLINE_MS_DEFAULT", 60000)
        patch(s, "RECALL_MAX_SUB_QUERIES", K if bounded else 0)
        patch(s, "RECALL_MAX_QUERY_CHARS", MAX_QUERY_CHARS if bounded else 0)
        patch(worker, "get_profile", lambda _n: _profile())
        patch(worker, "fetch_bus_synaptic_anomaly_fragments", counter.spy("bus_synaptic_anomaly", lambda *a, **k: [dict(feed_item)]))
        patch(worker, "fetch_falkor_chatturn_fragments", counter.spy("falkor_chat", lambda *a, **k: []))
        patch(worker, "fetch_chat_history_pairs", counter.spy("sql_chat_pairs", lambda *a, **k: []))
        patch(worker, "fetch_chat_messages", counter.spy("sql_chat_msgs", lambda *a, **k: []))
        patch(worker, "fetch_recent_fragments", counter.spy("sql_timeline_recent", lambda *a, **k: []))
        patch(worker, "fetch_related_by_entities", counter.spy("sql_timeline_related", lambda *a, **k: []))
        patch(worker, "fetch_falkor_neighborhood_fragments", counter.spy("falkor_neighborhood", _lexical(corpus)))
        patch(worker, "fetch_exact_fragments", counter.spy("anchor_exact", lambda *a, **k: []))
        patch(worker, "fetch_chat_turn_timestamps", _no_ts)
        patch(recall_v2, "run_recall_v2_shadow", _shadow)

        q = RecallQueryV1(fragment=row["query"] or "", verb=row.get("verb"), profile=row.get("profile") or "reflect.v1")
        started = time.perf_counter()
        bundle, decision = asyncio.run(worker.process_recall(q, corr_id="eval"))
        wall_ms = (time.perf_counter() - started) * 1000

    entity_sigs = [sig for sig in _signals(row, bounded) if sig not in (decision.query, row.get("verb"))]
    return {
        "source": decision.retrieval_query_source,
        "query_chars": decision.query_chars,
        "sub_queries": decision.sub_query_count,
        "stopword_sub_queries": [
            sig for sig in entity_sigs if sig.split()[0].lower() in worker._EXPANSION_STOPWORDS
        ],
        "calls": dict(counter.calls),
        "total_calls": sum(counter.calls.values()),
        "wall_ms": wall_ms,
        "top": list(decision.selected_ids[:TOP_N]),
        "deadline_hit": decision.deadline_hit,
    }


def _signals(row: Dict[str, Any], bounded: bool) -> List[str]:
    q = RecallQueryV1(fragment=row["query"] or "", verb=row.get("verb"))
    with mock.patch.object(worker.settings, "RECALL_MAX_SUB_QUERIES", K if bounded else 0), mock.patch.object(
        worker.settings, "RECALL_MAX_QUERY_CHARS", MAX_QUERY_CHARS if bounded else 0
    ):
        intake = worker._intake_query(q, profile_name=str(row.get("profile") or ""))
        return worker._expand_query(intake["search_text"], verb=row.get("verb"), intent=None, enable=True)


def run_eval() -> Dict[str, Any]:
    rows = load_rows()
    corpus = _corpus(rows)
    results = []
    for row in rows:
        new = _run_one(row, corpus, bounded=True)
        old = _run_one(row, corpus, bounded=False)
        overlap = None
        if len(row["query"] or "") <= 500 and old["top"]:
            overlap = len(set(new["top"]) & set(old["top"])) / float(len(old["top"]))
        results.append(
            {
                "verb": row.get("verb"),
                "chars": len(row["query"] or ""),
                "new": new,
                "uncapped": old,
                "old_sequential_projected_calls": old["sub_queries"] * OLD_CALLS_PER_SIGNAL,
                "top8_overlap": overlap,
            }
        )
    short = [r for r in results if r["top8_overlap"] is not None]
    long = [r for r in results if r["chars"] > MAX_QUERY_CHARS]
    return {
        "rows": len(results),
        "results": results,
        "max_sub_queries": max(r["new"]["sub_queries"] for r in results),
        "max_query_chars_searched": max(r["new"]["query_chars"] for r in results),
        "stopword_sub_queries": sorted({s for r in results for s in r["new"]["stopword_sub_queries"]}),
        "feeds_called_once": all(
            r["new"]["calls"].get(f, 0) <= 1 for r in results for f in FEED_NAMES
        ),
        "short_query_count": len(short),
        "short_top8_overlap_mean": (sum(r["top8_overlap"] for r in short) / len(short)) if short else None,
        "short_top8_overlap_min": min((r["top8_overlap"] for r in short), default=None),
        "long_query_count": len(long),
        "long_max_wall_ms": max((r["new"]["wall_ms"] for r in long), default=0.0),
        "long_max_calls_new": max((r["new"]["total_calls"] for r in long), default=0),
        "long_max_calls_old_projected": max((r["old_sequential_projected_calls"] for r in long), default=0),
    }


def main() -> int:
    report = run_eval()
    print(f"rows={report['rows']} (distinct query/verb/profile from recall_telemetry 2026-09-29)")
    print(f"max sub-queries (K={K}): {report['max_sub_queries']}  [bound K+2={K + 2}]")
    print(f"max chars searched: {report['max_query_chars_searched']}  [bound {MAX_QUERY_CHARS}]")
    print(f"stopword sub-queries: {report['stopword_sub_queries'] or 'none'}")
    print(f"context feeds called at most once per recall: {report['feeds_called_once']}")
    print(
        f"short queries (<=500 chars): n={report['short_query_count']} top-8 overlap bounded-vs-uncapped "
        f"mean={report['short_top8_overlap_mean']:.3f} min={report['short_top8_overlap_min']:.3f}"
    )
    print(
        f"long queries (>{MAX_QUERY_CHARS} chars): n={report['long_query_count']} max wall={report['long_max_wall_ms']:.0f}ms "
        f"(stub I/O {STUB_LATENCY_S * 1000:.0f}ms/call) backend calls new={report['long_max_calls_new']} "
        f"vs old sequential projected={report['long_max_calls_old_projected']}"
    )
    print("\nper distinct long query:")
    seen = set()
    for r in sorted(report["results"], key=lambda r: -r["chars"]):
        key = (r["verb"], r["chars"])
        if r["chars"] <= MAX_QUERY_CHARS or key in seen:
            continue
        seen.add(key)
        n, u = r["new"], r["uncapped"]
        print(
            f"  {r['verb']:<26} {r['chars']:>6} chars -> {n['source']:<9} {n['query_chars']:>4} chars, "
            f"{n['sub_queries']} sub-queries (uncapped {u['sub_queries']}), calls {n['total_calls']} "
            f"(old projected {r['old_sequential_projected_calls']}), wall {n['wall_ms']:.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
