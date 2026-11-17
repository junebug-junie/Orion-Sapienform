"""Committed lived-memory replay eval on made-up fixtures (issue #3533).

Runs the real pipeline (app/lived_memory.py: exact-word candidates, live gating, rerank, the
bar, ranking, render) with a fake embedder (no dense hits) and a fake cross-encoder (logit grows
with shared content words), then scores it with the same ``score_bar`` the live replay uses.
It guards the harness and the ranking contract, not the model's quality: the live numbers come
from ``run_lived_memory_replay_eval.py`` against the host-side bank.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import app.lived_memory as lm
from app.lived_memory import LivedMemoryConfig, LivedMemoryRecall, tokenize
from app.settings import settings
from orion.introspect.semantic_index import SearchConfig

from run_lived_memory_replay_eval import is_small_talk, score_bar

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lived_memory_replay_fixture.json"
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


class _Pool:
    def __init__(self, data):
        self.rows = {}
        for kind, recs in data.items():
            self.rows[kind] = [{
                "record_id": r["record_id"], "purpose": r.get("purpose", kind), "text": r["text"],
                "occurred_at": NOW - timedelta(days=r["days_ago"]), "importance": 0.8, "is_correction": False,
                "confirmed": False, "label": r.get("label"), "created_at": NOW - timedelta(days=r["days_ago"]),
            } for r in recs]

    async def fetch(self, sql, *args):
        kind = ("episode_memory" if "FROM episode_memory" in sql else
                "stance" if "FROM memory_crystallizations" in sql else "person_fact")
        rows = self.rows.get(kind, [])
        return [r for r in rows if not args or r["record_id"] in set(args[0])]


def _reranker(query, docs):
    q = set(tokenize(query))

    async def go():
        return [-4.0 + 3.0 * len(q & set(tokenize(d))) for d in docs]

    return go()


def test_fixture_replay_hits_every_target_and_keeps_small_talk_empty(monkeypatch):
    """At the shipped content-word floor: short messages that name something rare still search."""
    async def embed(*_a, **_k):
        return [0.0], "fake"

    async def nearest(*_a, **_k):
        return []

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cfg = LivedMemoryConfig(search=SearchConfig("http://c", "http://e", "orion_lived_memory", 0.0),
                            rerank_url="http://r", min_fit=0.0, max_lines=64, timeout_sec=1.0,
                            min_query_tokens=settings.RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS)
    reader = LivedMemoryRecall(_Pool(data["records"]), cfg, reranker=_reranker, clock=lambda: NOW)

    async def run():
        results = []
        for item in data["referring"]:
            out = await reader.recall(item["message"])
            results.append({"label": item["label"], "targets": item["targets"], "reranked": out.reranked, "now": NOW})
        noise = []
        for msg in data["small_talk"]:
            out = await reader.recall(msg)
            noise.append({"reranked": out.reranked, "now": NOW, "small_talk": is_small_talk(msg)})
        return results, noise

    results, noise = asyncio.run(run())
    bar = score_bar(results, noise, min_fit=0.5, now=NOW)
    assert bar["hit_rate"] == 1.0, bar["items"]
    assert bar["small_talk_messages"] == len(data["small_talk"])
    assert bar["small_talk_share_injecting"] == 0.0


def test_fixture_replay_implicit_references_need_the_rewrite(monkeypatch):
    """Step 2 (issue #3534): short implicit messages hit only with the rewrite; small talk, whose
    rewrite answers NONE, still injects nothing."""
    async def embed(*_a, **_k):
        return [0.0], "fake"

    async def nearest(*_a, **_k):
        return []

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    answers = {item["message"]: item["rewrites"] for item in data["implicit"]}

    async def rewriter(message, _context):
        return list(answers.get(message, []))  # small talk -> NONE

    def build(with_rewrite: bool) -> LivedMemoryRecall:
        cfg = LivedMemoryConfig(search=SearchConfig("http://c", "http://e", "orion_lived_memory", 0.0),
                                rerank_url="http://r", min_fit=0.0, max_lines=64, timeout_sec=2.0,
                                min_query_tokens=settings.RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS,
                                rewrite_timeout_sec=0.5, link_top_k=0)
        return LivedMemoryRecall(_Pool(data["records"]), cfg, reranker=_reranker,
                                 rewriter=rewriter if with_rewrite else None, clock=lambda: NOW)

    async def run(reader):
        results = []
        for item in data["implicit"]:
            out = await reader.recall(item["message"])
            results.append({"label": item["label"], "kind": "implicit", "targets": item["targets"],
                            "reranked": out.reranked, "now": NOW})
        noise = []
        for msg in data["small_talk"]:
            out = await reader.recall(msg)
            noise.append({"reranked": out.reranked, "now": NOW, "small_talk": is_small_talk(msg)})
        return results, noise

    before = score_bar(*asyncio.run(run(build(False))), min_fit=0.5, now=NOW)
    after = score_bar(*asyncio.run(run(build(True))), min_fit=0.5, now=NOW)
    assert before["by_kind"] == {"implicit": "0/3"}
    assert after["by_kind"] == {"implicit": "3/3"}, after["items"]
    assert after["small_talk_share_injecting"] == 0.0
