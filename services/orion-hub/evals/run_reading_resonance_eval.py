#!/usr/bin/env python3
"""Reading resonance replay, read-only: crosswalk recent reads against what Orion holds.

    PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_resonance_eval.py \
        [--days 14] [--out-dir /tmp/reading-resonance-eval] \
        [--labeling-md docs/superpowers/evals/2026-10-13-resonance-labeling-set.md] \
        [--judge-url http://127.0.0.1:8222/v1/chat/completions --judge-model agent]

Design: docs/superpowers/specs/2026-10-13-reading-resonance-design.md (PR #2642), patch A.

What it reads (never writes): Postgres (world_pulse_read_seed, reading_document_snapshot,
memory_crystallizations, temporal_self_arc) inside a READ ONLY transaction; FalkorDB
orion_worldview and orion_substrate through GRAPH.RO_QUERY only; Chroma collections
through /get; vector-host /embedding (persists nothing).

Steps:
1. Similarity diagnosis on labelled-by-construction sets (a read vs its own Stage 2
   claims; a prior vs its own Findings; a lived question vs its own answers; each vs a
   random mismatched partner), raw vs mean-centered cosine. Printed and written to
   <out-dir>/similarity.json.
2. Crosswalk: per read, every kind (priors + findings/revisions, lived questions/answers,
   atlas concepts + 1-hop substrate neighbourhood, crystallizations, curiosity findings,
   dreams, earlier reads' open threads, open concern arcs) scored by centered,
   hub-corrected cosine; top of each kind re-read from its record store; frozen packet.
3. Judge prompt per packet, written to <out-dir>/prompts/. The judge only runs with
   --judge-url; without it the judge step is UNVERIFIED and the labeling set carries the
   retrieval's top pieces instead of a judged connection.
4. Labeling set (markdown + CSV) with empty wanted/notes columns for Juniper.

Exit 0 on a completed replay; 2 when a store is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from orion.curiosity.worldview import CLOSED_STATUSES
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.reading_resonance import ResonanceNeighborV1, ResonancePacketV1
from orion.world_pulse_read.fetch_text import load_retained_texts
from orion.world_pulse_read.resonance import (
    KIND_BUDGET,
    PoolDoc,
    ReadInput,
    ScoredDoc,
    build_judge_prompt,
    build_packet,
    check_judgment,
    fallback_article_summary,
    strained_prior,
    with_recent_counts,
)
from orion.world_pulse_read.resonance_similarity import (
    CENTERED_OFFER_FLOOR,
    center,
    dot,
    hub_from_sims,
    mean_vector,
    rank_score,
)

READS_SQL = """
SELECT seed_id, url, title, root_request_id::text AS root, stage2_completed_at,
       stage2_result_json, handoff_json
FROM world_pulse_read_seed
WHERE stage2_status = 'done' AND stage2_completed_at > $2::timestamptz - make_interval(days => $1)
  AND stage2_completed_at <= $2::timestamptz
ORDER BY stage2_completed_at
"""
OPEN_THREADS_SQL = """
SELECT seed_id, root_request_id::text AS root, coalesce(title, '') AS title, url,
       stage2_result_json->'open_threads' AS threads
FROM world_pulse_read_seed
WHERE stage2_status = 'done' AND stage2_completed_at > $2::timestamptz - make_interval(days => $1)
  AND stage2_completed_at <= $2::timestamptz
  AND jsonb_typeof(stage2_result_json->'open_threads') = 'array'
"""
CRYS_SQL = """
SELECT crystallization_id::text AS id, kind, subject, summary, status
FROM memory_crystallizations WHERE crystallization_id::text = ANY($1::text[])
"""
CONCERN_SQL = """
SELECT arc_id, subject_ref, arc_json FROM temporal_self_arc WHERE kind = 'concern' AND status = 'open'
"""
CHROMA_KINDS = {
    "orion_memory_crystallizations": "crystallization",
    "orion_curiosity": "curiosity_finding",
    "orion_dreams": "dream",
    "orion_atlas_concepts": "atlas_concept",
}
# Per kind, how many top-ranked docs are hydrated from their record store before the
# packet budget is applied (a hit whose record is gone is dropped, the next one moves up).
HYDRATE_PER_KIND = 3 * max(KIND_BUDGET.values())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Lease states that mean something is running or waiting on a GPU (orion/schemas/gpu_pool.py).
POOL_BUSY_STATUSES = frozenset({"granted", "queued", "backlogged", "recall"})


def pool_busy_reasons(state: dict[str, Any]) -> list[str]:
    """Why the GPU pool is not quiet, from its /v1/pool snapshot. Empty = quiet.

    Conservative on purpose: the judge's ``agent`` route runs at system priority, ahead of
    background reading/curiosity holds, so ANY live or waiting lease (and any queue or
    backlog) pauses it. Dead-lettered and unavailable leases are history, not load."""
    reasons = []
    for lease in state.get("leases") or []:
        if lease.get("status") in POOL_BUSY_STATUSES:
            reasons.append(f"{lease.get('status')} {lease.get('priority')} {lease.get('work_class')} "
                           f"lease held by {lease.get('holder')}")
    for key in ("queue_depth", "backlog_depth"):
        for k, v in (state.get(key) or {}).items():
            if v:
                reasons.append(f"{key}[{k}]={v}")
    return reasons


async def wait_for_quiet_pool(client: httpx.AsyncClient, pool_url: str, wait_sec: float) -> list[str]:
    """Poll /v1/pool until quiet or ``wait_sec`` passes (a lease from the previous judge
    call can take a moment to release). Returns the busy reasons, or [] when quiet. An
    unreadable pool counts as busy: no state, no call."""
    deadline = time.monotonic() + max(0.0, wait_sec)
    while True:
        try:
            resp = await client.get(pool_url.rstrip("/") + "/v1/pool", timeout=10.0)
            resp.raise_for_status()
            reasons = pool_busy_reasons(resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            reasons = [f"pool state unreadable: {type(exc).__name__}"]
        if not reasons or time.monotonic() >= deadline:
            return reasons
        await asyncio.sleep(5.0)


class Falkor:
    """GRAPH.RO_QUERY only: the server refuses any write in a read-only query."""

    def __init__(self, url: str) -> None:
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def rows(self, graph: str, cypher: str, **params: Any) -> list[dict[str, Any]]:
        prefix = ""
        if params:
            prefix = "CYPHER " + " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in params.items()) + " "
        res = self.r.execute_command("GRAPH.RO_QUERY", graph, prefix + cypher)
        return [dict(zip(res[0], row)) for row in res[1]]


class Embedder:
    def __init__(self, client: httpx.AsyncClient, url: str, cache_path: Path) -> None:
        self.client, self.url, self.cache_path = client, url, cache_path
        self.cache: dict[str, list[float]] = (
            json.loads(cache_path.read_text()) if cache_path.exists() else {})
        self.calls = 0

    async def __call__(self, text: str) -> list[float]:
        key = _sha(text)
        if key not in self.cache:
            resp = await self.client.post(self.url, json={"doc_id": f"resonance-replay-{key[:12]}", "text": text})
            resp.raise_for_status()
            self.cache[key] = resp.json()["embedding"]
            self.calls += 1
        return self.cache[key]

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))


async def chroma_all(client: httpx.AsyncClient, base: str, name: str) -> list[dict[str, Any]]:
    colls = {c["name"]: c["id"] for c in (await client.get(f"{base}/api/v1/collections")).json()}
    if name not in colls:
        return []
    resp = await client.post(f"{base}/api/v1/collections/{colls[name]}/get",
                             json={"limit": 20000, "include": ["documents", "metadatas", "embeddings"]})
    resp.raise_for_status()
    b = resp.json()
    return [{"id": i, "doc": d or "", "meta": m or {}, "vec": v}
            for i, d, m, v in zip(b["ids"], b["documents"], b["metadatas"], b["embeddings"])]


def read_claims(s2: dict[str, Any]) -> list[str]:
    out = [str(c.get("claim") or "") for c in s2.get("candidate_priors") or []]
    for c in s2.get("concept_candidates") or []:
        lab, d = str(c.get("label") or "").strip(), str(c.get("definition") or "").strip()
        out.append(f"{lab}: {d}" if d else lab)
    out += [str(c.get("claim_ref") or "") for c in s2.get("priors_tested") or []]
    return [c.strip() for c in out if c and c.strip()]


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))] if s else float("nan")


def _stats(pos: list[float], neg: list[float]) -> dict[str, float]:
    return {"n_pos": len(pos), "n_neg": len(neg), "pos_mean": sum(pos) / len(pos), "neg_mean": sum(neg) / len(neg),
            "pos_p10": _pct(pos, 0.10), "neg_p50": _pct(neg, 0.5), "neg_p95": _pct(neg, 0.95),
            "auc": _auc(pos, neg)}


# ---------------------------------------------------------------------------------------
# Record-store re-reads and neighbourhoods
# ---------------------------------------------------------------------------------------

def worldview_pool(fk: Falkor) -> tuple[list[PoolDoc], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    priors = [p for p in fk.rows("orion_worldview",
                                 "MATCH (p:Prior) RETURN p.prior_id AS id, p.claim AS claim, p.status AS status, "
                                 "p.confidence AS conf, p.times_tested AS tt")
              if p["claim"] and p["status"] not in CLOSED_STATUSES]
    findings = fk.rows("orion_worldview",
                       "MATCH (f:Finding)-[r]->(p:Prior) RETURN p.prior_id AS pid, type(r) AS rel, "
                       "f.finding_id AS fid, f.text AS text")
    revisions = fk.rows("orion_worldview",
                        "MATCH (v:PriorRevision) RETURN v.prior_id AS pid, v.from_status AS fs, v.to_status AS ts, "
                        "v.from_confidence AS fc, v.to_confidence AS tc, v.from_claim AS from_claim, "
                        "v.note AS note, v.written_at AS at ORDER BY v.written_at DESC")
    questions = fk.rows("orion_worldview",
                        "MATCH (s:SelfQuestionMint) WHERE s.text IS NOT NULL RETURN s.question_id AS qid, s.text AS text")
    answers = fk.rows("orion_worldview",
                      "MATCH (a:LivedAnswer) RETURN id(a) AS nid, a.run_id AS run_id, a.question_id AS qid, "
                      "a.text AS text, a.revises AS revises, a.written_at AS at ORDER BY a.written_at DESC")
    f_by: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        if f["text"]:
            f_by.setdefault(f["pid"], []).append(f)
    r_by: dict[str, list[dict[str, Any]]] = {}
    for v in revisions:
        r_by.setdefault(v["pid"], []).append(v)
    a_by: dict[str, list[dict[str, Any]]] = {}
    for a in answers:
        if a["text"]:
            a_by.setdefault(a["qid"], []).append(a)
    docs: list[PoolDoc] = []
    for p in priors:
        nb = [ResonanceNeighborV1(relation=f"finding_{f['rel'].lower()}", ref_id=str(f["fid"]), text=f["text"][:600])
              for f in f_by.get(p["id"], [])[:3]]
        for v in r_by.get(p["id"], [])[:2]:
            txt = (f"revision {v['fs']}->{v['ts']} conf {v['fc']}->{v['tc']}"
                   + (f"; was: {v['from_claim']}" if v.get("from_claim") else "")
                   + (f"; note: {v['note']}" if v.get("note") else ""))
            nb.append(ResonanceNeighborV1(relation="prior_revision", ref_id=f"{p['id']}@{v['at']}", text=txt[:600]))
        docs.append(PoolDoc("self_prior", p["id"], "falkor:orion_worldview:Prior",
                            f"{p['claim']} (status {p['status']}, confidence {p['conf']}, tested {p['tt']}x)",
                            tuple(nb)))
    for q in questions:
        nb = []
        for a in a_by.get(q["qid"], [])[:2]:
            nb.append(ResonanceNeighborV1(relation="lived_answer" + ("_revision" if a.get("revises") else ""),
                                          ref_id=f"{q['qid']}@{a['run_id']}", text=a["text"][:600]))
        docs.append(PoolDoc("lived_question", q["qid"], "falkor:orion_worldview:SelfQuestionMint", q["text"], tuple(nb)))
    qtext = {q["qid"]: q["text"] for q in questions}
    for qid, alist in a_by.items():
        # Each answer once; its question and the answer it revised are its neighbourhood.
        for i, a in enumerate(alist):
            nb = []
            if qid in qtext:
                nb.append(ResonanceNeighborV1(relation="answers_question", ref_id=qid, text=qtext[qid][:600]))
            if i + 1 < len(alist):
                nb.append(ResonanceNeighborV1(relation="previous_answer", ref_id=f"{qid}@{alist[i + 1]['run_id']}",
                                              text=alist[i + 1]["text"][:600]))
            docs.append(PoolDoc("lived_answer", f"{qid}@{a['run_id']}", "falkor:orion_worldview:LivedAnswer",
                                a["text"], tuple(nb)))
    pairs = {"findings": findings, "answers": answers, "questions": questions}
    return docs, pairs, priors  # type: ignore[return-value]


def atlas_hydrate(fk: Falkor, node_id: str) -> tuple[PoolDoc, set[str]] | None:
    """Re-read a concept from orion_substrate plus its 1-hop neighbourhood over substrate
    edges (legacy_unreviewed excluded) and the assertions it is subject/object of."""
    node = fk.rows("orion_substrate", "MATCH (n:Concept {node_id: $nid}) RETURN n.label AS label, "
                   "n.node_kind AS kind, n.evidence_refs_json AS ev", nid=node_id)
    if not node or not node[0]["label"]:
        return None
    nbrs = fk.rows("orion_substrate",
                   "MATCH (n:Concept {node_id: $nid})-[e]-(m:Concept) "
                   "WHERE e.substrate_edge = true AND coalesce(e.edge_role, '') <> 'legacy_unreviewed' "
                   "RETURN type(e) AS rel, m.node_id AS mid, m.label AS label, e.confidence AS conf "
                   "ORDER BY e.confidence DESC LIMIT 4", nid=node_id)
    asserts = fk.rows("orion_substrate",
                      "MATCH (a:Assertion)-[e]->(n:Concept {node_id: $nid}) "
                      "WHERE type(e) IN ['assertion_subject', 'assertion_object'] "
                      "RETURN type(e) AS rel, a.node_id AS aid, a.statement_text AS text LIMIT 2", nid=node_id)
    nb = [ResonanceNeighborV1(relation=r["rel"], ref_id=r["mid"], text=str(r["label"])[:600])
          for r in nbrs if r["label"]]
    nb += [ResonanceNeighborV1(relation=r["rel"], ref_id=r["aid"], text=str(r["text"])[:600])
           for r in asserts if r["text"]]
    try:
        evidence = {str(e) for e in json.loads(node[0]["ev"] or "[]")}
    except (TypeError, ValueError):
        evidence = set()
    doc = PoolDoc("atlas_concept", node_id, "falkor:orion_substrate:Concept", str(node[0]["label"]), tuple(nb))
    return doc, evidence


# ---------------------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    import asyncpg

    out = Path(args.out_dir)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    try:
        conn = await asyncpg.connect(args.dsn)
        fk = Falkor(args.falkor)
        fk.r.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"UNKNOWN: store unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        async with conn.transaction(readonly=True):  # nothing here writes; the server enforces it
            return await _replay(args, conn, fk, out)
    except Exception as exc:  # noqa: BLE001 - any store/embedder failure is UNKNOWN, not a verdict
        print(f"UNKNOWN: replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        await conn.close()


async def _replay(args: argparse.Namespace, conn: Any, fk: "Falkor", out: Path) -> int:
    random.seed(13)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        emb = Embedder(client, args.embed_url, out / "embed_cache.json")
        try:
            rows = await conn.fetch(READS_SQL, args.days, args.until)
            reads: list[dict[str, Any]] = []
            for r in rows:
                s2 = json.loads(r["stage2_result_json"]) if isinstance(r["stage2_result_json"], str) else r["stage2_result_json"]
                ho = json.loads(r["handoff_json"]) if isinstance(r["handoff_json"], str) else (r["handoff_json"] or {})
                ev = [SourceFetchEvidenceV1.model_validate(e) for e in ho.get("read_evidence") or []]
                texts = await load_retained_texts(conn, ev)
                learned = ho.get("what_i_learned") if isinstance(ho.get("what_i_learned"), str) else ""
                reads.append({
                    "input": ReadInput(seed_id=r["seed_id"], root_request_id=r["root"], title=r["title"] or "",
                                       url=r["url"], summary=str(s2.get("summary") or ""), learned=learned or "",
                                       retained_texts=[(t.sha256, t.text) for t in texts]),
                    "claims": read_claims(s2), "done": r["stage2_completed_at"],
                    "own_concepts": [str(c.get("label") or "") for c in s2.get("concept_candidates") or []
                                     if c.get("label")],
                })
            print(f"reads={len(reads)} (stage2 done, last {args.days} days)")

            # ---- pool: worldview (embedded now; no index exists yet) ----
            wv_docs, pairs, _ = worldview_pool(fk)
            pool: list[tuple[PoolDoc, list[float]]] = []
            for d in wv_docs:
                text = d.excerpt.split(" (status ")[0] if d.ref_kind == "self_prior" else d.excerpt
                pool.append((d, await emb(text[:2000])))
            # ---- pool: Chroma-indexed kinds (vectors from the index; text re-read later) ----
            chroma_rows: dict[str, list[dict[str, Any]]] = {}
            for name, kind in CHROMA_KINDS.items():
                chroma_rows[kind] = await chroma_all(client, args.chroma_url, name)
                for c in chroma_rows[kind]:
                    store = {"crystallization": "pg:memory_crystallizations",
                             "atlas_concept": "falkor:orion_substrate:Concept"}.get(kind, f"chroma:{name} (primary)")
                    ref = c["meta"].get("crystallization_id") or c["meta"].get("node_id") or c["id"]
                    pool.append((PoolDoc(kind, str(ref), store, c["doc"]), c["vec"]))
            # ---- pool: earlier reads' open threads + open concern arcs ----
            thread_root: dict[str, str | None] = {}
            for r in await conn.fetch(OPEN_THREADS_SQL, max(args.days, 60), args.until):
                threads = json.loads(r["threads"]) if isinstance(r["threads"], str) else r["threads"]
                for i, t in enumerate(threads or []):
                    text = t if isinstance(t, str) else json.dumps(t)
                    if text.strip():
                        ref = f"{r['seed_id']}#open_thread{i}"
                        thread_root[ref] = r["root"]
                        doc = PoolDoc("open_thread", ref, "pg:world_pulse_read_seed.stage2_result_json.open_threads",
                                      f"{text} (from an earlier read: {r['title'] or r['url']})")
                        pool.append((doc, await emb(text[:2000])))
            concerns = await conn.fetch(CONCERN_SQL)
            concern_text = 0
            for c in concerns:
                arc = json.loads(c["arc_json"]) if isinstance(c["arc_json"], str) else c["arc_json"]
                label = str(arc.get("subject_label") or "").strip()
                body = arc.get("body")
                text = label or (json.dumps(body) if body else "")
                if text:
                    concern_text += 1
                    pool.append((PoolDoc("open_thread", c["arc_id"], "pg:temporal_self_arc", text, floorless=True),
                                 await emb(text)))
            print(f"open concern arcs={len(concerns)} with text={concern_text} (an arc with no label or body "
                  "is not offered: it would be an empty excerpt)")

            # ---- query vectors: what the article says, not the Stage 2 process notes ----
            for rd in reads:
                texts = [t for t in [rd["input"].learned, *rd["claims"]] if t.strip()]
                fallback = rd["input"].summary or rd["input"].title or rd["input"].url
                rd["qraw"] = [await emb(t[:2000]) for t in texts] or [await emb(fallback)]
                rd["summary_vec"] = await emb(rd["input"].summary) if rd["input"].summary else None
                rd["claim_vecs"] = [await emb(t[:2000]) for t in rd["claims"]]
            # sanity-set texts
            fvec = {f["fid"]: await emb(f["text"][:2000]) for f in pairs["findings"] if f["text"]}
            avec = {a["nid"]: await emb(a["text"][:2000]) for a in pairs["answers"] if a["text"]}
            emb.save()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            print(f"UNKNOWN: embedder/index: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if not reads:
            print("no completed reads in the window; nothing to replay")
            return 0
        print(f"pool={len(pool)} embed_calls={emb.calls} ({time.perf_counter() - t0:.0f}s)")

        # ---- 1. similarity diagnosis ----
        all_raw = [v for _, v in pool] + [v for rd in reads for v in rd["qraw"]]
        mu = mean_vector(all_raw)
        C = lambda v: center(v, mu)  # noqa: E731
        diag: dict[str, Any] = {}
        prior_vec = {d.ref_id: v for d, v in pool if d.ref_kind == "self_prior"}
        q_vec = {d.ref_id: v for d, v in pool if d.ref_kind == "lived_question"}
        sets: dict[str, tuple[list[tuple[list[float], list[float]]], list[tuple[list[float], list[float]]]]] = {}
        pos, neg = [], []
        domains = [urlparse(rd["input"].url).hostname for rd in reads]
        for i, rd in enumerate(reads):
            if rd["summary_vec"] is None:
                continue
            others = [j for j in range(len(reads)) if domains[j] != domains[i] and reads[j]["claim_vecs"]]
            if not others:
                continue
            for cv in rd["claim_vecs"]:
                pos.append((rd["summary_vec"], cv))
                j = random.choice(others)
                neg.append((rd["summary_vec"], random.choice(reads[j]["claim_vecs"])))
        sets["read summary vs its own Stage 2 claims"] = (pos, neg)
        pos, neg = [], []
        pids = list(prior_vec)
        for f in pairs["findings"]:
            if f["pid"] in prior_vec and f["fid"] in fvec and len(pids) > 1:
                pos.append((prior_vec[f["pid"]], fvec[f["fid"]]))
                neg.append((prior_vec[random.choice([p for p in pids if p != f["pid"]])], fvec[f["fid"]]))
        sets["prior vs its own Finding"] = (pos, neg)
        pos, neg = [], []
        qids = list(q_vec)
        for a in pairs["answers"]:
            if a["qid"] in q_vec and a["nid"] in avec and len(qids) > 1:
                pos.append((q_vec[a["qid"]], avec[a["nid"]]))
                neg.append((q_vec[random.choice([q for q in qids if q != a["qid"]])], avec[a["nid"]]))
        sets["lived question vs its own answer"] = (pos, neg)
        print("\n== similarity diagnosis (labelled by construction) ==")
        print(f"{'set':42s} {'n':>4s} | raw: related / unrelated / AUC | centered: related / unrelated / AUC")
        pooled_raw: tuple[list[float], list[float]] = ([], [])
        pooled_cen: tuple[list[float], list[float]] = ([], [])
        for name, (P, N) in sets.items():
            if not P or not N:
                print(f"{name:42s}    0 | (no pairs)")
                continue
            rp, rn = [dot(a, b) for a, b in P], [dot(a, b) for a, b in N]
            cp, cn = [dot(C(a), C(b)) for a, b in P], [dot(C(a), C(b)) for a, b in N]
            diag[name] = {"raw": _stats(rp, rn), "centered": _stats(cp, cn)}
            if name != "read summary vs its own Stage 2 claims":
                pooled_raw[0].extend(rp); pooled_raw[1].extend(rn)
                pooled_cen[0].extend(cp); pooled_cen[1].extend(cn)
            print(f"{name:42s} {len(P):4d} | {diag[name]['raw']['pos_mean']:.3f} / {diag[name]['raw']['neg_mean']:.3f} / "
                  f"{diag[name]['raw']['auc']:.3f} | {diag[name]['centered']['pos_mean']:.3f} / "
                  f"{diag[name]['centered']['neg_mean']:.3f} / {diag[name]['centered']['auc']:.3f}")
        if pooled_raw[0] and pooled_raw[1]:
            diag["belief_sets_pooled"] = {"raw": _stats(*pooled_raw), "centered": _stats(*pooled_cen)}
            b = diag["belief_sets_pooled"]
            print(
                f"belief sets pooled: raw related p10={b['raw']['pos_p10']:.3f} unrelated p50={b['raw']['neg_p50']:.3f} "
                f"p95={b['raw']['neg_p95']:.3f} | centered related p10={b['centered']['pos_p10']:.3f} "
                f"unrelated p50={b['centered']['neg_p50']:.3f} p95={b['centered']['neg_p95']:.3f} "
                f"(offer floor {CENTERED_OFFER_FLOOR})")

        # ---- 2. crosswalk ----
        cen_pool = [(d, C(v)) for d, v in pool]
        for rd in reads:
            qs = [C(v) for v in rd["qraw"]]
            rd["q"] = center(mean_vector(qs), [0.0] * len(qs[0]))  # renormalized mean of centered queries
        # Each doc's similarity to every replayed read, once; a read's hub penalty leaves that
        # read out (a doc near ONLY this read must not be penalised for it).
        sims = [[dot(v, rd["q"]) for rd in reads] for _, v in cen_pool]

        def hubs_for(ri: int) -> list[float]:
            return [hub_from_sims(row[:ri] + row[ri + 1:]) for row in sims]

        hubs_by_read = [hubs_for(ri) for ri in range(len(reads))]
        # Raw-vs-centered attractor: which crystallization is in the most reads' top-2.
        crys_idx = [i for i, (d, _) in enumerate(cen_pool) if d.ref_kind == "crystallization"]
        raw_pool = [v for _, v in pool]
        att: dict[str, dict[str, int]] = {"raw": {}, "centered_hub": {}}
        for ri_, rd in enumerate(reads):
            hubs = hubs_by_read[ri_]
            qraw = mean_vector(rd["qraw"])
            for mode in att:
                if mode == "raw":
                    sc = sorted(crys_idx, key=lambda i: -dot(qraw, raw_pool[i]))[:2]
                else:
                    sc = sorted(crys_idx, key=lambda i: -rank_score(rd["q"], cen_pool[i][1], hubs[i]))[:2]
                for i in {cen_pool[i][0].excerpt[:80]: i for i in sc}.values():
                    att[mode][cen_pool[i][0].excerpt[:80]] = att[mode].get(cen_pool[i][0].excerpt[:80], 0) + 1
        for mode, counts in att.items():
            top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("-", 0)
            diag[f"attractor_{mode}"] = {"reads": top[1], "excerpt": top[0], "distinct": len(counts)}
            print(f"crystallization attractor [{mode}]: in top-2 for {top[1]}/{len(reads)} reads: {top[0]!r}")

        crys_ids = [str(c["meta"].get("crystallization_id")) for c in chroma_rows["crystallization"]
                    if c["meta"].get("crystallization_id")]
        crys_rows = {r["id"]: r for r in await conn.fetch(CRYS_SQL, crys_ids)}
        packets: list[ResonancePacketV1] = []
        per_read_scores: list[dict[str, Any]] = []
        built_at = datetime.now(timezone.utc)
        for ri_, rd in enumerate(reads):
            hubs = hubs_by_read[ri_]
            ri: ReadInput = rd["input"]
            scored_by_kind: dict[str, list[ScoredDoc]] = {}
            for i, (d, v) in enumerate(cen_pool):
                cos = dot(rd["q"], v)
                scored_by_kind.setdefault(d.ref_kind, []).append(ScoredDoc(d, cos, rank_score(rd["q"], v, hubs[i])))
            hydrated: list[ScoredDoc] = []
            excluded = {ref for ref, root in thread_root.items()
                        if ref.startswith(ri.seed_id + "#") or (root and root == ri.root_request_id)}
            # The read's own concepts land in the atlas: near their read by construction.
            own = {c.split(":", 1)[0].strip().casefold() for c in rd["own_concepts"]}
            excluded |= {d.ref_id for d, _ in cen_pool if d.ref_kind == "atlas_concept"
                         and d.excerpt.split(": ", 1)[0].strip().casefold() in own}
            for kind, lst in scored_by_kind.items():
                lst.sort(key=lambda s: -s.rank)
                for s in [x for x in lst if x.doc.ref_id not in excluded][:HYDRATE_PER_KIND]:
                    d = s.doc
                    if kind == "crystallization":
                        row = crys_rows.get(d.ref_id)
                        if row is None or row["status"] in ("rejected", "retired"):
                            continue
                        d = replace(d, excerpt=f"[{row['kind']}] {row['summary'] or row['subject'] or ''}".strip())
                    elif kind == "atlas_concept":
                        h = atlas_hydrate(fk, d.ref_id)
                        # A concept this read (or its thread) minted is near it by construction.
                        if h is None or ri.url in h[1] or (ri.root_request_id and ri.root_request_id in h[1]):
                            continue
                        d = h[0]
                    hydrated.append(ScoredDoc(d, s.centered_cos, s.rank))
            packet = build_packet(ri, hydrated, built_at=built_at, exclude_ref_ids=excluded)
            packets.append(packet)
            offered = {(c.ref_kind, c.ref_id) for c in packet.candidates}
            per_read_scores.append({
                "seed_id": ri.seed_id,
                # Over what the packet actually offers (the read's own concepts excluded).
                "best_centered": max((s.centered_cos for s in hydrated
                                      if (s.doc.ref_kind, s.doc.ref_id) in offered), default=0.0),
                "best_centered_prior": max((s.centered_cos for s in scored_by_kind.get("self_prior", [])),
                                           default=0.0),
                "best_raw_prior": max((dot(mean_vector(rd["qraw"]), v) for d, v in pool if d.ref_kind == "self_prior"),
                                      default=0.0),
            })
        packets = with_recent_counts(packets)

        # ---- 3. judge (only with --judge-url) ----
        results: list[dict[str, Any]] = []
        judged_dir = out / "judgments"
        judged_dir.mkdir(parents=True, exist_ok=True)
        paused: list[str] | None = None
        for rd, packet in zip(reads, packets):
            prompt = build_judge_prompt(packet, rd["input"].retained_texts)
            (out / "prompts" / f"{packet.packet_id}.txt").write_text(prompt)
            item: dict[str, Any] = {"packet": packet, "prompt": prompt, "judgment": None, "check": None,
                                    "judge_status": "UNVERIFIED: judge not run", "lease_sec": None}
            cache_file = judged_dir / f"{_sha(packet.packet_id + chr(0) + prompt)[:24]}.json"
            if args.judge_url and cache_file.exists():
                # Resume: a judgment already made for this exact prompt is re-checked, not re-asked.
                cached = json.loads(cache_file.read_text())
                j, chk = check_judgment(packet, cached["raw"], retained_texts=rd["input"].retained_texts, prompt=prompt)
                item.update(judgment=j, check=chk, raw=cached["raw"], judge_status="judged",
                            lease_sec=cached.get("lease_sec"))
            elif args.judge_url and paused is None:
                busy = await wait_for_quiet_pool(client, args.pool_url, args.pool_wait_sec)
                if busy:
                    # Stop asking; everything after this stays unjudged until a re-run resumes.
                    paused = busy
                    print(f"judge paused before item {len(results) + 1}/{len(packets)}: pool busy ({'; '.join(busy)})")
                else:
                    started = time.perf_counter()
                    try:
                        resp = await client.post(args.judge_url, timeout=args.judge_timeout, json={
                            "model": args.judge_model, "max_tokens": args.judge_max_tokens, "temperature": 0.2,
                            "messages": [{"role": "user", "content": prompt}],
                            "chat_template_kwargs": {"enable_thinking": False},
                        })
                        resp.raise_for_status()
                        raw = resp.json()["choices"][0]["message"]["content"]
                        lease_sec = round(time.perf_counter() - started, 1)
                        cache_file.write_text(json.dumps({"packet_id": packet.packet_id, "raw": raw,
                                                          "lease_sec": lease_sec}))
                        j, chk = check_judgment(packet, raw, retained_texts=rd["input"].retained_texts, prompt=prompt)
                        item.update(judgment=j, check=chk, raw=raw, judge_status="judged", lease_sec=lease_sec)
                        print(f"judged {len(results) + 1}/{len(packets)} in {lease_sec}s: "
                              f"{chk.decision if chk.accepted else 'REJECTED ' + ','.join(chk.reasons)}")
                    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                        item["judge_status"] = f"judge_error: {type(exc).__name__}"
            if args.judge_url and item["check"] is None and paused is not None and item["judge_status"].startswith("UNVERIFIED"):
                item["judge_status"] = "UNVERIFIED: judge paused (GPU pool busy); re-run to resume"
            results.append(item)
    # ---- outputs ----
    (out / "similarity.json").write_text(json.dumps(diag, indent=1, default=str))
    (out / "packets.jsonl").write_text("\n".join(p.model_dump_json() for p in packets) + "\n")
    write_labeling_set(args, reads, results, per_read_scores, diag)
    judged = [r for r in results if r["check"] is not None]
    if judged:
        acc = [r for r in judged if r["check"].accepted]
        print(f"\njudged={len(judged)} accepted={len(acc)} share={sum(r['check'].decision == 'share' for r in acc)} "
              f"note={sum(r['check'].decision == 'note' for r in acc)} rejected="
              f"{sum(not r['check'].accepted for r in judged)}")
    else:
        print("\njudge: UNVERIFIED (no --judge-url); prompts written to", out / "prompts")
    sizes = [len(p.candidates) for p in packets]
    print(f"packets={len(packets)} candidates/packet min={min(sizes, default=0)} max={max(sizes, default=0)} "
          f"empty_kinds_per_packet=" + ",".join(str(len(KIND_BUDGET) - len({c.ref_kind for c in p.candidates}))
                                                 for p in packets))
    print(f"wrote {args.labeling_md}, {args.labeling_csv} and {args.labeling_html}; packets+prompts under {out}")
    if args.judge_url and len(judged) < len(results):
        print(f"judge INCOMPLETE: {len(judged)}/{len(results)} judged; re-run the same command to resume "
              f"(cached judgments under {out / 'judgments'})")
        return 3
    return 0


def _flat(text: Any) -> str:
    """One line, no trailing whitespace: stored texts carry newlines and line-end spaces."""
    return " ".join(str(text or "").split())


def _source(url: str) -> str:
    return urlparse(url).hostname or url


def write_labeling_set(args: argparse.Namespace, reads: list[dict[str, Any]], results: list[dict[str, Any]],
                       scores: list[dict[str, Any]], diag: dict[str, Any]) -> None:
    rows = []
    for n, (rd, res, sc) in enumerate(zip(reads, results, scores), 1):
        p: ResonancePacketV1 = res["packet"]
        j, chk = res["judgment"], res["check"]
        if j is not None and chk is not None and not ("summary_sentence_count" in chk.reasons):
            summary, summary_src = j.article_summary, "judge"
        else:
            summary = fallback_article_summary(rd["input"].learned) or rd["input"].summary
            if len(summary) > 900:
                summary = summary[:899].rsplit(" ", 1)[0] + "\u2026"
            summary_src = "Stage 1 what_i_learned (first sentences), not judge-written"
        # Best piece of each kind, up to five kinds: the label then tests every store.
        best: dict[str, Any] = {}
        for c in p.candidates:
            if c.ref_kind not in best or c.score > best[c.ref_kind].score:
                best[c.ref_kind] = c
        top_pieces = sorted(best.values(), key=lambda c: -c.score)[:5]
        cited: dict[tuple[str, str], str] = {}
        read_quote, explore, judged_flag = "", "", False
        if j is not None and chk is not None:
            judged_flag = True
            cited = {(c.ref_kind, c.ref_id): c.how for c in j.citations}
            pieces = [c for c in p.candidates if (c.ref_kind, c.ref_id) in cited]
            # Recall: what retrieval offered that the judge did not cite, so a "the judge
            # missed this" label is possible on every item, judged or rejected.
            passed_over = [c for c in top_pieces if (c.ref_kind, c.ref_id) not in cited]
            connection = j.connection or "(none)"
            decision = chk.decision if chk.accepted else f"REJECTED ({', '.join(chk.reasons)}); judge said {j.decision}"
            conf = f"{j.confidence:.2f}"
            strain = strained_prior(j, chk) or ""
            read_quote, explore = j.read_quote, j.explore_next or ""
        else:
            pieces, passed_over = top_pieces, []
            connection = "(judge not run: UNVERIFIED) retrieval's top pieces below, NOT a judged connection"
            decision, conf, strain = res["judge_status"], "", ""

        def piece(c: Any) -> dict[str, Any]:
            return {"ref_kind": c.ref_kind, "ref_id": c.ref_id, "store": c.store, "text": c.excerpt,
                    "how": cited.get((c.ref_kind, c.ref_id), ""),
                    "neighborhood": [f"[{x.relation}] {x.text[:200]}" for x in c.neighborhood[:3]],
                    "offered_in_other_reads": c.recent_packet_count}
        rows.append({
            "item": n, "seed_id": p.seed_id, "title": p.read_title or "(untitled)", "source": _source(p.read_url),
            "url": p.read_url, "article_summary": summary, "summary_source": summary_src,
            "proposed_connection": connection, "decision": decision, "judge_confidence": conf,
            "would_retest_prior": strain, "candidates_in_packet": len(p.candidates),
            "best_centered_cos": f"{sc['best_centered']:.3f}", "best_raw_prior_cos": f"{sc['best_raw_prior']:.3f}",
            "best_centered_prior_cos": f"{sc['best_centered_prior']:.3f}",
            "pieces": [piece(c) for c in pieces], "passed_over": [piece(c) for c in passed_over],
            "judged": judged_flag, "accepted": bool(chk is not None and chk.accepted),
            "read_quote": read_quote, "explore_next": explore,
            "wanted": "", "notes": "",
        })
    csv_path = Path(args.labeling_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["item", "seed_id", "title", "source", "url", "article_summary", "summary_source",
                    "proposed_connection", "decision", "judge_confidence", "would_retest_prior",
                    "best_centered_cos", "best_raw_prior_cos", "best_centered_prior_cos", "cited_pieces",
                    "offered_not_cited", "wanted", "notes"])
        for r in rows:
            r = {k: (_flat(v) if isinstance(v, str) else v) for k, v in r.items()}
            w.writerow([r["item"], r["seed_id"], r["title"], r["source"], r["url"], r["article_summary"],
                        r["summary_source"], r["proposed_connection"], r["decision"], r["judge_confidence"],
                        r["would_retest_prior"], r["best_centered_cos"], r["best_raw_prior_cos"], r["best_centered_prior_cos"],
                        " || ".join(_flat(f"{x['ref_kind']} {x['ref_id']} @ {x['store']}: {x['text']}") for x in r["pieces"]),
                        " || ".join(_flat(f"{x['ref_kind']} {x['ref_id']} @ {x['store']}: {x['text']}")
                                    for x in r["passed_over"]),
                        "", ""])
    md = [f"# Reading resonance labeling set ({datetime.now(timezone.utc):%Y-%m-%d})", "",
          "Generated by `services/orion-hub/evals/run_reading_resonance_eval.py` (read-only replay of every read "
          f"whose Stage 2 finished in the last {args.days} days). Fill in **wanted** (yes/no) and **notes** per "
          "item, here or in the CSV next to this file.", "",
          "How to read an item: *What the article says* is the summary you would get in the message. *Pieces* are "
          "the things Orion holds that the item cites, each with its stored text and where it lives. "
          "When the judge did not run, the pieces are retrieval's best piece of each kind (up to five), "
          "which is a recall net and is mostly adjacency: a `no` there is useful information too, and a "
          "`yes` with a note naming WHICH piece tells us what the judge should have found.", ""]
    b = diag.get("belief_sets_pooled", {})
    if b:
        md += [f"Similarity context: on pairs that are related by construction (a belief and its own finding, a "
               f"question and its own answer), raw cosine averages {b['raw']['pos_mean']:.2f} and unrelated pairs "
               f"{b['raw']['neg_mean']:.2f}. After centering, {b['centered']['pos_mean']:.2f} vs "
               f"{b['centered']['neg_mean']:.2f}. The `best_*` numbers below are there for audit only.", ""]
    for r in rows:
        md += [f"## {r['item']}. {r['title']}", "",
               f"- source: {r['source']} — {r['url']}",
               f"- decision: {r['decision']}" + (f" (confidence {r['judge_confidence']})" if r["judge_confidence"] else ""),
               f"- would re-test prior: {r['would_retest_prior'] or '-'}",
               f"- best centered cosine {r['best_centered_cos']}; nearest belief: raw {r['best_raw_prior_cos']}, "
               f"centered {r['best_centered_prior_cos']} (offer floor {CENTERED_OFFER_FLOOR}); "
               f"{r['candidates_in_packet']} pieces in the packet", "",
               f"**What the article says** ({r['summary_source']}): {_flat(r['article_summary'])}", "",
               f"**Proposed connection:** {_flat(r['proposed_connection'])}", "", "**Pieces:**", ""]
        if r["read_quote"]:
            md += [f"**Quote from the read:** \"{_flat(r['read_quote'])}\"", ""]
        if not r["pieces"]:
            md.append("- (none cited)" if r["judged"] else "- (none: nothing cleared the offer floor)")

        def md_piece(x: dict[str, Any]) -> None:
            md.append(f"- `{x['ref_kind']}` `{x['ref_id']}` ({x['store']}"
                      + (f"; offered for {x['offered_in_other_reads']} other reads" if x["offered_in_other_reads"] else "")
                      + f"): {_flat(x['text'])}")
            if x.get("how"):
                md.append(f"  - judge: {_flat(x['how'])}")
            for nb in x["neighborhood"]:
                md.append(f"  - {_flat(nb)}")

        for x in r["pieces"]:
            md_piece(x)
        if r["passed_over"]:
            md += ["", "**Offered but not cited** (did the judge miss one?):", ""]
            for x in r["passed_over"]:
                md_piece(x)
        if r["explore_next"]:
            md += ["", f"**Explore next:** {_flat(r['explore_next'])}"]
        md += ["", "**wanted:** ___ (yes/no)", "", "**notes:** ___", ""]
    Path(args.labeling_md).write_text("\n".join(md) + "\n")
    Path(args.labeling_html).write_text(labeling_html(rows, generated=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
                                                      days=args.days))


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Reading resonance labeling set</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:920px;margin:0 auto;padding:16px;background:#f6f5f2;color:#222}
header{position:sticky;top:0;background:#f6f5f2;padding:8px 0;border-bottom:1px solid #ccc;z-index:2}
button{font:inherit;padding:6px 14px;cursor:pointer}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin:16px 0}
.card.done{border-left:6px solid #4a8}
.meta{color:#666;font-size:13px}
.decision{display:inline-block;padding:1px 8px;border-radius:10px;font-size:13px;background:#eee}
.decision.share{background:#cfe9d6}.decision.note{background:#e2e6f5}.decision.rej{background:#f6d4d0}
.piece{border-left:3px solid #bbb;padding:2px 10px;margin:8px 0;font-size:14px}
.piece .tag{font-family:monospace;font-size:12px;color:#555}
.how{color:#335;font-style:italic}
.nb{color:#666;font-size:13px;margin-left:12px}
.passed{opacity:.8}
textarea{width:100%;min-height:52px;font:inherit}
label{margin-right:14px}
h2{font-size:17px;margin:0 0 4px}
</style></head><body>
<header><b>Reading resonance labeling set</b> &middot; generated __GENERATED__ &middot; reads from the last __DAYS__ days
&middot; <span id="progress"></span>
<button id="export">Export my labels (CSV)</button></header>
<p>For each item: is this a connection you would have <b>wanted</b> Orion to tell you about? Mark yes or no and add a
note (which piece mattered, what the judge missed). Labels are kept in this browser (localStorage) until you export;
nothing is sent anywhere.</p>
<div id="cards"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const rows = JSON.parse(document.getElementById("data").textContent);
const KEY = "orion-resonance-labels-__KEY__";
const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; return e; }
function pieceEl(x, extra) {
  const d = el("div", "piece" + (extra ? " " + extra : ""));
  d.appendChild(el("div", "tag", x.ref_kind + " · " + x.ref_id + " · " + x.store +
    (x.offered_in_other_reads ? " · offered for " + x.offered_in_other_reads + " other reads" : "")));
  d.appendChild(el("div", "", x.text));
  if (x.how) d.appendChild(el("div", "how", "judge: " + x.how));
  (x.neighborhood || []).forEach(n => d.appendChild(el("div", "nb", n)));
  return d;
}
function progress() {
  const n = rows.filter(r => (saved[r.seed_id] || {}).wanted).length;
  document.getElementById("progress").textContent = n + " / " + rows.length + " labelled";
}
const root = document.getElementById("cards");
rows.forEach(r => {
  const s = saved[r.seed_id] || (saved[r.seed_id] = {wanted: "", notes: ""});
  const c = el("div", "card" + (s.wanted ? " done" : ""));
  c.appendChild(el("h2", "", r.item + ". " + r.title));
  const m = el("div", "meta"); const a = el("a", "", r.source); a.href = r.url; a.rel = "noreferrer";
  a.target = "_blank"; m.appendChild(a); c.appendChild(m);
  const dc = r.accepted ? r.decision : (r.judged ? "rej" : "");
  const dl = el("div", "meta"); const badge = el("span", "decision " + dc, r.decision);
  dl.appendChild(badge);
  if (r.judge_confidence) dl.appendChild(document.createTextNode(" confidence " + r.judge_confidence));
  if (r.would_retest_prior) dl.appendChild(document.createTextNode(" · would re-test prior " + r.would_retest_prior));
  c.appendChild(dl);
  c.appendChild(el("p", "", "")).appendChild(el("b", "", "What the article says "));
  c.lastChild.appendChild(el("span", "meta", "(" + r.summary_source + "): "));
  c.lastChild.appendChild(document.createTextNode(r.article_summary));
  if (r.read_quote) { const q = el("p"); q.appendChild(el("b", "", "Quote from the read: "));
    q.appendChild(document.createTextNode("“" + r.read_quote + "”")); c.appendChild(q); }
  const pc = el("p"); pc.appendChild(el("b", "", "Proposed connection: "));
  pc.appendChild(document.createTextNode(r.proposed_connection)); c.appendChild(pc);
  c.appendChild(el("b", "", r.judged ? "Cited pieces" : "Retrieval's top pieces (judge not run)"));
  if (!r.pieces.length) c.appendChild(el("div", "meta", r.judged ? "(none cited)" : "(nothing cleared the offer floor)"));
  r.pieces.forEach(x => c.appendChild(pieceEl(x)));
  if (r.passed_over.length) { c.appendChild(el("b", "", "Offered but not cited (did the judge miss one?)"));
    r.passed_over.forEach(x => c.appendChild(pieceEl(x, "passed"))); }
  if (r.explore_next) { const e = el("p"); e.appendChild(el("b", "", "Explore next: "));
    e.appendChild(document.createTextNode(r.explore_next)); c.appendChild(e); }
  const f = el("div"); f.appendChild(el("b", "", "Wanted? "));
  ["yes", "no"].forEach(v => { const l = el("label"); const i = el("input"); i.type = "radio";
    i.name = "w-" + r.item; i.value = v; i.checked = s.wanted === v;
    i.addEventListener("change", () => { s.wanted = v; c.classList.add("done"); store(); });
    l.appendChild(i); l.appendChild(document.createTextNode(" " + v)); f.appendChild(l); });
  const t = el("textarea"); t.placeholder = "notes"; t.value = s.notes;
  t.addEventListener("input", () => { s.notes = t.value; store(); });
  f.appendChild(t); c.appendChild(f); root.appendChild(c);
});
function store() { localStorage.setItem(KEY, JSON.stringify(saved)); progress(); }
function cell(v) { v = String(v == null ? "" : v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
document.getElementById("export").addEventListener("click", () => {
  const head = ["item", "seed_id", "title", "url", "decision", "judge_confidence", "proposed_connection", "wanted", "notes"];
  const lines = [head.join(",")].concat(rows.map(r => { const s = saved[r.seed_id] || {};
    return [r.item, r.seed_id, r.title, r.url, r.decision, r.judge_confidence, r.proposed_connection,
            s.wanted || "", s.notes || ""].map(cell).join(","); }));
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "resonance-labels-__KEY__.csv"; document.body.appendChild(a); a.click(); a.remove();
});
progress();
</script></body></html>
"""


def labeling_html(rows: list[dict[str, Any]], *, generated: str, days: int) -> str:
    """One self-contained page: cards, a wanted yes/no + notes per item, and an export
    button that downloads a CSV. No network: inline CSS/JS, a CSP of default-src 'none',
    data embedded as JSON (every ``<`` escaped so text can never close the script tag)."""
    keep = ("item", "seed_id", "title", "source", "url", "article_summary", "summary_source",
            "proposed_connection", "decision", "judge_confidence", "would_retest_prior", "pieces",
            "passed_over", "judged", "accepted", "read_quote", "explore_next")
    data = json.dumps([{k: r.get(k) for k in keep} for r in rows], ensure_ascii=False)
    data = data.replace("<", "\\u003c")
    key = _sha("|".join(str(r.get("seed_id")) for r in rows))[:10]
    return (_HTML_TEMPLATE.replace("__GENERATED__", generated).replace("__DAYS__", str(int(days)))
            .replace("__KEY__", key).replace("__DATA__", data))


def _utc(text: str) -> datetime:
    t = datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--until", type=_utc, default=None,
                    help="window end (ISO time, default now). Pin it when resuming a judge run so a new "
                         "read cannot change the packets (and the cached judgments) between runs")
    ap.add_argument("--out-dir", default="/tmp/reading-resonance-eval")
    ap.add_argument("--labeling-md", default="docs/superpowers/evals/2026-10-13-resonance-labeling-set.md")
    ap.add_argument("--labeling-csv", default="docs/superpowers/evals/2026-10-13-resonance-labeling-set.csv")
    ap.add_argument("--labeling-html", default="docs/superpowers/evals/2026-10-13-resonance-labeling-set.html")
    ap.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap.add_argument("--falkor", default=os.environ.get("FALKORDB_URI", "redis://127.0.0.1:6380"))
    ap.add_argument("--chroma-url", default=os.environ.get("CHROMA_URL", "http://127.0.0.1:8500"))
    ap.add_argument("--embed-url", default=os.environ.get(
        "HUB_READING_SEARCH_EMBED_URL", "http://127.0.0.1:8320/embedding"))
    ap.add_argument("--judge-url", default="", help="OpenAI-style chat completions URL; empty = judge not run")
    ap.add_argument("--judge-model", default="agent")
    ap.add_argument("--judge-max-tokens", type=int, default=900)
    ap.add_argument("--judge-timeout", type=float, default=600.0)
    ap.add_argument("--pool-url", default=os.environ.get("GPU_POOL_URL", "http://127.0.0.1:8127"),
                    help="orion-gpu-pool HTTP base; GET /v1/pool is read before every judge call")
    ap.add_argument("--pool-wait-sec", type=float, default=30.0,
                    help="how long to wait for the pool to go quiet before pausing the judge")
    args = ap.parse_args()
    args.until = args.until or datetime.now(timezone.utc)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
