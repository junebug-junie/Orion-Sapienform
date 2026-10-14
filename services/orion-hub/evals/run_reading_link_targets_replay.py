#!/usr/bin/env python3
"""Live replay, read-only: how many link targets does Stage 2 offer each read, and how many
links could it make, with the raw-cosine meaning search (live) vs the centered shortlist?

    PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_link_targets_replay.py \
        [--days 14] [--out /tmp/reading-link-targets] \
        [--llm-url http://127.0.0.1:8210/v1/chat/completions --llm-model agent]

Reads (never writes): Postgres world_pulse_read_seed + reading_document_snapshot (SELECT
inside a READ ONLY transaction), Falkor through FalkorSubstrateStore reads, Chroma
``orion_atlas_concepts`` through /get (the vectors Hub's index already holds), vector-host
/embedding for a subject the index lacks (persists nothing). gpu-pool /v1/pool before each
LLM call.

Per read whose Stage 2 finished in the last --days days:

- texts: retained fetch text. No text = no claim block in the Stage 2 prompt at all
  (``ClaimContextV1.usable``), whatever the target list.
- before: exact-label objects only (assertions.object_candidates).
- live: exact + raw bge cosine >= 0.65, the code deployed 2026-10-13 (retired by this patch).
- after: exact + centered cosine >= OBJECT_MIN_CENTERED_SIMILARITY (concept_search, this patch).
- acceptable: offered objects whose stored label is in the retained text. Under
  reading_quote_rule_v1 a quote must be verbatim in the retained text AND contain the
  object's stored label, so this is a hard upper bound on accepted links.

Calibration (the floor's data): centered cosine between two concepts minted by the same
read (related by construction) vs a concept from a different read's evidence group.

With --llm-url, each usable read gets a claims-only prompt (the real
``claim_prompt_section``) with the live and the after object lists, and the model's claims
go through the real ``plan_claims`` (no journal). Stops when the pool is busy.
Exit 0 on completion; 2 when a store is unavailable; 3 when the LLM pass paused.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadRelationshipClaimV1
from orion.substrate.reconcile import is_identity_fenced
from orion.world_pulse_read.assertions import (
    MAX_OBJECTS,
    resolve_reading_subjects,
    MIN_LABEL_CHARS,
    EndpointV1,
    build_claim_context,
    claim_prompt_section,
    plan_claims,
)
from orion.world_pulse_read.concept_search import (
    ATLAS_CONCEPT_COLLECTION,
    OBJECT_MIN_CENTERED_SIMILARITY,
    centered_matrix,
    concept_doc,
)
from orion.world_pulse_read.fetch_text import load_retained_texts

# The raw-cosine floor concept_search used until 2026-10-14 (replaced by the centered one).
RETIRED_RAW_FLOOR = 0.65

SEEDS_SQL = """
SELECT seed_id, url, stage2_completed_at, handoff_json FROM world_pulse_read_seed
WHERE stage2_status = 'done' AND handoff_json IS NOT NULL
  AND stage2_completed_at > $2::timestamptz - make_interval(days => $1)
  AND stage2_completed_at <= $2::timestamptz
ORDER BY stage2_completed_at
"""

_POOL = None


def _pool_mod() -> Any:
    """The resonance eval's pool-quiet helpers (same rule, one implementation)."""
    global _POOL
    if _POOL is None:
        spec = importlib.util.spec_from_file_location(
            "run_reading_resonance_eval", Path(__file__).with_name("run_reading_resonance_eval.py"))
        _POOL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_POOL)  # type: ignore[union-attr]
    return _POOL


async def chroma_vectors(client: httpx.AsyncClient, base: str, name: str) -> dict[str, list[float]]:
    coll = (await client.get(f"{base}/api/v1/collections/{name}")).json()
    out: dict[str, list[float]] = {}
    offset = 0
    while True:
        body = (await client.post(f"{base}/api/v1/collections/{coll['id']}/get",
                                  json={"limit": 500, "offset": offset, "include": ["embeddings"]})).json()
        ids = body.get("ids") or []
        for i, v in zip(ids, body.get("embeddings") or []):
            out[str(i)] = v
        if len(ids) < 500:
            return out
        offset += 500


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    a = np.sort(np.asarray(xs))
    return {f"p{int(q * 100)}": round(float(a[min(len(a) - 1, int(len(a) * q))]), 3)
            for q in (0.1, 0.5, 0.9, 0.95, 0.99)} | {"n": len(a), "mean": round(float(a.mean()), 3)}


def claims_prompt(handoff: WorldPulseReadHandoffV1, ctx: Any) -> str:
    """Stage 2's claim block on a minimal frame: the read's own words plus the real section."""
    return ("You finished reading a source. What you learned:\n" + handoff.what_i_learned[:3000]
            + "\nReturn ONE JSON object with only the key \"relationship_claims\" (a list, possibly empty), "
            "no prose outside a fenced JSON block." + claim_prompt_section(ctx))


def parse_claims(raw: str) -> list[WorldPulseReadRelationshipClaimV1]:
    start = raw.find("{")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except ValueError:
        return []
    out = []
    for c in (obj.get("relationship_claims") or []) if isinstance(obj, dict) else []:
        try:
            out.append(WorldPulseReadRelationshipClaimV1.model_validate(c))
        except ValueError:
            continue
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--until", default=None, help="window end, ISO time (default now)")
    ap.add_argument("--out", default="/tmp/reading-link-targets")
    ap.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap.add_argument("--falkor", default=os.environ.get("FALKORDB_URI", "redis://127.0.0.1:6380"))
    ap.add_argument("--chroma-url", default=os.environ.get("HUB_READING_SEARCH_CHROMA_URL", "http://127.0.0.1:8500"))
    ap.add_argument("--embed-url", default=os.environ.get(
        "HUB_READING_SEARCH_EMBED_URL", "http://127.0.0.1:8320/embedding"))
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--llm-model", default="agent")
    ap.add_argument("--llm-max-tokens", type=int, default=1200)
    ap.add_argument("--llm-role", default="agent", help="gpu-pool role the LLM route lands on (pool gate)")
    ap.add_argument("--pool-url", default=os.environ.get("GPU_POOL_URL", "http://127.0.0.1:8127"))
    args = ap.parse_args()
    until = datetime.fromisoformat(args.until) if args.until else datetime.now(timezone.utc)
    args.until = until if until.tzinfo else until.replace(tzinfo=timezone.utc)
    out = Path(args.out)
    (out / "llm").mkdir(parents=True, exist_ok=True)

    import asyncpg

    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

    try:
        conn = await asyncpg.connect(args.dsn)
        store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=args.falkor, ensure_indexes=False))
    except Exception as exc:  # noqa: BLE001
        print(f"UNKNOWN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    random.seed(7)
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "reads": []}
    paused = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            vecs = await chroma_vectors(client, args.chroma_url.rstrip("/"), ATLAS_CONCEPT_COLLECTION)
            nodes = {n.node_id: n for n in store.read_concept_region(limit_nodes=5000, limit_edges=1).nodes
                     if not is_identity_fenced(n)}
            ids = [i for i in vecs if i in nodes]
            raw = np.asarray([vecs[i] for i in ids], dtype=np.float64)
            raw /= np.linalg.norm(raw, axis=1, keepdims=True)
            mu, cen = centered_matrix(raw)
            index = {i: k for k, i in enumerate(ids)}
            print(f"atlas concepts indexed={len(vecs)} live+indexed={len(ids)}")

            # ---- calibration: concepts minted by the same read vs a different read ----
            groups: dict[str, list[int]] = {}
            for i in ids:
                ev = [e for e in (nodes[i].provenance.evidence_refs or []) if str(e).startswith("http")]
                if i.startswith("sub-concept-wp-read-") and ev:
                    groups.setdefault(ev[0], []).append(index[i])
            keys = [k for k, v in groups.items() if len(v) >= 2]
            pos_raw, pos_c, neg_raw, neg_c = [], [], [], []
            for k in keys:
                g = groups[k]
                others = [x for kk in keys if kk != k for x in groups[kk]]
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        pos_raw.append(float(raw[g[a]] @ raw[g[b]]))
                        pos_c.append(float(cen[g[a]] @ cen[g[b]]))
                        o = random.choice(others)
                        neg_raw.append(float(raw[g[a]] @ raw[o]))
                        neg_c.append(float(cen[g[a]] @ cen[o]))
            calib = {"groups": len(keys), "related_raw": quantiles(pos_raw), "unrelated_raw": quantiles(neg_raw),
                     "related_centered": quantiles(pos_c), "unrelated_centered": quantiles(neg_c)}
            report["calibration"] = calib
            print("calibration (same-read concept pairs vs cross-read):", json.dumps(calib))

            url_of = {}
            for k, members in groups.items():
                for m in members:
                    url_of[ids[m]] = k
            read_minted = np.asarray([i in url_of for i in ids])

            async def subject_vec(node_id: str) -> np.ndarray:
                if node_id in index:
                    return raw[index[node_id]]
                node = store.get_node_by_id(node_id)
                resp = await client.post(args.embed_url, json={"doc_id": "replay-probe",
                                                               "text": concept_doc(node) or node_id})
                resp.raise_for_status()
                v = np.asarray(resp.json()["embedding"], dtype=np.float64)
                return v / np.linalg.norm(v)

            def offerable(i: str, taken: set[str]) -> bool:
                n = nodes.get(i)
                return (i not in taken and n is not None and n.node_kind in ("concept", "entity")
                        and len(str(n.label or "").strip()) >= MIN_LABEL_CHARS)

            totals = {k: 0 for k in ("reads", "with_text", "usable_before", "usable_live", "usable_after",
                                     "offered_before", "offered_live", "offered_after",
                                     "acceptable_before", "acceptable_live", "acceptable_after")}
            null_pool: list[float] = []
            async with conn.transaction(readonly=True):
                rows = await conn.fetch(SEEDS_SQL, args.days, args.until)
                for row in rows:
                    handoff = WorldPulseReadHandoffV1.model_validate(json.loads(row["handoff_json"]))
                    texts = await load_retained_texts(conn, handoff.read_evidence)
                    ctx = build_claim_context(store, handoff, texts)
                    retained = "\n".join(t.text for t in texts).lower()
                    totals["reads"] += 1
                    totals["with_text"] += bool(texts)
                    exact = list(ctx.objects)
                    live: list[tuple[str, float]] = []
                    after: list[tuple[str, float]] = []
                    if ctx.subjects and texts:
                        q = np.stack([await subject_vec(s.node_id) for s in ctx.subjects])
                        qc = q - mu
                        qc /= np.maximum(np.linalg.norm(qc, axis=1, keepdims=True), 1e-12)
                        raw_best = (q @ raw.T).max(axis=0)
                        cen_best = (qc @ cen.T).max(axis=0)
                        taken = {s.node_id for s in ctx.subjects} | {o.node_id for o in exact}
                        room = MAX_OBJECTS - len(exact)
                        for scores, floor, dest in ((raw_best, RETIRED_RAW_FLOOR, live),
                                                    (cen_best, OBJECT_MIN_CENTERED_SIMILARITY, after)):
                            seen = set(taken)
                            for k in np.argsort(-scores):
                                if scores[k] < floor or len(dest) >= room:
                                    break
                                if offerable(ids[k], seen):
                                    dest.append((ids[k], float(scores[k])))
                                    seen.add(ids[k])
                    # Null for the statistic Hub thresholds (best over ALL this read's subjects):
                    # concepts minted by OTHER reads (a different source URL). Impure: sibling
                    # reads on one topic (the CoreWeave pages) count as "other" here too.
                    null_subjects = resolve_reading_subjects(store, handoff)
                    if null_subjects:
                        q0 = np.stack([await subject_vec(e.node_id) for e in null_subjects])
                        q0c = q0 - mu
                        q0c /= np.maximum(np.linalg.norm(q0c, axis=1, keepdims=True), 1e-12)
                        own = {url_of.get(e.node_id) for e in null_subjects} | {row["url"]}
                        other = read_minted & np.asarray([url_of.get(i) not in own for i in ids])
                        best0 = (q0c @ cen[other].T).max(axis=0) if other.any() else np.zeros(0)
                        null_rec = {"subjects": len(null_subjects), "cross_read_concepts": int(other.sum()),
                                    "clear_floor": int((best0 >= OBJECT_MIN_CENTERED_SIMILARITY).sum()),
                                    "best_p50": round(float(np.median(best0)), 3) if best0.size else None,
                                    "best_p95": round(float(np.quantile(best0, 0.95)), 3) if best0.size else None}
                        null_pool.extend(float(x) for x in best0)
                    else:
                        null_rec = {"subjects": 0}
                    acc = lambda objs: sum(1 for i in objs if str(nodes[i].label).lower() in retained) if objs else 0  # noqa: E731
                    a_b = sum(1 for o in exact if o.label.lower() in retained)
                    rec = {"seed_id": row["seed_id"], "url": row["url"],
                           "completed": row["stage2_completed_at"].isoformat(),
                           "retained_chars": [len(t.text) for t in texts],
                           "subjects": [s.label for s in ctx.subjects],
                           "before": [o.label for o in exact],
                           "live": [(nodes[i].label, round(s, 3)) for i, s in live],
                           "after": [(nodes[i].label, round(s, 3)) for i, s in after],
                           "null_best_over_subjects": null_rec,
                           "acceptable": {"before": a_b, "live": a_b + acc([i for i, _ in live]),
                                          "after": a_b + acc([i for i, _ in after])}}
                    usable = bool(ctx.subjects and texts)
                    for tag, n_obj in (("before", len(exact)), ("live", len(exact) + len(live)),
                                       ("after", len(exact) + len(after))):
                        totals[f"usable_{tag}"] += bool(usable and n_obj)
                        totals[f"offered_{tag}"] += n_obj if usable else 0
                        totals[f"acceptable_{tag}"] += rec["acceptable"][tag]
                    # ---- optional: the model's actual claims, through the real plan_claims ----
                    if args.llm_url and usable:
                        rec["llm"] = {}
                        for tag, extra in (("live", live), ("after", after)):
                            objs = tuple(exact) + tuple(EndpointV1(node_id=i, kind=nodes[i].node_kind,
                                                                   label=str(nodes[i].label), via="meaning")
                                                        for i, _ in extra)
                            c2 = replace(ctx, objects=objs[:MAX_OBJECTS])
                            if not c2.usable:
                                continue
                            prompt = claims_prompt(handoff, c2)
                            # Keyed on what was asked: a changed object list (floor, atlas) is a new prompt.
                            key = hashlib.sha256(f"{args.llm_model}|{args.llm_max_tokens}|{prompt}".encode()).hexdigest()[:24]
                            cache = out / "llm" / f"{tag}-{key}.json"
                            if cache.exists():
                                raw_out = json.loads(cache.read_text())["raw"]
                            elif paused is None:
                                busy = await _pool_mod().wait_for_quiet_pool(client, args.pool_url, 30.0, args.llm_role)
                                if busy:
                                    paused = busy
                                    print("LLM paused: pool busy:", "; ".join(busy))
                                    continue
                                started = time.perf_counter()
                                resp = await client.post(args.llm_url, timeout=600.0, json={
                                    "model": args.llm_model, "max_tokens": args.llm_max_tokens, "temperature": 0.2,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "chat_template_kwargs": {"enable_thinking": False}})
                                resp.raise_for_status()
                                raw_out = resp.json()["choices"][0]["message"]["content"]
                                cache.write_text(json.dumps({"raw": raw_out,
                                                             "sec": round(time.perf_counter() - started, 1)}))
                            else:
                                continue
                            plans = plan_claims(parse_claims(raw_out), c2, seed_id=row["seed_id"],
                                                recorded_at=datetime.now(timezone.utc))
                            reasons: dict[str, int] = {}
                            for p in plans:
                                reasons[p.claim.receipt.reason] = reasons.get(p.claim.receipt.reason, 0) + 1
                            via = {o.node_id: o.via for o in c2.objects}
                            sub_label = {e.node_id: e.label.lower() for e in c2.subjects}
                            # Counterfactual only (NOT a rule in force): a meaning-object claim whose
                            # verbatim quote names the SUBJECT, the narrowest change on the table.
                            subject_named = [p for p in plans if p.claim.receipt.reason == "quote_not_about_endpoints"
                                             and via.get(p.claim.object_id) == "meaning"
                                             and sub_label.get(p.claim.subject_id, "\0") in p.claim.quote.lower()]
                            rec["llm"][tag] = {
                                "cf_meaning_quote_names_subject": len(subject_named),
                                "cf_meaning_quote_names_subject_associated_with": sum(
                                    p.claim.predicate == "associated_with" for p in subject_named),
                                "claims": len(plans), "reasons": reasons,
                                "accepted": sum(p.decision is not None for p in plans),
                                "to_meaning_objects": sum(via.get(p.claim.object_id) == "meaning" for p in plans),
                                "examples": [{"object": p.claim.object_id, "predicate": p.claim.predicate,
                                              "quote": p.claim.quote[:200], "reason": p.claim.receipt.reason}
                                             for p in plans[:6]]}
                    report["reads"].append(rec)
                    print(f"{row['seed_id'][:58]:58} text={rec['retained_chars']} subj={len(ctx.subjects)} "
                          f"before={len(exact)} live={len(exact) + len(live)} after={len(exact) + len(after)} "
                          f"acceptable={rec['acceptable']} null={null_rec}" + (f" llm={ {k: (v['claims'], v['accepted'], v['reasons']) for k, v in rec.get('llm', {}).items()} }" if rec.get("llm") else ""))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            print(f"UNKNOWN: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        finally:
            await conn.close()
    report["totals"] = totals
    report["null_best_over_subjects_pooled"] = quantiles(null_pool) | {
        "share_clearing_floor": round(sum(x >= OBJECT_MIN_CENTERED_SIMILARITY for x in null_pool)
                                      / max(1, len(null_pool)), 3)}
    print("null (best over subjects vs other reads' concepts):", json.dumps(report["null_best_over_subjects_pooled"]))
    if args.llm_url:
        agg: dict[str, dict[str, int]] = {}
        for r in report["reads"]:
            for tag, v in (r.get("llm") or {}).items():
                a = agg.setdefault(tag, {"reads": 0, "claims": 0, "accepted": 0, "to_meaning_objects": 0,
                                         "cf_meaning_quote_names_subject": 0,
                                         "cf_meaning_quote_names_subject_associated_with": 0})
                a["reads"] += 1
                for k in ("claims", "accepted", "to_meaning_objects", "cf_meaning_quote_names_subject",
                          "cf_meaning_quote_names_subject_associated_with"):
                    a[k] += v[k]
                for reason, n in v["reasons"].items():
                    a[f"reason:{reason}"] = a.get(f"reason:{reason}", 0) + n
        report["llm_totals"] = agg
        print("llm totals", json.dumps(agg))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("totals " + " ".join(f"{k}={v}" for k, v in totals.items()))
    print(f"wrote {out / 'report.json'}")
    return 3 if paused else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
