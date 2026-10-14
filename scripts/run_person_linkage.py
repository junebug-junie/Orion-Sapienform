#!/usr/bin/env python3
"""Person linkage DRY RUN: the one-time cleanup proposal (spec section 5, rev 3).

READ ONLY. Postgres inside a read-only session, FalkorDB through
``GRAPH.RO_QUERY`` only, vector-host ``/embedding`` (persists nothing). Writes
files only:

    /tmp/person-linkage/<run>/progress.log
    /tmp/person-linkage/<run>/decisions.csv      (stable: a rerun is byte-identical)
    /tmp/person-linkage/<run>/decisions.jsonl
    /tmp/person-linkage/<run>/merge_plan.jsonl   (what applying would write, with before-values)
    --html <path>                                (the review file for the interlocutor)

Needs the extractor stack (torch, gliner) plus rapidfuzz and jellyfish:
``pip install -r orion/person_model/requirements-extract.txt``. There is no
apply mode: applying decisions is a later, separate step after review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from orion.person_model import linkage as L  # noqa: E402
from orion.person_model.extract import PERSON_LABEL  # noqa: E402
from orion.person_model.merge_log import build_entry  # noqa: E402
from orion.person_model.normalize import fold, prompt_context_keys  # noqa: E402
from orion.person_model.person import interlocutor, load_people  # noqa: E402
from orion.person_model.review_html import render  # noqa: E402

GRAPH = "orion_substrate"


class Progress:
    def __init__(self, path: Path):
        self.path, self.t0 = path, time.time()
        path.write_text("", encoding="utf-8")

    def __call__(self, title: str, done: int = 0, total: int = 0, errors: int = 0, note: str = "") -> None:
        el = time.time() - self.t0
        pct = (100.0 * done / total) if total else 100.0
        rate = done / el if el > 0 else 0.0
        eta = ((total - done) / rate) if rate > 0 and total else 0.0
        line = f"{datetime.now(timezone.utc).isoformat()} {title} {pct:.0f}% {done}/{total} rate={rate:.1f}/s eta={eta:.0f}s errors={errors} {note}".rstrip()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


# ---------------------------------------------------------------- read-only IO
class Graph:
    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def ro(self, cypher: str, params: Optional[dict] = None) -> List[list]:
        if params:
            prefix = "CYPHER " + " ".join(f"{k}={json.dumps(v)}" for k, v in params.items()) + " "
            cypher = prefix + cypher
        res = self.r.execute_command("GRAPH.RO_QUERY", GRAPH, cypher)
        return res[1] if len(res) > 1 else []


def pg_connect(dsn: str):
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True, autocommit=False)
    return conn


def pg_all(conn, sql: str, args: tuple = ()) -> List[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


class ReadOnlyStore:
    """merge_log.Store over the live stores, reads only. Writes raise."""

    def __init__(self, graph: Graph, conn):
        self.g, self.conn = graph, conn

    def get_prop(self, node_id, prop):
        rows = self.g.ro(f"MATCH (n {{node_id: $id}}) RETURN n.{prop}", {"id": node_id})
        return rows[0][0] if rows else None

    def has_edge(self, src, dst, rel):
        rows = self.g.ro(f"MATCH (a {{node_id: $s}})-[r:{rel}]->(b {{node_id: $d}}) RETURN count(r)", {"s": src, "d": dst})
        return bool(rows and rows[0][0])

    def get_alias(self, node_id, alias_norm):
        rows = pg_all(self.conn, "select alias_text, alias_class, referent_kind, promotion_state, admitted_by, grounded_in, "
                      "valid_until::text from referent_alias where node_id=%s and alias_norm=%s", (node_id, alias_norm))
        if not rows:
            return None
        keys = ("alias_text", "alias_class", "referent_kind", "promotion_state", "admitted_by", "grounded_in", "valid_until")
        return dict(zip(keys, rows[0]))

    def set_prop(self, *a):
        raise RuntimeError("dry run: read only")

    add_edge = remove_edge = put_alias = set_prop


# ---------------------------------------------------------------- embeddings
class CenteredEmbedder:
    """Mean-centered cosine over vector-host bge vectors (PR #2647: raw cosine
    rests near 0.6 for unrelated text). The mean is taken over the whole label
    population of the graph, not just the pairs being scored."""

    def __init__(self, url: str, cache_path: Path):
        self.url, self.cache_path = url.rstrip("/"), cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.mean = None
        self.ok = True

    def _one(self, text: str) -> Optional[List[float]]:
        import urllib.request

        req = urllib.request.Request(self.url + "/embedding", data=json.dumps({"doc_id": "person-linkage-dryrun", "text": text}).encode(),
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())["embedding"]
        except Exception:
            return None

    def warm(self, texts: List[str], progress: Progress) -> None:
        todo = sorted({t for t in texts if t not in self.cache})
        errors = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, (t, v) in enumerate(zip(todo, ex.map(self._one, todo)), 1):
                if v is None:
                    errors += 1
                else:
                    self.cache[t] = v
                if i % 250 == 0 or i == len(todo):
                    progress("embed labels", i, len(todo), errors)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        import numpy as np

        if not self.cache:
            self.ok = False
            return
        mat = np.array([self.cache[k] for k in sorted(self.cache)], dtype="float64")
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        self.mean = mat.mean(axis=0)

    def sim(self, a: str, b: str) -> Optional[float]:
        import numpy as np

        if not self.ok or a not in self.cache or b not in self.cache:
            return None
        va, vb = (np.array(self.cache[x], dtype="float64") for x in (a, b))
        va, vb = va / np.linalg.norm(va) - self.mean, vb / np.linalg.norm(vb) - self.mean
        return round(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))), 4)


# ---------------------------------------------------------------- mentions
_LIST_SEP = re.compile(r"^\s*(,|and|&|,\s*and)\s*$", re.I)


def cluster_mentions(mentions_by_source: Dict[str, Tuple[str, list]]) -> Dict[str, dict]:
    clusters: Dict[str, dict] = {}
    for sid in sorted(mentions_by_source):
        text, ms = mentions_by_source[sid]
        persons = [m for m in ms if m.label == PERSON_LABEL]
        for m in persons:
            key = fold(m.text)
            if not key:
                continue
            c = clusters.setdefault(key, {"surfaces": defaultdict(int), "sources": [], "repeat_sources": [], "first_seen": None})
            c["surfaces"][m.text] += 1
            if sid not in c["sources"]:
                c["sources"].append(sid)
        # A name listed twice side by side ("Jay, Jay") in one source.
        for a, b in zip(persons, persons[1:]):
            if fold(a.text) == fold(b.text) and _LIST_SEP.match(text[a.span_end:b.span_start] or "x"):
                rs = clusters[fold(a.text)]["repeat_sources"]
                if sid not in rs:
                    rs.append(sid)
    for key, c in clusters.items():
        c["label"] = sorted(c["surfaces"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        c["node_id"] = "mention:" + key
        c["mentions"] = sum(c["surfaces"].values())
    return clusters


def cooccurring_pairs(mentions_by_source: Dict[str, Tuple[str, list]]) -> frozenset:
    pairs = set()
    for sid, (text, ms) in mentions_by_source.items():
        persons = [m for m in ms if m.label == PERSON_LABEL]
        for a, b in zip(persons, persons[1:]):
            fa, fb = fold(a.text), fold(b.text)
            if fa != fb and _LIST_SEP.match(text[a.span_end:b.span_start] or "x"):
                pairs.add(tuple(sorted((fa, fb))))
    return frozenset(pairs)


def window(text: str, needle: str, width: int = 90) -> Optional[str]:
    i = text.lower().find(needle.lower())
    if i < 0:
        return None
    a, b = max(0, i - width), min(len(text), i + len(needle) + width)
    return ("..." if a else "") + " ".join(text[a:b].split()) + ("..." if b < len(text) else "")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="postgresql://postgres:postgres@localhost:55432/conjourney")
    ap.add_argument("--graph-url", default="redis://localhost:6380/0")
    ap.add_argument("--vector-host", default="http://localhost:8320")
    ap.add_argument("--run", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/person-linkage"))
    ap.add_argument("--html", type=Path, default=None)
    ap.add_argument("--people", type=Path, default=ROOT / "config" / "person_model" / "people.yaml")
    ap.add_argument("--answers", type=Path, default=ROOT / "config" / "person_model" / "stated_answers.yaml")
    ap.add_argument("--no-extractor", action="store_true", help="skip GLiNER (extractor features read 'missing')")
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()

    out = args.out_dir / args.run
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress(out / "progress.log")
    cfg = L.load_config()
    people = load_people(args.people)
    me = interlocutor(people)
    answers = yaml.safe_load(args.answers.read_text(encoding="utf-8")) if args.answers.exists() else {}

    g = Graph(args.graph_url)
    conn = pg_connect(args.dsn)
    try:
        # --- graph population (read only)
        rows = g.ro("MATCH (n:SubstrateNode) WHERE (n:Entity OR n:Concept) AND n.label IS NOT NULL "
                    "RETURN n.node_id, n.label, labels(n), n.provenance_producer, n.entity_type, n.evidence_refs_json ORDER BY n.node_id")
        nodes = {}
        for nid, label, labels, producer, etype, ev in rows:
            try:
                refs = tuple(sorted(json.loads(ev or "[]")))
            except Exception:
                refs = ()
            nodes[nid] = {"node_id": nid, "label": label, "labels": tuple(labels), "producer": producer or "", "entity_type": etype or "", "evidence_refs": refs}
        prog("load graph nodes", len(nodes), len(nodes))

        alias_rows = [dict(zip(("node_id", "alias_norm", "alias_text", "alias_class", "referent_kind", "grounded_in", "valid_until"), r))
                      for r in pg_all(conn, "select node_id, alias_norm, alias_text, alias_class, referent_kind, grounded_in, valid_until::text "
                                      "from referent_alias order by node_id, alias_norm")]
        prompts = pg_all(conn, "select id, prompt, created_at::text from chat_history_log where prompt is not null and length(prompt) > 0 order by created_at, id")
        memories = pg_all(conn, "select memory_id, statement from episode_memory where statement is not null order by created_at, memory_id")
        prog("load postgres rows", len(prompts), len(prompts), note=f"aliases={len(alias_rows)} memories={len(memories)}")

        # --- anchors: person referents (+ their name aliases)
        person_ids = sorted(nid for nid, n in nodes.items() if n["entity_type"] == "person")
        anchors = []
        for nid in person_ids:
            aliases = tuple(sorted({r["alias_text"] for r in alias_rows if r["node_id"] == nid and r["alias_class"] in ("name", "descriptor")}))
            ev_rows = g.ro("MATCH (n {node_id: $id})-[:observed_in]->(e) RETURN e.content_ref ORDER BY e.content_ref", {"id": nid})
            anchors.append(L.Anchor(nid, nodes[nid]["label"], aliases, nodes[nid]["producer"], tuple(r[0] for r in ev_rows if r[0])))
        anchor_ids = {a.node_id for a in anchors}
        cands = [L.Candidate(n["node_id"], n["label"], "substrate_node", n["labels"], n["producer"], n["evidence_refs"])
                 for n in nodes.values() if n["node_id"] not in anchor_ids]

        # --- extractor over the interlocutor's chat prompts
        mentions_by_source: Dict[str, Tuple[str, list]] = {}
        extractor = None
        if not args.no_extractor:
            from orion.person_model.extract import GlinerExtractor

            extractor = GlinerExtractor()
            for i, (pid, text, _) in enumerate(prompts, 1):
                mentions_by_source[pid] = (text, extractor.extract(text, source_kind="chat_prompt", source_id=pid))
                if i % 50 == 0 or i == len(prompts):
                    prog("extract mentions", i, len(prompts))
            with (out / "mentions.jsonl").open("w", encoding="utf-8") as fh:
                for pid in sorted(mentions_by_source):
                    for m in mentions_by_source[pid][1]:
                        fh.write(m.model_dump_json() + "\n")
        clusters = cluster_mentions(mentions_by_source)
        first_seen = {pid: created for pid, _, created in prompts}
        for c in clusters.values():
            c["first_seen"] = min(first_seen[s] for s in c["sources"])[:10]
        agent_names = {fold(p.display_name) for p in people if p.role == "self"}
        cluster_cands = [L.Candidate(c["node_id"], c["label"], "mention_cluster", (), "person_extractor", tuple("chat_prompt:" + s for s in c["sources"]))
                         for k, c in sorted(clusters.items()) if k not in agent_names]

        # --- pairs: anchors x (graph nodes + clusters); clusters x clusters; clusters x unanchored entities
        pairs = L.candidate_pairs(anchors, cands + cluster_cands)
        anchored = {c.node_id for _, c in pairs}
        cl_anchors = [L.Anchor(c.node_id, c.label, (), c.producer, c.evidence_refs) for c in cluster_cands]
        cl_pairs = [(a, c) for a, c in L.candidate_pairs(cl_anchors, cluster_cands) if a.node_id < c.node_id]
        free_entities = [c for c in cands if c.node_id not in anchored and "Entity" in c.node_labels]
        cl_pairs += L.candidate_pairs(cl_anchors, free_entities)
        all_pairs = pairs + cl_pairs
        prog("blocking", len(all_pairs), len(all_pairs), note=f"anchors={len(anchors)} clusters={len(cluster_cands)}")

        # --- features needing models
        person_cache: Dict[str, Optional[bool]] = {}
        labels_needed = sorted({c.label for _, c in all_pairs})
        if extractor is not None:
            for i, lab in enumerate(labels_needed, 1):
                person_cache[lab] = extractor.whole_label_is_person(lab)
                if i % 100 == 0 or i == len(labels_needed):
                    prog("extractor on labels", i, len(labels_needed))
        emb = CenteredEmbedder(args.vector_host, out / "embedding_cache.json")
        if args.no_embeddings:
            emb.ok = False
        else:
            pop = sorted({n["label"] for n in nodes.values()} | {c.label for c in cluster_cands}
                         | {x for a in anchors for x in (a.label, *a.aliases)})
            emb.warm(pop, prog)
        ctx = L.LinkageContext(
            is_person_name=lambda lab: person_cache.get(lab),
            embedding_sim=emb.sim,
            prompt_keys=frozenset(prompt_context_keys([ROOT / "orion" / "cognition" / "prompts"])),
            cooccurring=cooccurring_pairs(mentions_by_source),
        )

        # --- evidence for cards
        mem_by_id = {m: s for m, s in memories}
        node_view = lambda nid, lab, kind, prod: {"node_id": nid, "label": lab, "kind": kind, "producer": prod}  # noqa: E731

        def snippets_for(a: L.Anchor, c: L.Candidate) -> List[dict]:
            out_s: List[dict] = []
            for ref in c.evidence_refs[:3]:
                kind, _, rid = ref.partition(":")
                if kind == "episode_memory" and rid in mem_by_id:
                    out_s.append({"source": ref, "text": mem_by_id[rid][:240]})
            if c.kind == "mention_cluster":
                for ref in c.evidence_refs[:3]:
                    pid = ref.partition(":")[2]
                    w = window(mentions_by_source.get(pid, ("", []))[0], c.label)
                    if w:
                        out_s.append({"source": ref, "text": w})
            else:
                needle = c.label.replace("_", " ") if "_" in c.label else c.label
                for mid, stmt in memories:
                    if needle.lower() in stmt.lower():
                        out_s.append({"source": f"episode_memory:{mid}", "text": stmt[:240]})
                        if len(out_s) >= 2:
                            break
                hits = 0
                for pid, text, _ in prompts:
                    w = window(text, needle)
                    if w:
                        out_s.append({"source": f"chat_prompt:{pid}", "text": w})
                        hits += 1
                        if hits >= 2:
                            break
                if c.kind == "substrate_node":
                    nb = g.ro("MATCH (n {node_id: $id})-[r]-(m) WHERE m.label IS NOT NULL RETURN type(r), m.label ORDER BY m.label LIMIT 4", {"id": c.node_id})
                    if nb:
                        out_s.append({"source": "graph neighbours", "text": "; ".join(f"{t}: {lab}" for t, lab in nb)})
            if not out_s and c.kind == "substrate_node" and not c.evidence_refs:
                pass
            return out_s

        decisions = []
        for i, (a, c) in enumerate(all_pairs, 1):
            ev = {"anchor_node": node_view(a.node_id, a.label, "person referent" if a.node_id in anchor_ids else "mention cluster", a.producer),
                  "candidate_node": node_view(c.node_id, c.label, "/".join(x for x in c.node_labels if x != "SubstrateNode") or c.kind, c.producer)}
            d = L.decide(a, c, cfg, ctx, ev)
            decisions.append(d)
            if i % 200 == 0 or i == len(all_pairs):
                prog("score pairs", i, len(all_pairs))
        decisions = L.best_per_candidate(decisions)
        # snippets only for the decisions that survive (cheap enough, deterministic)
        dec_final = []
        amap = {a.node_id: a for a in anchors + cl_anchors}
        cmap = {c.node_id: c for c in cands + cluster_cands}
        for d in decisions:
            ev = dict(d.evidence)
            ev["snippets"] = snippets_for(amap[d.anchor_id], cmap[d.candidate_id])
            dec_final.append(d.model_copy(update={"evidence": ev}))

        if me is not None:
            home = ((answers or {}).get("people") or {}).get(me.node_id, {}).get("home")
            if home:
                me_anchor = next(a for a in anchors if a.node_id == me.node_id)
                place_labels = {nid: n["label"] for nid, n in nodes.items()}
                for r in alias_rows:
                    place_labels.setdefault(r["node_id"], next((x["alias_text"] for x in alias_rows if x["node_id"] == r["node_id"] and x["alias_class"] == "name"), r["node_id"]))
                dec_final += L.stated_home_conflicts(alias_rows, person=me_anchor, stated_home=home["value"], place_labels=place_labels, cfg=cfg, stated_ref=home["source"])
        dec_final += L.repeated_name_splits({k: v for k, v in clusters.items() if k not in agent_names}, cfg)
        dec_final.sort(key=lambda d: (d.group, d.anchor_label, d.candidate_label, d.decision_id))

        # --- outputs
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["decision_id", "group", "relation", "rule", "score", "anchor_id", "anchor_label", "candidate_id", "candidate_label", "candidate_kind", "features"])
        for d in dec_final:
            w.writerow([d.decision_id, d.group, d.relation, d.rule or "", "" if d.score is None else f"{d.score:.3f}", d.anchor_id, d.anchor_label,
                        d.candidate_id, d.candidate_label, d.candidate_kind, ";".join(f"{f.name}={f.level}:{f.weight:+.1f}" for f in d.features)])
        (out / "decisions.csv").write_text(buf.getvalue(), encoding="utf-8")
        with (out / "decisions.jsonl").open("w", encoding="utf-8") as fh:
            for d in dec_final:
                fh.write(d.model_dump_json() + "\n")
        store = ReadOnlyStore(g, conn)
        now_iso = f"{args.run}T00:00:00+00:00"
        with (out / "merge_plan.jsonl").open("w", encoding="utf-8") as fh:
            for d in dec_final:
                if d.relation in ("same_as", "facet_of", "junk", "not_home") and not d.candidate_id.startswith("mention:") and not d.anchor_id.startswith("mention:"):
                    e = build_entry(d, store, decided_by="person_linkage_v1" if d.group != "needs_your_call" else "pending_interlocutor",
                                    decision="auto" if d.group != "needs_your_call" else "asked", now_iso=now_iso)
                    fh.write(e.model_dump_json() + "\n")
        counts = defaultdict(int)
        for d in dec_final:
            counts[d.group] += 1
        sha = hashlib.sha256((out / "decisions.csv").read_bytes()).hexdigest()
        prog("done", len(dec_final), len(dec_final), note=f"auto_merge={counts['auto_merge']} needs_your_call={counts['needs_your_call']} auto_reject={counts['auto_reject']} decisions_sha256={sha[:16]}")

        if args.html:
            info = [{"name": c["label"], "mentions": c["mentions"], "first_seen": c["first_seen"],
                     "example": window(mentions_by_source[c["sources"][0]][0], c["label"], 70)}
                    for k, c in sorted(clusters.items(), key=lambda kv: (-kv[1]["mentions"], kv[0]))
                    if k not in agent_names and not any(d.candidate_id == c["node_id"] and d.group == "auto_merge" and not d.anchor_id.startswith("mention:") for d in dec_final)]
            summary = [
                f"Dry run {args.run}, read only. Nothing here has been applied to any store.",
                f"{counts['auto_merge']} auto-merge, {counts['needs_your_call']} need your call, {counts['auto_reject']} auto-reject. "
                f"Thresholds: auto-merge at score >= {cfg.upper:+.1f}, auto-reject at <= {cfg.lower:+.1f} ({cfg.version}).",
                "Pick approve or reject on any card (auto groups start pre-approved), add notes, then Export decisions. Your CSV becomes the labels that re-tune the thresholds.",
            ]
            args.html.parent.mkdir(parents=True, exist_ok=True)
            args.html.write_text(render(dec_final, title="Person cleanup review", run_id=args.run, summary_lines=summary, info_rows=info), encoding="utf-8")
            prog("wrote html", 1, 1, note=str(args.html))
    finally:
        conn.rollback()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
