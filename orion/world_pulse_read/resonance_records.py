"""Authoritative graph excerpts shared by live resonance and its replay eval."""
from __future__ import annotations
import json
from typing import Any
from orion.curiosity.worldview import CLOSED_STATUSES
from orion.schemas.reading_resonance import ResonanceNeighborV1
from orion.world_pulse_read.resonance import PoolDoc


class Falkor:
    """Bounded-time, read-only graph access; callers run these synchronous reads in a thread."""
    def __init__(self, url: str):
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=10, socket_connect_timeout=5)

    def rows(self, graph: str, cypher: str, **params: Any) -> list[dict[str, Any]]:
        prefix = "CYPHER " + " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in params.items()) + " " if params else ""
        res = self.r.execute_command("GRAPH.RO_QUERY", graph, prefix + cypher)
        return [dict(zip(res[0], row)) for row in res[1]]


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


