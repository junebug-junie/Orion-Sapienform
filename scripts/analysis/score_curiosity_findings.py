#!/usr/bin/env python3
"""Score every finished curiosity run for REGISTER and NOVELTY. Read-only analysis.

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md
("Which findings compete").

For each finished ``curiosity.investigate`` run (both lines: ``investigate`` and
``self_inquiry``) this script assembles:

- the QUESTION the run took up: the self-question text (``curiosity_self_questions``)
  for ``self_inquiry``; the claim of the prior that moved most, else the first prior
  formed, for ``investigate``;
- the ANSWER Orion wrote: ``LivedAnswer.text`` / ``SelfDefinition.text`` /
  ``Finding.text`` from ``orion_worldview`` (FalkorDB, ``GRAPH.RO_QUERY`` only);
- NOVELTY: ``1 - max cosine`` between the answer's embedding and everything Orion
  wrote before the run started (earlier curiosity answers, prior claims, Day
  letters). Embeddings come from vector-host ``/embedding`` (bge-large-en-v1.5,
  compute only, nothing stored), the same endpoint the personality graph uses;
- REGISTER: read from a labels file (``--labels``, JSONL ``{run_id, register}``),
  produced by the judge prompt in the design doc. More than one labels file may be
  given; the first is the judge, the rest are agreement raters;
- NOV_PCT: share of the PHILOSOPHICAL findings of the 7 days up to and including
  the run that sit strictly below its novelty, from the real
  ``compute_prediction_error_magnitude`` (the function ``judge_candidate`` uses).
  Band ``high``/``unusual`` (top tenth) = COMPETES; 0.75-0.9 = near-miss.

Outputs:

- ``--events-out``: JSONL events for ``replay_attention_world_candidates.py``
  (``source=curiosity_finding``, ``value=novelty``, ``register``).
- ``--table-out``: the markdown appendix table (one row per run).
- stdout: agreement between raters, measure distributions, verdict counts.

Run (SELECT-only reads; the embedding call computes and stores nothing):

    /mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/analysis/score_curiosity_findings.py \\
        --since 2026-09-29 --labels judge.jsonl --labels rater2.jsonl \\
        --cache /tmp/curiosity_emb.npz --events-out findings.jsonl --table-out table.md

Needs numpy, httpx, redis (repo venv), ``docker exec`` on the SQL container, and
FalkorDB on localhost:6380.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude  # noqa: E402

PSQL = ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc"]
EMBED_URL = "http://localhost:8320/embedding"
EMBED_CHARS = 2000  # bge-large reads ~512 tokens; the rest is truncated by the model anyway
WINDOW = timedelta(days=7)
MIN_HISTORY = 20  # same floor the design uses for the live candidate

RUNS_SQL = """
select json_build_object('run_id',s.run_id,'t',s.generated_at,
  'started',(select min(generated_at) from substrate_durable_run_state x where x.run_id=s.run_id),
  'line',s.detail->>'line','family',s.detail->>'self_question_family',
  'lived',s.detail->'lived_answer','selfdef',s.detail->'self_definition','finding',s.detail->>'finding_text',
  'nats',o.realized_nats,'n_tested',o.n_tested,'n_formed',o.n_formed,'per_prior',o.per_prior)
from substrate_durable_run_state s left join curiosity_run_outcomes o using(run_id)
where s.workflow='curiosity.investigate' and s.node='finish' order by s.generated_at
"""
QUESTIONS_SQL = ("select json_build_object('question_id',question_id,'text',text,'minted_by',minted_by) "
                 "from curiosity_self_questions")
LETTERS_SQL = "select json_build_object('id',run_id,'t',created_at,'text',note_md) from orion_day_letter"


def psql_json(sql: str) -> list[dict]:
    out = subprocess.run(PSQL + [" ".join(sql.split())], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def graph_rows(cypher: str) -> list[dict]:
    import redis

    r = redis.Redis(host="localhost", port=6380, decode_responses=True)
    res = r.execute_command("GRAPH.RO_QUERY", "orion_worldview", cypher)
    return [dict(zip(res[0], row)) for row in res[1]]


def clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def assemble() -> tuple[list[dict], dict, list[dict]]:
    priors = {p["id"]: p for p in graph_rows(
        "MATCH (p:Prior) RETURN p.prior_id AS id, p.claim AS claim, p.run_id AS run_id")}
    findings: dict[str, list[str]] = defaultdict(list)
    for f in graph_rows("MATCH (n:Finding) RETURN n.run_id AS run_id, n.text AS text"):
        findings[f["run_id"]].append(f["text"] or "")
    lived = {x["run_id"]: x for x in graph_rows(
        "MATCH (n:LivedAnswer) RETURN n.run_id AS run_id, n.text AS text, n.question_id AS qid")}
    selfdef = {x["run_id"]: x for x in graph_rows(
        "MATCH (n:SelfDefinition) RETURN n.run_id AS run_id, n.text AS text")}
    qs = {q["question_id"]: q for q in psql_json(QUESTIONS_SQL)}
    out = []
    for d in psql_json(RUNS_SQL):
        rid = d["run_id"]
        la = d["lived"] if isinstance(d.get("lived"), dict) else {}
        sd = d["selfdef"] if isinstance(d.get("selfdef"), dict) else {}
        pp = d.get("per_prior") or []
        if d["line"] == "self_inquiry":
            qid = "anatomy.made_of" if (d.get("family") == "anatomy" or sd.get("text")) else (
                la.get("question_id") or (lived.get(rid) or {}).get("qid"))
            question = (qs.get(qid) or {}).get("text") or qid or "(question not recorded)"
            qsrc = f"self_question:{qid}" if qid else "none"
            answer = (sd.get("text") or (selfdef.get(rid) or {}).get("text")) if qid == "anatomy.made_of" else (
                la.get("text") or (lived.get(rid) or {}).get("text"))
        else:
            tested = sorted((p for p in pp if p.get("kind") == "tested"), key=lambda p: -(p.get("nats") or 0))
            formed = [p for p in pp if p.get("kind") == "formed"]
            pick = (tested or formed or [None])[0]
            if pick:
                question = (priors.get(pick["prior_id"]) or {}).get("claim") or pick["prior_id"]
                qsrc = f"prior:{pick['prior_id']}"
            else:
                question, qsrc = "(no prior tested or formed)", "none"
            answer = None
        # Fallback order: the graph node Orion wrote, else the durable finish row's write-up.
        answer = answer or " ".join(findings.get(rid, [])) or d.get("finding") or ""
        out.append({"run_id": rid, "t": d["t"], "started": d["started"], "line": d["line"],
                    "family": d.get("family"), "question": clean(question), "question_src": qsrc,
                    "answer": clean(answer), "nats": d.get("nats"), "n_formed": d.get("n_formed"),
                    "per_prior": pp})
    return out, priors, psql_json(LETTERS_SQL)


async def _embed_all(texts: dict[str, str]) -> dict[str, list[float]]:
    import httpx

    out: dict[str, list[float]] = {}
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(timeout=60) as client:
        async def one(k: str, t: str) -> None:
            async with sem:
                r = await client.post(EMBED_URL, json={"doc_id": k, "text": t[:EMBED_CHARS]})
                r.raise_for_status()
                out[k] = r.json()["embedding"]
        await asyncio.gather(*(one(k, t) for k, t in texts.items() if t.strip()))
    return out


def embeddings(texts: dict[str, str], cache: Path | None) -> dict[str, np.ndarray]:
    have: dict[str, np.ndarray] = {}
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = {k: t for k, t in texts.items() if k not in have}
    if todo:
        for k, v in asyncio.run(_embed_all(todo)).items():
            have[k] = np.asarray(v, dtype=np.float32)
        if cache:
            ks = sorted(have)
            np.savez(cache, keys=np.array(ks), vecs=np.stack([have[k] for k in ks]))
    return {k: v / np.linalg.norm(v) for k, v in have.items() if k in texts}


def score(runs, priors, letters, labels: list[dict[str, str]], since: str, cache):
    run_t = {r["run_id"]: r["started"] for r in runs}
    texts = {f"run:{r['run_id']}": r["answer"] for r in runs}
    texts.update({f"prior:{k}": p["claim"] or k for k, p in priors.items()})
    texts.update({f"letter:{x['id']}": x["text"] or "" for x in letters})
    emb = embeddings(texts, cache)
    corpus = [(r["t"], f"run:{r['run_id']}", r["run_id"]) for r in runs]
    corpus += [(run_t.get(p["run_id"], "2000"), f"prior:{k}", p["run_id"]) for k, p in priors.items()]
    corpus += [(x["t"], f"letter:{x['id']}", None) for x in letters]
    corpus = [c for c in corpus if c[1] in emb]
    rows = []
    for r in runs:
        k = f"run:{r['run_id']}"
        if r["t"] < since or k not in emb:
            continue
        prev = [c for c in corpus if c[0] < r["started"] and c[2] != r["run_id"]]
        sims = np.stack([emb[c[1]] for c in prev]) @ emb[k] if prev else np.zeros(1)
        j = int(np.argmax(sims))
        regs = [lab.get(r["run_id"]) for lab in labels]
        rows.append({**r, "novelty": round(float(1 - sims.max()), 4),
                     "nearest": prev[j][1] if prev else None, "registers": regs,
                     "register": regs[0] if regs else None})
    return assign_verdicts(rows)


def assign_verdicts(rows: list[dict]) -> list[dict]:
    """Verdict per run, as the live candidate would judge it at the run's finish time.

    Only PHILOSOPHICAL findings are readings; operational and no_finding ones never
    compete and never enter the history. Novelty is judged by the real
    ``compute_prediction_error_magnitude`` against the philosophical findings of the 7
    days up to and including the run: band ``high``/``unusual`` (top tenth) competes.
    """
    phil = [x for x in rows if x["register"] == "philosophical"]
    for x in rows:
        t = datetime.fromisoformat(x["t"])
        hist = [(datetime.fromisoformat(y["t"]), y["novelty"]) for y in phil
                if t - WINDOW < datetime.fromisoformat(y["t"]) <= t]
        if x["register"] == "philosophical":
            mag = compute_prediction_error_magnitude(value=x["novelty"], observed_at=t, history=hist,
                                                     now=t, min_readings=MIN_HISTORY)
            x["nov_pct"], x["band"] = mag.percentile_now, mag.band
        else:
            x["nov_pct"], x["band"] = None, None
        x["phil_hist_7d"] = len(hist)
        if x["register"] != "philosophical":
            x["verdict"] = f"no: {x['register'] or 'unlabelled'}"
        elif x["band"] == "insufficient_history":
            x["verdict"] = "no: history<20"
        elif x["band"] in ("high", "unusual"):
            x["verdict"] = "COMPETES"
        elif (x["nov_pct"] or 0) >= 0.75:
            x["verdict"] = "near-miss"
        else:
            x["verdict"] = "no: not novel"
    return rows


def kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(x == "philosophical" for x in a) / n
    pb = sum(y == "philosophical" for y in b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return agree, (agree - pe) / (1 - pe)


def _load_privacy_redactor():
    """Word -> "[name]" for every term on the repo's personal-terms denylist.

    The table goes into a tracked doc in a public repo, and Orion's findings name
    real people. Uses ``scripts/check_personal_terms.py`` (the CI privacy gate) so
    the redaction and the gate can never disagree. Hashes only; no names here.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_personal_terms", REPO / "scripts" / "check_personal_terms.py")
    cpt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpt)
    hashes, _ = cpt.load_denylist()

    def redact(text: str) -> str:
        def sub(m: re.Match) -> str:
            return "[name]" if cpt.line_hits(m.group(0), hashes) else m.group(0)
        return re.sub(r"[^\W\d_]+", sub, text)

    return redact


def table(rows) -> str:
    redact = _load_privacy_redactor()

    def cell(s, n):
        s = clean(s).replace("|", "/")
        s = s if len(s) <= n else s[: n - 1] + "…"
        return redact(s)
    out = ["| date (UTC) | run | line | question | finding excerpt | register (judge/2nd) | novelty | pct in 7d phil | verdict |",
           "|---|---|---|---|---|---|---:|---:|---|"]
    for x in rows:
        regs = "/".join((r or "?")[:4] for r in x["registers"])
        pct = "" if x["nov_pct"] is None else f"{x['nov_pct']:.2f}"
        out.append(f"| {x['t'][:16].replace('T', ' ')} | {x['run_id'][:8]} | {x['line']}"
                   f"{'/' + x['family'] if x['family'] else ''} | {cell(x['question'], 90)} | "
                   f"{cell(x['answer'], 140)} | {regs} | {x['novelty']:.3f} | {pct} | {x['verdict']} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-29")
    ap.add_argument("--labels", action="append", default=[], help="JSONL {run_id, register}; first = judge")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--events-out", type=Path)
    ap.add_argument("--table-out", type=Path)
    args = ap.parse_args()
    labels = []
    for p in args.labels:
        lab = {}
        for line in Path(p).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                lab[d["run_id"]] = d["register"]
        labels.append(lab)
    runs, priors, letters = assemble()
    rows = score(runs, priors, letters, labels, args.since, args.cache)
    print(f"runs scored: {len(rows)}  since {args.since}")
    if len(labels) >= 2:
        both = [x for x in rows if x["registers"][0] and x["registers"][1]]
        agree, k = kappa([x["registers"][0] for x in both], [x["registers"][1] for x in both])
        print(f"judge vs rater 2: n={len(both)} agreement={agree:.3f} kappa={k:.3f}")
    print("register by line:", dict(Counter((x["line"], x["register"]) for x in rows)))
    print("novelty p10/p50/p90:", np.percentile([x["novelty"] for x in rows], [10, 50, 90]).round(3).tolist())
    print("verdicts:", dict(Counter(x["verdict"] for x in rows)))
    if args.events_out:
        with args.events_out.open("w") as f:
            for x in rows:
                if x["register"] == "philosophical":
                    f.write(json.dumps({"source": "curiosity_finding", "t": x["t"], "value": x["novelty"],
                                        "ref": x["run_id"], "detail": {"finding_text": x["answer"],
                                                                       "question": x["question"]}}) + "\n")
    if args.table_out:
        args.table_out.write_text(table(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
