#!/usr/bin/env python3
"""Judge-agreement eval for the curiosity finding register (read-only; compute-only model calls).

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md ("The judge, as
specified for the live patch"). A local lane may judge live findings only when, against
``scripts/analysis/fixtures/curiosity_register/reference_labels_2026-10-20.jsonl`` (Sonnet judge):

1. Cohen's kappa >= 0.8 over all reference runs;
2. agreement >= 90% on the runs the reference judge called philosophical;
3. all 6 replay winners labelled philosophical.

The input is assembled exactly as the live register loop assembles it
(``orion.curiosity.register.run_question_and_answer`` on the finish row, the outcome row's
``per_prior``, the self-question texts and the graph's prior claims), and the prompt is the pinned
``JUDGE_PROMPT_V2``. Each lane is called on its llama.cpp ``/v1/chat/completions`` with temperature 0
and thinking off. This is an offline eval (263 short calls); a LIVE judge must call orion-llm-gateway by
route, and note that the gpu-pool classes ``metacog`` and ``fast`` spill to the agent roles under load
(config/gpu_pool.yaml ``classes``), i.e. to the seats that write curiosity findings, so a live judge must
check the served model on every reply.

Run:

    /mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/analysis/eval_curiosity_register_judge.py \\
        --lane metacog=<metacog lane base url> --lane quick=<quick lane base url> \\
        --out /tmp/curiosity_register_judge

Needs ``docker exec`` on the SQL container (SELECT only) and FalkorDB on localhost:6380 (RO query).

Results 2026-11-12 (263 reference runs, all inputs assembled, 0 judge errors). No lane passes:

    lane                         model              kappa   agree  phil-agree  winners phil
    metacog                      Qwen3-8B Q5_K_M    0.580   0.772  0.96        5/6 (91241b6d operational)
    metacog, thinking on         Qwen3-8B Q5_K_M    0.586   0.776  0.96        5/6
    quick                        Qwen3-8B Q4_K_M    0.604   0.787  0.95        5/6

The failure is one-sided: 34-39 of 161 operational findings are called philosophical (reflective,
first-person plumbing: "the finding reflects on Orion's decision-making"), the dressing failure mode
the design names. Not tested: the Flash-Next seats (chat 8011, agent 8015) and the Bonsai gpu2 seat
write curiosity findings, so they are excluded; hecate (agent-deep) was offline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.curiosity.register import judge_messages, parse_judge_reply, run_question_and_answer  # noqa: E402

LABELS = REPO / "scripts/analysis/fixtures/curiosity_register/reference_labels_2026-10-20.jsonl"
WINNERS = ("2bf28e16", "987448e7", "8998e10d", "91241b6d", "1f7b1157", "a9ba948f")
PSQL = ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc"]

RUNS_SQL = """
select json_build_object('run_id', s.run_id, 'detail', s.detail, 'per_prior', o.per_prior)
from substrate_durable_run_state s left join curiosity_run_outcomes o using (run_id)
where s.workflow = 'curiosity.investigate' and s.node = 'finish' and s.run_id = any(string_to_array('{ids}', ','))
"""


def psql_json(sql: str) -> list[dict]:
    out = subprocess.run(PSQL + [" ".join(sql.split())], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def graph(cypher: str) -> list[list]:
    import redis

    r = redis.Redis(host="localhost", port=6380, decode_responses=True)
    return r.execute_command("GRAPH.RO_QUERY", "orion_worldview", cypher)[1]


def prior_claims() -> dict[str, str]:
    return {row[0]: row[1] for row in graph("MATCH (p:Prior) RETURN p.prior_id, p.claim") if row[0]}


def run_findings() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rid, text in graph("MATCH (n:Finding) RETURN n.run_id, n.text"):
        out.setdefault(str(rid), []).append(text or "")
    return out


def kappa(a: list[str], b: list[str]) -> tuple[float, float | None]:
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return agree, (None if pe >= 1.0 else (agree - pe) / (1 - pe))


def items() -> list[dict]:
    ref = [json.loads(x) for x in LABELS.read_text().splitlines() if x.strip()]
    rows = {r["run_id"]: r for r in psql_json(RUNS_SQL.format(ids=",".join(x["run_id"] for x in ref)))}
    qs = {q["question_id"]: q["text"] for q in psql_json(
        "select json_build_object('question_id', question_id, 'text', text) from curiosity_self_questions")}
    claims = prior_claims()
    findings = run_findings()
    out = []
    for x in ref:
        r = rows.get(x["run_id"])
        if r is None:
            out.append({**x, "missing": True})
            continue
        q, qref, a = run_question_and_answer(r["detail"] or {}, per_prior=r.get("per_prior") or [],
                                             questions=qs, prior_claims=claims,
                                             graph_findings=findings.get(x["run_id"], ()))
        out.append({**x, "question": q, "question_ref": qref, "answer": a})
    return out


async def judge_all(url: str, its: list[dict], concurrency: int, thinking: bool = False) -> list[dict]:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as client:
        async def one(it: dict) -> dict:
            if it.get("missing") or not it.get("answer"):
                return {**it, "label": None, "error": "no_input"}
            async with sem:
                try:
                    r = await client.post(f"{url}/v1/chat/completions", json={
                        "messages": judge_messages(it["question"], it["answer"]), "temperature": 0,
                        "max_tokens": 2048 if thinking else 160,
                        "chat_template_kwargs": {"enable_thinking": thinking}})
                    r.raise_for_status()
                    body = r.json()
                    text = body["choices"][0]["message"].get("content") or ""
                    model = body.get("model")
                except Exception as exc:  # noqa: BLE001
                    return {**it, "label": None, "error": f"call: {exc}"[:200]}
            reg, reason, err = parse_judge_reply(text)
            return {**it, "label": reg, "reason": reason, "error": err, "model": model}
        return await asyncio.gather(*(one(i) for i in its))


def report(name: str, res: list[dict]) -> dict:
    scored = [r for r in res if r.get("label")]
    a = [r["judge_sonnet"] for r in scored]
    b = [r["label"] for r in scored]
    agree, k = kappa(a, b) if scored else (0.0, None)
    phil = [r for r in res if r["judge_sonnet"] == "philosophical"]
    phil_agree = sum(r.get("label") == "philosophical" for r in phil) / max(1, len(phil))
    winners = {w: next((r.get("label") for r in res if r["run_id"].startswith(w)), "missing") for w in WINNERS}
    errors = Counter((r.get("error") or "").split(":")[0] for r in res if r.get("error"))
    confusion = Counter((r["judge_sonnet"], r.get("label")) for r in res)
    out = {"lane": name, "n": len(res), "scored": len(scored), "agreement": round(agree, 4),
           "kappa": None if k is None else round(k, 4), "philosophical_agreement": round(phil_agree, 4),
           "winners": winners, "errors": dict(errors),
           "confusion": {f"{x}->{y}": n for (x, y), n in sorted(confusion.items(), key=str)},
           "models": dict(Counter(r.get("model") for r in res if r.get("model")))}
    out["passes"] = bool(k is not None and k >= 0.8 and phil_agree >= 0.9
                         and all(v == "philosophical" for v in winners.values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", action="append", required=True, help="name=base_url (llama.cpp server)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--thinking", action="store_true", help="thinking on (more tokens per label)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    its = items()
    (args.out / "inputs.jsonl").write_text("".join(json.dumps(i) + "\n" for i in its))
    print(f"reference runs: {len(its)}  missing finish rows: {sum(1 for i in its if i.get('missing'))}")
    for spec in args.lane:
        name, _, url = spec.partition("=")
        res = asyncio.run(judge_all(url.rstrip("/"), its, args.concurrency, args.thinking))
        (args.out / f"labels_{name}.jsonl").write_text("".join(json.dumps(
            {k: r.get(k) for k in ("run_id", "judge_sonnet", "rater_haiku", "label", "reason", "error", "model")}) + "\n"
            for r in res))
        print(json.dumps(report(name, res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
