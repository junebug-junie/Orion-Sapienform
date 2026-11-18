#!/usr/bin/env python3
"""Precision eval for the supersede classifier (issue #3659).

Two commands. Real-data files live host-side under ~/.orion/evals/ (dir 700, files 600): they
hold memory statements and never go to git.

    # 1. Build the real labelled set from data that already exists (READ-ONLY on the database):
    .venv/bin/python scripts/memory_supersede_eval.py build

    # 2. Run the classifier over synthetic + real items on the local memory_distill route:
    .venv/bin/python scripts/memory_supersede_eval.py run
    #    ... and record the verdict where the distill step reads it (opens or closes write-time apply):
    .venv/bin/python scripts/memory_supersede_eval.py run --record-db

Real items (``build``):

- positives: Juniper's own confirmation-loop revisions (``episode_memory_event`` op ``revised``,
  actor ``juniper``): the revised row against her new wording. Judged by the classifier alone
  (``apply_rules`` false), since the old row's live state is already retired.
- hard negatives: the watched rows in the supersede manifest (trip rows against the home
  correction; the pet rows against the pet-species correction, which the open-ask rule protects);
  pairs the 2026-11-17 dry run vetoed (first question sure-yes, second said both true); and
  near-duplicates (cosine >= 0.85, a quarter of their words shared, first question said no).

``run`` prints counts only (no statements) and writes the per-item result to
~/.orion/evals/memory_supersede_eval_runs/<time>-<checker>.json. ``--record-db`` inserts one row in
``memory_supersede_eval_run`` (ids-free summary); write time applies only while the latest row for
the current check version says ``passed``.

DSN: $ORION_PAGES_DSN, else services/orion-durable-runs/.env POSTGRES_URI on the host port.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.memory.episode import supersede as S  # noqa: E402
from orion.memory.episode.supersede_eval import (  # noqa: E402
    SYNTHETIC_PATH, Decision, EvalItem, decision_dict, item_to_dict, load_items, rule_refusal, score,
)

EVAL_DIR = Path(os.path.expanduser("~/.orion/evals"))
REAL_PATH = EVAL_DIR / "memory_supersede_labelled.jsonl"
RUNS_DIR = EVAL_DIR / "memory_supersede_eval_runs"
DEFAULT_MANIFEST = Path("/mnt/orion-fcc/notes/memory_supersede_pass_manifest.json")
DEFAULT_CHECKS = Path("/tmp/memory-supersede-pass/checks.csv")
NEAR_DUP_COSINE = 0.85
NEAR_DUP_JACCARD = 0.25  # hand-checked 2026-11-18: the 6 pairs this admits are different events on one topic

ROW_SQL = """
SELECT m.memory_id::text AS memory_id, m.statement, m.purpose, m.created_at, m.occurred_at, m.episode_id,
       m.confirmation_state, m.status, m.voice,
       (m.voice = 'juniper_said' AND EXISTS (SELECT 1 FROM episode_memory_evidence e
            WHERE e.memory_id = m.memory_id AND e.source_kind = 'chat_prompt' AND e.verified)) AS juniper_quoted
FROM episode_memory m WHERE m.memory_id = ANY(%s::uuid[])
"""
REVISIONS_SQL = """
SELECT e.memory_id::text AS old_id, n.memory_id::text AS new_id, e.evidence->>'prior_confirmation_state' AS prior
FROM episode_memory_event e
JOIN episode_memory n ON n.supersedes_memory_id = e.memory_id
WHERE e.op = 'revised' AND e.actor = 'juniper'
ORDER BY e.created_at
"""


def default_dsn() -> str:
    if os.environ.get("ORION_PAGES_DSN"):
        return os.environ["ORION_PAGES_DSN"]
    for base in (REPO, Path("/mnt/scripts/Orion-Sapienform")):
        env = base / "services/orion-durable-runs/.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("POSTGRES_URI="):
                    return line.split("=", 1)[1].strip().replace("@orion-athena-sql-db:5432", "@localhost:55432")
    raise SystemExit("no DSN: set ORION_PAGES_DSN")


def private_write(path: Path, text: str) -> None:
    os.umask(0o077)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def _row_from_db(r: dict[str, Any], *, state: Optional[str] = None) -> S.MemoryRow:
    d = dict(r)
    if state:
        d["confirmation_state"] = state
    return S.row_from_db(d)


def _item(item_id: str, source: str, label: str, rel: str, hard: bool, old: S.MemoryRow, new: S.MemoryRow,
          apply_rules: bool) -> EvalItem:
    return EvalItem(item_id=item_id, kind="real", source=source, label=label, expected_relation=rel,
                    hard_negative=hard, old=old, new=new, apply_rules=apply_rules)


def _ordered(a: S.MemoryRow, b: S.MemoryRow) -> tuple[S.MemoryRow, S.MemoryRow]:
    """(older, newer) by created_at."""
    ka = (a.created_at or datetime.min.replace(tzinfo=timezone.utc), a.memory_id)
    kb = (b.created_at or datetime.min.replace(tzinfo=timezone.utc), b.memory_id)
    return (a, b) if ka <= kb else (b, a)


async def cmd_build(args: argparse.Namespace) -> int:
    import psycopg
    from psycopg.rows import dict_row

    watch = {}
    if Path(args.manifest).exists():
        for w in json.loads(Path(args.manifest).read_text()).get("watched", []):
            watch[w["label"]] = w["memory_id"]
    checks = list(csv.DictReader(Path(args.checks).open())) if Path(args.checks).exists() else []
    items: list[EvalItem] = []
    async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row) as conn:
        await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")

        async def rows(ids: list[str]) -> dict[str, dict[str, Any]]:
            cur = await conn.execute(ROW_SQL, (list(ids),))
            return {r["memory_id"]: dict(r) for r in await cur.fetchall()}

        revs = await (await conn.execute(REVISIONS_SQL)).fetchall()
        got = await rows([r["old_id"] for r in revs] + [r["new_id"] for r in revs])
        for r in revs:
            if r["old_id"] in got and r["new_id"] in got:
                old = _row_from_db(got[r["old_id"]], state=r["prior"] or "auto")
                new = _row_from_db(got[r["new_id"]])
                items.append(_item(f"rev:{r['old_id'][:8]}", "confirmation_revise", "act", "correction", False,
                                   old, new, apply_rules=False))

        pairs = []
        for t in ("trip_row_1", "trip_row_2"):
            if t in watch and "home_correction" in watch:
                pairs.append((t, watch[t], watch["home_correction"], "watched_trip_vs_home"))
        for p in ("pet_row_1", "pet_row_2"):
            if p in watch and "pet_species_correction" in watch:
                pairs.append((p, watch[p], watch["pet_species_correction"], "watched_pet_rows"))
        if "pet_row_1" in watch and "pet_row_2" in watch:
            pairs.append(("pet_rows", watch["pet_row_1"], watch["pet_row_2"], "watched_pet_rows"))
        got = await rows([a for _, a, _, _ in pairs] + [b for _, _, b, _ in pairs])
        for label, a, b, source in pairs:
            if a in got and b in got:
                old, new = _ordered(_row_from_db(got[a]), _row_from_db(got[b]))
                items.append(_item(f"watch:{label}", source, "no_act", "compatible", True, old, new, apply_rules=True))

        vetoed = [c for c in checks if c["corrects"] == "True" and c["sure"] == "True" and c["accepted"] != "True"
                  and c["both_true"] == "True"]
        near = []
        for c in checks:
            if c["corrects"] != "False" or not c["dense"] or float(c["dense"]) < NEAR_DUP_COSINE:
                continue
            near.append(c)
        got = await rows([c["old_id"] for c in vetoed + near] + [c["new_id"] for c in vetoed + near])
        for c in vetoed:
            if c["old_id"] in got and c["new_id"] in got:
                items.append(_item(f"veto:{c['old_id'][:8]}:{c['new_id'][:8]}", "dry_run_veto", "no_act", "compatible",
                                   True, _row_from_db(got[c["old_id"]]), _row_from_db(got[c["new_id"]]),
                                   apply_rules=True))
        for c in near:
            if c["old_id"] not in got or c["new_id"] not in got:
                continue
            old, new = _row_from_db(got[c["old_id"]]), _row_from_db(got[c["new_id"]])
            ta, tb = S.tokens(old.statement), S.tokens(new.statement)
            if not ta or not tb or len(ta & tb) / len(ta | tb) < NEAR_DUP_JACCARD:
                continue
            items.append(_item(f"dup:{c['old_id'][:8]}:{c['new_id'][:8]}", "near_duplicate", "no_act", "compatible",
                               True, old, new, apply_rules=True))
    seen, out = set(), []
    for it in items:
        if it.item_id not in seen:
            seen.add(it.item_id)
            out.append(it)
    private_write(Path(args.out), "".join(json.dumps(item_to_dict(it), default=str) + "\n" for it in out))
    counts: dict[str, int] = {}
    for it in out:
        counts[f"{it.source}:{it.label}"] = counts.get(f"{it.source}:{it.label}", 0) + 1
    print(json.dumps({"written": str(args.out), "items": len(out), "by_source": counts}, indent=2))
    return 0


async def gateway_ask(client: Any, url: str, route: str, prompt: str, *, temperature: float, timeout: float) -> str:
    started, tries = time.monotonic(), 0
    while True:
        resp = await client.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
            "model": route, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200,
            "temperature": float(temperature), "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}})
        if resp.status_code == 503 and time.monotonic() - started < 1800:
            await asyncio.sleep(15.0)
            continue
        if resp.status_code >= 500 and tries < 3:  # a transient gateway fault: the eval retries it
            tries += 1
            await asyncio.sleep(5.0 * tries)
            continue
        break
    resp.raise_for_status()
    choice = (resp.json().get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") or ""


async def decide_v3(it: EvalItem, ask: Any, votes: int) -> Decision:
    if it.apply_rules:
        why = rule_refusal(it.old, it.new)
        if why:
            return Decision(it.item_id, False, stopped_by="rule", detail=why)
    c = await S.classify_pair(it.old, it.new, ask, votes=votes)
    if c.error:
        return Decision(it.item_id, False, c.relation, "error", c.abstain, [v.audit() for v in c.votes], c.error)
    acted = c.acts or c.juniper_acts
    return Decision(it.item_id, acted, c.relation, None if acted else "model",
                    "juniper_path" if c.juniper_acts else c.abstain, [v.audit() for v in c.votes])


async def decide_v2(it: EvalItem, ask: Any, votes: int) -> Decision:  # baseline only (the 11-17 check)
    if it.apply_rules:
        why = rule_refusal(it.old, it.new)
        if why:
            return Decision(it.item_id, False, stopped_by="rule", detail=why)
    run_check = getattr(S, "run_check", None)
    if run_check is None:
        raise SystemExit("checker v2 is retired in this tree; its baseline numbers are in the PR report")

    async def ask0(prompt: str) -> str:
        return await ask(prompt, temperature=0.0)

    v = await run_check(it.old, it.new, ask0)
    err = v.error or (v.both_true_reason if v.both_true_reason.startswith("call_failed") else None)
    if err:
        return Decision(it.item_id, False, None, "error", error=err)
    return Decision(it.item_id, v.accepted, "correction" if v.accepted else "compatible",
                    None if v.accepted else "model")


async def cmd_run(args: argparse.Namespace) -> int:
    import httpx

    if args.route.startswith(("fast", "quick", "metacog")):
        raise SystemExit("the supersede check never runs on the fast or metacog lanes")
    items = []
    if args.set in ("all", "synthetic"):
        items += load_items(SYNTHETIC_PATH)
    if args.set in ("all", "real"):
        items += load_items(Path(args.real))
    decide = {"v2": decide_v2, "v3": decide_v3}[args.checker]
    started = datetime.now(timezone.utc)
    decisions: list[Decision] = []
    calls = [0]
    async with httpx.AsyncClient() as client:
        async def ask(prompt: str, *, temperature: float = S.VOTE_TEMPERATURE) -> str:
            calls[0] += 1
            return await gateway_ask(client, args.gateway, args.route, prompt, temperature=temperature,
                                     timeout=args.timeout)

        for i, it in enumerate(items, 1):
            try:
                decisions.append(await decide(it, ask, args.votes))
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                decisions.append(Decision(it.item_id, False, stopped_by="error", error=type(exc).__name__))
            if i % 10 == 0:
                print(f"{i}/{len(items)} items, {calls[0]} calls", file=sys.stderr)
    result = score(items, decisions)
    check_version = S.CHECK_VERSION if args.checker == "v3" else "supersede_check.v2"
    run = {"run_id": str(uuid.uuid4()), "checker": args.checker, "check_version": check_version,
           "started_at": started.isoformat(timespec="seconds"),
           "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "route": args.route, "votes": args.votes, "calls": calls[0], "summary": result,
           "decisions": [decision_dict(d) for d in decisions]}
    out = RUNS_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{args.checker}.json"
    private_write(out, json.dumps(run, indent=2, default=str))
    public = {k: v for k, v in result.items() if k != "false_acts"}
    public["false_act_items"] = [f["item_id"] for f in result["false_acts"]]
    print(json.dumps({"checker": args.checker, "check_version": check_version, "calls": calls[0],
                      "file": str(out), **public}, indent=2, default=str))
    if args.record_db:
        if args.checker != "v3":
            raise SystemExit("only the current checker's runs are recorded")
        await record(args.dsn, run)
        print(f"recorded eval run {run['run_id']} passed={result['passed']}")
    return 0 if result["passed"] else 1


RECORD_SQL = """
INSERT INTO memory_supersede_eval_run (run_id, check_version, started_at, finished_at, passed, n_items, n_real,
    precision, recall, false_acts, hard_negative_false_acts, errors, route, votes, summary)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


async def record(dsn: str, run: dict[str, Any]) -> None:
    import psycopg
    from psycopg.types.json import Jsonb

    s = run["summary"]
    summary = {k: v for k, v in s.items() if k != "false_acts"}
    summary["false_act_items"] = [f["item_id"] for f in s["false_acts"]]
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(RECORD_SQL, (
            run["run_id"], run["check_version"], run["started_at"], run["finished_at"], s["passed"], s["n"], s["real"],
            s["precision"], s["recall"], s["fp"], s["hard_negative_false_acts"], s["errors"], run["route"],
            run["votes"], Jsonb(summary)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the real labelled set (read-only)")
    b.add_argument("--out", default=str(REAL_PATH))
    b.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    b.add_argument("--checks", default=str(DEFAULT_CHECKS))
    r = sub.add_parser("run", help="run the classifier over the labelled sets")
    r.add_argument("--checker", choices=("v2", "v3"), default="v3")
    r.add_argument("--set", choices=("all", "synthetic", "real"), default="all")
    r.add_argument("--real", default=str(REAL_PATH))
    r.add_argument("--votes", type=int, default=S.VOTES)
    r.add_argument("--gateway", default=os.environ.get("ORION_GATEWAY_URL", "http://localhost:8210"))
    r.add_argument("--route", default="memory_distill")
    r.add_argument("--timeout", type=float, default=180.0)
    r.add_argument("--record-db", action="store_true")
    args = ap.parse_args()
    args.dsn = args.dsn or default_dsn()
    return asyncio.run({"build": cmd_build, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    raise SystemExit(main())
