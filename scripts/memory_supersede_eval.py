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
- hard negatives: pairs someone undid after an apply (``supersede_undone``: Juniper's own
  rejection); the watched rows in the supersede manifest (trip rows against the home
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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.memory.episode import supersede as S  # noqa: E402
from orion.memory.episode.supersede_eval import (  # noqa: E402
    HELDOUT_PATH, SYNTHETIC_PATH, Decision, EvalItem, decision_dict, item_to_dict, load_items, rule_refusal, score,
)

EVAL_DIR = Path(os.path.expanduser("~/.orion/evals"))
REAL_PATH = EVAL_DIR / "memory_supersede_labelled.jsonl"
RUNS_DIR = EVAL_DIR / "memory_supersede_eval_runs"
DEFAULT_MANIFEST = Path("/mnt/orion-fcc/notes/memory_supersede_pass_manifest.json")
# The 2026-11-17 v2 dry run's per-pair verdicts, frozen host-side (a v3 dry run writes a different
# checks.csv format): the source of the dry-run-veto and near-duplicate hard negatives.
DEFAULT_CHECKS = EVAL_DIR / "supersede_v2_checks_2026-11-17.csv"
PRODUCTION_ROUTE = S.EVAL_ROUTE
MIN_RECORD_REPEATS = 2
SEEN_MODELS: set[str] = set()   # model ids the gateway reported for the route during this run
# The rows #3587/#3660 watch (ids only, already public in those issues). The live manifest is
# rewritten by every pass run and may carry no --watch list, so the set never depends on it.
WATCHED_DEFAULT = {
    "trip_row_1": "4df8e60e-4c07-5685-8b1a-ef6239e6795f",
    "trip_row_2": "ac963bc8-2f8d-5b4a-8229-e4e6a83395e3",
    "home_correction": "c0dc86c8-26b6-5468-9928-be9376394b59",
    "pet_row_1": "8b51d185-8c9e-5a64-96d8-e9ad83e16fa5",
    "pet_row_2": "da376c17-529e-5dd1-a31d-b7d863ee5291",
    "pet_species_correction": "357bf161-f706-57c6-910b-4ad88c5c7910",
}
NEAR_DUP_COSINE = 0.85
NEAR_DUP_JACCARD = 0.25  # hand-checked 2026-11-18: the 6 pairs this admits are different events on one topic

ROW_SQL = """
SELECT m.memory_id::text AS memory_id, m.statement, m.purpose, m.created_at, m.occurred_at, m.episode_id,
       m.confirmation_state, m.status, m.voice,
       (m.voice = 'juniper_said' AND EXISTS (SELECT 1 FROM episode_memory_evidence e
            WHERE e.memory_id = m.memory_id AND e.source_kind = 'chat_prompt' AND e.verified)) AS juniper_quoted
FROM episode_memory m WHERE m.memory_id = ANY(%s::uuid[])
"""
UNDONE_SQL = """
SELECT DISTINCT ON (memory_id, evidence->>'superseded_by') memory_id::text AS old_id,
       evidence->>'superseded_by' AS new_id, actor
FROM episode_memory_event WHERE op = 'supersede_undone' AND evidence->>'superseded_by' IS NOT NULL
ORDER BY memory_id, evidence->>'superseded_by', created_at
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

    watch = dict(WATCHED_DEFAULT)   # ids only; a manifest's own watch list overrides
    if Path(args.manifest).exists():
        for w in json.loads(Path(args.manifest).read_text()).get("watched", []) or []:
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
                items.append(_item(f"rev:{r['old_id'][:8]}:{r['new_id'][:8]}", "confirmation_revise", "act", "correction", False,
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

        checks = [c for c in checks if "corrects" in c and "both_true" in c]   # v2-format rows only
        # Pairs someone put back after an apply (op supersede_undone). An undo is a human saying the
        # link was wrong (2026-11-18: both v3.1 proposals, rejected by Juniper and undone by the
        # operator). The undoing actor stays in the source so a test-rollback can be audited out.
        undone = await (await conn.execute(UNDONE_SQL)).fetchall()
        got = await rows([u["old_id"] for u in undone] + [u["new_id"] for u in undone])
        for u in undone:
            if u["old_id"] in got and u["new_id"] in got:
                items.append(_item(f"undone:{u['old_id'][:8]}:{u['new_id'][:8]}", f"undone_by_{u['actor']}", "no_act",
                                   "compatible", True, _row_from_db(got[u["old_id"]]), _row_from_db(got[u["new_id"]]),
                                   apply_rules=False))

        vetoed = [c for c in checks if c.get("corrects") == "True" and c.get("sure") == "True"
                  and c.get("accepted") != "True" and c.get("both_true") == "True"]
        near = [c for c in checks if c.get("corrects") == "False" and c.get("dense")
                and float(c["dense"]) >= NEAR_DUP_COSINE]
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
    for it in items:  # one item per pair (the first source wins: Juniper's own undo before a veto)
        key = (it.old.memory_id, it.new.memory_id)
        if key not in seen:
            seen.add(key)
            out.append(it)
    if len({it.item_id for it in out}) != len(out):
        raise SystemExit("duplicate item ids in the labelled set: scoring would overwrite decisions")
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
    body = resp.json()
    if body.get("model"):
        SEEN_MODELS.add(str(body["model"]))
    choice = (body.get("choices") or [{}])[0]
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
    if c.abstain == S.NUMBERED_ITEM_NOT_IN_B:   # the deterministic rule stopped it, not the model
        return Decision(it.item_id, False, None, "rule", c.abstain)
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
    if args.set in ("all", "synthetic", "heldout"):
        items += load_items(HELDOUT_PATH)
    if args.set in ("all", "real"):
        items += load_items(Path(args.real))
    # Votes are sampled above temperature 0, so one pass is noisy (2026-11-18: a held-out negative
    # that passed at 00:44 acted at 05:25). Every item runs --repeats times, scored pooled: one
    # false act in any repeat fails the run.
    if args.repeats > 1:
        items = [replace(it, item_id=f"{it.item_id}@{r}") for r in range(args.repeats) for it in items]
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
           "route": args.route, "votes": args.votes, "repeats": args.repeats, "calls": calls[0], "model": ",".join(sorted(SEEN_MODELS)),
           "summary": result,
           "decisions": [decision_dict(d) for d in decisions]}
    out = RUNS_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{args.checker}.json"
    private_write(out, json.dumps(run, indent=2, default=str))
    public = {k: v for k, v in result.items() if k != "false_acts"}
    public["false_act_items"] = [f["item_id"] for f in result["false_acts"]]
    print(json.dumps({"checker": args.checker, "check_version": check_version, "calls": calls[0],
                      "model": run["model"],
                      "file": str(out), **public}, indent=2, default=str))
    if args.record_db:
        record_guard(args)
        # Only a run that matches production counts for the gate: the current checker, the votes
        # write time uses, and the production route.
        await record(args.dsn, run)
        print(f"recorded eval run {run['run_id']} passed={result['passed']}")
    return 0 if result["passed"] else 1


def record_guard(args: argparse.Namespace) -> None:
    """Only a run that matches production counts for the gate: the current checker, the votes
    write time uses, the production route, the full set, and at least MIN_RECORD_REPEATS passes."""
    if (args.checker != "v3" or args.votes != S.VOTES or args.route != PRODUCTION_ROUTE or args.set != "all"
            or args.repeats < MIN_RECORD_REPEATS):
        raise SystemExit(f"only a full v3 run with --votes {S.VOTES} and --repeats >= {MIN_RECORD_REPEATS} "
                         f"on {PRODUCTION_ROUTE} is recorded")


RECORD_SQL = """
INSERT INTO memory_supersede_eval_run (run_id, check_version, started_at, finished_at, passed, n_items, n_real,
    precision, recall, false_acts, hard_negative_false_acts, errors, route, votes, model, summary)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


async def record(dsn: str, run: dict[str, Any]) -> None:
    import psycopg
    from psycopg.types.json import Jsonb

    s = run["summary"]
    summary = {k: v for k, v in s.items() if k != "false_acts"}
    summary["repeats"] = run.get("repeats")
    summary["false_act_items"] = [f["item_id"] for f in s["false_acts"]]
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(RECORD_SQL, (
            run["run_id"], run["check_version"], run["started_at"], run["finished_at"], s["passed"], s["n"], s["real"],
            s["precision"], s["recall"], s["fp"], s["hard_negative_false_acts"], s["errors"], run["route"],
            run["votes"], run.get("model") or None, Jsonb(summary)))


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
    r.add_argument("--set", choices=("all", "synthetic", "heldout", "real"), default="all")
    r.add_argument("--real", default=str(REAL_PATH))
    r.add_argument("--votes", type=int, default=S.VOTES)
    r.add_argument("--repeats", type=int, default=MIN_RECORD_REPEATS,
                   help="run every item this many times (sampled votes vary); any false act fails")
    r.add_argument("--gateway", default=os.environ.get("ORION_GATEWAY_URL", "http://localhost:8210"))
    r.add_argument("--route", default=PRODUCTION_ROUTE)
    r.add_argument("--timeout", type=float, default=180.0)
    r.add_argument("--record-db", action="store_true")
    args = ap.parse_args()
    args.dsn = args.dsn or default_dsn()
    return asyncio.run({"build": cmd_build, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    raise SystemExit(main())
