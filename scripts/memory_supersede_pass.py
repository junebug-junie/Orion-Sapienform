#!/usr/bin/env python3
"""The one-time (re-runnable) supersede pass over episode_memory (issue #3587).

Finds active rows that a newer active row corrects, using the same logic the distiller runs at
write time (orion/memory/episode/supersede.py): deterministic candidates from the lived-memory
index, then one strict local-model check per pair on the memory_distill route.

Three commands. Every file goes to --dir (default /tmp/memory-supersede-pass/, mode 700, files
600). Statements are private: they live only there, never in git, a PR, or an issue.

    # 1. Dry run (writes NOTHING to the database): snapshot, progress log, pair list, report.
    .venv/bin/python scripts/memory_supersede_pass.py dry-run
    tail -f /tmp/memory-supersede-pass/progress.log

    # 2. Apply the pairs Juniper reviewed (every pair in pairs.jsonl, or --only/--exclude ids).
    .venv/bin/python scripts/memory_supersede_pass.py apply --pairs /tmp/memory-supersede-pass/pairs.jsonl --yes

    # 3. Undo one pair: the old row active again, the pointer cleared, an event logged.
    .venv/bin/python scripts/memory_supersede_pass.py undo --old <old memory_id> --new <new memory_id> \
        --actor juniper --note "<why>"

    # 4. Rewrite the manifest from the database (read-only; e.g. after an undo by hand SQL).
    .venv/bin/python scripts/memory_supersede_pass.py manifest

Manifest (ids and statuses only, never statements): --manifest, default
/mnt/orion-fcc/notes/memory_supersede_pass_manifest.json, so Orion can verify the pass ran. It lists
the proposed and applied pair ids and the final status of every --watch LABEL=MEMORY_ID row.

Idempotent: a pair is handled once its ``superseded`` event exists (event_id =
uuid5(SUPERSEDE_NAMESPACE, "superseded|<old>|<new>")); apply logs it as already_handled and writes
nothing, no second event.

Re-running dry-run is cheap: verdicts are cached in --dir/verdicts.jsonl, keyed by both ids, both
statements and the check version, so only new pairs reach the model. After an apply, a second
dry-run proposes nothing (superseded rows are no longer live) -- that is the idempotency check.
The July-September backfill (#3537) runs dry-run, review, apply as its last step.

DSN: $ORION_PAGES_DSN, else services/orion-durable-runs/.env POSTGRES_URI with the container host
swapped for the host port (localhost:55432).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
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

from orion.memory.episode.supersede import (  # noqa: E402
    ACTOR_PASS, CHECK_VERSION, Candidate, CheckRecord, Corpus, Link, MemoryRow, candidates_for, check_pairs,
    cosine, dedupe_links, load_corpus, undone_pairs,
)

DEFAULT_DIR = Path("/tmp/memory-supersede-pass")
SNAPSHOT_MAX_ROWS = 100_000
SNAPSHOT_SQL = "SELECT * FROM episode_memory WHERE status = 'active' ORDER BY created_at, memory_id"


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


def private_dir(path: Path) -> Path:
    os.umask(0o077)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_private(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Progress:
    def __init__(self, path: Path, total: int) -> None:
        self.path, self.total, self.started = path, max(1, total), time.monotonic()
        self.done = self.errors = 0
        write_private(path, "")

    def line(self, title: str, anomalies: str = "") -> None:
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total
        msg = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {title} | {pct:5.1f}% | "
               f"ETA {eta:,.0f}s | {self.done}/{self.total} checks | {rate:.2f}/s | errors {self.errors}"
               f"{' | ' + anomalies if anomalies else ''}\n")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg)


async def gateway_ask(client: Any, url: str, route: str, prompt: str, *, timeout: float,
                      temperature: float = 0.0, queue_wait: float = 1800.0) -> str:
    """One vote on the gateway's OpenAI passthrough (leased per call), JSON mode. 503 = seat busy: wait.
    A transient fault -- another 5xx, or a dropped connection -- is retried up to 3 times (5, 10,
    15 s); a timeout is not (it already spent the full budget). A read-only vote, so safe to repeat."""
    import httpx

    started, tries = time.monotonic(), 0
    while True:
        try:
            resp = await client.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
                "model": route, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200,
                "temperature": float(temperature), "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False}})
        except httpx.TimeoutException:
            raise
        except httpx.TransportError:
            if tries >= 3:
                raise
            tries += 1
            await asyncio.sleep(5.0 * tries)
            continue
        if resp.status_code == 503 and time.monotonic() - started < queue_wait:
            await asyncio.sleep(15.0)
            continue
        if resp.status_code >= 500 and tries < 3:  # a transient gateway fault: retried, not an error vote
            tries += 1
            await asyncio.sleep(5.0 * tries)
            continue
        break
    resp.raise_for_status()
    choice = (resp.json().get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") or ""


async def dense_matrix(chroma: str, embed_url: str, collection: str) -> Optional[dict[str, list[float]]]:
    """Every episode_memory vector in the lived-memory index, or None when it cannot answer."""
    import httpx

    from orion.introspect.semantic_index import SearchConfig, SearchUnavailableError, all_vectors
    from orion.memory.lived_memory import KIND_EPISODE, split_doc_id

    cfg = SearchConfig(chroma_url=chroma, embed_url=embed_url, collection=collection, min_similarity=0.0)
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            vecs = await all_vectors(client, cfg)
    except SearchUnavailableError as exc:
        print(f"dense unavailable ({exc}); candidates from shared words/referents only -- PROPOSALS ONLY: "
              "apply refuses pairs found this way (issue #3659)", file=sys.stderr)
        return None
    out = {}
    for doc, vec in vecs:
        parts = split_doc_id(doc)
        if parts and parts[0] == KIND_EPISODE:
            out[parts[1]] = vec
    return out


def dense_for(new: MemoryRow, corpus: Corpus, vecs: Optional[dict[str, list[float]]]) -> Optional[dict[str, float]]:
    if vecs is None:
        return None
    v = vecs.get(new.memory_id)
    if v is None:
        return None  # not indexed yet: fall back to shared words/referents for this row
    out = {}
    for old in corpus.rows:
        w = vecs.get(old.memory_id)
        if w is not None and old.memory_id != new.memory_id:
            sim = cosine(v, w)
            if sim is not None:
                out[old.memory_id] = sim
    return out


def cacheable(answer: str) -> bool:
    from orion.memory.episode.supersede import parse_vote

    return parse_vote(answer).error is None


def plan_candidates(corpus: Corpus, vecs: Optional[dict[str, list[float]]],
                    excluded: frozenset[tuple[str, str]] = frozenset()) -> list[tuple[MemoryRow, list[Candidate]]]:
    """(new row, its candidates) for every live row that has any."""
    out = []
    for new in corpus.rows:
        cands = candidates_for(new, corpus, dense=dense_for(new, corpus, vecs), excluded=excluded)
        if cands:
            out.append((new, cands))
    return out


async def cmd_dry_run(args: argparse.Namespace) -> int:
    import httpx
    import psycopg
    from psycopg.rows import dict_row

    if args.route.startswith(("fast", "quick", "metacog")):
        raise SystemExit("the supersede check never runs on the fast or metacog lanes")
    out = private_dir(Path(args.dir))
    async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row) as conn:
        await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        snap = await (await conn.execute(SNAPSHOT_SQL)).fetchall()
        if len(snap) > SNAPSHOT_MAX_ROWS:
            raise SystemExit(f"{len(snap)} active rows > {SNAPSHOT_MAX_ROWS}: stop and ask Juniper (CLAUDE.md s14)")
        write_private(out / "snapshot_active_rows.jsonl",
                      "".join(json.dumps(dict(r), default=str) + "\n" for r in snap))
        corpus = await load_corpus(conn)
        excluded = await undone_pairs(conn)
    vecs = None if args.no_dense else await dense_matrix(args.chroma, args.embed_url, args.collection)
    groups = plan_candidates(corpus, vecs, excluded)
    if args.max_checks:  # cap on candidate PAIRS, at whole rows: a row's candidates are never split
        capped, n = [], 0   # (a split list could fake Juniper's "one clear target"); the first row
                            # is always kept, so the cap can be exceeded by up to one row
        for new_row, cands in groups:
            if n and n + len(cands) > args.max_checks:
                break
            capped.append((new_row, cands))
            n += len(cands)
        groups = capped
    n_pairs = sum(len(c) for _, c in groups)
    by_id = {r.memory_id: r for r in corpus.rows}

    # Answer cache keyed by the exact prompt (both statements and dates; a prompt change is a new
    # key), the temperature, and the vote's place (vote 1, 2, 3 are separate samples): a re-run
    # only asks what it has not asked.
    # Failed calls are not cached.
    cache_path = out / "verdicts.jsonl"
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                cache[d["key"]] = d["answer"]
            except (ValueError, KeyError):
                continue
    progress = Progress(out / "progress.log", n_pairs)
    progress.line(f"start: {len(snap)} active rows snapshotted, {len(corpus.rows)} live, "
                  f"{n_pairs} candidate pairs, dense={'on' if vecs is not None else 'off'}, "
                  f"{len(cache)} cached answers")
    records: list[CheckRecord] = []
    links: list[Link] = []
    started = datetime.now(timezone.utc)
    async with httpx.AsyncClient() as client:
        with cache_path.open("a", encoding="utf-8") as cache_f:
            os.chmod(cache_path, 0o600)
            seen_prompt: dict[str, int] = {}

            async def ask(prompt: str, *, temperature: float = 0.0) -> str:
                base = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                n = seen_prompt.get(base, 0) + 1
                seen_prompt[base] = n
                key = f"{base}:{temperature}:{n}"
                if key in cache:
                    return cache[key]
                answer = await gateway_ask(client, args.gateway, args.route, prompt, timeout=args.timeout,
                                           temperature=temperature)
                if cacheable(answer):  # a malformed answer is asked again next run, never replayed
                    cache[key] = answer
                    cache_f.write(json.dumps({"key": key, "answer": answer}) + "\n")
                    cache_f.flush()
                return answer

            for new_row, cands in groups:
                found, recs = await check_pairs(new_row, cands, ask)
                progress.done += len(recs)
                progress.errors += sum(1 for r in recs if r.verdict.error)
                records.extend(recs)
                links.extend(found)
                if progress.done % 20 < len(recs) or progress.done >= n_pairs:
                    progress.line("checking", f"accepted so far {len(links)}")
    links = dedupe_links(links)
    progress.line(f"done: {len(links)} proposed pairs")
    dense_state = "on" if vecs is not None else "off"
    write_outputs(out, links, records, by_id, snap, vecs is not None)
    plan = {"links": [l.as_dict() for l in links], "held_links": [], "apply": "proposal_only",
            "refused_because": [] if vecs is not None else ["dense_off"], "asked": len(records),
            "check_errors": sum(1 for r in records if r.verdict.error), "dense": dense_state,
            "check_version": CHECK_VERSION, "decisions": [r.audit() for r in records]}
    run_id = f"pass_dry_run:{started.strftime('%Y%m%dT%H%M%SZ')}:{uuid.uuid4().hex[:6]}"
    if args.record_db:
        await record_db(args.dsn, run_id, "pass_dry_run", plan, [], started)
    if not args.no_manifest:
        async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row) as conn:
            await write_manifest(conn, Path(args.manifest), mode="dry_run", proposed=[l.as_dict() for l in links],
                                 applied=None, watch=parse_watch(args.watch) or None, run_id=run_id, dense=dense_state,
                                 records=records, counts={"asked": len(records), "proposed": len(links),
                                                          "check_errors": plan["check_errors"]},
                                 undone=await undone_events(conn))
    print(f"{len(links)} proposed pairs from {len(records)} checks; files in {out}")
    return 0


def write_outputs(out: Path, links: list[Link], records: list[CheckRecord], by_id: dict[str, MemoryRow],
                  snap: list[dict[str, Any]], dense_on: bool) -> None:
    pair_rows = []
    for l in links:
        old, new = by_id[l.old_id], by_id[l.new_id]
        # Issue #3659: each pair says whether its candidate came with a meaning-index cosine.
        pair_rows.append({**l.as_dict(), "dense_on": l.dense is not None, "old_statement": old.statement, "new_statement": new.statement,
                          "old_date": old.date, "new_date": new.date,
                          "old_sha": sha(old.statement), "new_sha": sha(new.statement),
                          "old_confirmation_state": old.confirmation_state})
    write_private(out / "pairs.jsonl", "".join(json.dumps(p) + "\n" for p in pair_rows))
    md = ["# Proposed supersede pairs (DRY RUN, nothing applied)", "",
          f"{len(pair_rows)} pairs from {len(records)} checks. Each: the OLD row would be marked "
          "superseded; the NEW row would point at it.", ""]
    for i, p in enumerate(pair_rows, 1):
        md += [f"## {i}. old `{p['old_id']}` -> new `{p['new_id']}`", "",
               f"- OLD ({p['old_date']}, {p['old_confirmation_state']}): {p['old_statement']}",
               f"- NEW ({p['new_date']}): {p['new_statement']}",
               f"- relation: {p['relation']} (path {p['path']}, votes {', '.join(p['votes'])})",
               f"- check's reason: {p['reason']}",
               f"- cosine {p['dense']}, shared: {', '.join(p['shared']) or '-'}", ""]
    write_private(out / "pairs.md", "\n".join(md))

    checks_path = out / "checks.csv"
    with checks_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_id", "new_id", "relation", "acts", "juniper_acts", "linked", "abstain", "error", "dense",
                    "rare_tokens", "rare_referents", "rank", "shared", "votes", "reasons"])
        for r in records:
            v, legs = r.verdict, (r.candidate.legs() if r.candidate else {})
            w.writerow([r.old_id, r.new_id, v.relation, v.acts, v.juniper_acts, r.linked, v.abstain, v.error or "",
                        r.dense, legs.get("rare_tokens"), legs.get("rare_referents"), legs.get("rank"),
                        " ".join(r.shared), " ".join(f"{x.relation}:{x.confidence:.2f}" for x in v.votes),
                        " | ".join(x.reason for x in v.votes)])
    os.chmod(checks_path, 0o600)

    superseded = {l.old_id: l.new_id for l in links}
    pointed = {l.new_id: l.old_id for l in reversed(links)}
    ba_path = out / "before_after.csv"
    with ba_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["memory_id", "before_status", "after_status_if_applied", "superseded_by",
                    "before_supersedes", "after_supersedes_if_applied"])
        for r in snap:
            mid = str(r["memory_id"])
            if mid not in superseded and mid not in pointed:
                continue
            before_ptr = str(r["supersedes_memory_id"]) if r.get("supersedes_memory_id") else ""
            w.writerow([mid, r["status"], "superseded" if mid in superseded else r["status"],
                        superseded.get(mid, ""), before_ptr, before_ptr or pointed.get(mid, "")])
    os.chmod(ba_path, 0o600)

    errors = sum(1 for r in records if r.verdict.error)
    by_rel: dict[str, int] = {}
    for r in records:
        by_rel[str(r.verdict.relation)] = by_rel.get(str(r.verdict.relation), 0) + 1
    abstained = sum(1 for r in records if r.verdict.relation in ("update", "correction") and not r.linked)
    by_link_rel: dict[str, int] = {}
    for l in links:
        by_link_rel[l.relation] = by_link_rel.get(l.relation, 0) + 1
    report = [
        "# Memory supersede pass: report (DRY RUN)", "",
        f"- Verdict: DRY RUN. {len(links)} pairs proposed; nothing written to the database.",
        f"- Active rows snapshotted: {len(snap)} (snapshot_active_rows.jsonl).",
        f"- Candidate pairs classified: {len(records)}; check errors: {errors}; "
        f"final relation per pair (majority vote): {json.dumps(by_rel, sort_keys=True)}; "
        f"update/correction majority but not unanimous+confident (abstained): {abstained}.",
        f"- Proposed by relation: {json.dumps(by_link_rel, sort_keys=True)} "
        f"(juniper path: {sum(1 for l in links if l.path == 'juniper')}).",
        f"- Dense candidates: {'on (orion_lived_memory vectors)' if dense_on else 'off (shared words/referents only)'}.",
        f"- Check version: {CHECK_VERSION}.", "",
        "## Files", "",
        "- pairs.md / pairs.jsonl: the proposed pairs, both statements, the check's reason.",
        "- checks.csv: every pair asked, its legs, votes and verdict.",
        "- before_after.csv: each row the apply would touch, before and after.",
        "- verdicts.jsonl: answer cache (a re-run only asks new pairs).",
        "- progress.log, snapshot_active_rows.jsonl.", "",
        "## Next", "",
        "Juniper reviews pairs.md. Then, from the primary checkout:", "",
        f"    .venv/bin/python scripts/memory_supersede_pass.py apply --pairs {out}/pairs.jsonl --yes", "",
        "(--exclude OLD_ID:NEW_ID,... drops pairs she rejects.) Undo one pair:", "",
        "    .venv/bin/python scripts/memory_supersede_pass.py undo --old <OLD_ID> --new <NEW_ID> "
        "--actor juniper --note \"<why>\"", "",
        f"Needs another pass: {'yes, after apply (expect 0 pairs: the idempotency check)' if links else 'no'}.",
    ]
    write_private(out / "report.md", "\n".join(report) + "\n")


# /mnt/orion-fcc is mounted into Orion's FCC turns; notes/ is the operator-writable part of it.
DEFAULT_MANIFEST = Path("/mnt/orion-fcc/notes/memory_supersede_pass_manifest.json")
WATCH_SQL = ("SELECT memory_id::text AS memory_id, status, confirmation_state, "
             "supersedes_memory_id::text AS supersedes_memory_id, "
             "(SELECT c.memory_id::text FROM episode_memory c WHERE c.supersedes_memory_id = m.memory_id "
             " ORDER BY c.created_at LIMIT 1) AS superseded_by "
             "FROM episode_memory m WHERE memory_id = ANY(%s::uuid[])")
VERIFY_SQL = ("SELECT e.memory_id AS old_id, e.evidence->>'superseded_by' AS new_id, m.status, e.actor, e.created_at "
              "FROM episode_memory_event e JOIN episode_memory m USING (memory_id) "
              "WHERE e.op = 'superseded' ORDER BY e.created_at")
# Every undone pair, who undid it, when, and the note they gave (the note is the undo event's
# evidence->>'note', supersede.py undo_link). An undone pair is never re-applied.
UNDONE_EVENTS_SQL = ("SELECT memory_id::text AS old_id, evidence->>'superseded_by' AS new_id, actor, "
                     "created_at, coalesce(evidence->>'note', '') AS note "
                     "FROM episode_memory_event WHERE op = 'supersede_undone' ORDER BY created_at, event_id")


async def undone_events(conn: Any) -> list[dict[str, Any]]:
    """The undone pairs for the manifest: ids, actor, time and note (no statements)."""
    cur = await conn.execute(UNDONE_EVENTS_SQL)
    rows = [r if isinstance(r, dict) else dict(zip([d.name for d in cur.description], r))
            for r in await cur.fetchall()]
    return [{"old_id": str(r["old_id"]), "new_id": str(r["new_id"]), "actor": r["actor"],
             "undone_at": str(r["created_at"]), "note": r["note"]} for r in rows]


def parse_watch(items: list[str]) -> list[tuple[str, str]]:
    """--watch LABEL=MEMORY_ID (repeatable): rows whose final status the manifest reports."""
    out = []
    for item in items or []:
        label, sep, mid = item.partition("=")
        if sep and mid.strip():
            out.append((label.strip(), mid.strip()))
    return out


def watched_trace(records: list[CheckRecord], memory_id: str) -> list[dict[str, Any]]:
    """Every pair this run asked that involves a watched row: which candidate legs fired and the
    votes (relation, confidence), without reasons or text (issue #3660)."""
    out = []
    for r in records:
        if memory_id not in (r.old_id, r.new_id):
            continue
        out.append({"old_id": r.old_id, "new_id": r.new_id, "side": "old" if r.old_id == memory_id else "new",
                    "legs": r.candidate.legs() if r.candidate else None, "relation": r.verdict.relation,
                    "acts": r.verdict.acts, "juniper_acts": r.verdict.juniper_acts, "abstain": r.verdict.abstain,
                    "linked": r.linked, "error": r.verdict.error,
                    "votes": [{"relation": v.relation, "same_attribute": v.same_attribute,
                               "confidence": round(v.confidence, 3)} for v in r.verdict.votes]})
    return out


def append_history(path: Path, line: dict[str, Any]) -> None:
    """One line per run, never rewritten (issue #3660). Ids and counts only."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        os.chmod(path, 0o644)
    except OSError as exc:
        print(f"run history NOT appended to {path}: {type(exc).__name__}: {exc}", file=sys.stderr)


def history_path(manifest: Path) -> Path:
    return manifest.with_name("memory_supersede_pass_runs.jsonl")


async def record_db(dsn: str, run_id: str, source: str, plan: dict[str, Any], applied: list[dict[str, Any]],
                    started: datetime) -> None:
    """The run's audit row in memory_supersede_run (the same table the distill writes). A missing
    table (migration not applied) is said loudly and does not fail the pass."""
    import psycopg

    from orion.memory.episode.supersede import record_run

    try:
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            await record_run(conn, run_id=run_id, source=source, plan=plan, applied=applied, started_at=started)
    except Exception as exc:  # noqa: BLE001
        print(f"run audit row NOT written: {type(exc).__name__}: {exc}", file=sys.stderr)


async def write_manifest(conn: Any, path: Path, *, mode: str, proposed: list[dict[str, Any]] | None,
                         applied: list[dict[str, Any]] | None, watch: list[tuple[str, str]] | None,
                         run_id: str = "", dense: str = "", records: Optional[list[CheckRecord]] = None,
                         counts: Optional[dict[str, Any]] = None,
                         undone: Optional[list[dict[str, Any]]] = None) -> None:
    """Ids and statuses only -- never statements -- so Orion can verify the pass ran.

    Written where Orion's FCC turns can read it (default under /mnt/orion-fcc/notes, mode 644). The
    database is the other proof: once orion_readonly has SELECT on episode_memory (PR #3593's grant)
    and episode_memory_event (scripts/sql/2026-11-17_grant_orion_readonly_supersede_events.sql),
    ``verify_sql`` below reads every applied pair directly.

    ``proposed`` / ``applied`` / ``watch`` / ``undone`` None keep the previous manifest's value (an
    undo or a refresh rewrites the file without a new proposal). ``undone`` lists every undone pair;
    an ``applied`` entry whose pair was undone carries ``undone_at``, so the file never says a pair
    is in effect after someone put it back."""
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            previous = {}
    applied_this_run = list(applied or [])  # the history line records only what this run did
    proposed_this_run = proposed or []
    # The check version that found the proposed pairs: an undo or a refresh proposes nothing new,
    # so it keeps the version of the run that did.
    check_version = CHECK_VERSION if proposed is not None else str(previous.get("check_version") or CHECK_VERSION)
    if proposed is None:
        proposed = list(previous.get("proposed") or [])
    dense = dense or str(previous.get("dense") or "")
    if applied is None:
        applied = [{k: v for k, v in a.items() if k != "undone_at"} for a in previous.get("applied") or []]
    if watch is None:
        watch = [(str(w.get("label")), str(w.get("memory_id"))) for w in previous.get("watched") or []
                 if w.get("memory_id")]
    if undone is None:
        undone = list(previous.get("undone") or [])
    undone_at = {(u["old_id"], u["new_id"]): u["undone_at"] for u in undone}
    applied_now = [{**a, **({"undone_at": undone_at[(a["old_id"], a["new_id"])]}
                            if (a["old_id"], a["new_id"]) in undone_at else {})} for a in applied]
    rows = {}
    if watch:
        cur = await conn.execute(WATCH_SQL, ([m for _, m in watch],))
        rows = {r["memory_id"]: dict(r) for r in await cur.fetchall()}
    proposed_by_old = {p["old_id"]: p["new_id"] for p in proposed}
    manifest = {
        "issue": 3587,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_mode": mode,
        "check_version": check_version,
        "pair_handled_key": "episode_memory_event row with op='superseded' and event_id = "
                            "uuid5(SUPERSEDE_NAMESPACE, 'superseded|<old_id>|<new_id>') "
                            "(orion/memory/episode/supersede.py superseded_event_id); "
                            "re-applying a handled pair writes nothing",
        "proposed": [{"old_id": p["old_id"], "new_id": p["new_id"]} for p in proposed],
        "applied": applied_now,
        "in_effect": [{"old_id": a["old_id"], "new_id": a["new_id"]} for a in applied_now if "undone_at" not in a],
        "undone": undone,
        "undone_rule": "an undone pair is never re-applied (supersede.py apply_link; issue #3629)",
        "run_id": run_id,
        "dense": dense,
        "history": str(history_path(path)),
        "watched": [{"label": label, "memory_id": mid, **{k: v for k, v in (rows.get(mid) or {"status": "missing"}).items()
                                                            if k != "memory_id"},
                     "proposed_superseded_by": proposed_by_old.get(mid),
                     **({"trace": watched_trace(records, mid)} if records is not None else {})}
                    for label, mid in watch],
        "verify_sql": VERIFY_SQL,
        "statements": "not here: they live only in the operator's private pass directory",
    }
    append_history(history_path(path), {
        "run_id": run_id, "mode": mode, "written_at": manifest["written_at"], "check_version": CHECK_VERSION,
        "dense": dense, "counts": counts or {},
        "proposed": [{"old_id": p["old_id"], "new_id": p["new_id"]} for p in proposed_this_run],
        "applied": [{"old_id": a["old_id"], "new_id": a["new_id"]} for a in applied_this_run]})
    text = json.dumps(manifest, indent=2, default=str) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        os.chmod(path, 0o644)
    except OSError as exc:  # the pass's own results stand; say loudly where the manifest did not go
        print(f"manifest NOT written to {path}: {type(exc).__name__}: {exc}", file=sys.stderr)


def found_with_dense(pair: dict[str, Any]) -> bool:
    """True when the pair was found with a meaning-index cosine. Files from before #3659 carry no
    ``dense_on``: their ``dense`` value decides (None = found by words alone)."""
    if "dense_on" in pair:
        return bool(pair["dense_on"])
    return pair.get("dense") is not None


def _pair_set(spec: str) -> set[tuple[str, str]]:
    out = set()
    for item in (spec or "").split(","):
        if ":" in item:
            a, b = item.strip().split(":", 1)
            out.add((a.strip(), b.strip()))
    return out


def pair_decision(pair: dict[str, Any], outcome: str) -> dict[str, Any]:
    """One reviewed pair's decision trace for the apply run's ``memory_supersede_run`` row: the
    candidate legs (cosine, shared rare words/referents), the votes, the reason and what apply did.
    No statements: the shared words are the same ones a ``superseded`` event's evidence keeps."""
    return {"old_id": pair["old_id"], "new_id": pair["new_id"],
            "legs": {"cosine": pair.get("dense"), "shared": list(pair.get("shared") or [])},
            "relation": pair.get("relation"), "path": pair.get("path"), "votes": list(pair.get("votes") or []),
            "reason": str(pair.get("reason") or "")[:400], "check_version": pair.get("check_version"),
            "outcome": outcome}


async def cmd_apply(args: argparse.Namespace) -> int:
    import psycopg
    from psycopg.rows import dict_row

    from orion.memory.episode.supersede import apply_link

    if not args.yes:
        raise SystemExit("apply writes to the live database: re-run with --yes after Juniper reviewed the pairs")
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not args.allow_no_eval:
        # Same gate as write time (issue #3659): no apply without a passing precision eval for this
        # check version, unless the operator says so on purpose for this run.
        import psycopg
        from psycopg.rows import dict_row

        from orion.memory.episode.supersede import eval_gate

        async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row, autocommit=True) as gc:
            open_, why = await eval_gate(gc)
        if not open_:
            raise SystemExit(f"eval gate closed ({why}): run scripts/memory_supersede_eval.py run --record-db first, "
                             "or pass --allow-no-eval after reviewing every pair by hand.")
    only, exclude = _pair_set(args.only), _pair_set(args.exclude)
    pairs = [p for p in pairs if not ((only and (p["old_id"], p["new_id"]) not in only)
                                      or (p["old_id"], p["new_id"]) in exclude)]
    stale = [p for p in pairs if str(p.get("check_version") or "") != CHECK_VERSION]
    if stale:
        raise SystemExit(f"{len(stale)} pair(s) in {args.pairs} came from another check version (current "
                         f"{CHECK_VERSION}); re-run dry-run and review again.")
    no_dense = [p for p in pairs if not found_with_dense(p)]
    if no_dense and not args.allow_no_dense:
        # Issue #3659: a pair found without the meaning index (word overlap only) is never applied
        # unless someone says so on purpose for this run (or --exclude drops it).
        raise SystemExit(f"{len(no_dense)} pair(s) in {args.pairs} were found without the meaning index "
                         "(dense off); refusing. Re-run dry-run with the index up, --exclude them, or pass "
                         "--allow-no-dense after reviewing those pairs by hand.")
    out = private_dir(Path(args.pairs).parent)
    log = out / "applied.log"
    from orion.memory.episode.supersede import superseded_event_id

    applied = skipped = 0
    applied_ids: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row, autocommit=True) as conn:
        undone = await undone_pairs(conn)
        for p in pairs:
            key = (p["old_id"], p["new_id"])
            if (only and key not in only) or key in exclude:
                continue
            if key in undone:  # someone put this pair back: never re-applied, never reported applied
                decisions.append(pair_decision(p, "undone_never_reapplied"))
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} undone_never_reapplied "
                            f"old={key[0]} new={key[1]}\n")
                skipped += 1
                continue
            cur = await conn.execute("SELECT created_at FROM episode_memory_event WHERE event_id = %s",
                                     (superseded_event_id(*key),))
            done = await cur.fetchone()
            if done:  # already handled: a no-op, no second event
                applied_ids.append({"old_id": key[0], "new_id": key[1], "applied_at": str(done["created_at"])})
                decisions.append(pair_decision(p, "already_handled"))
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} already_handled old={key[0]} new={key[1]}\n")
                skipped += 1
                continue
            async with conn.transaction():
                cur = await conn.execute("SELECT memory_id::text AS id, statement FROM episode_memory "
                                         "WHERE memory_id = ANY(%s::uuid[])", ([p["old_id"], p["new_id"]],))
                now_text = {r["id"]: " ".join(str(r["statement"]).split()) for r in await cur.fetchall()}
                if sha(now_text.get(p["old_id"], "")) != p["old_sha"] or sha(now_text.get(p["new_id"], "")) != p["new_sha"]:
                    result, why = None, "statement_changed_since_review"
                else:
                    result = await apply_link(conn, Link.from_dict(p), actor=ACTOR_PASS)
                    why = "applied" if result else "skipped_state_changed"
            applied += bool(result)
            skipped += not result
            decisions.append(pair_decision(p, why))
            if result:
                applied_ids.append({"old_id": key[0], "new_id": key[1],
                                    "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} {why} old={p['old_id']} new={p['new_id']}\n")
            os.chmod(log, 0o600)
        run_id = f"pass_apply:{started.strftime('%Y%m%dT%H%M%SZ')}:{uuid.uuid4().hex[:6]}"
        if not args.no_manifest:
            await write_manifest(conn, Path(args.manifest), mode="apply",
                                 proposed=[{"old_id": p["old_id"], "new_id": p["new_id"]} for p in pairs],
                                 applied=applied_ids, watch=parse_watch(args.watch) or None, run_id=run_id,
                                 dense="on", counts={"applied": applied, "skipped": skipped},
                                 undone=await undone_events(conn))
    await record_db(args.dsn, run_id, "pass_apply",
                    {"links": [{"old_id": p["old_id"], "new_id": p["new_id"]} for p in pairs],
                     "apply": "on", "dense": "on", "check_version": CHECK_VERSION, "decisions": decisions},
                    applied_ids, started)
    print(f"applied {applied}, skipped {skipped}; log {log}")
    return 0


async def cmd_undo(args: argparse.Namespace) -> int:
    import psycopg
    from psycopg.rows import dict_row

    from orion.memory.episode.supersede import undo_link

    note = " ".join(str(args.note or "").split())
    if not note:
        raise SystemExit("undo needs --note: why the pair is put back (stored in the undo event; Orion reads it)")
    started = datetime.now(timezone.utc)
    async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row, autocommit=True) as conn:
        async with conn.transaction():
            restored = await undo_link(conn, old_id=args.old, new_id=args.new, actor=args.actor, note=note)
        if restored is None:
            print("nothing to undo: that pair is not superseded by this mechanism", file=sys.stderr)
            return 1
        if not args.no_manifest:  # the manifest must not keep saying the pair is applied
            await write_manifest(conn, Path(args.manifest), mode="undo", proposed=None, applied=None, watch=None,
                                 run_id=f"pass_undo:{started.strftime('%Y%m%dT%H%M%SZ')}:{uuid.uuid4().hex[:6]}",
                                 counts={"undone": 1, "old_id": args.old, "new_id": args.new, "actor": args.actor},
                                 undone=await undone_events(conn))
    print(f"restored old={args.old} to active ({restored['restored_confirmation_state']}); "
          f"pointer on new={args.new} cleared")
    return 0


async def cmd_manifest(args: argparse.Namespace) -> int:
    """Rewrite the manifest from the database (read-only): every undo, and which applied pairs are
    still in effect. For a manifest written before undo kept it current."""
    import psycopg
    from psycopg.rows import dict_row

    started = datetime.now(timezone.utc)
    async with await psycopg.AsyncConnection.connect(args.dsn, row_factory=dict_row) as conn:
        await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        await write_manifest(conn, Path(args.manifest), mode="refresh", proposed=None, applied=None,
                             watch=parse_watch(args.watch) or None,
                             run_id=f"pass_refresh:{started.strftime('%Y%m%dT%H%M%SZ')}:{uuid.uuid4().hex[:6]}",
                             undone=await undone_events(conn))
    print(f"manifest rewritten: {args.manifest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="propose pairs; writes nothing to the database")
    d.add_argument("--dir", default=str(DEFAULT_DIR))
    d.add_argument("--gateway", default=os.environ.get("ORION_GATEWAY_URL", "http://localhost:8210"))
    d.add_argument("--route", default="memory_distill")
    d.add_argument("--timeout", type=float, default=180.0)
    d.add_argument("--chroma", default=os.environ.get("ORION_LIVED_CHROMA_URL", "http://localhost:8500"))
    d.add_argument("--embed-url", default=os.environ.get("ORION_LIVED_EMBED_URL", "http://localhost:8320/embedding"))
    d.add_argument("--collection", default="orion_lived_memory")
    d.add_argument("--no-dense", action="store_true")
    d.add_argument("--max-checks", type=int, default=0, help="stop after this many pairs (0 = all)")
    d.add_argument("--record-db", action="store_true",
                   help="also write the run's audit row to memory_supersede_run (otherwise dry-run writes "
                        "nothing to the database)")
    a = sub.add_parser("apply", help="apply reviewed pairs")
    a.add_argument("--pairs", required=True)
    a.add_argument("--only", default="", help="OLD:NEW,... apply only these")
    a.add_argument("--exclude", default="", help="OLD:NEW,... skip these")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--allow-no-eval", action="store_true",
                   help="apply although no passing precision eval is recorded (refused by default, issue #3659)")
    a.add_argument("--allow-no-dense", action="store_true",
                   help="apply pairs found without the meaning index (refused by default, issue #3659)")
    u = sub.add_parser("undo", help="put one pair back")
    u.add_argument("--old", required=True)
    u.add_argument("--new", required=True)
    u.add_argument("--actor", required=True,
                   help="who decided: 'juniper' when it is her rejection, else 'operator' (issue #3629: "
                        "only an operator undo may be reopened)")
    u.add_argument("--note", required=True, help="why the pair is put back (stored in the undo event)")
    m = sub.add_parser("manifest", help="rewrite the manifest from the database (read-only)")
    for sp in (d, a, u, m):
        sp.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="ids-only manifest Orion can read (no statements)")
        sp.add_argument("--no-manifest", action="store_true")
        sp.add_argument("--watch", action="append", default=[], metavar="LABEL=MEMORY_ID",
                        help="report this row's final status in the manifest (repeatable)")
    args = ap.parse_args()
    args.dsn = args.dsn or default_dsn()
    return asyncio.run({"dry-run": cmd_dry_run, "apply": cmd_apply, "undo": cmd_undo, "manifest": cmd_manifest}[args.cmd](args))


if __name__ == "__main__":
    raise SystemExit(main())
