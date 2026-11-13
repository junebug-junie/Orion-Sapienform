#!/usr/bin/env python3
"""Mutation check for attention patch 4b's key guards (camera asks bid before they open, app/ask_bind.py).

Each mutant breaks one guard and the tests must FAIL on every one:

* held does not open (open_camera_ask; the held branch of ask_bind; the rolling no-deadline end);
* the backoff (a held candidate is not re-bid every tick);
* no duplicate want (the shadow loop skips a card that bid at open, stamped or decided after a client timeout; the
  stamp itself; no stamp without an answer);
* the per-minute bid budget;
* Juniper's own card never bids; ``shadow`` never bids;
* every opener goes through the bid site (a direct ``_insert_ask`` in the handoff opener);
* Prime's menu drops a funded want whose card never opened.

Files are rewritten IN PLACE and restored in a ``finally``; a SIGKILL mid-run would leave a mutant in the tree
(``git diff`` shows it). Needs a throwaway Postgres for the opener lifecycle test (it creates and drops its own
schema):

    ORION_SQL_WRITER_TEST_DATABASE_URI=postgresql://postgres:postgres@127.0.0.1:55447/postgres \\
        python services/orion-sql-writer/evals/ask_bind_mutations.py [--lane sql-writer|durable-runs|all]
"""
import os
import subprocess
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
SW = "services/orion-sql-writer"
PY = sys.executable
# --lane sql-writer (CI: orion-sql-writer-tests, needs the throwaway Postgres) | durable-runs (CI:
# orion-durable-runs-tests, Prime's menu; no database) | all (default).
LANE = sys.argv[sys.argv.index("--lane") + 1] if "--lane" in sys.argv else "all"
assert LANE in ("sql-writer", "durable-runs", "all"), LANE
if LANE != "durable-runs" and not os.environ.get("ORION_SQL_WRITER_TEST_DATABASE_URI"):
    sys.exit("set ORION_SQL_WRITER_TEST_DATABASE_URI (a throwaway Postgres): the opener lifecycle test needs it")
SW_TESTS = [f"{SW}/tests/test_ask_bind.py", f"{SW}/tests/test_ask_bind_integration.py",
            f"{SW}/tests/test_ask_event_wants.py"]
SW_ENV = f"{W}:{W}/{SW}:{W}/{SW}/tests"
DR_TESTS = ["services/orion-durable-runs/tests/test_event_wants_menu.py"]
DR_ENV = f"{W}:{W}/services/orion-durable-runs"

BIND = f"{SW}/app/ask_bind.py"
LOOP = f"{SW}/app/ask_event_wants.py"
MUTANTS = {
    "held opens anyway": (BIND, "    if not verdict.admit:\n        return None\n", ""),
    "held branch admits": (BIND, '    return BindVerdict(False, verdict.why, retry_at=retry_at, detail={**detail, "n_held": n})',
                           '    return BindVerdict(True, verdict.why, retry_at=retry_at, detail={**detail, "n_held": n})'),
    "a deadline returns (end = now)": (BIND, "verdict = bind_verdict(decision, now, now + NO_DEADLINE)",
                                       "verdict = bind_verdict(decision, now, now)"),
    "backoff ignored": (BIND, "    if held is not None and now < held.retry_at:", "    if False:"),
    "shadow loop posts a second want": (LOOP, '        if p.get("bid_want") is not None:', "        if False:"),
    "no stamp on a funded card": (BIND, '    if verdict.detail.get("market_answered") and verdict.detail.get("want_id"):',
                                  "    if False:"),
    "stamp without a market answer": (BIND, '    answered = decision is not None and str(getattr(decision, "reason", "") or "") not in NO_ANSWER_REASONS',
                                      "    answered = True"),
    "bid budget removed": (BIND, "        over = len(_bids) >= BID_BUDGET", "        over = False"),
    "shadow loop ignores a late-decided bid": (LOOP, """                    if f"orion_ask:{p['ask_id']}" in late:""",
                                               "                    if False:"),
    # Issue #3350 review: the writer's family share must be the market's family mean (same acts, same window).
    "grades windowed on close, not open": (LOOP, '''    "AND coalesce(opened_at, closed_at) >= :since AND coalesce(opened_at, closed_at) <= :now"''',
                                           '''    "AND closed_at >= :since AND CAST(:now AS timestamptz) IS NOT NULL"'''),
    "grades resolver-only": (LOOP, "AND outcome_source IN ('producer', 'resolver') ", "AND outcome_source = 'resolver' "),
    "juniper card bids": (BIND, "    if is_juniper_card(ask.evidence_refs, None):", "    if False:"),
    "shadow mode bids": (BIND, "    if mode() != MODE_ON:", "    if False:"),
    "handoff opener skips the bid": (f"{SW}/app/vision_handoff_loop.py",
                                     "        if open_camera_ask(ctx.conn, ask, now=ctx.now) is not None:",
                                     "        from app.vision_individuals import _insert_ask\n        if _insert_ask(ctx.conn, ask):"),
    "menu shows a card that never opened": ("services/orion-durable-runs/app/event_wants_menu.py",
                                            '            gone |= {f"{ASK_PREFIX}{i}" for i in ask_ids} - found', ""),
}


def run(tests, path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": path}
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], cwd=W, env=env,
                          capture_output=True, text=True)


def last(r):
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]


def lane_of(rel):
    return "durable-runs" if rel.startswith("services/orion-durable-runs") else "sql-writer"


results = {}
for lane, tests, path in (("sql-writer", SW_TESTS, SW_ENV), ("durable-runs", DR_TESTS, DR_ENV)):
    if LANE not in (lane, "all"):
        continue
    base = run(tests, path)
    print("baseline:", lane, last(base))
    assert base.returncode == 0, base.stdout[-2000:]
    assert "skipped" not in last(base), "the lifecycle test must run, not skip"
for name, (rel, a, b) in MUTANTS.items():
    if LANE not in (lane_of(rel), "all"):
        continue
    f = W / rel
    orig = f.read_text()
    try:
        assert orig.count(a) == 1, f"mutant {name!r} does not apply"
        f.write_text(orig.replace(a, b))
        tests, path = (DR_TESTS, DR_ENV) if lane_of(rel) == "durable-runs" else (SW_TESTS, SW_ENV)
        r = run(tests, path)
        results[name] = "KILLED" if r.returncode != 0 else "SURVIVED"
        print(f"{results[name]:9s} {name}  ({last(r)})")
    finally:
        f.write_text(orig)
assert results, "no mutant ran"
sys.exit(0 if all(v == "KILLED" for v in results.values()) else 1)
