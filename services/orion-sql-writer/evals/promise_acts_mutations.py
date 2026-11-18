#!/usr/bin/env python3
"""Mutation check for the promise ledger's shadow and honesty guards (intention ledger D2, 2026-11-16).

Each mutant breaks one guard, and the tests must FAIL on every one:

* shadow: a promise act graded into substrate_action_outcomes / the posterior; a void close graded;
  promise acts given a cost; the self-monitor building a bet from a shadow act;
* void vs expired: the resolver's "did Juniper write in the horizon" check removed or inverted;
* honesty: a kept mark accepted without its delivering quote in the reply, or one that only repeats the promise; an offer opening an act
  before Juniper accepts it; an item without force/done_when opening an act;
* settle classes / stable refs (#3595, #3596): a conditional promise never closing not-due, the promising turn's
  own condition mark counting, a run promise graded 0, a re-publish replacing the stored promise list.

Files are rewritten IN PLACE and restored in a ``finally``; a SIGKILL mid-run would leave a mutant in the
tree (``git diff`` shows it). Needs a throwaway Postgres for the lifecycle tests:

    ORION_SQL_WRITER_TEST_DATABASE_URI=postgresql://postgres:postgres@127.0.0.1:55499/conjourney \\
        python services/orion-sql-writer/evals/promise_acts_mutations.py
"""
import os
import subprocess
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
PY = sys.executable
if not os.environ.get("ORION_SQL_WRITER_TEST_DATABASE_URI"):
    sys.exit("set ORION_SQL_WRITER_TEST_DATABASE_URI (a throwaway Postgres): the lifecycle mutants need it")
TESTS = ["services/orion-sql-writer/tests/test_promise_acts_integration.py",
         "orion/grammar/tests/test_promise_acts.py",
         "orion/harness/tests/test_promise_gate.py",
         "orion/self_monitor/tests/test_self_monitor_promise_shadow.py"]
LEDGER = "orion/grammar/act_ledger.py"
ACTS = "orion/grammar/promise_acts.py"
GATE = "orion/harness/promise_gate.py"
ADAPTERS = "orion/self_monitor/adapters.py"
WRITER = "services/orion-sql-writer/app/harness_turn_trace_persist.py"
MUTANTS = {
    "shadow signal graded": (LEDGER, "or is_shadow_signal(exp.signal_id) or is_void_note(outcome.note)",
                             "or is_void_note(outcome.note)"),
    "shadow set emptied": (ACTS, 'SHADOW_ACT_SIGNALS: frozenset[str] = frozenset({PROMISE_SIGNAL})',
                           'SHADOW_ACT_SIGNALS: frozenset[str] = frozenset()'),
    "promise act gets a cost": (LEDGER, "cost = None if (exp is not None and exp.signal_id == PROMISE_SIGNAL) else",
                                "cost = None if False else"),
    "void is never void": (LEDGER, "                if wrote is None:\n", "                if False:\n"),
    "void and expired swapped": (LEDGER, "                if wrote is None:\n", "                if wrote is not None:\n"),
    "self-monitor bets on shadow acts": (ADAPTERS, "if is_void_note(row.get(\"outcome_note\")) or is_shadow_signal(row.get(\"signal_id\")):",
                                         "if False:"),
    "kept without delivering quote": (GATE, "if len(norm_quote) > PROMISE_QUOTE_MAX_CHARS or not quote_in_text(quote, reply_text):",
                                      "if False:"),
    "re-promise counts as kept": (GATE, "if own and (norm_quote in own or own in norm_quote):", "if False:"),
    "offer opens before acceptance": (ACTS, "        if promise.offer:\n            plan.offers_not_opened += 1\n            continue\n", ""),
    "old row without force opens": (ACTS, "        if not is_firm_ledger_promise(promise):\n            plan.skipped_no_force += 1\n            continue\n", ""),
    # Settle classes and stable refs (#3595, #3596, 2026-11-18).
    "conditional never not-due": (ACTS, "    if promise_condition(item) is not None and not condition_met:",
                                  "    if False:"),
    "condition-met counts the promising turn": (LEDGER, "       AND t.correlation_id <> coalesce(:correlation_id, '')\n", ""),
    "run promise graded 0": (ACTS, '    if cls == "run":\n        return VOID_NOT_CHAT_NOTE\n', ""),
    "re-publish replaces the promise list": (WRITER, "    if column == \"run_artifact\":\n        set_[\"run_artifact\"] = literal_column(RUN_ARTIFACT_ON_CONFLICT_SQL)\n", ""),
}
originals = {rel: (W / rel).read_text() for rel in {f for f, _, _ in MUTANTS.values()}}
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONPATH": f"{W}:{W}/services/orion-sql-writer:{W}/services/orion-sql-writer/tests"}


def run() -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "integration or not integration",
                           *TESTS], cwd=W, env=env, capture_output=True, text=True)


results = {}
try:
    base = run()
    print("baseline:", (base.stdout.strip().splitlines() or ["?"])[-1])
    assert base.returncode == 0, base.stdout[-2000:]
    for name, (rel, a, b) in MUTANTS.items():
        orig = originals[rel]
        assert orig.count(a) == 1, f"mutant {name!r} does not apply ({orig.count(a)} matches)"
        (W / rel).write_text(orig.replace(a, b))
        try:
            results[name] = run().returncode != 0
        finally:
            (W / rel).write_text(orig)
finally:
    for rel, text in originals.items():
        (W / rel).write_text(text)
for name, caught in results.items():
    print(("CAUGHT  " if caught else "SURVIVED") + "  " + name)
sys.exit(0 if results and all(results.values()) else 1)
