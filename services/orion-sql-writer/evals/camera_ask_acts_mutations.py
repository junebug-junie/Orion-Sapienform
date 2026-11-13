#!/usr/bin/env python3
"""Mutation check for attention patch 4a's key guards (orion/grammar/camera_ask_acts.py).

Each mutant breaks one guard -- Juniper's own story cards excluded (both clauses of the SQL filter and of
the Python twin), the grade mapping (answered 1, dismissed / don't know 0, silence at expiry a 0 grade,
withdrawn not a grade) and the Laplace prior -- and the tests must FAIL on every one. The file is rewritten IN
PLACE and restored in a ``finally``; a SIGKILL mid-run would leave a mutant in the tree (``git diff`` shows it).

Needs a throwaway Postgres for the lifecycle tests (they create and drop their own schema):

    ORION_SQL_WRITER_TEST_DATABASE_URI=postgresql://postgres:t@127.0.0.1:55613/postgres \
        python services/orion-sql-writer/evals/camera_ask_acts_mutations.py
"""
import os
import subprocess
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
F = W / "orion/grammar/camera_ask_acts.py"
PY = sys.executable
if not os.environ.get("ORION_SQL_WRITER_TEST_DATABASE_URI"):
    sys.exit("set ORION_SQL_WRITER_TEST_DATABASE_URI (a throwaway Postgres): the SQL mutants need the lifecycle tests")
TESTS = ["services/orion-sql-writer/tests/test_camera_ask_acts.py",
         "services/orion-sql-writer/tests/test_camera_ask_acts_integration.py",
         "services/orion-sql-writer/tests/test_ask_event_wants.py"]
MUTANTS = {
    "sql: drop marker exclusion": ('''    "AND NOT coalesce(a.evidence_refs @> '[\\"" + JUNIPER_STORY_CARD_REF + "\\"]'::jsonb, false) "\n''', ""),
    "sql: drop legacy story_id exclusion": ('''    "AND NOT coalesce(a.answer_detail ? 'story_id', false)"''', '''    "AND true"'''),
    "py: juniper card ignores marker": ('return JUNIPER_STORY_CARD_REF in _as_list(evidence_refs) or ', 'return '),
    "py: juniper card ignores story_id": (' or "story_id" in _as_dict(answer_detail)', ''),
    "grade: answered -> 0": ('return AskGrade(1.0, "resolver", answered_at or now, "juniper answered")',
                             'return AskGrade(0.0, "resolver", answered_at or now, "juniper answered")'),
    "grade: dismissed -> 1": ('''        return AskGrade(0.0, "resolver", answered_at or now,\n                        "juniper said''',
                              '''        return AskGrade(1.0, "resolver", answered_at or now,\n                        "juniper said'''),
    "grade: expiry not graded": ('''        return AskGrade(0.0, "resolver", expires_at or now, "unanswered at expiry")''',
                                 '''        return AskGrade(0.0, "expired", expires_at or now, "unanswered at expiry")'''),
    "grade: open past expiry stays open": ('''        if expires_at is not None and now >= expires_at:\n            return AskGrade(0.0, "resolver", expires_at, "unanswered at expiry")\n''', ""),
    "grade: withdrawn graded": ('''            return AskGrade(0.0, "expired", now, WITHDRAWN_NOTE + " before expiry")''',
                                '''            return AskGrade(0.0, "resolver", now, "withdrawn by orion before expiry")'''),
    "grade: stamped withdrawal graded": ('''            return AskGrade(0.0, "expired", answered_at, WITHDRAWN_NOTE + " before expiry")''',
                                         '''            return AskGrade(0.0, "resolver", answered_at, "x")'''),
    "grade: moot card counts as answered": ('''        if "resolved_by" in _as_dict(ask.get("answer_detail")):''',
                                            '''        if False:'''),
    "laplace: no prior": ('return {r: (answered[r] + 1) / (closed[r] + 2) for r in RULES}',
                          'return {r: (answered[r] / closed[r]) if closed[r] else 0.5 for r in RULES}'),
}
orig = F.read_text()
results = {}
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONPATH": f"{W}:{W}/services/orion-sql-writer:{W}/services/orion-sql-writer/tests"}
try:
    base = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], cwd=W, env=env,
                          capture_output=True, text=True)
    print("baseline:", base.stdout.strip().splitlines()[-1])
    assert base.returncode == 0
    for name, (a, b) in MUTANTS.items():
        assert orig.count(a) == 1, f"mutant {name!r} does not apply"
        F.write_text(orig.replace(a, b))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS], cwd=W, env=env,
                           capture_output=True, text=True)
        results[name] = "KILLED" if r.returncode != 0 else "SURVIVED"
        print(f"{results[name]:9s} {name}  ({r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''})")
finally:
    F.write_text(orig)
sys.exit(0 if all(v == "KILLED" for v in results.values()) else 1)
