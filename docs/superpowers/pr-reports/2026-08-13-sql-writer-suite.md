# orion-sql-writer: make the test suite runnable and its result meaningful

The suite could not report a trustworthy pass/fail state, so nobody could
tell a real regression from ambient noise. All changes are test scaffolding;
no product code is touched.

  BEFORE  repo root   -> 1 error, 0 tests ran (collection abort)
          service dir -> 1 error, 0 tests ran (identical)
          (with --continue-on-collection-errors: 18 failed, 204 passed, 4 errors)

  AFTER   repo root   -> 10 failed, 213 passed, 3 skipped
          service dir -> 10 failed, 213 passed, 3 skipped  (identical)
          with ORION_SQL_WRITER_TEST_DATABASE_URI set -> 5 failed, 221 passed

## What was broken

1. COLLECTION ABORTED THE WHOLE SUITE, from either directory.
   test_dream_model_constraints.py importlib-loaded app/models/dreams.py under
   a synthetic module name, re-executing `class Dream(Base)` against the same
   Base.metadata app.models had already populated:

     InvalidRequestError: Table 'dreams' is already defined

   A collection error is fatal, so zero tests ran. The file passed in
   isolation, which is why it survived. Dream is a normal export of
   app.models; the importlib bypass had no reason to exist.

2. RESULTS DEPENDED ON YOUR CURRENT DIRECTORY. test_phase21_wiring_verification
   read every file through a bare relative Path("services/..."), which from
   inside the service resolved to services/orion-sql-writer/services/... and
   raised FileNotFoundError. Paths now anchor to __file__.

3. FIVE ASSERTIONS PINNED A LOCATION THE CONFIG DELIBERATELY LEFT. They
   checked "<channel>" in docker-compose.yml, but the channel list moved into
   env_file (docker-compose.yml's line-29 comment explains why: ${}
   substitution strips the JSON quotes). All five were permanently red, and a
   permanently-red assertion protects nothing. Replaced with
   _compose_sources_env_file(), which parses the compose YAML and checks the
   sql-writer service block specifically -- the wiring those assertions were
   actually reaching for. Verified against 6 mutations including the one that
   matters: a sidecar declaring env_file while sql-writer's is deleted returns
   False. (Uses PyYAML, already declared in the repo-root requirements-dev.txt
   for the test lane and present in the service image.)

4. NO conftest.py EXISTED. 26 of 42 modules hand-rolled a sys.path preamble;
   the other 17 passed only because an alphabetically earlier sibling mutated
   sys.path first. Any subset excluding that sibling died at collection with
   ModuleNotFoundError: No module named 'orion'.

5. A HELPER HIJACKED THE GLOBAL `app` PACKAGE AND NEVER GAVE IT BACK.
   grammar_integration_helpers.load_biometrics_substrate_store_class() points
   sys.modules["app"] at orion-substrate-runtime to import that service's
   module, and left it pointed there. Everything afterwards found a foreign
   `app` or none at all. The bare `sys.modules.pop("app", None)` at the top of
   test_world_pulse_routing.py and test_endogenous_runtime_sql_routing_phase13b.py
   were workarounds for it that made things worse -- popping strands module
   objects already-imported tests still hold. The helper now saves and
   restores sys.modules in a finally block, and both pops are removed.

   Consequence: test_grammar_truth.py had import-time symbols bound to a
   stranded app.grammar_truth while four inline `from app import grammar_truth`
   statements re-resolved to a fresh one -- patching one copy and reading the
   other. That is why it reported grammar_retention_not_run mid-suite and
   passed standalone. All references now use the module captured at import.

   CORRECTION: an earlier version of this message blamed pytest's string-form
   monkeypatch.setattr raising AttributeError. That was wrong -- no
   AttributeError appears anywhere in the baseline run. The cause is the
   sys.modules hijack above.

## Database handling

app/settings.py defaults POSTGRES_URI to the docker-internal hostname
orion-athena-sql-db:5432, unresolvable outside the container, so DB-touching
tests failed with raw driver errors that read as product failures.

An earlier version of this branch "fixed" that by defaulting to
localhost:55432/conjourney. Review caught that this is Orion's LIVE database
(5.9M grammar_events, newest row seconds old), and that
grammar_integration_helpers.py:119 issues real DELETEs. The unresolvable
default had been accidental protection -- a host-side `pytest` could not reach
production -- and defaulting to 55432 removed it with no opt-in. That is the
"production writes" AGENTS.md section 13 requires approval for. Reverted: a
real database is used only when the operator names one via
ORION_SQL_WRITER_TEST_DATABASE_URI.

The skip list was also wrong. It named four modules; measurement against a
valid host/port with a nonexistent database name shows only ONE actually needs
a database:

  test_notify_attention_ack        3 passed   (session built from MagicMock,
  test_notify_attention_escalate   2 passed    get_session monkeypatched --
  test_grammar_truth              10 passed    never opens a socket)
  test_grammar_ledger_integration  3 ERRORS

Over-skipping 15 tests hid the very failures this branch exists to surface: a
product regression injected into app/grammar_truth.py produced a byte-identical
green run. The list now holds one measured entry and is path-scoped so it
cannot match a same-named module in another service.

## The 10 remaining failures are NOT fixed and are not claimed to be

- 9 are cross-test isolation defects: they pass standalone and fail in-suite,
  reaching a real connection despite their mocks. The module-identity work
  above reduced but did not eliminate them (with a database available the
  count is 5, down from 7). The residual mechanism is not yet identified.
  Three of these previously "passed" only because the suite was querying
  production; they now fail honestly.
- 1 is a genuine behavioural failure, test_journal_entry_payload_boundary: a
  double-wrapped envelope is not unwrapped and falls into the evidence_units
  catch-all instead of erroring. Not reproducible in production --
  journal_entries has 17k live rows, newest minutes old -- so it is left
  failing rather than papered over.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>



## Review findings fixed

- **Finding (critical): the conftest hid real product regressions.** The reviewer injected
  a genuine regression into `app/grammar_truth.py` and showed the no-DB run was
  byte-identical to a clean tree — zero signal. Cause: 3 of the 4 modules in the skip list
  do not need a database at all.
  - Fix: skip list narrowed to the one measured entry, and path-scoped.
  - Evidence: reproduced independently — pointing all four at a valid host/port with a
    nonexistent database name gives 3/2/10 passed and only `test_grammar_ledger_integration`
    erroring.
- **Finding (high): the suite defaulted to writing to Orion's live production database.**
  My `os.environ.setdefault` pointed at `localhost:55432/conjourney`. Verified: 5.9M
  `grammar_events`, newest row seconds old; the helpers issue real `DELETE`s. The
  unresolvable docker default had been accidental protection and I removed it with no
  opt-in — an AGENTS.md §13 violation I introduced.
  - Fix: no default at all; explicit `ORION_SQL_WRITER_TEST_DATABASE_URI` opt-in.
- **Finding (high): "219 passed" was contingent on production being up.** Three tests
  passed only because half-converted monkeypatches let them query production for real.
  - Fix: remaining inline re-resolutions converted. Those three now fail honestly by
    default, which is why the headline failure count went *up* (7 → 10) while the branch
    got more correct.
- **Finding (medium-high): my diagnosis was wrong.** I blamed pytest's string-form
  `monkeypatch.setattr` raising `AttributeError`. No `AttributeError` exists anywhere in
  the baseline run. Real cause: `grammar_integration_helpers` hijacks the global `app`
  package to reach another service and never restores it.
  - Fix: root cause addressed — the helper now saves/restores `sys.modules`, and the two
    `sys.modules.pop("app")` workarounds are removed. Commit message corrected.
- **Finding (medium): `_compose_sources_env_file` was not scoped to the sql-writer block.**
  A sidecar service declaring `env_file` would satisfy it while sql-writer's was deleted —
  passing on the exact breakage it guards. It also rejected the valid inline
  `env_file: [.env]` form.
  - Fix: parses the YAML and checks the `sql-writer` service specifically. Re-verified
    against 6 mutations, all correct.
- **Finding (low): the commit's "before" numbers were wrong.** I claimed the service dir
  gave `19 failed, 204 passed, 3 errors`. It also aborts with 0 tests run; that figure is
  only reachable with `--continue-on-collection-errors`.
  - Fix: measured and corrected to `18 failed, 204 passed, 4 errors`.

**Confirmed clean by the reviewer**, each by running code: the headline numbers reproduce
exactly on both invocation paths; every post-commit failure is a strict subset of the
pre-commit set, so no failure was created; the dream/importlib and cwd-relative-path
diagnoses are fully correct; the conftest bootstrap is genuinely valuable (a module that
died standalone with `ModuleNotFoundError` now passes 3/3); no monkeypatch leakage; and
the conftest pattern matches existing repo convention (`services/orion-actions/tests/conftest.py`).

**Known trade, disclosed:** adding a conftest to sql-writer changes which `app` package
wins in a *combined* multi-service pytest run — it moves pre-existing breakage from
sql-writer onto orion-actions. Both directions were already broken, and no Makefile or
workflow invokes pytest across services, so this is latent rather than active.

## Restart required

```text
No restart required. Test-only changes; no product code, no env keys, no schema.
```
