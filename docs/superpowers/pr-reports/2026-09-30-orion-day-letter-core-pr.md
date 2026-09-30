# Orion's Day letter -- backend (gather, budget, admitted durable run, verbs)

## Summary

- Orion now has everything needed to write a daily note about yesterday: a read-only gatherer
  that pulls one America/Denver day out of Postgres at full length, a deterministic budget that
  turns it into a ~70k-token model input, and a new admitted durable workflow
  (`orion_day.letter`) that writes the note and a separate list of threads to carry forward.
- Two separate model calls, two separate fields: `orion_day_note_v1` writes a long first-person
  note whose prompt says nothing about future curiosity; `orion_day_carry_forward_v1` gets the
  day plus the finished note and lists threads for curiosity. They never share a field, a
  column, or a journal entry.
- The run holds an agent-lane GPU pool hold for both calls, releases it before writing, persists
  one row per day (`orion_day_letter`, first writer wins), and journals the NOTE only (journal
  email disabled -- the Hub PR sends the styled email).
- The dream-hypothesis blind rule is enforced in SQL and in the schema: only hypotheses whose
  offering run completed, never `arm` / `ref_a` / `ref_b` / `cycle_json`, and no hypothesis id in the
  model's view (ids are the dream scorecard's join key).
- The hold asks for enough context for the whole run (`minimum_context_tokens`), so a heavy day never
  lands on the 65,536-token chat card when the pool spills agent work there.
- This is the backend half. The Hub PR (scheduler at 08:30, HTML email, carry-forward injection
  into curiosity) consumes the interface described under "Interface for the Hub PR".

## Outcome moved

Before: nothing in the mesh could say "here is what Orion thought about yesterday". The material
existed in ten tables, never assembled. Now a single call assembles it (live 2026-09-29: 8
curiosity runs, 9 failed runs, 16 self-sense answers, 9 readings + 8 reading journals, 6 seen
dream hypotheses, 958 reverie thoughts / 290 chains, 15 visual reveries, the world-pulse digest),
and the durable run turns it into a persisted note + carry-forward that survives restarts,
pool take-backs and the 06:00 congestion. No model has been called yet (not deployed).

## Current architecture

- Admitted durable workflows in `services/orion-durable-runs` (curiosity, self-sense, reflect,
  reading, reverie.visual) each hold one GPU pool hold per run; the accepted request row
  (`durable_admission_runs.request`) is the durable demand; LangGraph checkpoints each node.
- Day windows already exist for the compactors (`previous_local_day_window`, inclusive end).
- Readings had a read-only introspect path capped at 900 chars / 5 items (the tool's budget).
- Dream SQL for introspect existed only as a plan on `feat/introspect-dreams` (no code).

## Architecture touched

- New package `orion/orion_day/` (window, gather, budget, brief, store) + `orion/schemas/orion_day.py`.
- `orion/schemas/durable_run.py`: `orion_day.letter` workflow, brief union, admission required.
- `services/orion-durable-runs`: new graph + store, `AdmissionRuntime` wiring (graph, WORK_NODES,
  finish detail, slim checkpoint brief, status block, no harness cancel), `runner._call_verb_text`
  built on `_cortex_orch_rpc` (copied verbatim from open PR #2431 so the two merge to one copy).
- Journaler: `trigger_kind=orion_day_letter`, `source_kind=orion_day`, mode `daily`, dispatch policy
  email/in-app disabled.
- Cortex: two verbs + prompts; cortex-exec completion budgets (new env keys) and default `agent`
  route for the two verbs.
- Postgres: `orion_day_letter` table (manual migration).
- Bus catalog: durable-runs listed as producer of `orion:cortex:request` and consumer of
  `orion:cortex:result*` (identical hunks to #2431; it was already true for self_study.reflect).

## Files changed

- `orion/schemas/orion_day.py`: material / llm-view / brief / letter contracts, stable run and journal ids.
- `orion/orion_day/window.py`: half-open Denver day (same calendar date as the compactors, DST-safe).
- `orion/orion_day/gather.py`: read-only asyncpg SQL per source; one failing source records `error`, never cancels.
- `orion/orion_day/budget.py`: deterministic model view; reveries condensed (one thought per chain, prefix dedupe, hollow skipped, themes), water-fill clipping only under pressure, bracketed refs.
- `orion/orion_day/brief.py`: `build_orion_day_brief`, `build_orion_day_request`, empty-day refusal.
- `orion/orion_day/store.py`: `fetch_letter` for Hub.
- `orion/dream/introspect_sql.py`: shared dream SQL with the blind rule (from the slice-2 plan).
- `orion/world_pulse_read/introspect.py`: `text_cap` kwarg on `_item` / `reading_results` (default unchanged), `reading_items_between`.
- `orion/schemas/durable_run.py`, `orion/schemas/registry.py`: workflow + registration.
- `orion/journaler/{schemas,worker,dispatch_registry}.py`: new trigger/source kind, email off.
- `services/orion-durable-runs/app/orion_day_graph.py`, `app/orion_day_store.py`: the graph and the one writer.
- `services/orion-durable-runs/app/admission_runtime.py`, `app/runner.py`: wiring, verb helper.
- `orion/cognition/verbs/orion_day_{note,carry_forward}_v1.yaml`, `orion/cognition/prompts/orion_day_*.j2`: the verbs.
- `services/orion-cortex-exec/app/{executor,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: budgets, route default.
- `services/orion-sql-db/manual_migration_orion_day_letter_v1.sql`: the table.
- `orion/bus/channels.yaml`: catalog fix (see above).
- `config/metrics/metric_definitions.lock.json`: re-locked for that catalog change (2 declared-routing
  deltas: durable-runs as a producer of `orion:cortex:request` and a consumer of `orion:cortex:result*`;
  CI's definition-drift gate caught it on the first push). #2431 re-locks for the same edit.
- Tests: `orion/orion_day/tests/*`, `services/orion-durable-runs/tests/test_orion_day_{graph,postgres}.py`, `services/orion-durable-runs/tests/test_verb_text.py`, `services/orion-cortex-exec/tests/test_orion_day_verbs.py`.
- Eval: `orion/orion_day/evals/run_orion_day_eval.py`. Smoke: `scripts/smoke_orion_day_letter.py`.
- CI: `.github/workflows/orion-durable-runs-tests.yml` (paths + orion_day tests + eval), `orion-gpu-pool-tests.yml` (cortex-exec verb test).
- Docs: `services/orion-durable-runs/README.md`, this report.

## Schema / bus / API changes

- Added: `OrionDayMaterialV1`, `OrionDayLlmViewV1`, `OrionDayRunBriefV1`, `OrionDayLetterV1` (registered, `resolve()` verified); workflow `orion_day.letter`; journal `trigger_kind=orion_day_letter`, `source_kind=orion_day`; table `orion_day_letter`; verbs `orion_day_note_v1`, `orion_day_carry_forward_v1`.
- Removed: none. Renamed: none.
- Behavior changed: `reading_results(..., text_cap=)` optional (default 900, unchanged).
- Compatibility notes: additive Literal values on `extra="forbid"` models (`DurableWorkflowV1`, `JournalTriggerKind`, `JournalSourceKind`). Every parser (sql-writer, actions, durable-runs, cortex-orch, Hub) must be on this build before anything submits the workflow -- nothing does until the Hub PR.

## Env/config changes

- Added keys: `LLM_ORION_DAY_NOTE_MAX_TOKENS=12000`, `LLM_ORION_DAY_CARRY_FORWARD_MAX_TOKENS=4000` (orion-cortex-exec).
- Removed / renamed keys: none.
- `.env_example` updated: yes (orion-cortex-exec), compose passes both with defaults.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: the default run skipped them (prefix not in its sync list); `--all-keys orion-cortex-exec` added both (verified in `services/orion-cortex-exec/.env` lines 449-450). No other keys changed.
- Skipped keys requiring operator action: none.

## Tests run

```text
# orion_day package + touched neighbours (repo venv)
python -m pytest orion/orion_day/tests orion/world_pulse_read/tests orion/introspect/tests \
  tests/test_check_journal_dispatch_registry.py tests/test_schema_registry_import_light.py -q
  -> 212 passed, 2 skipped
# durable-runs, whole suite, on a throwaway postgres:16 (ORION_ADMISSION_TEST_DSN)
cd services/orion-durable-runs && python -m pytest tests -q
  -> 290 passed   (incl. test_orion_day_graph.py 17, test_orion_day_postgres.py 3, test_verb_text.py 8)
# same suites in a FRESH venv built only from CI's installs (durable-runs reqs + requirements-dev + acceptance)
  -> orion/orion_day/tests 45 passed; durable-runs 275 passed (before the review-fix commit; re-run above after it)
# cortex-exec
cd services/orion-cortex-exec && python -m pytest tests/test_orion_day_verbs.py tests/test_gpu_lease_forwarding.py -q
  -> 15 passed
# gates
check_chat_route_poachers PASS | check_env_template_parity PASS | check_journal_dispatch_registry OK (11 kinds)
check_bus_reply_channels 0 uncovered | check_metric_lineage PASS | check_definition_drift PASS
check_inner_state_registry OK | check_compose_no_relative_mounts PASS | check_async_routes_not_blocking PASS
# mutation checks (each reverted after): drop the early hold release, route every grant to write_note,
# append the carry-forward to the journal body, drop orion_day.letter from WORK_NODES -> each caught by a test
# pre-existing failures, identical on clean main d3c09c9cb (not touched here):
tests/test_journaler_worker.py::test_draft_from_cortex_result_raises_structured_parse_error
tests/test_sql_writer_journal.py (2, need the orion-athena-sql-db hostname)
services/orion-cortex-exec/tests/test_chat_general_route_mapping.py::test_introspect_spark_uses_quick_route
```

## Evals run

```text
python orion/orion_day/evals/run_orion_day_eval.py            (fixture day; also a pytest + a CI step)
  PASS separation    note-instruction leaks=[]
  PASS grounding     refs=96 unresolved=[] missing=[]
  PASS blind         hypotheses=1 ids_in_view=[]
  PASS full_text     bodies=7 absent=[]
  PASS condensation  shown 80/301, themes 4/4, hollow_shown=0
  PASS budget        digest~9208 tokens (budget 70000); note prompt~9768 + 12000 < 131072
  PASS determinism   rebuilt from JSON
python orion/orion_day/evals/run_orion_day_eval.py --live     (live 2026-09-29, read-only)
  PASS separation    note-instruction leaks=[]
  PASS grounding     refs=161 unresolved=[] missing=[]
  PASS blind         hypotheses=6 ids_in_view=[]
  PASS full_text     bodies=41 absent=[]
  PASS condensation  shown 80/958, themes 9/9, hollow_shown=0
  PASS budget        digest~69521 tokens (budget 70000); note prompt~70082 + 12000 < 131072
  PASS determinism   rebuilt from JSON
Real token count (agent lane's own tokenizer, POST :8015/tokenize, read-only): digest 244,783 chars ->
65,020 tokens (estimate 69,938; 3.76 chars/token, so the 3.5 estimate is ~7% conservative); full note
prompt 65,486 tokens. No model has been called: note quality is UNVERIFIED until the first live run.
```

## Docker/build/smoke checks

```text
python scripts/smoke_orion_day_letter.py        (live Postgres, session default_transaction_read_only=on)
letter_date=2026-09-29 window=[2026-09-29T06:00:00+00:00, 2026-09-30T06:00:00+00:00) read_only_session=on
== sections (material, full length)
  curiosity_runs     count=    8 json_chars=91038
  curiosity_failed   count=    9 json_chars=1549
  self_sense         count=   16 json_chars=52498
  readings           count=    9 json_chars=13954
  reading_journals   count=    8 json_chars=51901
  dream_narratives   count=    0 json_chars=0
  dream_hypotheses   count=    6 json_chars=2581
  reverie_thoughts   count=  958 json_chars=572257
  reverie_chains     count=  290 json_chars=60646
  visual_reveries    count=   15 json_chars=11034
  chat_compactor     present=False   (run at 01:51 Denver; the 09-29 digest is written at 06:00)
  github_compactor   present=False   (no GitHub compactor entry since 2026-09-28 03:46 -- separate issue)
  world_pulse_digest present=True json_chars=3948
== model view (budgeted)
  digest_chars=243322 approx_tokens=69521 budget_tokens=70000
  condensed: reverie_thoughts 80 of 958 (361 duplicate openings, 373 same-chain skipped), 9/9 themes,
             51 full-text items, 0 clipped
  included_refs=161 unresolved_refs=0
== prompts
  note_prompt_chars=245287 (~70082 tokens estimated; 65,486 real)
== durable request
  run_id=orion-day-2026-09-29-1 workflow=orion_day.letter resource=llm.route.agent priority=background
  deadline_at=2026-10-01T06:00:00+00:00 minimum_context_tokens=87021
  request_json_bytes=1094664 material_json_bytes=843205
Docker: not built or deployed (task said do not deploy). Postgres e2e ran against a throwaway postgres:16
container with the real in-process GPU pool fixture and the real migration.
```

## Review findings fixed

Code review ran in a subagent (explicit target: origin/main...feat/orion-day-letter-core).

- Finding (BLOCKER): a reasoning dump or an error string could be stored as the note --
  `extract_cortex_payload_text` falls back to `reasoning_content` when `final_text` is empty; no
  error-text or `finish_reason=length` check.
  - Fix: `runner.strict_final_text` -- `final_text` only, refuses `looks_like_error_text` and truncation
    (step `finish_reason` or `truncation_detected`); carry-forward needs a list item.
  - Evidence: `tests/test_verb_text.py` (8), `test_carry_forward_without_a_list_item_is_an_attempt`.
- Finding: the blind dream experiment could be re-exposed / its scorecard polluted -- the digest used
  `[dream_hypothesis:<id>]`, the scorecard's `formed_from` join key, and "offered" does not mean seen
  (a failed run keeps its offer).
  - Fix: only hypotheses whose offering run completed (`DREAM_HYPOTHESES_SEEN_SQL`), rendered as
    `[dream_offered:N]` with no id; material (email) keeps ids.
  - Evidence: `test_hypothesis_ids_never_reach_the_model`, eval `blind` check; live 15 -> 6 hypotheses.
- Finding: LLM retries had no backoff (a cortex restart could burn all attempts in seconds).
  - Fix: failed attempt -> `retry_wait` with exponential backoff, routed by `retry_node`.
  - Evidence: `test_a_failed_attempt_waits_out_a_backoff_before_the_next_one`.
- Finding: after the row was inserted, journal trouble could fail the run and a later run would never
  publish the journal.
  - Fix: journal is published from the STORED row (note, created_at, writer run id) by every run; its own
    retry budget; never fails a run whose letter exists (`journal_published=false`).
  - Evidence: `test_journal_trouble_never_fails_a_run_whose_letter_is_written`,
    `test_a_second_run_for_the_same_day_is_a_no_op`, Postgres e2e (identical second entry).
- Finding: GitHub rolling-mode digests share the day digest's stable id.
  - Fix: compactor rows must be written after the day ends (`created_at >= window_end`).
  - Evidence: `test_a_compactor_row_written_during_the_day_is_not_the_day_digest`.
- Finding: a release error after the carry-forward would discard the finished text.
  - Fix: release moved outside the try. Evidence: `test_a_release_error_after_the_carry_forward_keeps_the_text`.
- Finding: real restart recovery untested.
  - Fix: runtime-level Postgres test driving `AdmissionRuntime._recover`.
  - Evidence: `test_restart_mid_carry_forward_resumes_under_the_same_hold_without_rewriting_the_note`
    (mutation: removing the WORK_NODES entry makes it fail).
- Nits fixed: RPC timeout 30 s under the node budget; journal `created_at` from the row; readings
  `truncated` flag; carry-forward ref stats in the finish detail; deploy order (below).
- Found while fixing (not in the review): the chat card serves 65,536 tokens (live `/props`), below a heavy
  day's prompt + note. Fix: admission `requirements.minimum_context_tokens` (the pool honours it).
- Not fixed, recorded under Risks: 1.1 MB request row re-read on hot paths; `_cortex_orch_rpc`'s
  compactor-named reply prefix.

## Restart required

Not deployed. Deploy order (each step only after the previous one is live):

```bash
# 1. table
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_orion_day_letter_v1.sql
# 2. parsers of the new Literal values
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
# 3. verbs (orch loads verb YAML/prompts and parses DurableRunRequestV1; exec renders + budgets);
#    orion/ is baked into both images, and without exec the note would get the 512-token default
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
# 4. the workflow
scripts/safe_docker_build.sh orion-durable-runs up -d --build
# 5. Hub: only with the Hub PR that submits orion_day.letter
```

## Risks / concerns

- Severity: medium. Concern: the accepted request row is ~1.1 MB on a heavy day (843 KB of it material,
  572 KB reverie thoughts) and `registry_store.list_pending` / `get_run` read `SELECT *` every admission
  tick and heartbeat while the run waits (up to its deadline, ~40 h worst case). The slim checkpoint keeps
  it out of LangGraph, not out of these reads. Mitigation: bounded to one run a day; follow-up: explicit
  column lists on the registry store's hot paths (shared by every workflow, so not changed here).
- Severity: medium. Concern: note quality is UNVERIFIED -- no model call was made (not deployed). The
  grounding eval checks the INPUT (every ref resolves, nothing invented in the digest), not the note.
  Mitigation: first live run should be read by Juniper before the Hub PR emails anything.
- Severity: low. Concern: a timed-out verb call is not cancelled at cortex; the model may finish it on the
  card after the hold is released (same exposure as self_study.reflect). Mitigation: RPC gives up 30 s
  before the node budget; documented in the README.
- Severity: low. Concern: `_cortex_orch_rpc` is a verbatim copy from open PR #2431 (so the two merge to one
  copy; `git merge-tree` confirmed runner.py merges cleanly with it). Its reply prefix
  `orion:cortex:result:compactor-digest:*` and log label now also carry orion_day calls. Follow-up once both
  land: parameterize the prefix. Remaining conflicts with #2431 and #2419 are the workflow Literal/union,
  the admission graph map, and the CI workflow paths -- trivial.
- Severity: low. Concern: the digest shows the day's offered hypotheses' claims (approved), which the
  journaled note may echo into recall. Their ids are withheld so nothing can be credited to the blind
  offer; showing claims at all is Juniper's approved design.
- Severity: low. Concern: GitHub compactor has written nothing since 2026-09-28 03:46 (separate issue; the
  letter reports the source as empty).

## Interface for the Hub PR

```python
from orion.orion_day.brief import build_orion_day_brief, build_orion_day_request, OrionDayEmptyError
from orion.orion_day.store import fetch_letter
from orion.orion_day.window import yesterday_letter_date

letter_date = yesterday_letter_date(now)                    # Denver calendar day
if await fetch_letter(conn, letter_date) is None:           # asyncpg conn, read-only is fine
    brief = await build_orion_day_brief(conn, letter_date)  # raises OrionDayEmptyError on a blank day
    request = build_orion_day_request(brief, attempt=1)     # run_id "orion-day-<date>-<attempt>"
    # POST {durable-runs}/runs  json=request.model_dump(mode="json")   (202 + receipt)
    #   or publish on orion:durable:run:request, kind durable.run.request.v1
letter = await fetch_letter(conn, letter_date)              # OrionDayLetterV1 | None
# letter.note_md, letter.carry_forward_md, letter.material (full texts), letter.sources,
# letter.carry_forward_expires_at; Hub stamps emailed_at / email_notification_id /
# carry_forward_offered_at / carry_forward_offered_run_id.
```

- Key on `detail.persist_outcome` (`written` / `already_written`), not on `completed`, and on
  `detail.journal_published`.
- Resubmitting the same `run_id` with a regathered brief is refused by the durable store
  (`SubmissionConflict`); check `GET /runs/{run_id}` first, and after a FAILED run submit
  `attempt=2`. A second run for a day whose row exists completes with
  `persist_outcome=already_written` and writes nothing.
- Terminal state arrives on `orion:durable:run:state` with `detail.line == "orion_day"`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2435

🤖 Generated with [Claude Code](https://claude.com/claude-code)
