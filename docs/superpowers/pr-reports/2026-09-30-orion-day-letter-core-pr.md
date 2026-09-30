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
- The dream-hypothesis blind rule is enforced in SQL and in the schema: only offered hypotheses,
  never `arm` / `ref_a` / `ref_b` / `cycle_json`.
- This is the backend half. The Hub PR (scheduler at 08:30, HTML email, carry-forward injection
  into curiosity) consumes the interface described under "Interface for the Hub PR".

## Outcome moved

Before: nothing in the mesh could say "here is what Orion thought about yesterday". The material
existed in ten tables, never assembled. Now a single call assembles it (live 2026-09-29: 8
curiosity runs, 9 failed runs, 16 self-sense answers, 9 readings + 8 reading journals, 15 offered
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
- Tests: `orion/orion_day/tests/*`, `services/orion-durable-runs/tests/test_orion_day_{graph,postgres}.py`, `services/orion-cortex-exec/tests/test_orion_day_verbs.py`.
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
REVIEW_PLACEHOLDER_TESTS
```

## Evals run

```text
EVAL_PLACEHOLDER
```

## Docker/build/smoke checks

```text
SMOKE_PLACEHOLDER
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Not deployed. Deploy order (each step only after the previous one is live):

```bash
# 1. table
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_orion_day_letter_v1.sql
# 2. parsers of the new Literal values
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
# 3. the workflow
scripts/safe_docker_build.sh orion-durable-runs up -d --build
# 4. verbs (orch loads verb YAML/prompts; exec renders + budgets) -- orion/ is baked into both images
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
# 5. Hub: only with the Hub PR that submits orion_day.letter
```

## Risks / concerns

RISKS_PLACEHOLDER

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

- Resubmitting the same `run_id` with a regathered brief is refused by the durable store
  (`SubmissionConflict`); check `GET /runs/{run_id}` first, and after a FAILED run submit
  `attempt=2`. A second run for a day whose row exists completes with
  `persist_outcome=already_written` and writes nothing.
- Terminal state arrives on `orion:durable:run:state` with `detail.line == "orion_day"`.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
