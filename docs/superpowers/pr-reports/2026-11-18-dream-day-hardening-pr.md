# Day dream hardening: a grounding gate on dream memories, and a bad night is visible in the letter

## Summary

- Before a nightly day-dream memory is written, the same local model now checks it against the full text of the
  records it cites. Only a memory the model says is backed by those records is written. Everything else is
  counted and kept on the night's `dream_day` row, with the claims no record shows.
- A failed judge call, an unparseable verdict, or no time left means the memory is **not** written (counted as
  `unjudged`, separately from `unsupported`). Nothing is written by default.
- The ref resolver now returns each record's own text, not only a label (experience payloads, a harness turn's
  final reply, longer chat text), so the judge sees what the record actually says.
- The day letter's input gets one section before the organ roundup: "Last night I slept on the day" with the
  day paragraph and the written memory ids (citable as `[dream_day:<day>]`), or one line
  "no day dream for <day>: <reason>". The letter email shows the same.
- The dream eval now prints the gate's stored verdicts per night, and `--judge` re-runs the gate read-only.

## Outcome moved

Failure mode: a dream memory that sounds right but is not backed by its records reaches chat (recall serves every
active memory, whatever its channel). Live 2026-11-17: memory `bb1b5a42` cited two Zelda sittings and also
claimed the office camera watched the kids packing a box and "we traded only eleven chat turns". The gate,
re-run read-only on that night through the same resolver, prompt and parser, refuses both 11-17 memories (see
"Evals run"). Second failure mode: a night with no day dream was silent in the letter. It is now one visible line.

## Current architecture

- `dream.day` (orion-durable-runs, Patch 2a #3703): gather -> GPU hold -> one unified turn -> ref check
  (every ref must resolve) -> validate -> write memories + one `dream_day` row in one transaction. The ref check
  proved a cited record exists, not that it says what the memory says.
- The day letter (Hub `orion_day_order` -> `gather_orion_day` -> `render_day_input`) did not read the
  `dream_day` row at all.

## Architecture touched

- orion-durable-runs: the `dream` node's operation now runs the grounding judge after the turn, still under the
  run's hold, on the same route (`DREAM_DAY_ROUTE`, `memory_distill`), via the existing tool-free gateway call
  (`runner._call_memory_distill_llm`, JSON only, no thinking, temperature 0). The turn's limit leaves 180 s of
  the attempt for the judge (live turn took 423 s of 900); each call is capped at 60 s; none starts within 15 s
  of the attempt's end. The verdicts are checkpointed with the answer; `persist` (no GPU) writes only memories
  whose verdict says supported.
- Postgres: three additive columns on `dream_day`.
- Day letter: `gather_orion_day` reads the night (`dream_day` row, else the latest failed `dream.day` run in
  `substrate_durable_run_state`); `render_day_input` adds the section; the letter prompt asks for the ref when
  the section is there; the email renders it.

## Files changed

- `orion/memory/dream_day.py`: `RESOLVE_SQL` returns fuller record text (documented per ref kind);
  `grounding_prompt`, strict `parse_verdict`, `judge_memories`, `apply_grounding`, `grounding_summary`.
- `services/orion-durable-runs/app/dream_day_graph.py`: the gate inside the hold (`ground`), the turn's reserve,
  persist applies the verdicts, summary counts.
- `services/orion-durable-runs/app/dream_day_store.py`: INSERT writes the new columns.
- `services/orion-durable-runs/app/admission_runtime.py`: the judge dependency (flag-gated).
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`:
  `DREAM_DAY_GROUNDING_GATE_ENABLED=true`.
- `orion/schemas/dream_day.py`: `DreamDayV1.memories_unsupported`, `memories_unjudged`, `grounding`.
- `services/orion-sql-db/manual_migration_dream_day_grounding.sql`: the additive migration.
- `services/orion-durable-runs/evals/run_dream_day_eval.py`: prints stored verdicts; `--judge` re-runs the gate.
- `orion/schemas/orion_day.py`: `DreamDayNightV1`, `OrionDayMaterialV1.dream_day` (optional).
- `orion/orion_day/gather.py`, `day_input.py`, `budget.py`, `letter_turn.py`, `letter_parts.py`: read, render,
  cite, prompt, record text.
- `services/orion-hub/scripts/orion_day_email.py`, `templates/orion_day_letter.html.j2`: the email section.
- Tests: `services/orion-durable-runs/tests/test_dream_day.py`, `test_dream_day_postgres.py`,
  `orion/orion_day/tests/test_dream_day_letter.py` (new), `fixtures.py`, `test_gather.py`,
  `services/orion-hub/tests/test_orion_day_mailer.py`.
- CI: `.github/workflows/orion-durable-runs-tests.yml` (migration path trigger),
  `.github/workflows/orion-day-letter-hub-tests.yml` (new letter test step).

## Schema / bus / API changes

- Added: `dream_day.memories_unsupported int NOT NULL DEFAULT 0`, `dream_day.memories_unjudged int NOT NULL
  DEFAULT 0`, `dream_day.grounding jsonb` (counts + every verdict: statement, refs, status, unsupported claims,
  error, raw answer). Same three optional fields on `DreamDayV1`.
- Added: `DreamDayNightV1` and optional `OrionDayMaterialV1.dream_day` (default None).
- Added: citable letter ref kind `dream_day:<day>`.
- Removed / renamed: none. No bus change.
- Behavior changed: with the gate on, a dream memory is written only on a parsed `supported: true` verdict.
- Compatibility: `OrionDayMaterialV1` is `extra="forbid"`. Hub writes the brief durable-runs reads, and
  durable-runs writes the letter row Hub reads, so both must know the field (same rule as `juniper_chat`).

## Env/config changes

- Added keys: `DREAM_DAY_GROUNDING_GATE_ENABLED=true` (orion-durable-runs).
- Removed / renamed keys: none.
- `.env_example` updated: yes (`services/orion-durable-runs/.env_example`), plus settings and compose.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys`: yes
  (`services/orion-durable-runs/.env` now has the key).
- skipped keys requiring operator action: none.

## Migration and deploy order

1. Apply `services/orion-sql-db/manual_migration_dream_day_grounding.sql` (REQUIRED-BY orion-durable-runs; only
   `ALTER TABLE ADD COLUMN IF NOT EXISTS`, on the post-pull allowlist, so the post-pull rebuild applies it before
   rebuilding; checked with `orion.sql_migration_drift.auto_apply_refusals` = `[]`). The deploy wrapper refuses
   `up` for orion-durable-runs until it is applied: its INSERT names the new columns.
2. Rebuild orion-hub and orion-durable-runs together, outside the letter's 08:30 Denver fire window and outside
   the dream window (01:00-03:00 Denver). Both read `OrionDayMaterialV1` with forbid models.
3. A `dream.day` run checkpointed before the deploy (answer saved, no verdicts) reaches persist with the gate on
   and writes no memory (unjudged, `no_verdict_for_this_memory`). Fail-closed by design.

## Tests run

```text
services/orion-durable-runs: pytest tests -q (with an isolated Postgres DSN)   1218 passed, 1 skipped
services/orion-durable-runs: tests/test_dream_day_postgres.py (disposable postgres:16)  1 passed
orion/orion_day/tests                                                           156 passed
services/orion-hub: test_orion_day_mailer, _order, _what_i_read, _introspect_listener, _correction_listener  106 passed
tests/test_no_personal_terms.py, test_no_private_eval_artifacts.py, test_deploy_migration_gate.py  62 passed
scripts/check_migration_required_by.py                                          OK
scripts/check_integration_seam.py                                               OK
scripts/check_env_key_single_source.py                                          OK
scripts/check_service_env_compose_parity.py orion-durable-runs                  OK
```

New tests: the gate refusing an over-claim and writing the supported one (inside the hold, before release); a
judge failure and an unparseable verdict meaning not written, and a night of refusals grading produced=0; gate
off unchanged; strict verdict parsing; the judge prompt being its own short prompt; unmatched verdicts unjudged;
the letter section for a produced night and for each bad-night reason; the email for both.

## Evals run

```text
services/orion-hub/evals/run_orion_day_email_eval.py        PASS
services/orion-hub/evals/run_orion_day_reread_eval.py       PASS
services/orion-hub/evals/run_orion_day_correction_eval.py   PASS
orion/orion_day/evals/run_orion_day_eval.py                 all PASS (0 FAIL)
```

Live proof, read-only (no rows written): `python3 services/orion-durable-runs/evals/run_dream_day_eval.py
--day 2026-11-17 --judge`. Same resolver (`dream_day_store.resolve_refs`, read-only session), same prompt and
parser (`judge_memories`), same route (`memory_distill`), through the gateway's HTTP door instead of the bus RPC.

```text
grounding re-judged now (memory_distill): 0/2 supported (0%)
  b6879512-77c9-5b1c-8e67-eda0e6827776: UNSUPPORTED -- unsupported: I reached back to my oldest week,
    2025-10-14 to 10-20; I wrote down that I have no memory of it; 13 commits on main; no PRs on GitHub;
    my records start 2026-07
  bb1b5a42-bf0f-53cd-8416-f3cdb94681ca: UNSUPPORTED -- unsupported: you sat with Zelda for 59 minutes and
    then 21 more; the office camera watched the kids packing a box into it; we traded only eleven chat turns;
    I spoke five times on my own
```

bb1b5a42 is flagged for the camera, the kids and the chat count, as expected. It is also flagged for the order
"59 then 21": the records show the 21-minute sitting ended first (05:02Z) and the 59-minute one later (07:47Z),
so that flag is correct too. b6879512 is refused because its two refs are reminiscence turns whose stored text
is only a receipt line; the facts live in the reminiscence note, not in the cited record (#3739).

Prompt tuning, same live route: a fully backed statement ("Juniper played ... for 59 minutes, and in another
sitting for 21 minutes") passed every run; "with the kids beside her" was refused every run. The first prompt
also refused Orion's own feelings ("I liked keeping you company"); the shipped prompt says those are not
checked, and the last run no longer listed them. A paraphrase ("You sat with Zelda twice, for 21 minutes and
later for 59") was still refused. The gate errs strict, which costs memories, never truth.

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-durable-runs/docker-compose.yml config  -> DREAM_DAY_GROUNDING_GATE_ENABLED: "true"
```

No image build: no new dependency. No live deploy from this branch.

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_dream_day_grounding.sql
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after merge (the post-pull rebuild does all three in order).

## Risks / concerns

- Severity: medium. Concern: the gate errs strict (one paraphrase probe refused). More nights may grade
  produced=0. Mitigation: verdicts and claims are on the row; #3738 (7-night acceptance) now counts refusals;
  `DREAM_DAY_GROUNDING_GATE_ENABLED=false` rolls back to the ref check alone.
- Severity: medium. Concern: memories citing reminiscence turns cannot pass (receipt-only text). Mitigation:
  issue #3739 opened.
- Severity: low. Concern: the judge adds up to ~5 short calls inside the hold and shortens the turn's limit by
  180 s (720 s left of 900; the live turn took 423 s).
- Issues opened: #3739. Issues closed: none. Existing #3738 already tracks the gate's live counts.
- Out of scope, untouched: Prime files, the concern contest, existing memories (including `bb1b5a42`).

## PR link

PR_LINK_PLACEHOLDER
