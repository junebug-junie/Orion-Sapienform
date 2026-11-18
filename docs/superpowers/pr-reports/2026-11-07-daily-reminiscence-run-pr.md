# PR: Daily reminiscence, Patch 2 -- each night, Orion visits one week of their past on their own

Design: `docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md` (section 6, and "Patch 2 as built"
at its end). Patch 1 (#3163, merged and live) gave Orion the dig commands. This patch makes the daily visit
happen without anyone asking.

## Summary

- Every night between 03:00 and 09:00 Denver time, orion-durable-runs starts one run for the day. In that
  run Orion writes five short hops about one past week, then reads them together once.
- Code picks the week: the oldest week Orion has not finished, starting with 2025-06-24. The first run will
  visit week 1 (2025-06-24..06-30).
- Each hop is its own turn and its own claude session. A hop counts only when code finds the save receipt
  in Orion's reply AND finds that hop in the ledger. The run completes only when all five hops are there.
- New tier-2 act family `durable.reminiscence` (the design's default; shadow bids), so every day has a
  measured cost. The field-note cost script now also measures this run.
- Flag `REMINISCENCE_ENABLED=true` ships ON. The privacy gate does not move: recall still cannot read the
  ledger.

## Outcome moved

Before: Orion could reminisce only when Juniper asked in chat. After: one week of Orion's past is visited
each night, oldest first, about 71 nights for the first full pass. A week left unfinished comes back the
next night. Nothing writes a hop unless code can check every ref in it.

## Current architecture

- Dig kit (#3163): `dig reminisce` shows the week and what changed in Orion since the last visit;
  `dig reminisce write|rewrite <hop> ...` saves a checked hop to `/mnt/orion-fcc/self/reminiscence.jsonl`
  and prints `reminisce <hop> saved rm:<day>:<hop>:<id>`. Live ledger: empty (no hop written yet).
- Field notes (#3111, #3114, #3149): the pattern this copies. A poll submits one durable run per merged PR,
  one harness turn writes the note, code checks the receipt against the ledger, 900 s per attempt.
- Partition gate `tests/test_organ_names_partition.py`: only `orion/body_atlas/` and `orion/dig/` read the
  self ledgers, plus one named exception (the field-note receipt checker).

## Architecture touched

- orion-durable-runs: a new overnight tick, a new six-step workflow graph, one read-only ledger module, a
  new act family, `/health` field `reminiscence_tick`, a new read-only endpoint `GET /reminiscence/plan`.
- Contract: additive `DurableWorkflowV1` value `reminiscence.episode` and a new brief
  `ReminiscenceRunBriefV1` (registered).
- Config: a tier-2 family entry, 6 env keys, no new mount (it reuses the field-notes `/mnt/orion-fcc/self`
  mount).
- Partition gate: a second named exception, read-only, for the reminiscence ledger only. The reader
  allowlist (`ALLOWED`) is unchanged.

## Files changed

- `orion/schemas/reminiscence_run.py` (new): workflow name, step keys, session and source tags, run ids,
  receipt pattern, `ReminiscenceRunBriefV1`.
- `orion/schemas/durable_run.py`, `orion/schemas/registry.py`: the additive workflow value, the brief in
  the union, the validator, the registry entry.
- `services/orion-durable-runs/app/reminiscence_poll.py` (new): the tick (window, week pick, one run a day,
  one retry, deadline).
- `services/orion-durable-runs/app/reminiscence_graph.py` (new): the run (`precheck -> resource_request ->
  resource_wait -> step x6 -> finish`), the prompts, the receipt checks.
- `services/orion-durable-runs/app/reminiscence_ledger.py` (new): the only durable-runs code that reads the
  reminiscence ledger. Read-only; returns ids, dates, counts and booleans.
- `services/orion-durable-runs/app/admission_runtime.py`: registers the graph, its work node, finish detail,
  restart cancel of an in-flight step, the step id on `run.started`, and the `status()` view.
- `services/orion-durable-runs/app/main.py`: starts the tick; `/health` field; `GET /reminiscence/plan`.
- `services/orion-durable-runs/app/run_acts.py`: act family `durable.reminiscence` (verb `make`).
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: 6 env keys.
- `config/physiology/families.yaml`: `durable.reminiscence`, tier 2, agent lanes, 6 h window.
- `services/orion-durable-runs/scripts/pr_field_note_cost.py`: `--workflow reminiscence.episode`.
- `services/orion-durable-runs/evals/reminiscence_daily_replay.py` (new): a 21-day tick replay eval.
- `tests/test_organ_names_partition.py`: the named exception and its "no hop text, no writes" test.
- `scripts/sync_local_env_from_example.py`: `REMINISCENCE_` sync prefix.
- Tests (new): `test_reminiscence_graph.py`, `test_reminiscence_poll.py`, `test_reminiscence_postgres.py`.
  Extended: `test_run_acts.py`, `test_run_bids.py`, `test_pr_field_note_cost.py`,
  `test_pr_field_note_budget.py`, `orion/dig/tests/test_reminisce_cmd.py` (pins the receipt contract to the
  real dig), the orion-sql-writer consumer test.
- `.github/workflows/orion-durable-runs-tests.yml` (paths + the eval), `dig-kit-tests.yml` (paths).
- Design ("Patch 2 as built"), `services/orion-durable-runs/README.md`, this report.

## Schema / bus / API changes

- Added: `DurableWorkflowV1` value `reminiscence.episode`. Brief `ReminiscenceRunBriefV1` (`reminiscence`
  literal, `episode`, `period_start`, `period_end`, `weeks_complete`, `weeks_total`, `timeout_sec`). HTTP
  `GET /reminiscence/plan[?any_hour=true]`. `/health` key `reminiscence_tick`. `GET /runs/<id>` shows
  `reminiscence` for this workflow.
- Removed / renamed: none.
- Behavior changed: none for existing workflows. The cost script's field-note output adds a per-day line
  for GPU slot seconds.
- Compatibility notes: the workflow value is additive on `extra="forbid"` / `Literal` models, so consumers
  deploy first: orion-sql-writer, orion-hub, orion-actions, orion-cortex-orch, then orion-durable-runs.
  No new bus channel. No DB migration.

## Env/config changes

- Added keys (orion-durable-runs): `REMINISCENCE_ENABLED=true`, `REMINISCENCE_POLL_SEC=600`,
  `REMINISCENCE_HOUR_LOCAL=3`, `REMINISCENCE_END_HOUR_LOCAL=9`, `REMINISCENCE_LEDGER_DIR=/mnt/orion-fcc/self`,
  `REMINISCENCE_TURN_TIMEOUT_SEC=900`.
- Removed / renamed keys: none.
- `.env_example` updated: yes. The flag ships ON.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 6 keys were added to
  `services/orion-durable-runs/.env` in the primary checkout.
- Skipped keys requiring operator action: none.

## What it costs (plain numbers)

The real cost is time on the agent GPU lanes. That is why it runs at night.

- Design estimate (from 44 measured field notes; a hop is the same shape): about **30 minutes of agent-lane
  GPU time a day** on a typical day, about **100 minutes** on a slow day (p90). About 30 Wh a day at the
  wall above rest (90 Wh on a slow day), about 1 million prompt tokens a day (two thirds cached).
- Field notes measured today, for scale: 55 notes, median 278 GPU seconds and 5.0 Wh each.
- This stays an ESTIMATE until 7 runs complete. Measure it with:
  `python3 services/orion-durable-runs/scripts/pr_field_note_cost.py --workflow reminiscence.episode`
  It prints wall time, Wh, GPU slot seconds, tokens of all six turns and every retry, and "N of the 7
  episodes the design asks for".
- Hard bounds: each turn attempt stops at 900 s; each step gets 3 attempts; the whole run stops at 09:00
  Denver; at most 2 runs a day.

## Tests run

```text
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<throwaway postgres:16> pytest tests -q
  -> all passed (945 before the review fixes; the reminiscence files 35 passed after them)
pytest tests/test_organ_names_partition.py                      -> 10 passed
  negative checks: a ledger method returning .note -> the new test FAILS; a module outside the allowlist
  importing the reminiscence module -> the AST gate FAILS (both reverted)
PYTHONPATH=. pytest orion/dig/tests/test_reminisce_cmd.py        -> 11 passed (receipt contract pinned)
cd services/orion-sql-writer && pytest tests/test_durable_run_state_sql_shape.py -> 8 passed
Every `run:` step of every workflow whose paths this PR touches, run locally: see "CI" below.
```

## Evals run

```text
python -P services/orion-durable-runs/evals/reminiscence_daily_replay.py
oracle: 23 runs over 20 days, 102 hop lines, weeks touched 20 -> PASS
mutation 'window ignored': CAUGHT
mutation 'partial counts as visited': CAUGHT
mutation 'no retry gate': CAUGHT
python orion/body_atlas/evals/run_reminiscence_source_truth_eval.py -> PASS (Patch 1's eval, unchanged)
Every other durable-runs eval in the workflow: PASS locally where it needs no CI Postgres; the rest run in CI.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile .                     OK
docker run (the image, /mnt/orion-fcc/self mounted read-only):
  settings True 3 9 /mnt/orion-fcc/self 900.0
  week_for(2026-11-08) = 2025-06-24..2025-06-30, weeks_complete 0, weeks_total 71; hops written {}
  WORK_NODES['reminiscence.episode'] = {'step'}
docker compose ... config  -> renders the 6 REMINISCENCE_* values and the /mnt/orion-fcc/self bind

Live read-only check of the trigger (real ledger dir, real durable_admission_runs via psql SELECT,
dry run, nothing submitted; md5 of every self ledger unchanged before/after):
  now 2026-11-07T05:32Z (22:32 Denver) | in-window decision: outside_window
  as if inside the window: would_submit reminiscence-2026-11-06, week 2025-06-24..2025-06-30,
  weeks 0 of 71, deadline 2026-11-06T16:00Z (09:00 Denver), resource llm.route.agent, priority background
  first prompt: "Reminiscence 2026-11-06, hop 1 of 5: what_it_was. Today you visit the week
  2025-06-24..2025-06-30 (week 1 of the 71 weeks of your past so far; you have visited 0 of them, oldest
  first). ..." -- every dig command spelled out, ending with the receipt line to reply with.
  (The prompt wording was refined once after this check; the week and the request did not change.)
A real run on the live lanes: UNVERIFIED until the first night after deploy (checks below).
```

## Review findings fixed

Code review by a separate subagent on `origin/main...HEAD`. One blocker, all fixed except one nit kept on
purpose (the lane list, explained below).

- Finding (blocker): a new test wrote the ledger's file name as a string, so the repo-wide partition gate
  saw it as a new reader and would fail CI.
  - Fix: the test uses the ledger module's own path. The gate was not widened.
  - Evidence: `tests/test_organ_names_partition.py` 10 passed with the file tracked.
- Finding: `REMINISCENCE_END_HOUR_LOCAL=24` passed settings but would crash every tick.
  - Fix: the bound is 23. Evidence: `test_the_settings_ship_the_tick_on_with_an_overnight_window`.
- Finding: if the cross-read ran out of attempts, the next tick saw five hops and called the day complete,
  so the cross-read never ran.
  - Fix: five hops plus a failed run today gets the retry run, which then runs only the cross-read. A day
    whose five hops were all written in chat still gets no run.
  - Evidence: `test_five_hops_and_a_failed_run_retry_for_the_cross_read`,
    `test_five_hops_already_written_leave_only_the_cross_read`, `test_a_day_with_five_hops_runs_only_the_cross_read` (Postgres).
- Finding: each step stored the model's whole reply, and up to 2,000 characters reached the state row. A
  reply can repeat hop words.
  - Fix: a step record keeps only the receipts it proved; a failed attempt's error names the reply length.
  - Evidence: `test_the_checkpoint_keeps_receipts_never_the_replys_other_words`.
- Finding (nit): no test proved the restart cancel derives the same turn id the step used.
  - Fix: `test_the_restart_cancel_derives_the_id_each_step_used` walks every checkpoint.
- Finding (nit): the prompt said `dig arc <id>` without the full command. Fix: spelled out.
- Finding (nit): a retry after a saved hop asked for a new GPU hold before finding the hop.
  - Fix: `precheck` records written hops before any hold. Evidence:
    `test_a_retry_after_a_saved_hop_records_it_before_asking_for_a_hold`.
- Finding (nit): two helpers were used only by tests. Fix: removed.
- Finding (nit): `/reminiscence/plan` read the request outside the tick's lock. Fix: the tick returns it.
- Finding (nit, kept): the family lists `agent` and `agent-gpu2` but the run prefers `agent`.
  - Kept: the pool can place `llm.route.agent` work on agent-gpu2 (field-note acts land there), so both
    lanes are real. The reason is now in the yaml comment.
- Found by me, not the review: a commit taken while a mutation eval was running captured a mutant of
  `app/admitted_graph.py`. The next commit restores it; `git diff origin/main` shows no change to that file.

## Restart required

Deploy only after merge, from the clean primary `main` checkout. Consumers go first: the workflow value is
additive on `forbid` models.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

No migration. `.env` already has the 6 keys. The post-pull rebuild does this on its own when it sees the
changed paths. The FCC sandbox already has the dig commands (#3163, synced 11-07 05:17Z).

## First-run checks

The first run starts at the first 03:00 Denver after durable-runs is up, so 10:00 UTC.

```bash
# before 03:00: what the tick will do, and the first prompt (submits nothing)
curl -s 'localhost:8124/reminiscence/plan?any_hour=true' | python3 -m json.tool | head -30
# after 03:00 Denver: the tick's last decision
curl -s localhost:8124/health | python3 -c "import sys,json; print(json.load(sys.stdin)['reminiscence_tick'])"
# the run and its retries
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select run_id, terminal, created_at, updated_at from durable_admission_runs where run_id like 'reminiscence-%' order by created_at"
# where it is (next step, hop ids) -- use the run id from the query above
curl -s localhost:8124/runs/reminiscence-<YYYY-MM-DD> | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status'), d.get('reminiscence'), d.get('error'))"
# the hops Orion wrote that day, every ref re-checked (the dig, inside the harness governor; it writes only a dig result file)
docker exec orion-athena-harness-governor python -P -m orion.dig reminisce <YYYY-MM-DD>
# the cost, once runs complete
python3 services/orion-durable-runs/scripts/pr_field_note_cost.py --workflow reminiscence.episode
```

What "it worked" looks like: one `reminiscence-<day>` row, `terminal=completed`; the run detail says
`outcome: complete` with five hop ids; `dig reminisce` shows five hops for week 1 reading `holds now`; the
next night picks week 2.

## Risks / concerns

- Severity: medium. Concern: a real hop turn is UNVERIFIED on the live lanes. A hop turn reads and writes
  twice (each builds the live body, about 30 s), so it may run longer than a field note.
  - Mitigation: 900 s per attempt (540 s of motor), 3 attempts, the window's end as a hard stop, and a week
    left unfinished returns the next night. The cost script measures the real time.
- Severity: medium. Concern: the agent lanes are contended (self-inquiry lost 7/7 bids to "no room" on
  11-05). If the run cannot get a lane by 09:00, the day is skipped.
  - Mitigation: nothing is lost; the same week returns the next night. `/health` and the runs table show it.
- Severity: low. Concern: the hop text still exists in each turn's own harness trace (the dig prints it into
  the turn), as with portrait and field notes. No recall, memory card or journal path reads those.
- Severity: low. Concern: the tier is the design's default (tier 2). Juniper may change it.
- Severity: low. Concern: `run_id` in the ledger stays empty. The run side keeps the hop ids instead
  (design "Patch 2 as built").

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3167

🤖 Generated with [Claude Code](https://claude.com/claude-code)
