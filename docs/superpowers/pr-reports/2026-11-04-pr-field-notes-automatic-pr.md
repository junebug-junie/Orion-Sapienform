# PR: PR field notes, Patch 2 -- after every merge, Orion writes a field note on their own

Spec: `docs/superpowers/specs/2026-11-04-pr-field-notes-design.md` (approved by Juniper 2026-11-04;
"Patch 2 as built" at its end). Patch 1 (#3100, merged and deployed) gave Orion the dig commands.
This patch makes the notes happen on their own.

## Summary

- Every 15 minutes, orion-durable-runs asks GitHub's public API which PRs merged in the last 72 hours.
  It uses no token: the repo is public, and this container has no `gh`, no `git` and no clone.
- A dependency-only or tiny docs-only PR gets a code record at once. No model turn runs for it
  (Juniper's decision 2).
- Every other merged PR gets one durable run, `pr.field_note`. The run is one harness turn on the #2958
  step seam. Orion reads the packet with `orion.dig field-note <n>` and writes the note with
  `field-note <n> write`.
- The run counts as done only when code finds the save receipt in Orion's reply AND finds that note
  id in the ledger. A reply without it is a failed attempt and is retried. It is never a success.
- New tier-1 family `durable.pr_field_note` in `config/physiology/families.yaml`, with a 24 h window
  (Juniper's decision 1). Each run is an act, so every note has a measured cost. A script reads the
  real per-note time, tokens and whole-machine energy.
- `harness.echo_smoke` ran live first and PASSED. It is the first smoke row ever.

## Outcome moved

Before: a note was written only when Juniper asked for one in chat. Her first test took 15 tool calls,
and Hub showed about 60 steps. After: each merged PR gets a note or a code record within about
15 minutes plus the wait for the agent lane, with no one asking. The turn is kept lean: one read, one
write, and the receipt as the reply. Code checks are allowed only when a sentence claims something
about code that the packet does not state.

## Current architecture

- Dig kit field notes (#3100): `field-note <n>` builds the packet and `field-note <n> write` checks
  and appends to `$ORION_DIG_HOME/self/pr_field_notes.jsonl`. These run only inside
  orion-harness-governor, which has `gh` and the `/mnt/orion-fcc` mount.
- Harness step seam (#2958, `app/harness_step.py`): one durable-run step becomes one Hub harness turn
  under the run's GPU pool hold. Its smoke had never run live (0 admission rows).
- durable-runs had no access to GitHub and no ledger mount. The partition test let only
  `orion/body_atlas` and `orion/dig` touch the ledger.

## Architecture touched

- orion-durable-runs: a new poll loop, a new workflow graph, one ledger-access module, a new act
  family, `/health` field `field_note_poll`, and a new endpoint `POST /pr-field-notes/poll`.
- Contract: additive `DurableWorkflowV1` value `pr.field_note` and a new brief `PrFieldNoteBriefV1`
  (registered).
- Config: a tier-1 family entry, 8 env keys, and one bind mount (`/mnt/orion-fcc/self`).
- Partition gate: one named exception for the field-notes ledger only. A new test checks that the
  exception never hands back note text.

## Files changed

- `orion/schemas/pr_field_note_run.py` (new): the workflow name, step key, session, tag, run-id
  helpers, the receipt pattern, and `PrFieldNoteBriefV1`.
- `orion/schemas/durable_run.py`, `orion/schemas/registry.py`: the additive workflow value, the brief
  in the union, the validator, and the registry entry.
- `services/orion-durable-runs/app/pr_field_note_poll.py` (new): the public-API client, the
  classification, code records, run submission, the bounds and the gap warning.
- `services/orion-durable-runs/app/pr_field_note_graph.py` (new): `precheck -> resource_request ->
  resource_wait -> field_note -> finish`, the lean prompt, and `verify_receipt`.
- `services/orion-durable-runs/app/pr_field_note_ledger.py` (new): the only durable-runs code that
  touches the ledger. It returns ids and booleans only.
- `services/orion-durable-runs/app/admission_runtime.py`: registers the graph, its work node, its
  finish detail, its cancel id after a restart, its step id on `run.started`, and its `status()` view.
- `services/orion-durable-runs/app/main.py`: starts the poll loop, adds the health field and the
  manual poll endpoint.
- `services/orion-durable-runs/app/run_acts.py`: the act family `durable.pr_field_note` (verb `make`).
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: the env keys
  and the ledger mount.
- `config/physiology/families.yaml`: `durable.pr_field_note`, tier 1, `deferrable_window_s: 86400`,
  citing Juniper's decision.
- `orion/body_atlas/field_notes.py`: an additive `LEDGER_FILE` / `ledger_path()` helper, the one path
  rule. `orion/dig/__main__.py` now uses it.
- `tests/test_organ_names_partition.py`: the narrow exception, and a test that every `.note` use in that
  module is an `is not None` check.
- `scripts/sync_local_env_from_example.py`: a `PR_FIELD_NOTES_` sync prefix, so the default sync adds
  the keys.
- `services/orion-durable-runs/scripts/pr_field_note_cost.py` (new): the measured cost per note.
- Tests (new): `test_pr_field_note_poll.py`, `test_pr_field_note_graph.py`,
  `test_pr_field_note_postgres.py`, `test_pr_field_note_cost.py`. Extended: `test_run_acts.py`,
  `test_run_bids.py`, and the orion-sql-writer consumer test.
- `.github/workflows/orion-durable-runs-tests.yml`: also runs on changes to the brief and the ledger
  format.
- Spec ("Patch 2 as built"), `services/orion-durable-runs/README.md`, and this report.

## Schema / bus / API changes

- Added: `DurableWorkflowV1` value `pr.field_note`. New brief `PrFieldNoteBriefV1` (`field_note`
  literal, `pr_number`, `merged_at`, `title`, `timeout_sec`). HTTP `POST /pr-field-notes/poll[?since=]`.
  A new `/health` key, `field_note_poll`. `GET /runs/<id>` shows `pr_field_note` for this workflow.
- Removed / renamed: none.
- Behavior changed: none for existing workflows.
- Compatibility notes: the workflow value is additive on `extra="forbid"` / `Literal` models, so
  consumers deploy first. Deploy orion-sql-writer, orion-hub, orion-actions and orion-cortex-orch,
  then orion-durable-runs (the same list as #2958). No new bus channel. No DB migration.

## Env/config changes

- Added keys (orion-durable-runs): `PR_FIELD_NOTES_ENABLED=true`, `PR_FIELD_NOTES_POLL_SEC=900`,
  `PR_FIELD_NOTES_LOOKBACK_H=72`, `PR_FIELD_NOTES_GITHUB_REPO=junebug-junie/Orion-Sapienform`,
  `PR_FIELD_NOTES_NOT_BEFORE=2026-11-04T06:00:00+00:00`, `PR_FIELD_NOTES_SELF_HOST=/mnt/orion-fcc/self`,
  `PR_FIELD_NOTES_LEDGER_DIR=/mnt/orion-fcc/self`, `PR_FIELD_NOTES_TURN_TIMEOUT_SEC=420`.
- Removed / renamed keys: none.
- `.env_example` updated: yes. The flag ships ON.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 8 keys were added
  to `services/orion-durable-runs/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<throwaway postgres:16> PYTHONPATH=<repo> pytest tests -q
  867 passed, 1 skipped   (includes test_pr_field_note_{poll,graph,postgres,cost}.py, test_run_acts, test_run_bids)
pytest orion/dig/tests orion/body_atlas/tests tests/test_organ_names_partition.py
  all passed (partition: 7 passed, 2 of them new)
cd services/orion-sql-writer && pytest tests -q
  1185 passed, 183 skipped (new: test_pr_field_note_state_is_accepted_by_real_writer_route)
pytest orion/schemas/tests -q                                  79 passed
every `run:` step of .github/workflows/orion-static-gates.yml  no FAIL
python scripts/check_env_template_parity.py                    PASS
python scripts/check_env_key_single_source.py                  OK
```

## Evals run

```text
No eval harness in this patch. The spec's truth eval (orion/body_atlas/evals/run_field_note_truth_eval.py)
and the measured cost table are Patch 3 ("Eval and cost"); they need real notes first.
The measuring tool ships here and runs live today:
python3 services/orion-durable-runs/scripts/pr_field_note_cost.py
  pr.field_note runs ended: 0 (0 completed) -- no completed runs yet: the spec's cost table is still an ESTIMATE
Same joins checked against the live echo smoke run: 1 turn, 8 LLM calls, 81,980 prompt tokens
(32,972 cached), 1,681 completion, 11 Hub harness steps.
```

## Docker/build/smoke checks

```text
harness.echo_smoke, LIVE, before any build (the spec's "run the smoke first"):
python3 services/orion-durable-runs/scripts/harness_echo_smoke.py
  submitted harness-echo-smoke-20261104T055700Z-55ce10 -> waiting_resource -> admitted -> completed
  PASS step text: 'harness echo smoke ok 2026-11-04T05:58:38Z'   (117 s submit -> completed)
  turn faa6ab2c-03f4-503d-8ad3-e65c2f0c582f: cockpit_turn_sighting stages present; finalize ran, did not change the text
  durable_admission_runs now has its first harness.echo_smoke row (terminal=completed)

docker build -f services/orion-durable-runs/Dockerfile -t fn-patch2-durable-runs:test .   OK
docker run ... fn-patch2-durable-runs:test python -c "import app.main, app.admission_runtime ..."
  settings: PR_FIELD_NOTES_ENABLED True, not_before 2026-11-04 06:00:00+00:00, ledger dir /mnt/orion-fcc/self
  public GitHub search from inside the image: 13 PRs merged in the last 3 h, 1 call, no token
Live dry run of the classification (ledger mounted read-only, nothing written, nothing submitted):
  ledger live records: [3090]
  #3104 note run (8 files, 245 lines)   #3105 note run (20 files, 568 lines)   #3106 note run (5 files, 491 lines)
docker compose --env-file .env --env-file services/orion-durable-runs/.env -f services/orion-durable-runs/docker-compose.yml config
  renders the 7 PR_FIELD_NOTES_* env values and the bind /mnt/orion-fcc/self -> /mnt/orion-fcc/self
```

## Review findings fixed

Review by a separate subagent on the full diff. All findings fixed except #9, which is accepted below.

- Finding: a test docstring named the partition test's file, which holds the grep term `organ_names`,
  so the repo-wide partition gate failed (blocker).
  - Fix: reworded the docstring.
  - Evidence: `tests/test_organ_names_partition.py` 7 passed.
- Finding: a retry went straight back to a turn. An attempt that saved the note but replied badly
  would resume a session that only repeats the receipt. That is too few steps for Hub's lookup gate,
  so the run would fail with a note in the ledger.
  - Fix: `retry_wait -> precheck`. A note found there finishes the run as `already_recorded`.
  - Evidence: `test_a_retry_after_the_note_was_saved_runs_no_second_turn`.
- Finding: one PR's error (a 404 on its files, a refused code record) ended every tick at that PR and
  starved later merges for up to 72 h.
  - Fix: each PR is handled in its own try. Only a rate limit, a 5xx, a network error or the ledger
    itself ends the tick. Per-PR errors go to `pr_errors`.
  - Evidence: `test_one_bad_pr_never_blocks_the_others`,
    `test_a_404_is_one_prs_problem_and_a_rate_limit_is_the_ticks`, `test_a_rate_limit_mid_tick_ends_the_tick`.
- Finding: the in-flight count read only runs inside the poll window, so a manual `since=` poll could
  start a second run.
  - Fix: a separate count of every unfinished `pr-field-note-*` run, with no time filter.
  - Evidence: `test_in_flight_counts_runs_from_before_the_window`.
- Finding: `PR_FIELD_NOTES_NOT_BEFORE` with no offset made the endpoint compare naive and aware times
  (a 500 error).
  - Fix: a settings validator makes it UTC-aware once.
  - Evidence: `test_a_not_before_without_an_offset_means_utc`.
- Finding: the "never hands back note text" check looked only at `.note`. A later `return records`
  would have passed it.
  - Fix: a new gate. Every public function must declare an id- or boolean-shaped return, and no
    `return` may hand back a bare variable, a load or live result, or a model dump.
  - Evidence: `test_the_field_note_writer_returns_only_ids_and_booleans`.
- Finding (nit): the gap warning fired after any quiet 72 h, even with no merges.
  - Fix: it now searches the stretch the lookback lost and names only merges that have no record and
    no run.
  - Evidence: `test_gap_warning_names_only_merges_that_fell_out_of_the_lookback`.
- Finding (nit): a PR whose first file page is full could be called a code record.
  - Fix: a full page (100 files) always gets a note run.
  - Evidence: `test_a_full_page_of_files_is_always_a_note_pr`.
- Finding (nit, optional): the prompt did not ban backslashes in the note.
  - Fix: it does now.
- Finding (nit, accepted): a run that ends `already_recorded` after it held a GPU grant closes its act
  as artifact-produced. A note does exist (most likely from its own first attempt), but this run did
  not prove it with a receipt. A run that finds a record at the start never opens an act. This is
  left as is and listed under Risks.

## Restart required

Deploy only after merge, from the clean primary `main` checkout. Consumers go first: the workflow value is additive on `forbid` models.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

There is no migration. `.env` already has the 8 new keys. orion-durable-runs needs the
recreate (`up -d --build`, not a restart), because the new bind mount only applies to a new container.

Live check, about 15 min after durable-runs is up:

```bash
curl -s localhost:8124/health | python3 -c "import sys,json; print(json.load(sys.stdin)['field_note_poll'])"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select run_id, terminal, created_at from durable_admission_runs where run_id like 'pr-field-note-%' order by created_at"
tail -n 3 /mnt/orion-fcc/self/pr_field_notes.jsonl
python3 services/orion-durable-runs/scripts/pr_field_note_cost.py
```

The first tick runs at boot. It should submit `pr-field-note-3104`, the oldest merge since
2026-11-04T06:00Z, and the later ones one at a time. A merge after deploy should get its run within
one poll (15 min). It gets its ledger line once the agent lane gives the run a hold. A second poll
submits nothing new: `field_note_poll.submitted` stays `[]` for PRs that already have a run.

## Risks / concerns

- Severity: medium.
  - Concern: Hub's finalize step could rewrite the one-line receipt reply, so `verify_receipt` misses it.
  - Mitigation: on the live smoke, finalize ran and left the one-line reply unchanged. If this happens,
    the retry passes `precheck`, finds the saved note and finishes `already_recorded` (no third turn).
- Severity: medium.
  - Concern: real per-note cost is UNVERIFIED until runs complete. The 420 s turn bound and one run in
    flight cap the worst case at about 3 x 7 min per PR. The spec's 1-2 h/day is still an estimate.
  - Mitigation: `scripts/pr_field_note_cost.py` (wall time, whole-machine Wh, tokens of every attempt,
    Hub steps). Patch 3 replaces the table after 20 notes.
- Severity: low.
  - Concern: the "about 3 tool calls, never more than 6" budget is only in the prompt. Hub's turn
    request has no step-limit field. The hard bound is time.
  - Mitigation: `harness_step_count` is kept in every step record, and the cost script reports it.
    Adding a real step cap means a consumer-first Hub change, not this patch.
- Severity: low.
  - Concern: durable-runs now mounts `/mnt/orion-fcc/self` read-write (this includes the organ-name and
    capacity ledgers).
  - Mitigation: only `app/pr_field_note_ledger.py` touches it, and it uses only the field-notes file.
    The partition gate names that one file and checks it returns no note text. No credential was added.
- Severity: low.
  - Concern: code records carry an empty `body_map`, because durable-runs has no repo or live atlas.
  - Mitigation: a code record makes no claims. `dig field-note <n>` still builds the full map on demand.
- Severity: low.
  - Concern: when a retry finds the note already saved, the act closes as artifact-produced without
    this run proving it (review nit #9).
  - Mitigation: the run detail says `outcome: already_recorded` with `prior_record`, so the cost
    report can tell these runs apart.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3111

🤖 Generated with [Claude Code](https://claude.com/claude-code)
