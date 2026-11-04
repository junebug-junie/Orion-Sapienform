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
REVIEW_PLACEHOLDER_TESTS
```

## Evals run

```text
REVIEW_PLACEHOLDER_EVALS
```

## Docker/build/smoke checks

```text
REVIEW_PLACEHOLDER_DOCKER
```

## Review findings fixed

REVIEW_PLACEHOLDER_FINDINGS

## Restart required

REVIEW_PLACEHOLDER_RESTART

## Risks / concerns

REVIEW_PLACEHOLDER_RISKS

## PR link

REVIEW_PLACEHOLDER_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
