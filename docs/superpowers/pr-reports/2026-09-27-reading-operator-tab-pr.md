## Summary

- Hub gets a **Reading** tab that lists every read Orion has queued and shows what each one produced. Stage 1 shows what they took from the source, the beliefs they might adopt, new concepts, open questions and proof they fetched it. Stage 2 shows the follow-up summary, beliefs tested, what the follow-up did and its round trips. The tab also shows matching journal entries, durable runs and duplicate requests.
- A write-up from a read that did not finish is shown as **"Rejected, not learned"**, never as learning. The list preview only quotes accepted write-ups.
- Operator controls: **submit a URL** (same ingress as the chat tool), **cancel** a waiting or running read, **retry Stage 1 or 2**, and **Read this URL again** (pre-fills the submit form when no retry applies).
- Controls reuse the workers' own transitions and row locks. Cancel never charges a wallet. A running stage is cancelled at the durable runner and then finished by the worker's existing cancel path.
- New reading provenance `invocation_context="operator"` → `requested_by="juniper"`. The model's reading tool binding cannot claim it.

## Outcome moved

Before, Juniper could not see what a read produced without querying Postgres by hand. Live data (read-only, 2026-09-27) had 331 queue rows, 13 Stage 1 write-ups and 4 Stage 2 results. One finished digest item's Stage 2 output existed only in the journal, so the tab joins journal entries into the detail view. Failed or stuck reads could only be fixed with SQL; now they can be cancelled or retried from Hub, and every refusal comes with a plain-language reason.

## Current architecture

- The reading queue is the `world_pulse_read_seed` table. It holds each read's Stage 1 and Stage 2 status and output, trace ids and duplicate aliases.
- `reading_durable_turn` binds each stage attempt to a durable-runs `run_id`.
- Workers (`world_pulse_read_pipeline.py`, `world_pulse_read_stage2.py`) claim a row with `FOR UPDATE SKIP LOCKED`, bind a run, poll it, then mark the row done, failed or cancelled.
- A Stage 1 tick skips pending digest items older than `HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS`.
- Hub only exposed `GET /world-pulse-read/api/status` (counts and wallets). There was no per-read view and no controls.

## Architecture touched

- `orion/world_pulse_read/operator.py` (new): list, detail, cancel, retry and submit.
- `orion/world_pulse_read/queue.py`: extracted the pure `derive_reading_status`.
- `orion/world_pulse_read/durable.py`: added the shared constant `OPERATOR_CANCEL_REASON`.
- `orion/schemas/reading.py`: added the `operator` context.
- Hub routes, the `/reading` page, and an iframe tab in `index.html`, following the GPU pool tab pattern.
- CI: the Reading browser smoke now runs in `schedule-browser-smoke.yml`.

## Files changed

- `orion/world_pulse_read/operator.py`: SQL for list and detail; cancel, retry and submit rules.
- `orion/world_pulse_read/queue.py`: `derive_reading_status` is reused by the status endpoint and the tab.
- `orion/world_pulse_read/durable.py`: `OPERATOR_CANCEL_REASON` is used by the worker's `cancel_claim` and by the operator cancel.
- `orion/schemas/reading.py`: the `"operator"` context, which must be requested by `"juniper"`.
- `services/orion-hub/scripts/world_pulse_read_routes.py`: 5 API endpoints, the `/reading` page, the control guard and the durable cancel call.
- `services/orion-hub/scripts/main.py`: mounts the page router.
- `services/orion-hub/templates/reading.html`, `static/js/reading.js`, `static/js/reading_tab.js`, `templates/index.html`: the tab UI.
- `services/orion-hub/static/js/reading.test.js`: tests for the page script's pure helper functions, plus a check that the script never writes read content as HTML.
- `services/orion-hub/tests/test_world_pulse_read_operator_postgres.py`: 13 real-SQL tests.
- `services/orion-hub/tests/test_world_pulse_read_operator_routes.py`: tests for the guard (and that no token is needed), error mapping, durable cancel, submit and page wiring.
- `services/orion-hub/tests/test_reading_panel_browser_smoke.py`: Playwright smoke that loads the real template and script.
- `.github/workflows/schedule-browser-smoke.yml`: runs the smoke.
- `services/orion-hub/README.md`: documents the endpoints, guard and refusal codes.
- `docs/superpowers/specs/2026-09-27-reading-operator-tab-design.md`: design spec.

## Schema / bus / API changes

- Added:
  - `ReadingContext` value `"operator"`, which must be requested by `"juniper"`.
  - HTTP endpoints `GET /world-pulse-read/api/reads`, `GET /world-pulse-read/api/reads/{seed_id}`, `POST /world-pulse-read/api/reads`, `POST .../{seed_id}/cancel` and `POST .../{seed_id}/retry`.
  - Page `GET /reading`.
- Removed: none.
- Renamed: none.
- Behavior changed: `cancel_claim` now takes its reason from a parameter set to the same constant; the stored value is unchanged (`reading_cancelled_by_operator`).
- Compatibility notes:
  - No new bus channel or schema kind. Submit publishes the existing `orion:reading:requested` event through `enqueue_reading`.
  - `ReadingToolBindingV1` is unchanged, so the model tool cannot mark its requests as coming from the operator.

## Env/config changes

- Added keys: none. Existing settings are reused: `HUB_READING_DURABLE_URL` and `HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
RUN_READING_POSTGRES=1 pytest <full orion-reading-tests.yml selection>   737 passed, 1 skipped
pytest test_world_pulse_read_operator_postgres.py test_world_pulse_read_operator_routes.py   31 passed
node --test services/orion-hub/static/js/reading.test.js                  9 pass, 0 fail
node --test services/orion-hub/static/js/*.test.js (earlier run)          190 pass, 0 fail
scripts/check_async_routes_not_blocking.py                                no blocking calls
scripts/check_env_template_parity.py                                      PASS
```

11 Hub UI tests (for example `test_agent_trace_debug_panel`, `test_response_feedback_ui`, `test_biometrics_view_ui`) fail identically on `main` at `d16e9adf9`. I diffed the failure lists and they match, so these failures predate this change.

## Evals run

```text
pytest services/orion-hub/evals/test_reading_handoff_eval.py   11 passed
```

There is no separate eval for the tab itself. It is a viewer and control surface, so the browser smoke is its behavior check.

## Docker/build/smoke checks

```text
uv run --with pytest --with playwright pytest test_reading_panel_browser_smoke.py   1 passed
```

The smoke loads the real template and script in Chromium with the Hub API intercepted. It checks the list, the detail outputs, and that injected `<img onerror>` markup stays inert text. It checks that retry POSTs with the CSRF header and an encoded id, that the rejected label appears, that a duplicate row shows as "merged into another read", that "Read this URL again" pre-fills the form, and that submit works.

The Docker rebuild and the live tab are **UNVERIFIED**. Hub has not been redeployed from this branch.

## Review findings fixed

The review ran in a subagent. It found no blockers and reported security, the new provenance context, and lock/wallet behavior between operator actions and the workers as clean.

- Finding (should): retrying a *failed* digest item older than the stale-sweep age reported "requeued", but the next Stage 1 tick skipped it again. The reviewer confirmed this on disposable Postgres.
  - Fix: `retry_read` takes `digest_item_max_age_sec` (the route passes `HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS`) and refuses with `stale_digest_item_would_be_reskipped`. The tab points to "Read this URL again" instead.
  - Evidence: `test_retry_refuses_failed_digest_item_past_the_stale_sweep_age`. An old item is refused. A young item is requeued, and `skip_stale_digest_items` then touches 0 rows.
- Finding (nit): cancel said it succeeded when the run had already completed or failed.
  - Fix: the response carries `run_already_finished`, and the tab says "too late, the run had already …".
  - Evidence: a parametrized route test covering cancelled, running, completed and failed, plus a JS test.
- Finding (nit): a run the worker hadn't collected yet was labelled "open".
  - Fix: relabelled "not yet collected by the worker" / "collected <time>".
  - Evidence: `reading.js`.
- Finding (nit): duplicate requests showed as "skipped".
  - Fix: `rowStatus` shows "merged into another read".
  - Evidence: JS test and browser smoke.
- Finding (nit): the list preview quoted a rejected Stage 1 write-up.
  - Fix: the preview only uses `what_i_learned` when Stage 1 is `done`.
  - Evidence: a new assertion in the rejected-handoff real-SQL test.
- Finding (nit): some failed rows could be neither retried nor cancelled.
  - Fix: a "Read this URL again" button pre-fills the submit form, which creates a new read.
  - Evidence: browser smoke.

- Finding (CI): the `reading` job failed 3 submit-route tests with `No module named 'aiohttp'`. The submit route imported all of Hub's `main` to find the bus, and CI installs only `requirements-reading.txt`.
  - Fix: a `_bus()` helper (which `_redis()` now uses too) that the tests stub, the same way they stub `_pool` and `_source_ref`.
  - Evidence: the CI selection under `uv run --with-requirements requirements-reading.txt` gives 738 passed, and the PR's `reading` check is green.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-reading-operator-tab   # or main after merge
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: controls use no operator token, by Juniper's decision. The only guard is the cross-site check the GPU pool panel uses (`X-Requested-With: orion-hub` plus a JSON body). That stops another website from triggering them, but anyone who can reach Hub can cancel, retry or submit reads.
  - Mitigation: cancel never charges a wallet, and retries and submits stay inside the normal daily wallet caps.
- Severity: low
  - Concern: a retry or submit spends a normal wallet slot when it runs.
  - Mitigation: the button tooltips say so. The daily caps are unchanged.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2375
