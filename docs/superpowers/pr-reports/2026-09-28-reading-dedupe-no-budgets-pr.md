## Summary

- Orion no longer reads the same article twice. Once a URL has really been read (Stage 1 finished *with* fetch evidence), any new request for it is folded onto that earlier read. This covers World Pulse, curiosity, and Juniper asking through chat or the Hub.
- When Juniper asks for an already-read URL, Orion tells her it was blocked as a duplicate by design and shares the earlier read's summary. The receipt carries `duplicate: "already_read"`, and the automatic footer on the chat reply now says "Not read again … blocked as a duplicate by design" instead of "accepted".
- Rows already waiting when this ships are passed on by a sweep each tick. The operator "retry" refuses with `already_read`, and a Stage 2 follow-up link that was already read is skipped without using a round trip.
- Reading budgets are gone: no daily cap and no cooldown between reads, for both stages. The on/off switch, the optional reading window and the refund backoff remain.
- The Hub Reading tab shows "N read today", labels passed-on rows "already read, passed on", and the submit form reports the block.

## Outcome moved

- **Failure mode:** the NVIDIA Rubin page finished Stage 1 three times (Sep 13, Sep 26, Sep 27), because the ingress only folded new requests onto reads still in flight. Live history shows the Sep 26 read had fetch evidence, so this change would have blocked the Sep 27 repeat. The Sep 13 row has no fetch evidence (it predates the evidence gate) and correctly does not count as a read.
- **Workflow:** the Sep 27 caps (12/day per stage) stopped reading at 21:16 and 22:16 UTC until local midnight. Pacing is now one read at a time per stage, plus GPU admission and the refund backoff.

## Current architecture

- `enqueue_seeds` (`orion/world_pulse_read/queue.py`) folded a request onto an *active* same-URL row (`ACTIVE_URL_SQL`). A URL whose earlier read had finished was queued and read again.
- `wallet_a.py` and `wallet_b.py` blocked on daily cap and paced cooldown, set by 4 env keys. The refund backoff used the cooldown as its base.

## Architecture touched

- **Queue ingress:**
  - `READ_URL_SQL` looks for the earliest evidence-backed read and is checked before `ACTIVE_URL_SQL`, under the same per-URL advisory lock.
  - `INSERT_SQL` takes `last_error` (`$13`).
- **Per-tick sweeps:**
  - `skip_already_read_stage1` passes on a waiting row, folds it onto the earlier read, and re-points requests that had joined it.
  - `skip_already_read_stage2` passes on a waiting follow-up when the URL's follow-up already finished.
  - Both leave rows with an open durable run alone.
- **Receipt:** `_reading_status_row` adds a `duplicate` field (`already_read` / `already_queued` / null).
- **Chat grounding:** `ReadingRecommendationOutcomeV1.duplicate` feeds a separate footer line in `enforce_reading_receipt_grounding` (`orion/harness/reading_receipts.py`).
- **Operator retry:** refuses `already_read` for Stage 1 and Stage 2. The Stage 2 retry now takes the URL lock and refuses while another follow-up for the URL is running.
- **Wallets:** only `disabled`, `outside_window` and `refund_backoff` remain. Refund backoff is a fixed 1800 s base with a 4 h cap, the same values that were live before.

## Files changed

- `orion/world_pulse_read/queue.py`: already-read ingress, sweeps, receipt `duplicate`.
- `orion/world_pulse_read/operator.py`: retry refusals and the Stage 2 URL lock.
- `orion/world_pulse_read/tools.py`: tool description and brief line tell Orion to report the block.
- `orion/harness/reading_receipts.py`: footer for `already_read`.
- `orion/schemas/reading.py`: `duplicate` on the status receipt and the recommendation outcome.
- `orion/world_pulse_read/wallet_a.py`, `wallet_b.py`, `wallet_refund.py`: caps and cooldowns removed.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `world_pulse_read_stage2.py`: sweeps wired in, fixed refund backoff, reentry skips already-read URLs.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`: 4 budget keys retired.
- `services/orion-hub/scripts/world_pulse_read_routes.py`: status and schedule no longer report `daily_cap` / `cooldown_sec`.
- `services/orion-hub/static/js/reading.js` (+ `reading.test.js`, browser smoke): "N read today", "already read, passed on", submit text, retry hints.
- `services/orion-hub/README.md`: "No reading budgets" and "A URL is read once" sections.
- Tests:
  - new `test_world_pulse_read_already_read_postgres.py` (real SQL);
  - ingress chat-tool test;
  - harness footer test;
  - Stage 2 reentry tests;
  - wallet, pipeline and route tests updated for no budgets.
- `services/orion-hub/tests/reading_queue_fakes.py`: the fake now answers the already-read lookup. Previously it returned a count for every `fetchval`.
- `.github/workflows/orion-reading-tests.yml`: runs `orion/harness/tests/test_reading_receipts.py` and the receipt-truth eval.

## Schema / bus / API changes

- **Added:**
  - `duplicate: "already_read" | "already_queued" | null` on `ReadingStatusReceiptV1` (inherited by `DurableReadingReceiptV1`) and on `ReadingRecommendationOutcomeV1`;
  - operator refusal code `already_read`.
- **Removed:** `daily_cap` and `cooldown_sec` from `/world-pulse-read/api/schedule`; `daily_cap` from `/api/status` `wallet_a` / `wallet_b`.
- **Renamed:** none.
- **Behavior changed:**
  - a URL with an evidence-backed finished Stage 1 is never read again, whoever asks;
  - there is no daily cap or cooldown.
- **Compatibility notes:**
  - the new fields are optional; older receipts validate;
  - no bus channel or migration change;
  - the existing `idx_reading_active_url` index covers the new lookups.

## Env/config changes

- **Added keys:** none.
- **Removed keys:**
  - `HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC`
  - `HUB_WORLD_PULSE_READ_DAILY_CAP`
  - `HUB_WORLD_PULSE_READ_STAGE2_MIN_COOLDOWN_SEC`
  - `HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP`
- **Renamed keys:** none.
- **`.env_example` updated:** yes.
- **Local `.env` synced with `python scripts/sync_local_env_from_example.py`:** yes.
  - The sync script only adds keys, so the 4 retired keys were removed from the primary checkout's `services/orion-hub/.env` by hand. Backup: `/tmp/reading-dedupe-nobudget/orion-hub.env.bak`.
  - The parity check warns about those 4 keys until this merges, because it compares against the primary checkout's template. Leftover keys are harmless either way, since `Settings` ignores extras.
- **Skipped keys requiring operator action:** none.

## Tests run

```text
reading CI selection (RUN_READING_POSTGRES=1, real disposable Postgres): 779 passed
orion/harness/tests/test_reading_receipts.py: 35 passed
node --test services/orion-hub/static/js/reading.test.js: 10 passed
services/orion-hub/tests/test_reading_panel_browser_smoke.py (Playwright): 1 passed
scripts/check_async_routes_not_blocking.py: no blocking calls inside async routes
scripts/check_env_template_parity.py: PASS
```

## Evals run

```text
services/orion-hub/evals/test_reading_handoff_eval.py + test_reading_receipt_truth_eval.py: 16 passed
```

## Docker/build/smoke checks

```text
Live read-only snapshot of rows the first sweep would change: 0 rows
(/tmp/reading-already-read-sweep/before.csv, report.md).
Hub not rebuilt from this branch: runtime behaviour after deploy is UNVERIFIED.
```

## Review findings fixed

- **Finding (must):** the chat footer still said "Reading recommendation accepted" for a blocked duplicate.
  - Fix: `duplicate` added to the outcome, plus a separate "Not read again … blocked as a duplicate by design" footer line.
  - Evidence: `test_already_read_receipt_footer_says_blocked_not_accepted`, which runs through `enforce_reading_receipt_grounding`.
- **Finding (should):** rows marked done before the read-evidence gate would block a URL forever.
  - Fix: only Stage 1 done *with* `read_evidence` counts.
  - Evidence: `test_done_without_fetch_evidence_does_not_block_a_new_read`; the live Sep 13 Rubin row has no evidence.
- **Finding (should):** swept rows kept `duplicate_of = NULL`, so requests that had joined them lost the result, and the ingress could fold a request onto a row about to be swept.
  - Fix: the sweep folds the row onto the earliest read and re-points joined requests; the ingress checks past reads first; the receipt flags `already_read` from `last_error` or `stage2_error`.
  - Evidence: the extended sweep test asserts the joined request shows the earlier read's summary.
- **Finding (should):** a Stage 2 retry could run alongside another follow-up for the same URL.
  - Fix: URL advisory lock, and refusal with `url_already_active`.
  - Evidence: `test_stage2_retry_refuses_while_another_follow_up_for_the_url_runs`.
- **Finding (nit):** a closed Wallet A gate stopped the follow-up loop on an already-read URL.
  - Fix: the already-read check runs before the gate.
  - Evidence: `test_reentry_passes_on_an_already_read_url_even_when_wallet_a_is_backing_off`.
- **Finding (nit):** "Read this URL again" still showed on swept rows.
  - Fix: hidden when `last_error == already_read`.
- **Finding (nit):** `already_queued` read as if it were live.
  - Fix: the tool text says "still in progress when you asked".

## Restart required

```bash
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** a URL whose earlier read was poor can't be re-read, even by Juniper.
  - **Mitigation:** by design (Juniper's call). The receipt and tab say so. An operator can still clear `read_evidence` or delete the earlier row by hand if one ever must be re-read.
- **Severity: low.**
  - **Concern:** without caps, a large backlog reads continuously.
  - **Mitigation:** one read at a time per stage, GPU admission, the refund backoff, and the on/off switch.
- **Severity: low.**
  - **Concern:** the first post-deploy tick modifies live rows.
  - **Mitigation:** the snapshot shows 0 rows would change today.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2379
