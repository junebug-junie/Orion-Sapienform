# Reading pipeline progress and live acceptance

## Outcome

This patch fixes two queue policies that prevent recovery, and supplies a read-only acceptance check. It does not claim the live pipeline is repaired. A completed reading through the current reader, Stage 2, and journal rail remains UNVERIFIED; production deployment and a controlled recovery run require Juniper's approval under AGENTS.md section 13.

## Live findings

Read-only observations on 2026-09-26:

- `https://arxiv.org/abs/2310.19279`: latest request `09d40faa-07fa-57d1-b3a0-211bcd72142f` aliases `reading:b0aec83b-0458-5a06-b934-ffa6934d1566`, pending since September 13. One recorded attempt failed before reading due to GPU admission.
- At first inspection, non-alias queue rows included 66 pending Stage 1, one claimed, 13 done/failed, and three done/done. Only two had confirmed landings, both September 13 and predating the new read-evidence requirement. These are not evidence that today's pipeline works.
- Seven Stage 2 failures were `fcc_stream_stalled`, most recently September 23. Newer failures included GPU admission and stance timeout. These historical rows do not prove every failure mode is still active.
- Live Stage 1 attempt `73db662a-e0ea-4a6a-b9c0-0666769262b0` failed `gpu_pool_unavailable:deadline`; its wallet refunded, but its seed spent an attempt.
- Another attempt observed during this investigation, `d84ee62b-68f7-4958-9ee1-ebb916f65b71`, obtained agent-gpu2 after waiting, then failed `timeout:caller_budget_exhausted`. Gateway logged 208.4 seconds remaining for execution; the stance verb's current ceiling is 240 seconds. The reader never started. Raising that ceiling without measuring the complete path would be another unverified guess.
- Current Stage 1 cap is six/day. Stage 2 is enabled but restricted to 08:00-22:00 in the configured local timezone. Its current `outside_window` log is expected scheduling, not proof of a dead worker.

## Current architecture

- Hub owns two background loops: `scripts/world_pulse_read_pipeline.py` and `scripts/world_pulse_read_stage2.py`.
- Both use `execute_unified_turn`, including Thought stance, Cortex Exec, GPU-pool/Gateway admission, and FCC harness execution. Stage 1 also materializes Concept Atlas candidates.
- Shared queue: `orion/world_pulse_read/queue.py`, `world_pulse_read_seed` in Postgres.
- Wallet A/B backoff: existing Redis keys and `wallet_refund.py`.
- Journal landing: `journal_entries` via existing bus writer and `confirm_landings`.
- Config: Hub `app/settings.py`, `.env_example`, `docker-compose.yml`, `requirements.txt`; no changes required.
- Contracts/channels: no new schema, event, channel, metric, or telemetry consumer.
- Tests/evals: Hub queue/pipeline/Postgres suites and existing offline handoff/receipt evals.

## Changes

Both stages now claim FIFO within priority. Removing attempts-first ranking prevents fresh feed arrivals from continually jumping ahead of older retries. Queue-position reporting uses the identical ordering.

Known pre-reader admission/caller-budget deferrals stay pending without incrementing attempts. Existing wallet refund backoff continues to pace them. Unknown failures, intentional stance refusals, malformed output, and failures after the reader starts retain their bounded behavior. Historical attempt counts and terminal rows are not rewritten.

`python3 -m orion.world_pulse_read.verify --url URL` runs SELECTs in a read-only repeatable-read transaction. It resolves aliases and checks stored typed artifacts, existing source-fetch evidence, matching seed/trace IDs, nonempty journal bodies, and landing confirmation. Exit 0 is a pass, 2 is incomplete, and 1 is unavailable. It does not infer semantic correctness or introduce another pipeline detector.

## Acceptance evidence

The new verifier was executed inside the live Hub container with the worktree module supplied to a temporary Python process, without altering deployed files. For the arXiv URL it returned exit 2, `verified_complete=false`, both stages pending, missing handoff/result/journals, and unconfirmed landing. This correctly reproduces the failure rather than treating queue presence as success.

Regression coverage includes repeated admission failures beyond the configured attempt cap, then actual reader failures exhausting it; FIFO in both real PostgreSQL claim queries; direct/alias/missing read-only verification; refusal vs capacity classification; and rejection of hollow or unlanded artifacts. Existing offline reading evals remain part of the checks. Final check counts and CI status are recorded in the PR body.

## Review findings fixed

- Verifier test expected a custom empty-text label, but existing schemas already reject blank learning/summary. Reused the schema validation and corrected the assertion.
- Added database coverage for direct, alias, and missing URL verification, with unchanged rows before/after.
- Updated the wallet documentation and test comment to distinguish admission failures from spent reading attempts.

Independent requesting-code-review review found no remaining material code issues. Operational limitation: FIFO and non-exhausting admission retries preserve work but cannot provide compute; persistent admission failure can still hold younger same-priority work behind the oldest row.

## Controlled production recovery proposal

Approval scope: deploy the reviewed Hub patch; snapshot and temporarily prioritize only the pending arXiv seed; run that source through both normal stages with their normal evidence/materialization/journal code; permit a temporary schedule/backoff override for this bounded smoke only, recording and restoring affected values. Do not bulk-requeue historical failures, mark success manually, suppress stance refusals, or bypass source evidence.

Before writes, store the target queue row and any changed Redis/env values in an owner-only `/tmp/reading-recovery-<timestamp>/` directory. Preserve existing request/alias lineage. Capture stage correlation IDs and actual admission decisions. Stop at the first execution failure and inspect that exact trace before choosing the next change. A successful smoke requires the verifier to exit 0; then restore temporary priority/scheduling settings and observe a subsequent scheduled reading. If the first run stalls in stance or FCC, the pipeline is still not repaired.

Deploy command from the worktree with normal local env provisioning:

```sh
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

Rollback: restore the prior Hub image and any snapshotted operator settings. No schema migration is needed. No production writes, restart, retry, or budget override were performed in this patch.
