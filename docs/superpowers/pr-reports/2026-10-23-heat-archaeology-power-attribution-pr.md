## Summary

- Orion can now ask "where is my power (and so the cabinet's heat) going, and since when?" and get a numbers-backed answer instead of a guess. It is a new orion-introspect tool, `power_attribution`.
- The method that found the 2026-10-14 circe step by hand is now deterministic code (`orion/physiology/heat_archaeology.py`, pure Python, no numpy). It finds steps in hourly wall watts and splits each step into GPU, CPU and rest watts. A least-squares fit then credits the watts to GPU-pool workloads and to a CPU-load regime. It also reports CPU-load shifts that no GPU lease can see.
- One implementation serves two callers. The tool is on the orion-introspect MCP server, which unified chat turns and curiosity runs both get. An orion-hub responder answers it from Postgres in a READ ONLY transaction. The urgent-run prompt names the tool beside its psql sources.
- Every answer is labeled correlational (`epistemic_status=unsettled`, `verified=false`, "UNVERIFIED until stopping the suspect drops the watts"). Every answer carries the evidence refs and the exact SQL.

## Outcome moved

Curiosity runs kept ending at "my cabinet runs hot and I don't know why." With this tool, the same question returns this on live data (14-day window, 2026-10-23):

> circe: about 1001 W since around 2026-10-14 03:00 UTC ... +181 W around 2026-10-14 03:00 UTC (820 -> 1001 W; GPU +44 W, CPU +80 W, rest +57 W): CPU-load regime (load_15m > 7.19) +83 W, which no GPU lease explains (check host processes and container restarts); harness +46 W; vision-council +36 W; cortex-exec +12 W; unexplained +3 W. Floor with no listed workload running: ~670 W. CPU load (15 min) shifts: 0.97 -> 13.3 at 2026-10-14 04:00 UTC. Correlational; UNVERIFIED ...

The 30-day window finds the same change through the CPU-load shift: wall power +130 W across it, of which +119 W is the CPU-load regime. That matches the hand-found ~144 W crash loop.

## Current architecture

- Orion's tools during a harness turn come from MCP servers rendered per turn (`orion/fcc/mcp_config.py`). `orion-introspect` (`orion/introspect/`) is attached to every turn that has a reading binding: `unified_chat` (Hub chat) and `curiosity` (every curiosity run, urgent runs included). Each tool sends an RPC over the bus to the service that owns the data. `node_history` is the closest pattern: Hub answers from Postgres in a READ ONLY transaction.
- Curiosity sandboxes also have `psql` with the `orion_readonly` role. Urgent runs are told to read `orion_biometrics_summary`, `home_cooling_sample`, `gpu_pool_events` and `durable_run_workflow` by hand.
- Before this patch, no code attributed watts to workloads. The Energy tab (`energy_routes.py`) shows kWh and dollars. The Body organ (`orion/physiology/body.py`) reads levels against lines. sql-writer's `orion_energy_rollup.py` writes hourly kWh.

## Architecture touched

- New pure analysis module: `orion/physiology/heat_archaeology.py`.
- New SQL reader: `orion/introspect/power_attribution.py`.
- New Hub bus responder: `services/orion-hub/scripts/power_attribution_introspect_listener.py`, started in `main.py` next to `node_history`.
- New introspect tool spec, argument contract and harness brief line.
- New bus channel `orion:introspect:power_attribution:request`.

## Files changed

- `orion/physiology/heat_archaeology.py`: step detection, autocorrelation-corrected statistics, OLS, attribution, plain-English summary.
- `orion/schemas/heat_archaeology.py`: typed result (`NodeHeatArchaeologyV1` and parts).
- `orion/introspect/power_attribution.py`: the three SELECTs, row merging, item building with an MCP-budget shed order.
- `orion/introspect/tools.py`, `brief.py`, `transport.py`: tool spec, description, brief line, channel constant.
- `orion/schemas/introspect.py`: `PowerAttributionArguments`, the operation literal, window bounds.
- `orion/schemas/registry.py`, `orion/schema_skew_discovery.py`, `orion/bus/channels.yaml`: contract registration.
- `services/orion-hub/scripts/power_attribution_introspect_listener.py`, `main.py`: the responder.
- `orion/curiosity/urgent_prompt.py`: names the tool beside the Postgres sources (only when Postgres is offered).
- `config/physiology/contract.yaml`: registers the reader as `display` (it gates nothing).
- `config/metrics/metric_definitions.lock.json`: re-locked for the new bus channel.
- `.github/workflows/orion-reading-tests.yml`, `curiosity-urgent-prompt-tests.yml`: CI paths and tests.
- `services/orion-hub/README.md`, `services/orion-harness-governor/README.md`: responder section and tool row.
- Tests and evals: `orion/physiology/tests/{heat_fixture,test_heat_archaeology}.py`, `orion/introspect/tests/{power_attribution_fake,test_power_attribution}.py`, `orion/introspect/tests/test_introspect_tools.py`, `services/orion-hub/tests/test_power_attribution_introspect_listener.py`, `services/orion-hub/evals/{run_power_attribution_eval,test_power_attribution_eval,run_power_attribution_live_eval}.py`, `tests/test_curiosity_urgent_prompt.py`.

## Schema / bus / API changes

- Added: channel `orion:introspect:power_attribution:request` (`IntrospectRequestV1`, producer orion-harness-governor, consumer orion-hub, single consumer). Added `PowerAttributionArguments` (registered) and the `power_attribution` operation. Added result models in `orion/schemas/heat_archaeology.py`; these travel inside `IntrospectItemV1.extra`.
- Removed: none.
- Renamed: none.
- Behavior changed: the urgent-run prompt gains two lines naming the tool.
- Compatibility notes: additive. An old governor never sends the operation. An old Hub never answers it, so the tool returns "answer unknown" after its 15 s timeout until Hub is redeployed.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Proposal mode (Orion capability)

- **What capability changes:** Orion can attribute their own machines' watts to workloads over time, from chat or from curiosity.
- **What data is touched:** read-only SELECTs on `orion_energy_hourly`, `orion_biometrics_cluster`, `gpu_pool_events` and `durable_run_workflow`. The reads run in a READ ONLY transaction through Hub's existing pool. Nothing is written, no container is touched, and no grant change is needed (Hub's own pool, not `orion_readonly`).
- **Privacy boundary:** none. These are hardware telemetry and job names only.
- **What trace proves it worked:** the Hub log line `power_attribution_introspect_answered corr=<id> node=<n> days=<d> items=<n> steps=<n>`, the reply on `orion:introspect:result:<corr>`, and the tool call in the turn's MCP stream. The bus path stays UNVERIFIED until Hub and the governor are redeployed (see Risks).
- **Dangerous failure mode:** a confident-sounding wrong attribution. Juniper could then stop the wrong service. Three things limit this: the statistics are autocorrelation-corrected, the text always says UNVERIFIED and names an intervention, and a term is named only when it is 2 corrected SEs from zero.
- **Disable / rollback:** revert the PR, or remove the `ToolSpec` line in `orion/introspect/tools.py`. The responder only answers requests, so with no requests it does nothing.

## Metric quality gate (for the new derived signals)

1. Provenance: wall watts come from `orion_energy_rollup.py` (chassis_watts integrated). GPU/CPU watts and load come from biometrics `measurements_by_node`. Busy-hours come from `gpu_pool_events.held_ms` on end rows.
2. Independence: GPU and CPU watts are parts of wall watts. They are reported as a decomposition and never fitted as regressors. The regime indicator is derived from `load_15m`, which can partly follow GPU work; that is stated as a confound.
3. Theory: an additive power model, wall = floor + Σ watts per busy-hour × busy-hours + regime. Steps are found by binary segmentation with Bartlett AR(1) effective-sample correction. The regime is an Otsu two-class split, kept only above 0.85 separability. A one-moded normal scores 2/π ≈ 0.64, a uniform 0.75, an exponential ~0.66.
4. Live sanity: on live data, circe load is clearly two-moded (separability 0.97: idle ~1, crash loop ~13). athena (0.61) and hecate (0.21) are correctly refused a regime. The circe residual autocorrelation is 0.45-0.62.
5. Existing mechanism: none attributes watts to workloads (checked the Energy tab, the Body organ, the energy rollup, power-guard and prediction).
6. Reversibility: no stored state. Nothing writes, so removing it is a revert.

## Tests run

```text
orion/introspect/tests orion/physiology/tests orion/fcc/tests/test_reading_mcp.py orion/harness/tests/test_harness_prefix.py tests/test_curiosity_urgent_prompt.py -> 406 passed
services/orion-hub/tests/test_power_attribution_introspect_listener.py services/orion-hub/evals/test_power_attribution_eval.py services/orion-hub/tests/test_curiosity_urgent_start.py -> 36 passed
  (includes test_unified_chat_turn_reaches_power_attribution_through_the_mcp_server: the real MCP server
   with a unified_chat binding lists the tool; a power_attribution call travels over a bridged bus to the
   real Hub responder and returns the 10-14 attribution with evidence; READ ONLY transaction asserted)
orion-static-gates.yml commands (all 42, incl. definition drift, body-economy contract, schema skew) -> pass
```

## Evals run

```text
python services/orion-hub/evals/run_power_attribution_eval.py   (fixture, 25 seeds + negative control)
{"recall": 1.0, "median_hour_error": 0.0, "max_hour_error": 4.0, "median_regime_error_w": 3.0,
 "median_vision_error_w": 2.5, "false_alarm_rate": 0.0, "pass": true}

POSTGRES_URI=... python services/orion-hub/evals/run_power_attribution_live_eval.py --days {7,14,21,30}   (live smoke, read-only)
7d:  10-14 outside window, PASS
14d: 10-14 (wall_step) +181 W at 2026-10-14 03:00, regime +83 W, R2 0.863, PASS
21d: 10-14 (load_shift) +163 W at 2026-10-14 04:00, regime +107 W, R2 0.857, PASS
30d: 10-14 (load_shift) +130 W at 2026-10-14 04:00, regime +119 W, R2 0.797, PASS
(every step's before/after watts cross-checked against a direct AVG on orion_energy_hourly; results 8-11k chars, under the 12k MCP budget)
```

## Docker/build/smoke checks

```text
Not deployed, per task. No Docker build was run: no dependency, Dockerfile or compose change (both images already COPY orion/).
The bus path (governor -> Hub responder) is UNVERIFIED live until both are redeployed.
```

## Review findings fixed

- Finding: SEs assumed independent hours. Unrelated wandering columns were rated credible 74.5% of the time, and flat wandering series showed a "step" 62% of the time.
  - Fix: Bartlett AR(1) correction on the step t and on every OLS SE (`lag1`, `ar1_inflation`). The step bar was raised to 4.0 corrected SEs.
  - Evidence: `test_wandering_flat_series_rarely_reads_as_a_step` and `test_unrelated_wandering_column_is_rarely_credible`. False steps at live-like rho 0.6: 1.5-2.5%.
- Finding: the stricter bar then lost the real 10-14 step in 30-day windows.
  - Fix: attribute across each CPU-load shift too (`anchor="load_shift"`).
  - Evidence: the live eval passes at 14, 21 and 30 days.
- Finding: the budget clip could still overflow with uneven summaries (13007 chars).
  - Fix: clip the longest text first and re-measure in a loop. The docstring was corrected.
  - Evidence: `test_uneven_summaries_still_fit_the_budget`.
- Finding: leases still open at the window's end, or ending more than 1 day later, were missed.
  - Fix: an `opens` CTE (granted with no end row, 6 h ghost cap) and a 7-day end lookahead. A caveat was added.
  - Evidence: `test_reader_pins_utc_hours_inside_its_transaction`; live pool query ~1 s.
- Finding: the summary parts did not add up.
  - Fix: everything not named folds into "other terms".
  - Evidence: `test_sentence_parts_add_up_to_the_delta`.
- Finding: the regime column was dropped when it aliased a lease.
  - Fix: the regime column is listed first, so the lease is the one dropped.
- Finding: the attribution delta was measured on different rows from the step delta.
  - Fix: the attribution reports its own `before_w`/`after_w`/`delta_w` on the fitted hours (documented). The sentence shows both when they differ.
- Findings: segment rows are not hours; there was no "why no fit"; buckets used the session time zone; lease-hours are not card-hours.
  - Fix: docstrings, `fit_note`, `set local timezone = 'UTC'` inside the transaction, and the FitTermV1 doc.
- Finding: the collinearity test never checked the surviving coefficient.
  - Fix: it now asserts 2.0 or 1.0.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

(Run from the clean primary `main` checkout after merge. The governor needs the new tool spec. Hub needs the responder.)

## Risks / concerns

- Severity: medium. Concern: the fit is correlational. On live data, harness and the regime both moved on 10-14, so their split depends on the window (regime +83 W at 14 d, +119 W at 30 d). Mitigation: the text says UNVERIFIED and names the intervention. A follow-up could record intervention outcomes (stop X, watts moved by Y) as verification.
- Severity: low. Concern: a 30-day read takes ~4 s (biometrics jsonb aggregation ~3 s) against the 15 s introspect RPC timeout. Mitigation: the default window is 14 days. If the DB is under load (backups), a 30-day ask could read as "answer unknown".
- Severity: info. Concern: circe's CPU-load regime (load_15m ~13) has stayed on from 2026-10-14 to now in every window. Whatever started then may still be running. Mitigation: worth a host check (open PRs #2894/#2895 touch affectgpt and circe crash loops).
- Severity: low. Concern: `orion/physiology/**` is a CODEOWNERS-protected path. Mitigation: Juniper's review is required.

## PR link

<filled after push>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
