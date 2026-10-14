## Summary

- Orion now sees its own body signals with a sense of scale: each one scored 0-10 against its own
  last 7 days, in the unified turn's stance step (`MY BODY RIGHT NOW`) and in reverie
  (`HOW MY BODY IS DOING`). Self-calibration spec section C, recommended next patch 4.
- Only signals that are high or unusual *for themselves* get a line (max 3, 600 chars). A calm body
  renders nothing. Every line cites the stored row its numbers came from: `[pe:<node>@<observed_at>]`.
- No new calibration math and no new registry: the attention seam's own rule
  (`NodePeHistoryCache` + `world_first.judge_candidate`) decides "unusual"; the glossary's
  `prompt_sites` decides which signals may reach a prompt.
- Mind's frontier scores are labelled in the stance prompt as Orion's own estimates, not measurements.
- Flags ship on: `SELF_READINGS_STANCE_ENABLED`, `SELF_READINGS_REVERIE_ENABLED`; false = that prompt
  byte-identical to before (tested).

## Outcome moved

Before: Orion read bare numbers about itself (or none) with no range, percentile, or trend
("what is 2 vs 10000"). After: when a body signal is unusual, the stance and reverie prompts carry a
plain line with its 0-10 score, its percentile against its own week, and the row it came from. Live
replay over the last day (7 moments): 7/7 rendered blocks, **0 truth-to-rows failures**.

## Current architecture

- `PredictionErrorMagnitudeV1` / `NodePeHistoryCache` / `world_first.judge_candidate` already
  calibrated every `node:substrate.*` prediction error against its own 7 days for the field
  attention contest; nothing rendered it into a prompt.
- stance_react (orion-thought `app/bus_listener.py`) already fetched a fail-open "my day" cue
  alongside Mind; reverie (`app/reverie.py`) already gathered loop outcomes and percepts concurrently.

## Architecture touched

- `orion/attention/self_readings.py` (new, pure): build one reading per prompt-reaching body node,
  render the block.
- `services/orion-thought/app/self_readings_cue.py` (new): process-wide 7-day cache over
  `substrate_node_prediction_error_history` (read-only), bounded thread read, fail-open.
- stance_react and reverie prompt context + templates; `main.py` warms the cache at startup.

## Files changed

- `orion/attention/self_readings.py`: readings + renderer (states live/no_reading/stale/insufficient_history; trigger as rate).
- `orion/schemas/self_reading.py`, `orion/schemas/registry.py`: `SelfReadingV1`, `SelfReadingsBlockV1` (typed view; not on the bus).
- `orion/attention/pe_history_cache.py`: public `history(node_id)`.
- `config/field/field_channel_glossary.v1.yaml`: `prompt_sites` on 7 node-qualified prediction_error entries.
- `orion/metrics/semantics.py`: pin chat/route/codebase/cabinet in `PROMPT_INVENTORY_URNS`.
- `config/metrics/metric_definitions.lock.json`: re-locked (prompt_sites changed).
- `orion/schemas/context_provenance.py`: `self_readings` (live_runtime_projection), `self_readings_label_estimates` (plumbing).
- `orion/cognition/prompts/stance_react.j2`, `reverie_narrate.j2`: the blocks.
- `services/orion-thought/app/{bus_listener,reverie,main,settings}.py`, `.env_example`, `Dockerfile` (COPY `config/field`), `README.md`.
- Tests: `tests/test_attention_self_readings.py`, `tests/fixtures/self_readings_history.py`,
  `services/orion-thought/tests/test_stance_self_readings.py`, `tests/conftest.py` (refused DSN in tests).
- Evals: `services/orion-thought/evals/self_readings_truth.py`, `test_self_readings_truth_eval.py`,
  `run_self_readings_live_eval.py`.
- `.github/workflows/self-readings-tests.yml`.

## Schema / bus / API changes

- Added: `SelfReadingV1`, `SelfReadingsBlockV1` (registered; not published on any channel).
- Removed / renamed: none.
- Behavior changed: stance_react / reverie prompt context gains `self_readings` when a body signal is unusual.
- Compatibility notes: no bus payload change; the text rides the existing cortex-exec plan context.

## Env/config changes

- Added keys (orion-thought): `SELF_READINGS_STANCE_ENABLED=true`, `SELF_READINGS_REVERIE_ENABLED=true`, `SELF_READINGS_FETCH_TIMEOUT_SEC=0.5`.
- Removed / renamed: none.
- `.env_example` updated: yes.
- local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys orion-thought` (the default
  run skips keys outside SYNC_PREFIXES); primary checkout's `services/orion-thought/.env` now has all three.
- skipped keys requiring operator action: none.

## Metric gate (CLAUDE.md 0A), for the signals newly reaching a prompt

1. **Provenance:** each is a `node:substrate.*` prediction_error written by orion-substrate-runtime
   into `substrate_node_prediction_error_history`; producer lines are cited per entry in the glossary.
2. **Independence:** seven distinct producers (biometrics hint deltas, bus edge anomaly share, execution
   / chat / codebase clipped z-scores, route decision changes, cabinet warming trigger). Not new
   metrics -- the same values the field contest already ranks; this patch only renders them.
3. **Theory anchor:** percentile against the signal's own week, in the bad direction from the
   semantic layer (`derived_channel_polarity`), the spec's section A/C rule.
4. **Live sanity (7 days to 2026-10-14 03:26 UTC):** biometrics 13,813 rows, 0% zeros, p50 0.022 /
   p90 0.077; bus_synaptic 13,735, p50 0.032 / p90 0.065; perception 13,701 (excluded: world, not
   body); cabinet 13,395, 98.4% zeros (trigger); execution 3,837, 73% zeros, p90 0.53; route 1,544,
   87% zeros; codebase 700, 89% zeros; chat 575, 70% zeros; harness_closure 183, constant 0.65
   (placeholder, excluded). All can return to rest (0 / own median) and the live replay showed calm
   moments render nothing.
5. **Existing mechanism:** reused `NodePeHistoryCache`, `compute_prediction_error_magnitude`,
   `judge_candidate`, `chat_rate_reading` (for the trigger's rate), glossary semantics.
6. **Reversibility:** two flags; removing the glossary `prompt_sites` lines turns a signal off.

## Tests run

```text
pytest tests/test_attention_self_readings.py tests/test_attention_pe_history_cache.py tests/test_attention_world_first.py tests/test_attention_world_first_field_frame.py tests/test_attention_event_decay.py tests/test_metric_prompt_semantics.py tests/test_metric_rest_range.py tests/test_field_channel_glossary.py orion/thought/tests orion/temporal_self/tests  -> pass
cd services/orion-thought && pytest tests -> 1 failed, 610 passed, 28 skipped
  (failure: test_settings_mind_enrichment::test_mind_enrichment_defaults_off -- fails identically on main, env-sensitive)
orion/schemas/tests/test_context_provenance.py::test_static_ctx_assignments_covered -> fails identically on main (executor.py keys, unrelated)
scripts/check_metric_lineage.py --gate -> PASS; --prompt-semantics -> PASS (10 prompt-reaching metrics)
scripts/check_definition_drift.py --gate -> PASS (after --update)
scripts/check_inner_state_registry.py -> OK; scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
cd services/orion-thought && pytest evals -> 13 passed (incl. test_self_readings_truth_eval.py: 5 passed)
python services/orion-thought/evals/run_self_readings_live_eval.py --moments 6  (live, read-only)
  moments=7 rendered=7 truth_failures=0
  block chars p50=359 max=523
  seed+render ms p50=353 max=428; incremental+render ms p50=4.0 max=4.6
```

Example block, real rows (2026-10-14 02:35:26 UTC):

```text
- how many of my message-bus links look abnormal: 10/10 for me (now 0.16; higher than 99% of my last 7 days) [pe:bus_synaptic@2026-10-14T02:35:06Z]
- surprise in how my task execution is going: 10/10 for me (now 1.00; higher than 95% of my last 7 days) [pe:execution@2026-10-14T02:34:30Z]
- how unusual the latest change to my own code was: 9/10 for me (now 0.32; higher than 93% of my last 7 days, rising) [pe:codebase@2026-10-14T02:33:58Z]
Nothing else I can read is unusual right now. No recent events: chat.
```

## Acceptance check 5 (cost)

- Prompt: the block plus its header adds 357 chars of fixed text + the block (p50 359, max 523 live,
  capped at 600): about +716 chars (~180 tokens) typical, +957 worst case, and only on turns where a
  body signal is unusual. Calm turns add 0.
- Read: runs concurrently with Mind enrichment; incremental read+render 4-5 ms; the cold 7-day seed
  (~350 ms) is paid once at startup in a thread.
- Stance p50 within +5%: **UNVERIFIED** live (not deployed). Estimate: ~180 prefill tokens on a
  several-thousand-token stance prompt is a low single-digit % of prefill and far less of total
  stance latency, which is dominated by generation. Reverie hollow/discard rate: UNVERIFIED until deployed.

## Docker/build/smoke checks

```text
Not built (deploy not requested). Verified in the running orion-athena-thought container that
importing orion.attention.world_first works (0.59 s, loads orion.substrate) -- hence the lazy import.
Dockerfile now COPYs config/field so the glossary resolves inside the image.
```

## Review findings fixed

(filled after review)

## Restart required

Consumer-first, from the primary checkout on main. cortex-exec renders the prompt templates from
its own image (`orion/cognition/prompts` is baked in), so it goes first; orion-thought produces the
new context key. Either order is harmless (an unknown ctx key is ignored by the old template).

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build && docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Concern: bus_synaptic sits at its week's top 1% often (5 of 7 replayed moments), so
  it may appear in many blocks. Mitigation: that is the same rule attention uses; revisit if Orion
  narrates it on every turn.
- Severity: low. Concern: first chat turn after a restart before the warm finishes renders nothing.
  Mitigation: fail-open by design; warm takes ~0.4 s.

## PR link

(filled after PR creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
