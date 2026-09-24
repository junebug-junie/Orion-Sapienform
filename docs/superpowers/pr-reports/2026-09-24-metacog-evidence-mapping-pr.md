## Summary

Metacog rows now describe the event that caused them, instead of how confident the writing model felt. Every `orion_metacog` row's severity, density score, evidence list and "touches" are now computed straight from the trigger's own numbers (the `upstream` block every gate already attaches), by one pure function per trigger kind.

- New `orion/metacog/evidence_map.py`: per-kind mappers (transport in all four upstream shapes, including the new condition-shaped one the section-A gate will emit; telemetry_anomaly, chat_turn, relational, repair_pressure_trend, insight, flow, baseline, llm_surface_instability, manual, dense/pulse). Malformed or unknown upstream maps to nominal / magnitude 0 / `no_evidence`, never an exception.
- `MetacogPublishService` (cortex-exec) takes severity, `causal_density`, `what_changed` and `touches` from the mapper. The pipeline's own step log moves out of `what_changed.evidence` into a new `provenance.pipeline_steps`.
- The second LLM call per row (the logprob "uncertainty probe") is gone, along with its three env keys. That halves LLM calls per metacog row (about 2,400 a day).
- The draft prompt no longer shows `zen_state` / `pressure` (flat at 0.965 and the source of the "zen persists" narration). It shows the mapped event evidence instead. The LLM now writes only summary, mantra and tags. If the draft fails, the row publishes a deterministic summary built from the evidence.
- A read-only replay (`scripts/analysis/replay_metacog_capture.py`) over 7 days of stored triggers, a fixture-driven eval, and a focused CI workflow.

## Outcome moved

On 16,375 real triggers from the last 7 days (replay, read-only):

| check | old (published) | new |
|---|---|---|
| transport latency: Spearman(severity, p95 / limit) | -0.024 | 0.614 |
| transport timeouts: Spearman(severity, timeout_count) | -0.001 | 0.961 |
| telemetry: Spearman(severity, recon_loss / threshold) | -0.154 | 0.843 |
| bus_synaptic: Spearman(severity, error / threshold) | -0.009 | 0.632 |
| distinct `causal_density` scores per day | 1-2 | 140-1,355 |
| nominal rows above the median critical row of the kind | n/a | 0 |

The honest reading: the old severity was **uncorrelated with the event** (rho ≈ 0 or negative), and baseline rows with identical inputs split 110 critical / 43 degraded. The new rho values are *not* independent validation. The proxies are the same upstream numbers the mapper bands, so a monotone mapper passes by construction. The check is a regression guard that the mapper reads and orders the right fields. In each gated row the new rho equals its "tie-limited ceiling", meaning severity is perfectly ordered by the raw number. The ceiling is below 1 only because three severity levels can't rank thousands of distinct values. Whether the band *edges* are right is still provisional (spec table).

## Current architecture

cortex-orch dispatches `log_orion_metacognition`. cortex-exec then runs Context, Draft (LLM) and Probe (a second LLM call) steps, then Publish. Publish scored severity with `compute_severity(llm_uncertainty, non_ok_step_count)`, scored density with `compute_causal_density(MetacogRealState)` (a blend of global state, only ever 0 or 0.25), and appended the last 20 pipeline log lines to `what_changed.evidence`. The draft prompt rendered `Pressure:` and `Zen state:` lines.

## Architecture touched

- `orion/metacog` (shared library): new mapper plus replay analysis; `service.py` reduced to density and provenance assembly.
- cortex-exec: Draft, Context and Publish steps in `executor.py`, plus settings, `.env_example` and compose.
- Shared prompt `log_orion_metacognition_draft.j2`, which cortex-orch's image also ships.
- `MetacogProvenance` schema gains an optional field (additive).

## Files changed

- `orion/metacog/evidence_map.py`: new pure per-kind mapper.
- `orion/metacog/capture_replay.py`: new pure replay analysis (Spearman, tie-limited ceiling, checks 4-6).
- `orion/metacog/service.py`: old severity/density/touches scorers retired; density and provenance now come from the mapping.
- `orion/metacog/__init__.py`: exports.
- `orion/schemas/metacog_entry.py`: `MetacogProvenance.pipeline_steps` added; field comments now describe the new meanings.
- `orion/cognition/prompts/log_orion_metacognition_draft.j2`: zen/pressure removed; measured-event block added; LLM contract is now summary/mantra/tags.
- `services/orion-cortex-exec/app/executor.py`: probe call and helpers removed; publish rewired; `metacog_event_evidence` cue; `Pressure:` dropped from `context_summary`.
- `services/orion-cortex-exec/app/settings.py`, `.env_example`, `docker-compose.yml`: three probe keys retired.
- `services/orion-llm-gateway/README.md`: stale reference to the retired keys removed.
- `scripts/analysis/replay_metacog_capture.py`: read-only replay CLI.
- `orion/metacog/evals/run_capture_eval.py`, `test_capture_eval.py`: acceptance eval (checks 4-6).
- `orion/metacog/tests/test_evidence_map.py`, `test_capture_replay.py`, `test_service.py`, `fixtures/metacog_trigger_sample.jsonl` (377 real trigger rows: stratified, free text truncated, IPs scrubbed).
- `services/orion-cortex-exec/tests/test_metacog_publish_lane.py`, `test_metacog_two_pass_draft.py`: updated for the new meanings; added prompt and fallback-summary tests.
- `.github/workflows/metacog-capture-tests.yml`: new focused CI gate.

## Schema / bus / API changes

- Added: `MetacogProvenance.pipeline_steps: list[str]` (optional, default `[]`).
- Removed: none (no columns, no channels).
- Renamed: none.
- Cut-over marker: every new row carries the tag `severity_def:event_v1`, and its `causal_density.rationale` starts with `event_magnitude[` or `no_evidence[`. Queries that span 2026-09-24 must split on that tag: the `severity` column holds two definitions across the deploy. Juniper approved the redefinition via the spec (PR #2309, section B).
- Behavior changed (metric-definition change): `severity`, `causal_density`, `is_causally_dense`/`snapshot_kind`, `touches`, `provenance.impacts`, `what_changed` and `state.llm_uncertainty` on `metacog.entry.v1` now mean "derived from the trigger's own upstream". `state.llm_uncertainty` is now only set for `llm_surface_instability` triggers (copied from their upstream).
- Compatibility notes: `MetacogProvenance` is `extra="ignore"`. An un-rebuilt sql-writer silently drops `pipeline_steps` rather than failing, so rebuild sql-writer to persist it. There are no production readers of `orion_metacog`. Historical rows are not rewritten (spec non-goal), so rows before the deploy use the old meanings.

## Metric quality gate (severity / causal_density redefinition)

1. **Provenance.** Values come from `metacog_trigger.upstream`, built by the producing gates. transport: `services/orion-equilibrium-service/app/transport_metacog_gate.py` (`build_transport_metacog_trigger_from_snapshot`/`_from_grammar_atom`/`_from_bus_synaptic`). telemetry: `telemetry_anomaly_metacog_gate.py`. chat_turn: `chat_turn_metacog_gate.py::evaluate_chat_turn_gate_conditions`. relational and trend: `repair_pressure_metacog_gate.py` and `repair_pressure_trend_gate.py`. insight and flow: `insight_metacog_gate.py` and `flow_metacog_gate.py`. Mapped by `orion/metacog/evidence_map.py::map_trigger`.
2. **Independence.** Severity and density are two views of one number: density score = magnitude, and severity = the band that magnitude falls in. They are one signal, not two, and the code says so (`banded()` returns both). Within transport, timeouts and latency come from the same sensor (timeouts are its censored tail), so they are combined by max, not summed. `bus_synaptic` is a separate sensor. The removed probe was not independent of the event; it was independent of *everything* (it measured the writer).
3. **Theory anchor.** Severity is an ordinal read of each gate's own exceedance ratio against the threshold that gate already uses (recon ratio vs 3x p95; p95 vs the gate limit; z vs the gate's z). The band edges are the spec's provisional table. A minimum sample count comes from the spec (A3.2): a two-call "p95" is the slowest call, so it is capped at degraded.
4. **Live-data sanity.** Replay over real data: density is not degenerate (140-1,355 distinct values per day, versus 1-2 before), and it reads 0 exactly when the magnitude is 0. There is a genuine rest state: baseline and no-evidence rows sit at 0. Severity is not saturated; transport splits 7,225 nominal / 7,880 degraded / 524 critical. Most degraded rows are single `timeout_count=1` windows, which is real evidence on today's pooled snapshot. Section A's per-hop gate is the fix for the volume.
5. **Existing mechanism.** `orion/metacog/service.py` held the old scorers; they are replaced in place, not duplicated. No other mapper from trigger upstream to row existed.
6. **Reversibility.** No columns changed and nothing feeds training defaults. Reverting the PR restores the old scoring. Rows written in between keep the new meanings; they are distinguishable by `causal_density.rationale` starting `event_magnitude[` or `no_evidence[`.

## Env/config changes

- Added keys: none.
- Removed keys: `CORTEX_METACOG_RETURN_LOGPROBS`, `CORTEX_METACOG_LOGPROB_PROBE_MODE`, `CORTEX_METACOG_UNCERTAINTY_PROBE_ENABLED` (cortex-exec). They were removed from settings, `.env_example` and compose.
- Renamed keys: none.
- `.env_example` updated: yes (removal plus a comment).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran for orion-cortex-exec; "No changes needed". The sync script only adds keys, never removes them. The three retired keys are still in the primary checkout's `services/orion-cortex-exec/.env`. They are inert, because settings use `extra="ignore"` and no code reads them. Operator action: delete those three lines.
- Skipped keys requiring operator action: the three removals above.

## Tests run

```text
orion/metacog + orion/schemas/tests/test_metacog_entry.py      198 passed
services/orion-cortex-exec: test_metacog_publish_lane.py, test_metacog_two_pass_draft.py,
  test_metacog_trend_cue_prompt_render.py, test_metacog_draft_trigger_kind_type_mapping.py,
  test_metacog_trigger_lineage.py                                 all passed
tests/test_metacog.py tests/test_metacog_phase_contract.py      28 passed (PYTHONPATH incl. service dir)
services/orion-sql-writer/tests/test_metacog_entry_sql_shape.py 10 passed
static gates: metric_lineage, definition_drift, inner_state_registry, scripts stdlib shadow,
  hostname refs, compose relative mounts, sentience instruments, env key single source,
  async routes, chat route poachers                              all PASS
```

## Evals run

```text
python orion/metacog/evals/run_capture_eval.py   PASS
  rows=394 (377 real + 17 synthetic transport_baseline) distinct_density=158
  rho == tie-limited ceiling for every gated proxy (stratified fixture; the live replay gates the flat 0.6)

python scripts/analysis/replay_metacog_capture.py --days 7   acceptance=PASS (16,375 rows)
  report: /tmp/metacog-capture-replay/report.md, summary: /tmp/metacog-capture-replay/summary.json
```

Acceptance check 6 for LLM-written summaries ("under 5% mention zen") is **UNVERIFIED** until deployed rows exist. Only the deterministic half is proven: fallback summaries contain no "zen" (0 of 16,375), and the prompt no longer renders zen or pressure.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-cortex-exec build   exit 0 (all 4 lane images built)
scripts/safe_docker_build.sh orion-cortex-orch build   exit 0
Not deployed (no `up`), per task instructions.
```

## Review findings fixed

A code review subagent ran against the `origin/main...HEAD` diff. It found no blockers. The fixes:

- Finding: `llm_surface_instability` read only `unstable_span_count`, but orion-mind's gate also fires on `mean_top1_margin < 0.75` or low-logprob ratio > 0.15, so those rows mapped to "no event".
  - Fix: all three firing conditions are now banded as "how far past its own line"; the worst one wins; the evidence names `instability_detail`. Reads `low_logprob_token_count` (was the wrong key).
  - Evidence: `test_llm_surface_instability_every_mind_gate_condition` (3 cases), plus calm-input → `no_evidence`.
- Finding: the ContextService line that fills the prompt's MEASURED EVENT block was untested; the prompt test set it by hand.
  - Fix: `test_context_service_populates_measured_event_evidence_for_the_prompt` drives the real MetacogContextService.
  - Evidence: mutation check. Renaming the ctx key makes the test fail (1 failed); restoring it passes.
- Finding: check 4 was described as validation, but its proxies are the numbers the mapper bands.
  - Fix: reworded as a regression guard in `capture_replay.py`, the eval docstring, the CI step name and this report.
- Finding: producer→mapper key drift was ungated.
  - Fix: `test_evidence_map_producer_contract.py` builds triggers with the real equilibrium gate builders (transport ×3, telemetry, chat_turn, relational, flow, insight) plus mind's three conditions, and asserts they are not `no_evidence`. The producer paths were added to the CI workflow `paths:`.
- Finding: the severity column changes meaning in place with no row marker.
  - Fix: `severity_def:event_v1` tag stamped on every row (asserted in the fallback publish test).
- Nits fixed:
  - Transport routes on `evidence_source` only, not on any `condition` key.
  - A latency row without a threshold is `no_evidence` instead of a fake nominal 0.
  - dense/pulse rows label with their real kind.
- Not changed:
  - `what_changed.summary` duplicates `summary` in fallback mode (acceptable).
  - The replay DSN default matches repo practice.
  - The `transport_baseline` mapper has no producer on this branch (see Risks).

## Restart required

Rebuild and restart both cortex services, because cortex-orch ships the draft prompt template in its own image (PR #2065 gotcha). Also rebuild sql-writer so `provenance.pipeline_steps` is persisted:

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: medium. Concern: transport still produces about 2,300 triggers a day, and now about half are "degraded" (mostly single-timeout windows on the pooled snapshot). The rows are honest but still many. Mitigation: section A (per-hop EWMA gate, episodes) is the volume fix, being built in parallel. This mapper already handles its condition-shaped upstream.
- Severity: low. Concern: band edges are provisional (spec table). Mitigation: they are named constants in `evidence_map.py`, and the replay script re-checks them against live data.
- Severity: low. Concern: the Spearman check is near-tautological for single-proxy kinds (telemetry, bus_synaptic), so it is a regression guard, not independent validation. It is reported as such.
- Severity: low. Concern: the `transport_baseline` (condition-shaped) mapper has no producer on this branch. It is consumer-first for section A's gate, and its z / saturation bands are uncalibrated. Mitigation: it is small and tested, and falls back to `no_evidence` on any shape mismatch. Section A should run this replay once its log-only week produces data.
- Severity: low. Concern: the producer contract test calls `transport_metacog_gate.py` builders, which section A is editing concurrently. A signature change there will fail this test. That is intended: it is the contract.
- Severity: low. Concern: the three retired env keys remain in the primary `.env` until an operator deletes them. They are inert.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
