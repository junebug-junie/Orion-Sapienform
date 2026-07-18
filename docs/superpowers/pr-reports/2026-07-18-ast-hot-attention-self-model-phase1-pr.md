# PR report: AST/HOT attention self-model reducer (Phase 1)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1196
Branch: `feat/ast-hot-attention-self-model-phase1`
Status: **DONE_WITH_CONCERNS**

## Summary

- Phase 1 of `docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`: a read-only, pure-function reducer (`reduce_attention_self_model`) that unifies the two real, previously-disconnected attention self-models — the GWT-dispatch lane (`AttentionBroadcastProjectionV1`, Lamme `voluntary_override`) and the general field lane (`FieldAttentionFrameV1` + `SelfStateV1`) — into one inspectable `AttentionSelfModelV1` artifact.
- New schema `orion/schemas/attention_self_model.py`, registered in `orion/schemas/registry.py`. Not published to any bus channel and not wired into any live decision path — measurement instrument only, per the spec's explicit Phase 1 scoping.
- Two new read-only Postgres replay scripts under `scripts/analysis/`, mirroring `measure_origination_gate.py`'s pure-replay/I/O-layer split and CLAUDE.md sec 14's background-job report contract: `measure_ast_hot_reducer.py` (the acceptance-check replay) and `measure_self_state_signal_quality.py` (the Phase 1 "hard gate" signal-quality pass).
- **Load-bearing finding, discovered while building the replay script**: `substrate_attention_broadcast_projection` is a singleton upsert table (PRIMARY KEY on `projection_id`, exactly one row, always) — not a history table. There is no per-tick history for the GWT-dispatch lane in Postgres, so the spec's acceptance check ("find a real historical `voluntary_override` tick") is **NOT MET via Postgres replay** — honestly reported, not smoothed over.
- **Hard-gate finding**: real 48h replay of `SelfStateV1` confirms the coherence/uncertainty sawtooth named in the program charter's Missing Question 4 is still live in `SelfStateV1`'s own values (median 5-tick oscillation period, 3500+ zero-crossings each over ~84k samples) — reported for Juniper's sign-off, not fixed here (explicitly out of scope).
- 30 new unit tests (reducer + both scripts' pure layers), all passing; `orion/sentience_striving_program/README.md` updated with Phase 1 status and a correction to two stale "no `SelfStateV1`" lines in Sec 9b that predated the roadmap doc's Phase 1 scope revision.

## Outcome moved

The AST/HOT consciousness-theory scaffolding gap named in the program charter (§9b items 2/4) now has a real, inspectable, unit-tested artifact instead of "nothing builds a model of that attention." Objective 3's routing math (Phase 2+) has something real to be informed by once Juniper signs off, per the roadmap doc's own gating.

## Current architecture

Before this patch: `FieldAttentionFrameV1` (Layer 5, `orion-attention-runtime`, ~2s tick) and `AttentionBroadcastProjectionV1`/`AttentionFrameV1.voluntary_override` (GWT-dispatch/Lamme lane, `orion/substrate/attention_broadcast.py`, ~30s tick) existed as two live, real, but disconnected attention self-models — nothing unified them into a single "what's salient, why, how confident" artifact. `SelfStateV1` (Layer 6) already carried `attention_schema_type`/`attention_dwell_ticks` fields but no reducer consumed the broadcast lane at all.

## Architecture touched

- `orion/schemas/` — new schema file + registry entries (read/contract layer only).
- `orion/substrate/` — new pure reducer module + its unit tests (no existing files' behavior changed).
- `scripts/analysis/` — two new standalone, read-only measurement scripts + their unit tests, following the existing `measure_origination_gate.py`/`measure_autonomy_gate.py` pattern.
- `orion/sentience_striving_program/README.md` — status + correction only, no rewrite of the roadmap spec doc itself.

No service, Docker, bus, or env surface touched.

## Files changed

- `orion/schemas/attention_self_model.py`: new `AttentionSelfModelV1` schema.
- `orion/schemas/registry.py`: registers `AttentionSelfModelV1`, `AttentionBroadcastProjectionV1`, `VoluntaryOverrideV1` (the latter two existed as real schemas but were never registered — found while wiring this patch).
- `orion/substrate/attention_self_model.py`: `reduce_attention_self_model()`, pure function, no I/O.
- `orion/substrate/tests/test_attention_self_model.py`: 13 unit tests (override present/absent, fresh/stale/future broadcast, each input `None`, predicted-shift derivation).
- `scripts/analysis/measure_ast_hot_reducer.py`: real-data replay + report/csv/progress-log, matching CLAUDE.md sec 14.
- `scripts/analysis/measure_self_state_signal_quality.py`: Phase 1 hard-gate signal-quality pass.
- `scripts/analysis/tests/test_measure_ast_hot_reducer.py`, `scripts/analysis/tests/test_measure_self_state_signal_quality.py`: 17 unit tests for both scripts' pure layers.
- `orion/sentience_striving_program/README.md`: Phase 1 status note (§6 item 2) + correction of stale "no `SelfStateV1`" constraint text (§9b items 2/4).

## Schema / bus / API changes

- Added: `AttentionSelfModelV1` (`attention.self_model.v1`), registered but **not published to any bus channel**.
- Removed: none.
- Renamed: none.
- Behavior changed: none (read-only, additive).
- Compatibility notes: `AttentionBroadcastProjectionV1`/`VoluntaryOverrideV1` were real, already-live schemas that had never been added to `orion/schemas/registry.py` — added in the same patch since this reducer imports them; no behavior change to their existing producer/consumers.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: n/a (no service touched).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a, not needed.
- skipped keys requiring operator action: none.

## Tests run

```
source venv/bin/activate && export PYTHONPATH=<worktree>
python -m pytest orion/substrate/tests/test_attention_self_model.py scripts/analysis/tests/ -q
# 111 passed (30 new: 13 reducer + 8 ast-hot-reducer-script + 9 signal-quality-script;
# 81 pre-existing scripts/analysis + orion/substrate tests, unaffected)

python -m pytest orion/substrate/tests/ -q
# 370 passed, 8 warnings (pre-existing warnings, unrelated to this patch)
```

## Evals run

Both new scripts ARE the eval/measurement harness for this patch (per CLAUDE.md sec 11's "if the repo has no eval harness, add the smallest useful one" — these are it, following the `measure_origination_gate.py` precedent exactly). Run against real live Postgres (`postgresql://postgres:postgres@localhost:55432/conjourney`), `--window-hours 48`:

```
python scripts/analysis/measure_ast_hot_reducer.py --window-hours 48
```
Headline: 84,992 field-lane ticks replayed. `substrate_attention_broadcast_projection` confirmed live as a singleton (row count = 1). 0/84,992 ticks could be honestly joined to the single broadcast snapshot (it postdates the entire window). `attention_reason` distribution: `field_salience_only`=84,992 (100%), all others 0. **Acceptance check: NOT MET via Postgres replay** — reasoning and compensating unit-test evidence in the report; also ran a live 15-minute background poll of the broadcast projection table during this session (30s cadence, matching its real update interval) — no live `voluntary_override` observed in that window either. Full report: `/tmp/ast-hot-reducer/report.md`.

```
python scripts/analysis/measure_self_state_signal_quality.py --window-hours 48
```
Headline: 84,413 `substrate_self_state` rows replayed across all 12 real `SelfStateDimensionV1` dimensions. `coherence`: median oscillation period 5 ticks, 3,529 zero-crossings. `uncertainty`: median oscillation period 5 ticks, 3,531 zero-crossings. `transport_integrity`: median period 2 ticks, 17,572 zero-crossings (fastest of all 12). 8/12 dimensions flagged `pinned_or_flat`; 4 flagged `fast_oscillation_sawtooth_suspect`; `agency_readiness`/`execution_pressure` came back `nominal`. Full report: `/tmp/self-state-signal-quality/report.md`.

## Docker/build/smoke checks

Not applicable — no service, Dockerfile, or compose file touched. `scripts/safe_graphify_update.sh` was run per repo convention; it hit the known, pre-existing 2026-07-14 incremental-update bug (unrelated to this patch, node count dropped from 32,529 to 2,496) and correctly auto-restored `graph.json`/`manifest.json` — nothing graphify-related committed.

## Review findings fixed

- Finding: `pinned_or_flat` and `fast_oscillation_sawtooth_suspect` can co-occur on the same dimension (confirmed live for `coherence`/`uncertainty`) in a way that reads as contradictory without explanation (local rolling-std flatness vs. global zero-crossing frequency are different measurements).
  - Fix: added a "Reading `pinned_or_flat` + `fast_oscillation_sawtooth_suspect` together" paragraph to `measure_self_state_signal_quality.py`'s report renderer explaining why both can be true simultaneously (step/plateau sawtooth pattern).
  - Evidence: `scripts/analysis/measure_self_state_signal_quality.py` diff, commit `06f60c17`; re-ran `--window-hours 48` after the fix, report now includes the explanation, `coherence`/`uncertainty` numbers unchanged.
- Three additional nits noted by review (coalition_stability_score default-1.0 honesty risk one layer up in the producer, missing boundary-condition unit test for the exact staleness threshold, a cosmetic missing comment header) — reviewed and judged genuinely low-risk / out of this patch's scope; not fixed.

Full review (dispatched via `orion-repo-agent` subagent, independently verified Postgres state and re-ran tests/scripts rather than trusting prior claims): verdict **ship as-is**, no blocking or should-fix issues.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: **informational, not blocking**
  Concern: The Phase 1 acceptance check as literally written in the spec ("real historical window containing a `voluntary_override` event") cannot currently be satisfied via Postgres replay, because `substrate_attention_broadcast_projection` is a singleton upsert table with no history. The reducer's why-branching logic is proven correct via unit tests against the real `VoluntaryOverrideV1` schema, and a live 15-minute poll during this session found no override firing either — but a genuine end-to-end historical proof is currently impossible with the existing table design.
  Mitigation: Recommendation recorded in both the replay script's report and the README: if Phase 2+ (e.g. Phase 3's shadow comparison) needs real historical replay of the GWT-dispatch lane, `substrate_attention_broadcast_projection` needs an append-only companion or `AttentionBroadcastProjectionV1` needs to be published to a bus channel with retained history — a schema/bus contract change per CLAUDE.md sec 6, explicitly out of scope for this patch.

- Severity: **needs Juniper sign-off (per the spec's own hard-gate framing, not a blocker I can resolve)**
  Concern: The Phase 1 hard gate confirms the coherence/uncertainty sawtooth named in the program charter's Missing Question 4 is still live in `SelfStateV1`'s own values today (not just historically, not just at the field level) — median 5-tick oscillation period, 3,500+ zero-crossings each over 84k real samples in the last 48h.
  Mitigation: Reported plainly per the spec's own instruction ("do NOT block Phase 1 on fixing any such finding yourself... just report it clearly as a finding for Juniper's sign-off"). No `SelfStateV1` v2 attempted here. This is the concrete decision point the spec anticipated: whether `SelfStateV1` needs replacing before anything downstream of Phase 1's reducer output can be trusted.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1196
