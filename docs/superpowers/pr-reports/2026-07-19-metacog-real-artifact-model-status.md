# Metacog real-artifact model — status readout

Branch: `feat/metacog-real-artifact-model` · PR: [#1208](https://github.com/junebug-junie/Orion-Sapienform/pull/1208) · Not yet merged.

Design doc: `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` (has full narrative history; this doc is the dense current-state snapshot).

## What shipped, verified live (not just code-reviewed)

- **`collapse_mirror` split.** Strict-lane only (Juniper's Hub-authored entries) going forward, zero schema change, zero history migration. `orion-dream`'s aggregation query scoped to match (`services/orion-dream/app/aggregators_sql.py`).
- **New `MetacogEntryV1` schema + `orion_metacog` table** (`orion/schemas/metacog_entry.py`, `services/orion-sql-writer/app/models/metacog_entry.py`). Real-artifact only — `biometrics`, `turn_effect`, `substrate_eventfulness_score`, `llm_uncertainty`, `reasoning_excerpt`, `repair_pressure` — no self-report field (`numeric_sisters` dropped entirely, not decontaminated, gone). `snapshot_kind` is a real `Literal["baseline","confirmed_dense"]`, not the free string that drifted into 38 garbage values on the old table.
- **Relational trigger re-sourced from `repair_pressure_v2`**, replacing `turn_change_classify`'s SHIFT gate. Architecture: Option A (new bus channel `orion:repair_pressure:appraisal`, `orion-hub` publishes post-appraisal, `orion-equilibrium-service`'s `repair_pressure_metacog_gate.py` subscribes) — chosen over inlining the gate into cortex-exec, because the data already exists in `orion-hub`'s process at appraisal time and a new channel avoids a round-trip.
- **`severity`/`touches`/dynamic `provenance`** — added in a correction pass after the first implementation flattened this content despite the design doc specifying it. `severity` (nominal/degraded/critical) off real failure-prefixed log lines + `orion-llm-gateway`'s logprob-margin telemetry. `touches` mechanically names which real-artifact fields are populated. `provenance.source`/`impacts` derived from `trigger_kind`/`touches`, not a hardcoded constant.
- **`repair_pressure_appraisal_log`** — new standalone Postgres table, durable log of *every* `repair_pressure_v2` appraisal (gated or not), fed by `orion-sql-writer` as a second consumer on the existing appraisal channel. Built because the signal had zero durable observability before this (ephemeral docker logs only, wiped by every container restart; nothing in Postgres).
- **Full pipe verified live tonight**, not just tested: synthetic publish → `orion-sql-writer` consume → Postgres insert, confirmed via a real row landing in `repair_pressure_appraisal_log` and then deleted. `orion_metacog`'s wiring is the same pattern, same confidence level, not yet independently synthetic-tested post-`.env`-fix.

## Bugs found and fixed this session (not in the original plan)

| Bug | Found via | Fix |
|---|---|---|
| `severity`/`touches` silently dropped on every `orion_metacog` insert | Checking SQL model columns before adding the appraisal table | Added matching SQLAlchemy columns, regression test |
| `channels.yaml` `schema_id: MetacogRepairPressure` for a payload that includes `correlation_id` (a field that schema doesn't have) | Building the dedicated appraisal schema | Real dedicated `RepairPressureAppraisalV1` schema, `schema_id` corrected |
| Live `.env` for `orion-sql-writer` never synced across **two separate deploys** (missing `orion:metacog:sql-write` from the first PR *and* `orion:repair_pressure:appraisal` from this one) | `orion_metacog`/`repair_pressure_appraisal_log` staying at 0 rows despite triggers firing | Edited the shared-checkout `.env` directly (backed up first), verified 62/62 channels live, verified end-to-end with a synthetic publish |
| Severity-classification logic counted routine `"exec ->"` log markers as failures, would have made every ordinary turn read `"degraded"` | A test I wrote for the fix itself | Narrowed to `fail`/`error`/`exception` prefixes only |

## Open, unresolved — real, not hedging

- **`repair_pressure`'s confidence structurally floors at exactly `0.0`** unless some evidence kind scores `>0.5` (`repair_pressure_v2.py:91-97`, `min()` over active confidences, empty → `0.0`). Every real sample observed tonight (9 total, two container restarts) showed the *identical* `level=0.087 confidence=0.000 evidence=7`. Plausible explanation traced (a text-only fallback path scores discrete YES/NO answers with fixed constants when logprobs aren't usable — would legitimately repeat on uniformly-"NO" boring turns) but **not confirmed** which path is actually firing. Whether `confidence_floor=0.7` on the relational trigger is a sane threshold is unanswerable until `repair_pressure_appraisal_log` has real accumulated volume.
- **Independence checks never done.** `substrate_eventfulness_score` and `turn_effect` both partly derive from `self_state` — real redundancy risk in `causal_density`'s blend, untested.
- **Live-data sanity checks incomplete** for `substrate_eventfulness_score`, `turn_effect`, `biometrics`, `llm_uncertainty`, `reasoning_excerpt` — provenance traced for all, independence and degeneracy-checked for none, due to short container uptime (fleet restarted twice tonight) limiting the observable window.
- **`causal_density` blend weights (0.5/0.3/0.2) and `severity` thresholds are uncalibrated starting defaults** — same caveat the old `collapse_mirror` scoring carried, never fit against real outcome data.
- **`substrate_eventfulness_score`'s own upstream dependency (`SelfStateV1`) has a separate, already-open, unresolved signal-quality gate** in this codebase (PR #1191: confirmed masking bug, dead channel entries, an unverified pre-fix sawtooth) — this model inherits that uncertainty without re-flagging it per-instance.
- **STANCE handling** — still not decided (does a STANCE shift belong in the relational trigger, alongside REPAIR/TOPIC-equivalent evidence, or not).
- **`emergent_entity`-as-phase-of-run** — deliberately left open; the original design conversation's own framing of this was a caution, not a decision.
- **`turn_change_classify.py` retirement** — no longer used by metacog (replaced by `repair_pressure_v2`), still feeds `retrieval_intent.py`'s recall routing. Open question, not touched.

## Blocked on

- **Nothing code-side.** All touched-service tests green (124 across 6 services, run standalone due to a pre-existing, unrelated pytest cross-service module-name collision — confirmed via `git stash` not introduced by this branch).
- **Merge is a judgment call, not a technical block.** Explicit standing instruction this session: not merging until quality concerns clear. Code is tested/reviewed/live-verified; the *calibration* question (is the confidence floor sane) can only be answered with real post-merge traffic regardless of merge timing — that's a property of the question, not something more code fixes.
- **Real-data accumulation for `repair_pressure_appraisal_log`** — needs hours-to-days of real chat volume before the confidence-floor question is answerable from data instead of a single repeated sample.

## Quality-process findings, worth remembering past this PR

- **The first implementation pass flattened real, already-worked-out design content** (sequential trigger evidence, severity, topology, dynamic provenance) despite the dispatch spec citing the design doc by line number — because the spec was paraphrased from memory instead of transcribed. The gap was between what got *written into the spec*, not what the doc said or what got executed. Root cause, not blame-shifted to the implementing pass.
- **The metric quality gate (CLAUDE.md §0A) was not run systematically on the first pass** — provenance and existing-mechanism checks happened ad hoc for some fields, independence and live-data checks for none. When actually run properly (this session, against live containers/Postgres, not code-tracing), it surfaced three real problems in about twenty minutes: the confidence-floor structural issue, zero durable observability, and the silently-dropped SQL columns. The gate works when actually run; the risk is skipping it under time pressure, twice, before being pushed to do it for real.
- **`.env_example` → live `.env` sync gap recurred across two separate deploys tonight**, for two different channels, from two different PRs. This isn't a one-off — worth treating as a standing operational risk on this repo, not just tonight's incident.
- **A real, separate bug was found live and unrelated to this branch**: the FCC harness's step-preview UI showed content (a full alternate draft response) that does not appear anywhere in the persisted backend record for that turn (`chat_history_log`, full `spark_meta`, checked byte-for-byte) — most likely a frontend staleness/race-condition bug in `orion-hub`'s step-rendering, not a generation-quality issue. Not filed as its own ticket yet.
