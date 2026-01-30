# Orion Metacog/Spark — Surgical Patch Tracker

## Step 0 Invariant: Collapse Mirror split (MUST HOLD)

**Two mirror classes, never mixed by default:**

A) **Strict Collapse Mirror** (Juniper-authored)
- `observer = "juniper"`
- `origin = "collapse_mirror_service"` (we will represent this using `source_service` to avoid DB migrations)
- Goes through: **collapse-mirror** intake → triage → enrichment pipeline.

B) **Metacog Mirror** (Orion-authored)
- `observer = "orion"` **OR** `origin = "metacog"`
- Produced by equilibrium → cortex → `log_orion_metacognition` pipeline.
- MUST **NOT** go through Juniper’s triage/enrichment by default.

**Operational rule:** triage/enrichment is only for Strict Collapse Mirrors unless explicitly overridden.

---

## Current repo anchors (confirmed)

- Schema: `orion/schemas/collapse_mirror.py`
  - Contains `_coerce_change_type_payload()` (dict → string + score extraction + telemetry meta)

- Collapse-mirror service runtime:
  - `services/orion-collapse-mirror/app/bus_runtime.py`
  - `handle_intake()` currently always publishes to triage + sql-write.

- Metacog enrich/publish path:
  - `services/orion-cortex-exec/app/executor.py`
  - `MetacogEnrichService` merges LLM patch into draft and validates `CollapseMirrorEntryV2`.

- Existing smoke for change_type dict coercion:
  - `scripts/smoke_change_type_dict.py`

---

## Status (as implemented)

✅ **Patch Set 1 + Patch Set 2 implemented and tested**
- Added Juniper/Orion mirror classification helpers with strict-only triage routing.
- collapse-mirror intake now **skips triage** for non-strict mirrors (still publishes to sql-write) + logs skip reason.
- Metacog enrich/publish now validates through the shared normalization/coercion path.
- Added `scripts/smoke_mirror_split.py` and pytest coverage.

✅ **Patch Set 3 implemented and tested (defense-in-depth + origin stamping)**
- Metacog drafts and enrich merges are stamped with `source_service="metacog"` (and observer defaults) in cortex-exec.
- Added a defense-in-depth triage gate in meta-tags to skip non-strict payloads using `should_route_to_triage`.
- Added `scripts/smoke_metacog_source_service.py` and test coverage for fallback draft stamping.

**Green checks reported:**
- ✅ `python scripts/smoke_change_type_dict.py`
- ✅ `python scripts/smoke_mirror_split.py`
- ✅ `python scripts/smoke_metacog_source_service.py`
- ✅ `pytest -q tests/test_metacog.py`

---

## Patch Set 1: Collapse Mirror split (schema discriminator + routing gates): Collapse Mirror split (schema discriminator + routing gates)

### Goal
Guarantee that **metacog mirrors never hit triage/enrichment** when they pass through collapse-mirror intake.

### Smallest safe approach (no DB migrations)
1) Add **classification helpers** in schema module (discriminator logic):
   - `mirror_kind(payload_or_entry) -> "strict" | "metacog" | "unknown"`
   - `should_route_to_triage(payload_or_entry) -> bool`
   - Use **observer** and optionally **source_service/origin** if present.
   - Default: only `observer==juniper` (or origin collapse_mirror_service) routes to triage.

2) Add **routing gate** in `services/orion-collapse-mirror/app/bus_runtime.py:handle_intake()`:
   - If `should_route_to_triage(entry) == False`, **skip triage publish**.
   - Still publish to `orion:collapse:sql-write` for persistence.
   - Log with clear reason: `skip_triage reason=metacog_or_non_juniper`.

### Notes
- We will treat `source_service` as the persisted “origin” (no DB schema changes).
- This gate also protects the optional legacy metacog → collapse-mirror path.

---

## Patch Set 2: change_type dict validation (schema coercion, smallest)

### Goal
If the LLM returns `change_type` as a dict:
- unwrap → `change_type: str`
- move numeric keys → `change_type_scores`
- stash remaining metadata → `state_snapshot.telemetry.change_type_meta` (size-limited)
- prevent giant nested blobs from breaking validation / SQL writes

### Implemented
- Central coercion in schema (`_coerce_change_type_payload`).
- Metacog enrich/publish validates through `normalize_collapse_entry(...)` so coercion always applies.

---

## Patch Set 4: Turn effect signal (pre/post + post-reply) into metacog

### Goal
Expose **effect-of-turn** to metacog mirrors (and optionally per-turn introspection) without changing DB schema:
- user impact: `phi_after - phi_before`
- assistant impact: `phi_post_after - phi_post_before`
- whole turn: `phi_post_after - phi_before`

### Implemented
- Added turn-effect extraction + summary helper.
- Attached `turn_effect` to Spark snapshot metadata.
- Surfaced through Metacog context + mirror telemetry injection (draft/enrich), plus a compact context-summary line.
- Updated `introspect_spark.j2` and metacog prompts to include assistant post-ingest + TURN EFFECT section.
- Added smoke + test coverage for turn-effect extraction.

### Testing status
- ✅ `python scripts/smoke_turn_effect_signal.py`
- ⚠️ `pytest -q` fails in some environments due to **test collection** pulling in non-unit script/tests (e.g. `scripts/test_*.py`, `orion/cognition/tests/*`) which require optional deps (redis) or package aliases (orion_cognition).

---

## Patch Set 4B: Pytest collection guard (make pytest -q reliable)

### Goal
Make `pytest -q` run the intended unit suite under `./tests` without collecting integration scripts or package-internal tests.

### Smallest safe approach
- Add `pytest` config to restrict collection:
  - `testpaths = ["tests"]`
  - optionally `norecursedirs = ["scripts", "orion/cognition/tests"]`
- Prefer `pyproject.toml` `[tool.pytest.ini_options]` to avoid adding new files.

### Verification
- ✅ `pytest -q` (collects only `./tests`)

## Patch Set 5: Traces-as-heartbeat upgrade (quiet spaces)

### Goal
Keep Spark UI/state “alive” during idle periods using **heartbeat traces** without drifting φ or spamming durable telemetry.

### Implemented
- Added equilibrium spark heartbeat settings + env/compose wiring.
- Added opt-in publisher loop for `cognition.trace` heartbeats.
- Added heartbeat detection/handling in spark-introspector:
  - publish snapshot + UI update
  - skip propagation/learning
  - skip durable `spark.telemetry`
- Filtered heartbeat traces out of metacog context summaries.
- Added smoke + pytest coverage for heartbeat trace detection.

### Testing status
- ✅ `python scripts/smoke_heartbeat_trace.py`
- ✅ `pytest -q`

---

## Patch Set 6: Metacog mirror turn-effect rendering + guardrails

### Goal
Make turn-effect **visible and durable** in metacog mirrors in a deterministic way (not just “available in prompt context”), while protecting brittle fields from LLM overwrite.

### Implemented
- Added a system-owned telemetry guard.
- Enforced `turn_effect` + `turn_effect_summary` injection for Metacog draft/enrich paths.
- Added test coverage for telemetry guard and a smoke script that exercises the render path through normalization.

### Testing status
- ✅ `python scripts/smoke_metacog_turn_effect_render.py`
- ✅ `pytest -q`

---

## Patch Set 7: Turn-effect persistence in chat_history_log.spark_meta

### Goal
Persist `turn_effect` + `turn_effect_summary` into `chat_history_log.spark_meta` JSON (same correlation_id) via existing spark.telemetry → sql-writer merge.

### Implemented
- Added `turn_effect` metadata (and summary) to candidate `spark.telemetry` emissions.
- Propagated `turn_effect` + summary into `chat_history_log.spark_meta` while avoiding JSON cycles during minimal spark_meta construction.
- Added a unit test that loads sql-writer’s spark_meta helper with stubbed dependencies and verifies turn_effect fields are preserved.

### Testing status
- ✅ `pytest -q`

---

## Patch Set 8: Turn-effect analytics helper + CLI

### Goal
Provide a small analytics helper + CLI to inspect recent turn-effect deltas from `chat_history_log.spark_meta`.

### Implemented
- Added `compute_deltas_from_turn_effect` helper for safe per-section delta dictionaries.
- Added `scripts/print_recent_turn_effects.py` (prints recent rows; optional CSV).
- Extended unit tests for the new delta helper.

### Testing status
- ✅ `pytest -q`
- ⚠️ CLI runtime failed in this environment: DB host `orion-athena-sql-db` not resolvable.

---

## Patch Set 8B: CLI connection robustness (portable runs)

### Goal
Make the turn-effect CLI runnable outside Docker networks by allowing explicit DB URL/host overrides and a safe dry-run mode.

### Implemented
- Added portable DB resolution with override precedence:
  - `--db-url` > `ORION_SQL_URL` > component flags/envs > defaults.
- Default host now falls back to `localhost` when no overrides are set.
- Added `--dry-run` to print resolved connection info (redacting secrets) + query without connecting.
- Added `--sqlite-path` override.
- Added unit coverage for resolution and dry-run output.

### Testing status
- ✅ `python scripts/print_recent_turn_effects.py --dry-run --limit 1`
- ✅ `python scripts/print_recent_turn_effects.py --dry-run --db-url postgresql://user:pass@localhost:5432/db --limit 1`
- ✅ `pytest -q`

---

## Patch Set 9: Spark UI “Last Turn Effect” panel

### Goal
Surface `turn_effect` from Spark snapshot metadata in the UI without changing φ computation or DB schema.

### Implemented
- Added a “Last Turn Effect” panel to the Spark UI.
- Panel stays hidden when no `turn_effect` metadata is present.
- Wired websocket snapshot metadata through to the front-end.
- Rendered user/assistant/turn deltas with two-decimal formatting and a JSON tooltip for full details.

### Testing status
- (UI change) Verify manually in hub: send messages; panel updates; remains hidden when metadata absent.

---

## Patch Set 10: Turn-effect alerts (opt-in)

### Goal
Trigger lightweight alerts when turn-effect deltas cross configured thresholds, with cooldown to avoid spam.

### Implemented
- Added turn-effect alert evaluation + cooldown helpers.
- Added alert emission wiring in spark-introspector telemetry handling.
- Added alert configuration knobs to spark-introspector settings + env/compose wiring.
- Added heartbeat blocking.
- Added greppable audit + suppression logging.
- Added optional core-event notifications for alert fires.
- Updated smoke script output and added pytest coverage for audit logging, cooldown suppression, and heartbeat blocking logic.

### Testing status
- ✅ `python scripts/smoke_turn_effect_alerts.py`
- ✅ `pytest -q`

---

## Patch Set 11: Metacog Recent Alerts (core events)

### Goal
Include recent turn-effect alert events in metacog context and prompts as deterministic evidence cues.

### Implemented
- Added core-event caching + subscription in cortex-exec, with core events channel wiring in settings/env/compose.
- Included recent turn-effect alert JSON + summary line in `MetacogContextService`.
- Added prompt sections for recent alerts in both metacog draft/enrich templates.
- Tagged turn-effect notifications with `event_type` for filtering.
- Added tests for alert filtering and summary formatting.

### Testing status
- ✅ `pytest -q`

---

## Patch Set 12: Alert governance (severity + dedupe + linkage + deterministic tags)

### Goal
Improve turn-effect alert operational quality:
- severity levels
- dedupe (in addition to cooldown)
- corr_id/trace_id linkage in payload + metacog summaries
- deterministic metacog alert tags preserved across draft/enrich

### Implemented
- Added dedupe controls alongside severity mapping and richer logging/payload fields in spark-introspector + config wiring.
- Extended core-event alert normalization and metacog context tagging/summary helpers to include severity and corr/trace linkage.
- Preserved deterministic alert tags in metacog draft/enrich flows.
- Expanded tests for linkage formatting, deterministic tag merging, severity mapping, and dedupe window logic.

### Testing status
- ✅ `pytest -q`

---

## Patch Set 13: Deterministic alert → action policy

### Goal
Provide deterministic guidance derived from recent alerts (no autonomous behavior changes).

### Implemented
- Added deterministic turn-effect action policy helper + summary formatter.
- Injected recommended action policy into Metacog context summaries.
- Persisted the policy as system-owned telemetry in metacog draft/enrich flows.
- Updated prompts to include policy evidence cues.
- Added smoke + unit tests.

### Testing status
- ✅ `python scripts/smoke_turn_effect_policy.py`
- ✅ `pytest -q`

---

## Patch Set 14: Deterministic alert explanations (no LLM inference)

### Goal
Add a structured, deterministic “why this alert” explanation block alongside recommended actions, so metacog mirrors can reference causes/questions safely.

### Implemented
- Added deterministic alert explanation helper + summary formatter for turn-effect alerts.
- Injected alert explanations into metacog context, prompts, and system-owned telemetry in draft/enrich flows.
- Added smoke script + unit tests for alert explanation mapping and summary output.

### Testing status
- ✅ `python scripts/smoke_turn_effect_explanations.py`
- ✅ `pytest -q`

---

## Patch Set 15: Retrieval hook profile (alerts + turn-effect)

### Goal
Enable recall to prioritize:
- metacog mirrors tagged `metacog.alert.*`
- chat turns with large Δ_turn (from `spark_meta.turn_effect`)

### Implemented
- Added SQL timeline turn-effect delta extraction from `spark_meta` and carried it into recall candidates for boosting.
- Implemented alert tag-prefix and turn-effect delta boosts in recall fusion scoring.
- Added `reflect.alerts.v1` recall profile and documented intent in profiles README.
- Added unit tests for boosting behavior.

### Testing status
- ✅ `pytest -q`

---

## Patch Set 16: Dashboard query + CLI export

### Goal
Provide a canonical query (and optional CLI wrapper) to chart Δ_user/Δ_assistant/Δ_turn over time with alert overlays.

### Implemented
- Added canonical Postgres turn-effect time series SQL query with optional alert overlay join for metacog tags.
- Extended `scripts/print_recent_turn_effects.py` with `--since-hours`, wide/csv output, and flattened delta columns.
- Added a unit test for flattened delta helper used by CLI output.

### Testing status
- ✅ `python scripts/print_recent_turn_effects.py --dry-run --since-hours 24 --include-deltas`
- ✅ `pytest -q`

---

## Patch Set 17: Metacog phase contract (Draft text vs Enrich scores)

### Goal
Enforce stricter roles:
- **Draft** produces qualitative text fields only (writer role) using structured evidence cues.
- **Enrich** (mini council) produces numeric/classification/scoring fields only.
- Prevent LLM clobbering of system-owned fields (`id`, `timestamp`, `observer`, `source_service`, `correlation_id`, telemetry) and reduce hallucinated nested dicts.

### Implemented (17A-REDUX)
- Added strict metacog draft/enrich patch schemas and registered them in the schema registry.
- Enforced whitelist patch application with truncation and scoring-only enrichment in the metacog pipeline.
- Preserved system-owned telemetry and IDs during both phases.
- Added tests and a smoke script validating patch rejection and system-owned invariants.

### Testing status
- ✅ `pytest -q`
- ✅ `python scripts/smoke_metacog_phase_contract.py` (warns about pydantic protected namespace)


---

## Patch Set 18: ID + correlation lineage hardening

### Goal
Make `id` first-class and stop `correlation_id == id` by default for metacog:
- `id` = new UUID per mirror row.
- `correlation_id` = upstream trigger correlation (chat turn correlation_id if chat-triggered; otherwise heartbeat correlation).
- Record explicit trigger lineage in telemetry.

### Proposed approach
1) Add system-owned telemetry:
   - `trigger_kind`, `trigger_correlation_id`, `trigger_trace_id`.
2) Reassert `id`, `correlation_id`, `source_service` after draft/enrich merges.
3) Add tests ensuring LLM cannot overwrite these fields.

### Draft prompt
See Prompt 18A.

---

## Patch Set 19: Turn-effect provenance (raw φ blocks)

### Goal
Persist compact raw φ evidence alongside derived deltas so metacog can always audit:
- `phi_before`, `phi_after`, `phi_post_before`, `phi_post_after` (compact dims only)
- stored as system-owned `turn_effect_evidence`.

### Proposed approach
1) Extend turn-effect helper to optionally return `turn_effect_evidence`.
2) Inject into Spark snapshot metadata (optional) and metacog mirror telemetry (required).
3) Add compact summary line and tests.

### Draft prompt
See Prompt 19A.

---

## Patch Set 20: Telemetry key normalization + phi_hint schema normalization

### Goal
Eliminate inconsistent telemetry keys (spaces) and inconsistent `phi_hint` shapes.

### Implemented
- Added telemetry key normalization (space cleanup, explicit fixes) and canonicalized `phi_hint` into a versioned bands/numeric structure during collapse mirror normalization.
- Added unit coverage for telemetry key cleanup and `phi_hint` canonicalization.
- Added a smoke script to validate normalization output.

### Testing status
- ✅ `pytest -q`
- ✅ `python scripts/smoke_telemetry_normalization.py`

---

## Patch Set 21: Spark introspector wildcard publish guard

### Goal
Fix runtime crash when spark-introspector attempted to publish to a wildcard candidate pattern (`orion:spark:introspect:candidate*`) under catalog enforcement.

### Implemented
- Added `_is_publishable_channel()` helper to reject wildcard/pattern channels for publish.
- Guarded the legacy completed-envelope publish in `handle_candidate`; logs a warning and skips publish when channel is non-concrete.
- Added a small unit test verifying wildcard channels are rejected and concrete channels allowed.

### Notes
- Keeps service alive but leaves a warning log each time (tech debt: remove legacy republish later).

### Testing status
- ✅ `pytest -q`

---

## Patch Set 22: Metacog prompt templates aligned with phase contract

### Goal
Stop MetacogDraft/Enrich LLM outputs from being rejected by strict patch schemas by updating prompts to request **patch JSON only**:
- Draft: `MetacogDraftTextPatchV1`
- Enrich: `MetacogEnrichScorePatchV1`

### Motivation (observed)
- Draft previously returned full `CollapseMirrorEntryV2` keys (event_id/id/observer/state_snapshot/tags/scores/etc.) which violated `extra=forbid` and caused patch rejection.

### Proposed approach
1) Update `log_orion_metacognition_draft.j2` to request ONLY DraftTextPatch JSON and include an example that contains ONLY allowed keys.
2) Update `log_orion_metacognition_enrich.j2` to request ONLY ScorePatch JSON and remove "Return FULL CollapseMirrorEntryV2" instruction.
3) Add a unit test ensuring templates reference patch schema names and example blocks do not include forbidden keys.

### Status
- Pending implementation; Codex drifted into unrelated services. Re-run with file-locked prompt.

---

## Patch Set 23: Metacog trigger typing + fail-fast step gating

### Goal
Fix metacog runtime failures:
- `CollapseMirrorEntryV2.trigger` expects a string but metacog base entry is receiving a trigger dict.
- Enrich/publish should not run when draft fails.

### Proposed approach
1) In metacog base entry builder, set `trigger` = `trigger_kind` (string) and store full trigger payload under system-owned telemetry (`trigger_payload`).
2) Add unit test verifying trigger is a string and trigger_payload is stored.
3) Fail-fast: if draft step fails, skip enrich and publish steps (record error in ctx/prior_step_results).

### Draft prompt
See Prompt 23A.
