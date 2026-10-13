## Summary

- Orion now checks its own gauges for having stopped telling the truth (Temporal Self rev 4, R4). When arousal is idle, at most every 6 h, a new `immune` step on the durable-runs temporal_self thread looks at 24 h of the field channels that declare semantics, for four failures: **constant**, **floor-locked** (never reaches rest), **replayed** (one frozen reading re-written ≥30 times) and **decayed-to-zero** (exact geometric fade, the 07-26 `substrate.route` bug). It records findings; it never fixes anything.
- Gauges get a machine-readable calm band, `rest_range: [lo, hi]`, next to the prose `rest` in the existing semantic layer (glossary + metric lock). 10 of 11 semantic glossary channels carry one, chosen from prose and checked against 24 h of live data (the 11th is a placeholder; a band there is refused).
- Findings land as `immune_finding` rows in `temporal_self_event` (idempotent per metric/kind/day), are counted in `RegulationStateV1.immune_findings_open`, and the Orion's Day closed-day block renders them as plain sentences ("My route prediction error gauge looked stuck (decayed to zero) from 03:10 to 09:40"). Not wired to daily_metacog (off on purpose).

## Outcome moved

The "calm vs dead" bug class Juniper and Claude kept catching by hand (bus_synaptic 0.27 floor, route decayed to zero, constant ontology_sparse_region, replayed repair pressure) becomes a scheduled, persisted self-check Orion can read about its own day.

## Current architecture

Temporal Self thread: ingest → regulate → chronicle. Semantics (`value_kind`, `rest`, `sparsity`, `absent_means`) existed as prose only; `classify_channel_series` gave dead/ratchet/quiet/live with no notion of rest.

## Architecture touched

- `orion/metrics/semantics.py`, `lineage.py`, `definitions.py`: `rest_range` field, validator, projection, lock recording (ordered, never sorted).
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: bands + re-lock (10 `rest_range` deltas).
- `orion/regulation/immune.py` (new, pure, never raises).
- `services/orion-durable-runs`: `immune` node (regulate → immune → chronicle), idle-gated, 6 h interval carried across midnight, 90 s timeout, failure never breaks regulation; one read-only per-minute downsampled query (~3 s); idempotent writer; health `temporal_self.immune`.
- `orion/schemas/regulation.py`: `immune_findings_open`; omitted from the Redis body (`REDIS_PROJECTION_EXCLUDE`) so old readers of the forbid model don't fall back to `unknown` before they are rebuilt. Full model at `GET /regulation/state`.
- `orion/temporal_self/day_block.py`, `orion/orion_day/gather.py`: findings rendered at priority 3 (never push out conversation/attention), max 5 lines, as recorded at close.

## Files changed

See `git diff --stat origin/main`. New: `orion/regulation/immune.py`, `tests/test_regulation_immune.py`, `tests/test_metric_rest_range.py`, evidence under `docs/superpowers/evidence/2026-10-13-immune-sweep/`. CI: `regulation-arousal.yml` runs the immune tests (triggers on glossary changes); `orion-static-gates.yml` runs `test_metric_rest_range.py`.

## Schema / bus / API changes

- Added: `RegulationStateV1.immune_findings_open` (int ≥0, default 0); `temporal_self_event.source_kind = 'immune_finding'` (no CHECK constraint, no migration); semantic field `rest_range`.
- Compatibility: Redis `orion:regulation:latest` body unchanged (field excluded). Follow-up: empty `REDIS_PROJECTION_EXCLUDE` once hub, dream, thought and proposal-runtime are rebuilt.

## Env/config changes

- Added: `TEMPORAL_SELF_IMMUNE_ENABLED=true`, `TEMPORAL_SELF_IMMUNE_INTERVAL_SEC=21600` (orion-durable-runs).
- `.env_example` updated; local `.env` synced with `scripts/sync_local_env_from_example.py` (exit 2 from unrelated pre-existing diverged keys, none changed). No skipped keys.

## Metric quality gate (`immune_findings_open`, `rest_range`)

1. Provenance: `orion/regulation/immune.py::sweep` over `substrate_field_state.field_json->node_vectors` + `node_vector_updated_at`.
2. Independence: derived from field channels but measures their *honesty*, not their level; it feeds no model input.
3. Theory anchor: the four failure signatures are each a recorded production incident (CLAUDE.md metric gate step 4; #2528 rev 2).
4. Live sanity: dry run over 3×24 h windows, zero findings, every channel verified calm by hand (vision router dispatching every 5 s; bus/biometrics 520–600 distinct values/day, reach 0). Planted faults (0.92 decay on route, 45 min frozen bus_synaptic) caught with correct kind and window. Re-run with the real merged `rest_range`: all four checks judge 9 present channels, zero findings (`live-dry-run-real-rest-range.txt`).
5. Existing mechanism: extends glossary semantics + `classify_channel_series`; no new registry.
6. Reversibility: `TEMPORAL_SELF_IMMUNE_ENABLED=false`; rows are append-only events; `rest_range` is optional.

Live vs prose contradictions recorded (bands not widened): bus_synaptic median 0.0299 at the top of its prose band; route rests at 0 only 68% (prose 87%); both vision channels exactly 0.0 on all 123k rows since 10-10 — calm per the router logs, but whether they can move is UNVERIFIED.

## Tests run

```text
tests/test_regulation_immune.py tests/test_metric_rest_range.py tests/test_metric_prompt_semantics.py
tests/test_field_channel_glossary.py orion/temporal_self/tests orion/orion_day/tests      356 passed
services/orion-durable-runs/tests (isolated postgres:16)                                 491 passed, 1 skipped
metric drift/lineage tests                                                                141 passed
hub test_orion_day_temporal_day.py test_orion_day_letter.py                              59 passed
gates: check_definition_drift --gate, check_metric_lineage --gate / --prompt-semantics, check_env_template_parity: all 0
```

## Evals run

No eval harness covers the sweep; the live dry run with planted faults is the behavioural check. Gap noted.

## Review findings fixed

- Rest checks inert without bands → bands now merged; `coverage.rest_unjudged` trace.
- Constant could fire on a carried value → requires re-writes in ≥20 distinct hours; stamp-less series cannot be judged.
- Day lines could push out attention/conversation → priority 3.
- Immune warnings never reached RegulationStateV1 → re-projected with warning.
- Silent no-run with arousal disabled → `health.immune.skipped` says why.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Medium: decayed-to-zero would flag a channel that decays by design (EWMA fed zeros, `NODE_DECAY_CHANNELS`). None of the semantic channels do today; add an exemption marker when one does.
- Low: field reads carry no source row, so a replay frozen exactly at 0.0 or 1.0 is not judged; and a gauge that lies *at* its rest value is invisible to all four checks.

## PR link

PENDING
