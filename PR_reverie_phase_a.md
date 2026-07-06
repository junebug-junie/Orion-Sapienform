## Summary

- Reconciles the reverie/dream weave spec with the **already-shipped `orion-thought` service** (unified-turn, #817–820): v2's premise "nothing gives a coalition a voice" was false — `stance_react`/`ThoughtEventV1` already narrates it. Adds a taxonomy: **evoked** thought (`ThoughtEventV1`) vs **spontaneous** thought (`SpontaneousThoughtV1`), split by trigger + destination, sharing `CoalitionSnapshotV1` grounding, both housed in `orion-thought`.
- Implements **Phase A**: a spontaneous-thought *mode* inside `orion-thought`. A self-driven tick reads the current rung-3 winning coalition (no `user_message`), narrates it via a new `reverie_narrate` cortex verb over the existing exec rail, and emits `SpontaneousThoughtV1` on `orion:reverie:thought`.
- Load-bearing `is_hollow()` guard rejects short **and** un-anchored text — the real Phase A fail-fast risk now that no user question anchors relevance.
- Everything is **default-off** (`ORION_REVERIE_ENABLED=false`) and **read-only** — no proposals, no memory writes, evoked path untouched.
- Salience is **deterministic** (§4): code owns it from coalition scores; the LLM owns only interpretation text.

## Outcome moved

New unprompted-cognition seam: `orion-thought` can now narrate its own current coalition when nobody asked (the first self-activated thought path), gated off. Before, coalition narration only fired in response to a user turn.

## Current architecture (before)

`orion-thought` = RPC turn-worker only: `orion:thought:request` (`StanceReactRequestV1`, requires `user_message`) → `stance_react` verb → `ThoughtEventV1` → Hub reply + `orion:thought:artifact`. Nothing narrated the coalition unprompted.

## Architecture touched

- `orion-thought`: second entrypoint (`run_reverie_worker`) beside `run_bus_worker`, wired into the FastAPI lifespan, default-off.
- New cortex verb `reverie_narrate` (reuses `LLMGatewayService` + `build_plan_for_verb` + `CortexExecClient`).
- New schema `SpontaneousThoughtV1` + registry + bus channel `orion:reverie:thought`.

## Files changed

- `docs/superpowers/plans/2026-07-05-reverie-dream-compaction-weave.md`: v3 reconciliation (premise, taxonomy, Phase A, rationale, component map, delivery status).
- `orion/schemas/reverie.py` (new): `SpontaneousThoughtV1`, `is_hollow()` guard, reuses `CoalitionSnapshotV1`.
- `orion/schemas/registry.py`: register `SpontaneousThoughtV1` → `reverie.thought.v1`.
- `orion/bus/channels.yaml`: `orion:reverie:thought` event channel.
- `orion/cognition/verbs/reverie_narrate.yaml` + `orion/cognition/prompts/reverie_narrate.j2` (new): mirror `stance_react`, reflection-shaped.
- `services/orion-thought/app/reverie.py` (new): producer — coalition read (fail-open), deterministic salience, narrate, hollow-drop, emit; degrades to None, never raises.
- `services/orion-thought/app/{settings,main}.py`: reverie env flags + gated self-driven task.
- `services/orion-thought/.env_example`: default-off flags.
- `services/orion-sql-db/manual_migration_substrate_reverie_thought.sql` (new): store table (contract-ahead; writer deferred).
- `services/orion-thought/tests/test_reverie_spontaneous_thought.py` + `evals/test_reverie_hollow_guard_eval.py` (new).

## Schema / bus / API changes

- **Added:** `SpontaneousThoughtV1` (`reverie.thought.v1`); channel `orion:reverie:thought` (event, producer `orion-thought`).
- **Removed / Renamed:** none.
- **Behavior changed:** none to existing paths — evoked `stance_react`/`ThoughtEventV1` untouched.
- **Compatibility:** additive; new channel/schema, default-off producer.

## Env/config changes

- **Added keys:** `ORION_REVERIE_ENABLED=false`, `ORION_REVERIE_INTERVAL_SEC=90`, `ORION_REVERIE_MIN_SALIENCE=0.0`, `CHANNEL_REVERIE_THOUGHT=orion:reverie:thought` (in `services/orion-thought/.env_example`).
- **`.env_example` updated:** yes. All safe non-secret defaults.
- **local `.env` sync:** ran `python scripts/sync_local_env_from_example.py`; `orion-thought` reported `no .env` (service has no local `.env` in this worktree). **Operator action:** on the athena host, run the sync after merge to add the four keys — the feature stays off without them (no functional impact until you enable it). Set `ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0` as always.
- **Skipped keys:** none.

## Tests run

```text
pytest services/orion-thought/tests/ services/orion-thought/evals/  → 23 passed
# schema hollow-guard (5), producer grounding/salience (4), tick gating +
# never-raise (6), default-reader coverage (3), existing suite (2), + eval (1)
build_plan_for_verb('reverie_narrate') → plan loads
registry import → SpontaneousThoughtV1 registered as reverie.thought.v1
channels.yaml parse → orion:reverie:thought present
```

## Evals run

```text
pytest services/orion-thought/evals/test_reverie_hollow_guard_eval.py → 1 passed
# un-anchored hollow-guard: 7/7 labeled cases (grounded vs drivel) separated
```

## Docker/build/smoke checks

```text
NOT RUN — no live mesh in this environment. Runtime evidence for Phase A
(stored SpontaneousThoughtV1 whose coalition matches a live broadcast,
rendered in hub) is UNVERIFIED until run on the athena host. This is an
honest §0A UNVERIFIED, not a claimed pass.
```

## Review findings fixed

- **Finding (HIGH):** `_default_broadcast_reader` built `SubstrateFeltStateReader` without the required `enabled` kwarg → `TypeError` swallowed by the broad except → producer dead-on-arrival, never emits; masked because every test injected a fake reader.
  - **Fix:** switch to the public fail-open `hydrate_felt_state_ctx` entrypoint; add 3 tests that actually exercise the default reader.
  - **Evidence:** `test_default_broadcast_reader_*` (hydrate / empty / never-raises) pass.
- **Finding (LOW):** parse + `bus.publish` ran outside the `try/except`; a bus failure could raise out of a tick, violating "adapters never raise."
  - **Fix:** wrap the whole tail; a tick degrades to a dropped `None`.
  - **Evidence:** `test_tick_swallows_publish_failure` passes.
- **Finding (MEDIUM):** orphan migration + no store-writer/hub panel.
  - **Fix:** explicitly marked store-writer + hub panel + live smoke as DEFERRED in the plan's Phase A "Delivery status"; evidence flagged `UNVERIFIED`.
  - **Evidence:** doc delivery-status block.

## Restart required

```bash
# After merge + pull on athena. Feature is default-off; these just deploy the code.
python scripts/sync_local_env_from_example.py   # adds the 4 reverie keys
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reverie_thought.sql
# restart orion-thought (self-driven tick is a no-op until ORION_REVERIE_ENABLED=true):
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
```

## Risks / concerns

- **Severity: medium.** Phase A's runtime evidence is `UNVERIFIED` (no live mesh here). Do not consider Phase A "done" per §0A until a real tick emits a coalition-matched, non-hollow `SpontaneousThoughtV1` on the athena host with the flag on.
- **Severity: low.** The store-writer + hub `_reverie_section` panel are deferred, so there is no inspection surface yet — enabling the flag emits to the bus channel only.
- **Severity: low.** The hollow guard is structural (evidence-ref anchoring); semantic "aboutness" is delegated to the prompt. Acceptable for Phase A; Phase H's resonance/efficacy evals are the real quality gate.

## Scope note

This PR is **spec reconciliation + Phase A only**. Phases B–H (thought→governed action, reverie chain, episode grounding, compaction request, dream REM, compaction applier, efficacy/resonance evals) remain proposal-mode and ship phase-by-phase with live verification per the plan's own escalating-blast-radius discipline — deliberately not bundled into one blind sprint.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
