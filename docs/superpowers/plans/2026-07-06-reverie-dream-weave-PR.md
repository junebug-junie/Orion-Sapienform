## Summary

- Builds the full reverie/dream/compaction weave (Phases A–H) as a tower on the existing self-modeling ladder — reverie is a *different kind of thought* inside `orion-thought`, not a new service.
- Adds spontaneous thought (A), governed proposals (B), reverie chains (C), episode/motif grounding (D), a compaction-request queue (E), staged REM compaction narration (F), a hard-gated memory-mutation applier (G), and efficacy + a resonance tripwire (H).
- **Every new gate defaults OFF.** The one memory-mutating rung (G) is hard-off pending proposal-mode sign-off + a live §14 verification.
- All schemas registered, channels wired, migrations applied, env synced, tests + evals green per-service, code review run per rung with all findings fixed.

## Outcome moved

Orion gains an unprompted inner voice that narrates the current winning coalition, chains trains of thought that habituate resolved themes, and (staged) narrates what sleep would do to memory — with an automated ouroboros tripwire that must fire clean before any memory compaction is ever enabled. No cognition path mutates memory in this PR.

## Current architecture (before)

`orion-thought` narrated only *evoked* thoughts (turn-bound `stance_react`). The substrate ladder emitted attention broadcasts (rung 3), episode summaries (rung 4), and consolidation motifs (Layer 11), but nothing narrated them spontaneously, and there was no path from a settled reverie to memory housekeeping.

## Architecture touched

- `orion-thought`: spontaneous-thought producer, reverie chain orchestrator, grounding, compaction-request emitter, resonance tripwire.
- `orion-dream`: REM compaction narrator (staged) + compaction applier (inert, injected store).
- `orion-hub`: observability sections ("what sleep would do", resonance alerts).
- `orion-proposal-runtime`: reverie → governed proposal candidate (Phase B).
- Contracts: `orion/schemas/reverie.py`, `orion/schemas/compaction.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.

## Schema / bus / API changes

- Added kinds: `reverie.thought.v1`, `reverie.chain.v1`, `reverie.refractory.entry.v1`, `dream.compaction.request.v1`, `dream.compaction.delta.v1`, `reverie.resonance.alert.v1` (all registered in both `_REGISTRY` and `SCHEMA_REGISTRY`).
- Added channels: `orion:reverie:thought`, `orion:reverie:chain`, `orion:dream:compaction-request`, `orion:dream:compaction-delta`, `orion:reverie:resonance-alert` (all experimental, producers scoped, dangerous consumers empty).
- `MemoryCompactionDeltaV1.proposal_marked` is `Literal[True]` — an applied-fact is unrepresentable.
- Compatibility: additive only; no existing payload meaning changed.

## Env/config changes

- Added keys (all default-safe): `ORION_REVERIE_*` (enabled/interval/min_salience/chain/refractory/ground/compaction-request/resonance), `CHANNEL_REVERIE_*`, `ORION_DREAM_REM_ENABLED`, `CHANNEL_DREAM_COMPACTION_DELTA`, `DREAM_REM_MAX_REQUESTS`, `ORION_DREAM_COMPACTION_APPLY_ENABLED`, `ORION_DREAM_COMPACTION_DOWNSCALE_ONLY`, `DREAM_COMPACTION_SNAPSHOT_DIR`.
- `.env_example` updated for orion-thought + orion-dream; local `.env` synced (gitignored, confirmed not staged).
- `scripts/sync_local_env_from_example.py`: added reverie/dream prefixes+exacts and `orion-dream` to the service list.

## Tests run (per-service — §11 gate model)

```text
reverie core (orion/reverie):                 27 passed
orion-thought (tests+evals):                  55 passed
orion-dream (tests+evals):                    35 passed
orion-hub (reverie observability):             8 passed
resonance determinism across PYTHONHASHSEED 0-3: 9 passed each
```

## Evals run

```text
orion-dream/evals/test_rem_compaction_eval.py            (proposal invariant over corpus)
orion/reverie/evals/test_reverie_efficacy_resonance_eval.py (tripwire fires on runaway, silent on healthy; pre/post recall + discharge)
services/orion-thought/evals/test_reverie_hollow_guard_eval.py
```

## Docker/build/smoke checks

```text
Not run (no live mesh in this environment). Migrations applied to conjourney via
docker exec orion-athena-sql-db psql; tables + indexes verified.
```

## Review findings fixed

- **F** — default request loader was keyword-only but called positionally (feature inert on the real path). Fix: positional signature + regression test.
- **F** — `MemoryCompactionDeltaV1` registered only in `SCHEMA_REGISTRY`, not `_REGISTRY` → every bus publish would raise. Fix: added to `_REGISTRY` + `resolve()` regression test.
- **G** — approval leaked across deltas sharing a `source_request_id`. Fix: require exact `delta_id` match + regression test.
- **G** — `autonomy_policy` gate accepted on the human-required rung. Fix: `execution_policy` only + regression test.
- **G** — fail-closed tests didn't isolate the guards. Fix: per-guard tests holding `execution_allowed=True`.
- **G** — gate check could raise on a malformed frame. Fix: wrapped, fails closed.
- **H** — nondeterministic winner on a full tie (set iteration / hashseed). Fix: sorted iteration + total order + cross-seed regression test.
- **H** — `uuid4` alert_id flooded the surface for a persisting runaway. Fix: deterministic theme+window id so `ON CONFLICT` dedups + regression test.

## Restart required

No restart required for merge (all gates off). If a flag is later enabled:

```bash
docker compose --env-file .env --env-file services/orion-dream/.env -f services/orion-dream/docker-compose.yml up -d
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml up -d
```

## Risks / concerns

- **Severity: HIGH (contained).** Phase G mutates memory. Mitigation: hard-off gate, no real store bound (inert on import), snapshot-first + rollback, downscale-only-first, human-only execution approval. Requires the sign-off doc's 5 acceptance conditions (`docs/superpowers/plans/2026-07-06-phase-g-compaction-applier-proposal-mode.md`) + a live §14 verify before the gate is flipped.
- **Severity: LOW.** All runtime claims are UNVERIFIED (no live mesh run in this environment). Everything is proven by unit/eval + applied migrations, not by a live path moving.

## PR link

<paste after creating>
