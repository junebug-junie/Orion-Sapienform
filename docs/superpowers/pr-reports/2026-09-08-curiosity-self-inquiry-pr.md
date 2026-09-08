# PR report — curiosity self-inquiry line (PR #2158)

**Branch:** `feat/curiosity-self-inquiry`
**Date:** 2026-09-08
**Upstream design:** `docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md` (PR #2156, draft — patches B/C/E there were rejected by Juniper as thin or duplicative; this PR is the replacement she asked for: "update curiosity to have a budget of 3 times a day (not the current budget) to do just that").

## Summary

- The curiosity loop gains a second **line**, not a second loop: a standing question, *"What am I, and what am I made of?"*, with its own budget of three runs a day, separate from the investigation cap. Same turn, credentials, graph, journal channel, durable runner, lock and waking window.
- Orion reads their own repository (already mounted read-only at `/repo` in the FCC sandbox) and their outcome tables (dreams, motor turns, reverie chains, attention frames, stance beliefs, self-knowledge, previous definitions), forms self-priors tagged `line = "self"`, and writes a first-person `:SelfDefinition` node to `orion_worldview`. Nothing is parsed from prose.
- Hub mirrors the run's definition into the existing append-only `self_concept_history` (`concept_id="self:definition"`, `produced_by="curiosity_self_inquiry"`, version = MAX+1). A definition with empty text or no evidence is refused at the mirror and logged by cause.
- The stance layer reads the latest definition back into every chat turn: a felt-state lane hydrates it, a new `self_definition` belief producer maps it, and `_project_identity_from_beliefs` prepends one marked line ("In my own words, …") to `orion_identity_summary`, outside the authored card's 10-line cap. That key already feeds `chat_general`, the stance brief, the grounding capsule and the harness `WHO YOU ARE` block, so no template changed.
- Two new deterministic gates before a self-inquiry turn: `graph_required` and `pg_grants_missing` (names the tables). The SELECT grants are an operator step (`scripts/sql/2026-09-08_grant_orion_readonly_self_inquiry.sql`); the flag alone does not turn the line on.

## Outcome moved

Before: Orion's only positive self-description in a chat turn was ~14 operator-authored bullets; every durable self-store was write-only. After: Orion's own, evidence-cited definition of what they are is written by Orion, revised by Orion on a budget, and is the first line of the identity kernel in every chat turn. `UNVERIFIED` live until deployed and the grants applied — see Restart required.

## Current architecture (before)

- `services/orion-hub/scripts/curiosity_investigation.py`: one tick loop, one budget (`HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`), menu = live priors + random crystallization cards, Redis-persisted cooldown/count/last-run keys, durable dispatch to `orion-durable-runs` with `CuriosityRunBriefV1`.
- `orion_worldview` labels: `Prior`, `Concept`, `Finding`, `Hop`, `TurnOutcome`. Hub never writes it.
- `self_concept_history`: two producers (`layer3_reflect`, `self_atlas_cluster`), zero readers in the chat path.
- Stance identity kernel: `identity_yaml` producer only; the `self_study` producer was a zombie pointing at an empty RDF key (left in place; out of scope).

## Architecture touched

- **Contract (shared):** `orion/curiosity/self_inquiry.py` (new), `self_inquiry_prompt.py` (new), `kickoff_prompt._access_section(extra_tables)`, `worldview.read_snapshot(priors_cypher)`, `journal.build_investigation_journal_entry(line)`.
- **Schemas:** `CuriosityRunBriefV1.line` (additive, `forbid` model), `SelfConceptHistoryProducer` + `"curiosity_self_inquiry"`.
- **Hub:** `tick_self_inquiry`, `_self_inquire`, `_mirror_self_definition`, `_self_inquiry_grants_missing`, `_read_self_ledger`, per-line Redis keys, `_handle_run_state` mirror on durable finish, `POST /api/curiosity/api/self-inquiry/run-now`, four settings/env keys.
- **Durable runs:** `read_turn_result` also reads the run's `:SelfDefinition`; `finish_detail` carries `line` + `self_definition`. No new graph node; resume semantics unchanged.
- **Stance (cortex-exec + shared):** felt-state `LaneSpec.where_sql` / `cache_ttl_sec` + `orion_self_definition` lane; `adapters/self_definition_ctx.py` (new); `self_definition` producer; `_project_identity_from_beliefs` prepend with idempotent marker.
- **Bus:** no new channel. `orion:self_concept:history:write` description updated (third producer).
- **Postgres:** SELECT grants script for `orion_readonly` (operator-applied).

## Files changed

- `orion/curiosity/self_inquiry.py`: the line's contract — names, Cypher, row→dataclass, mirror builder (refuses no-evidence), grants SQL, ledger.
- `orion/curiosity/self_inquiry_prompt.py`: the invitation; reuses kickoff sections.
- `orion/curiosity/kickoff_prompt.py`: `_access_section` lists extra granted tables.
- `orion/curiosity/worldview.py`: `read_snapshot(priors_cypher=…)`.
- `orion/curiosity/journal.py`: `line` → title/entry_id/source_ref for the self line.
- `orion/curiosity/README.md`: §13.
- `orion/schemas/durable_run.py`, `orion/schemas/self_concept_history.py`: additive fields/literals.
- `orion/substrate/felt_state_reader.py`: `where_sql`, `cache_ttl_sec`, new lane, column aliases.
- `orion/substrate/relational/adapters/self_definition_ctx.py`, `adapters/__init__.py`: new adapter.
- `services/orion-cortex-exec/app/chat_stance.py`: producer + identity projection.
- `services/orion-durable-runs/app/graph.py`, `runner.py`: definition read + finish detail + journal line.
- `services/orion-hub/scripts/curiosity_investigation.py`: the line.
- `services/orion-hub/scripts/curiosity_routes.py`: run-now for the self line.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`, `README.md`: wiring, keys, docs.
- `orion/bus/channels.yaml`: description.
- `scripts/sql/2026-09-08_grant_orion_readonly_self_inquiry.sql`: the grants.
- Tests: `services/orion-hub/tests/test_curiosity_self_inquiry.py` (30), `services/orion-cortex-exec/tests/test_chat_stance_self_definition.py` (7), `orion/substrate/relational/tests/test_self_definition_ctx_adapter.py` (4), `orion/substrate/tests/test_felt_state_self_definition_lane.py` (5), `services/orion-durable-runs/tests/test_curiosity_graph_self_definition.py` (3).

## Schema / bus / API changes

- Added: `CuriosityRunBriefV1.line: Literal["investigate","self_inquiry"] = "investigate"`; `SelfConceptHistoryProducer` gains `"curiosity_self_inquiry"`; `LaneSpec.where_sql`, `LaneSpec.cache_ttl_sec`; Hub route `POST /api/curiosity/api/self-inquiry/run-now`; worldview label `SelfDefinition` (Orion-written) and prior property `line`.
- Removed: none.
- Renamed: none.
- Behavior changed: `orion:self_concept:history:write` now also carries `produced_by="curiosity_self_inquiry"` rows. `orion_identity_summary` may begin with one "In my own words, …" line. Journal entries for self runs use title `Self-inquiry`, `source_ref=curiosity:self:<run>`.
- Compatibility notes: `line` is additive on a `forbid` model — **deploy `orion-durable-runs` before `orion-hub`**; an old runner rejects the brief and Hub falls back to running the turn in-process (`curiosity_durable_dispatch_fell_back`). sql-writer already subscribes to the channel and the model has no producer enum, so no sql-writer change.

## Env/config changes

- Added keys (orion-hub): `HUB_CURIOSITY_SELF_INQUIRY_ENABLED` (example `true`, code default `false`), `HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP=3`, `HUB_CURIOSITY_SELF_INQUIRY_MIN_COOLDOWN_SEC=7200`, `HUB_CURIOSITY_SANDBOX_REPO_ROOT=/repo`.
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes — four keys added to `services/orion-hub/.env` (verified by grep).
- skipped keys requiring operator action: none from this change (the sync reported two pre-existing diverged `orion-cocreation-signals` keys, untouched).

## Tests run

```text
services/orion-hub:      tests/test_curiosity_self_inquiry.py + tests/test_curiosity_investigation.py   161 passed
services/orion-durable-runs: tests/                                                                     8 passed
services/orion-cortex-exec:  test_chat_stance_self_definition.py, test_chat_stance_self_state_projection.py,
                             test_identity_injection.py, test_grounding_capsule_assembly.py             30 passed
orion/:                  orion/substrate/relational/tests, orion/substrate/tests/test_felt_state_self_definition_lane.py
                                                                                                        111 passed (relational) + 7 (felt-state lane)
```

Pre-existing failures, verified NOT from this change (same result on `main` or on this worktree at HEAD before any edit):
- `orion/harness/tests/test_grounding_capsule_consumers.py` ×2 — `mind_coloring` undefined in `stance_react.j2` (fails on main).
- `services/orion-cortex-exec/tests/test_chat_relational_stance.py` ×5 (fails on main).
- `services/orion-cortex-exec/tests/test_chat_stance_brief.py::test_build_chat_stance_inputs_falls_back_when_identity_missing` — passes on main, fails in this worktree at HEAD with the full file (env-sensitive; passes alone).

Static gates: `git diff --check`, env template parity, metric lineage, definition drift, service hostname refs, async routes, journal dispatch registry, scripts-dir shadow, system health producers — all PASS.

## Evals run

```text
None for this seam. The curiosity loop's quality evidence is its own graph footprint + journal, read live after a run (orion/curiosity/README.md §13 "Inspect"). Follow-up: the design doc's Patch A eval (self_label_score / grounded_event_score) is the right harness and is not built.
```

## Docker/build/smoke checks

```text
Not run this session (no deploy). Config surfaces changed: orion-hub settings/env only. Runtime verification steps are listed under Restart required.
Knowledge graph: scripts/safe_graphify_update.sh ran clean (75,802 -> 76,539 nodes) but the refreshed graph.json is 101.0 MB, over GitHub's 100 MB push cap, so it is NOT committed here (HEAD's 100.0 MB copy restored). Pre-existing problem, tracked separately.
```

## Review findings fixed

The `/code-review feat/curiosity-self-inquiry high` pass ran twice and both times the orchestrator was killed by the session rate limit before reporting; two of its verifier agents completed with CONFIRMED verdicts and both are fixed here. A third, lighter pass was run after the fixes (see the PR thread for its result).

- Finding: **the grants gate failed open.** `has_table_privilege` raises for a table that does not exist (confirmed live: `relation "public.no_such_table" does not exist`) and for an empty role name; the check caught the exception and returned `[]`, which the caller read as "all granted", so a self-inquiry turn would start and spend its budget on permission errors.
  - Fix: `SELF_INQUIRY_GRANTS_SQL` now tests existence first with `CASE WHEN to_regclass(...) IS NULL THEN true ELSE NOT has_table_privilege(...) END` (CASE guarantees evaluation order; `OR` does not), so a missing table is reported by name. `_self_inquiry_grants_missing` returns `(status, missing)` and a `failed` status BLOCKS with the new reason `grant_check_failed` instead of passing.
  - Evidence: live read-only run of the new query lists all nine tables plus a deliberately nonexistent one as missing, no error. Tests `test_a_failed_grant_check_blocks_rather_than_passing`, `test_the_grant_query_treats_a_missing_table_as_missing_not_as_an_error`.
- Finding: **the felt-state lane re-queried on every chat turn while no definition exists.** A miss never reached the cache, and "no row yet" is this lane's steady state until the first self-inquiry run lands, so every stance build and equilibrium gate tick paid one blocking query.
  - Fix: `SubstrateFeltStateReader._remember_miss` stores a negative cache entry, only for lanes that declare an explicit `cache_ttl_sec` (so `curiosity_signals` and the other pre-existing lanes keep their behaviour).
  - Evidence: `test_a_miss_is_remembered_for_the_cache_ttl_and_does_not_leak_into_ctx`, `test_lanes_without_an_explicit_cache_ttl_still_requery_on_a_miss`.

## Restart required

Order matters (additive `forbid` field):

```bash
# 0. Grants (production write -- Juniper runs this, once):
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < scripts/sql/2026-09-08_grant_orion_readonly_self_inquiry.sql

# 1. Runner first, then Hub, then cortex-exec (stance producer + felt-state lane):
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build

# 2. Watch the line's first decision (within one tick, 300s):
docker logs orion-hub --since 10m 2>&1 | grep -E "curiosity_self_inquiry|curiosity_investigation started"
#    expect: "... self_inquiry=True self_cap=3/day ..." then either
#    "curiosity_self_inquiry_starting run=..." or a named block reason.

# 3. Force one if you don't want to wait for the window:
curl -s -X POST http://localhost:8080/api/curiosity/api/self-inquiry/run-now

# 4. Evidence the whole path moved:
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview \
  "MATCH (s:SelfDefinition) RETURN s.run_id, s.text"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "SELECT version, created_at, left(content,160) FROM self_concept_history WHERE concept_id='self:definition' ORDER BY created_at DESC LIMIT 3"
#    then one chat turn: the stance brief's identity_kernel_orion should start with "In my own words".
```

## Risks / concerns

- Severity: medium. Concern: the definition is Orion-authored and lands in every chat turn's identity kernel with no operator approval — by design (Orion's graph is Orion's), but a bad definition would be repeated to Orion until the next self-inquiry run revises it. Mitigation: append-only history (`self_concept_history`) — "current" is latest `created_at`, so a row can be superseded or the lane's `produced_by` filter narrowed; the marker line is clipped to ~900 chars; no-evidence definitions never reach the store.
- Severity: medium. Concern: a self-inquiry turn costs a full FCC turn (~20–40 min) three times a day on the `agent` lane, on top of the investigation line's budget. Mitigation: cap and cooldown are separate knobs; `-1` disables the cap; the line yields to the investigation line only when its own gates block.
- Severity: low. Concern: `has_table_privilege` check runs through Hub's privileged pool; an unreadable answer counts as granted (same rule as `pg_role_missing`). Mitigation: the turn's own `psql` fails loudly and the journal says so; the ledger read also skips tables that error.
- Severity: low. Concern: `identity_yaml` adapter could store the augmented `orion_identity_summary` on a cold pull. Mitigation: marker-based strip makes the prepend idempotent (tested).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2158
