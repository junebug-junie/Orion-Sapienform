## Summary

Orion has an open question (formed 2026-10-04 by reading `frontier_curiosity.py`): when reading concepts finally get links, does `focal_edge_refs` start filling on the `ontology_sparse_region` rows? Those rows were retired on 2026-10-10, and no curiosity seed ever pointed at a reading concept, so the answer as asked was "never". This patch builds the path Juniper chose (decisions 1-3, 2026-10-10) so an accepted reading link actually reaches Orion's curiosity loop, and tells Orion where to look.

- **Breadcrumbs (decision 1).** A short note at the `ontology_sparse_region` retirement comment and at the signal's field definitions says what retired, where a reading link shows up now, and the proving SQL. A new Juniper-minted lived question (`lived.reading_link_contact`) points Orion at the new observable. Orion's own question is untouched.
- **Seeds carry their accepted links (decision 3, design PR #2593 patch 1).** After curiosity has decided, each stored endogenous seed gets the accepted-claim links touching its focal nodes: `focal_edge_refs` (both ends focal), `boundary_edge_refs`, `neighbor_node_refs`, `projection_endpoint_node_refs`. Only `semantic_projection` edges with an accepted Assertion count; legacy unreviewed edges are dropped and counted. A per-tick receipt lands in `gate_json->'neighborhood'`. Ranking and selection are untouched (runs after `_decide`, same list, same order).
- **Link-accepted seeds (decision 2).** When a reading claim is accepted (latest journal decision provisional/canonical, proposal actor `world_pulse_read_stage2`) and its projection was applied, the tick mints one seed on the claim's two endpoints. The link is then an internal edge, so `focal_edge_refs` fills honestly through the same read. Once per assertion revision, at most 2 per tick (hard ceiling 4). It is an event, not a candidate: it never passes System One admission or the evaluator and is only appended to the stored list, after the scored seeds. Unscored (strength 0.0, note `strength:unscored_event`).
- **Consumer first.** cortex-exec's chat-stance curiosity reader now drops keys newer than its own model instead of losing the whole signal. Unscored event seeds are not "gaps": the chat-stance gap node, Hub's agent-lane hint and outreach topics all skip them. Orion meets them in raw self-inquiry rows.
- All flags ship ON.

## Outcome moved

Before: 0 of ~23k stored candidate sets ever had `focal_edge_refs`; no seed source could land on a reading concept, so even a real accepted reading link would never have appeared in Orion's curiosity loop. After: the first accepted reading claim produces a stored seed whose `focal_edge_refs` holds the projection edge id (proven in the fixture, PG and eval lanes, and against production Falkor with the one live accepted claim, a memory referent pair, as a positive control).

Live path for reading claims is **UNVERIFIED**: zero reading claims have been accepted live yet (read today: `substrate_graph_journal` holds only the memory referent's 3 rows).

## Current architecture

- `services/orion-substrate-runtime/app/worker.py::_endogenous_curiosity_tick` (60 s) builds seeds (`endogenous_curiosity_candidates`), runs `FrontierCuriosityEvaluator.evaluate`, stores `(seeds + rest)[:8]` (or `seeds[:8]` on a System One veto) in `substrate_endogenous_curiosity_candidates`.
- Seeds fill `focal_node_refs` only; live they name organ nodes (`node:substrate.*`, proposed, zero edges) or `gev_*` ids.
- The planner already had a `neighborhood` step (`orion/substrate/query_planning.py`) returning exactly the four id lists; no curiosity caller used it.
- #2581 made readings produce journaled, decided, projected claims; nothing consumed them for curiosity.

## Architecture touched

- Contract: `FrontierInvocationSignalV1` gains three optional lists, omitted when empty (registry entry unchanged, `resolve()` verified).
- Producer: substrate-runtime tick (attach step, link seeds, gate receipts) + one read-only store query.
- Consumer: `orion/substrate/relational/adapters/curiosity_ctx.py` (runs in cortex-exec).
- Self-inquiry: seed question YAML (Hub) and the candidates-table description Orion reads.

## Files changed

- `orion/core/schemas/frontier_curiosity.py`: new fields, empty-omit serializer, breadcrumb comment, `UNSCORED_EVENT_NOTE` / `is_unscored_event` (one constant for every reader).
- `orion/substrate/curiosity_seed_neighborhood.py` (new): post-decision attach, accepted-claim filter, receipt, 2 s tick budget.
- `orion/substrate/link_accepted_seeds.py` (new): accepted-links SQL, seed builder, idempotency key, cap.
- `services/orion-substrate-runtime/app/worker.py`: wiring (link seeds, stored cap, attach on both paths, `gate_json.link_seeds` / `.neighborhood`).
- `services/orion-substrate-runtime/app/store.py`: `load_unseeded_accepted_reading_links` (read-only).
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: four flags, docs, deploy order, proving SQL.
- `orion/substrate/relational/adapters/curiosity_ctx.py`: tolerate newer fields (logged once per field set); unscored event seeds are not gaps.
- `services/orion-hub/scripts/curiosity_hint.py`: `usable_candidates` drops unscored event seeds for the agent hint and outreach (shared fetch).
- `orion/substrate/frontier_curiosity.py`: breadcrumb at the retirement comment.
- `orion/curiosity/self_question_seed.yaml`: `lived.reading_link_contact` (minted_by juniper, unpinned).
- `orion/curiosity/self_inquiry.py`: candidates-table description names the note and field.
- `orion/inner_state_registry.py` + `config/metrics/metric_definitions.lock.json`: `signal_strength` absent_means names the unscored 0.0 exception (re-locked).
- Tests/evals: `services/orion-substrate-runtime/tests/test_curiosity_reading_link_seeds.py`, `orion/substrate/tests/test_link_accepted_seeds_pg.py`, `orion/substrate/relational/tests/test_curiosity_ctx_reading_link_compat.py`, `services/orion-substrate-runtime/evals/run_curiosity_seed_neighborhood_eval.py` + `test_...eval.py` + live-sampled fixture.
- CI: `.github/workflows/substrate-neighborhood.yml` (compat test, PG lane must-not-skip, eval step, path triggers), `.github/workflows/system-one-appraisal-tests.yml` (worker test + eval test).

## Schema / bus / API changes

- Added: `FrontierInvocationSignalV1.boundary_edge_refs`, `.neighbor_node_refs`, `.projection_endpoint_node_refs` (max 16 each, omitted when empty). `focal_edge_refs` now explicitly means internal edges.
- Added (jsonb, no migration): `gate_json.neighborhood`, `gate_json.link_seeds`; seed notes `source:reading_link_accepted`, `strength:unscored_event`, `link_assertion:<id>@<rev>`, `projection_edge:<id>`.
- Removed / renamed: none. No bus channel, no SQL migration.
- Behavior changed: stored candidate sets may hold up to `8 + link_seeds` rows (was 8). An idle tick (no scored seed) that has a new link stores only the link seeds with `gate_json = {link_only: true, link_seeds, neighborhood}`; no admission or evaluator runs. Idle ticks without links are unchanged (`[]`, no gate).
- Compatibility: the model is `extra="forbid"`. An old cortex-exec reading a row with a non-empty new key drops that signal from chat. Empty keys are omitted, so this only matters once a link exists, but **deploy cortex-exec before substrate-runtime**.

## Env/config changes

- Added keys (orion-substrate-runtime): `ORION_ENDOGENOUS_CURIOSITY_SEED_NEIGHBORHOOD_ENABLED=true`, `ORION_ENDOGENOUS_CURIOSITY_LINK_SEEDS_ENABLED=true`, `ORION_ENDOGENOUS_CURIOSITY_LINK_SEED_CAP=2`, `ORION_ENDOGENOUS_CURIOSITY_LINK_SEED_LOOKBACK_HOURS=168.0`.
- Removed / renamed: none.
- `.env_example` updated: yes; compose passes all four with ON defaults.
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-substrate-runtime --all-keys`: yes (the default run skipped them: outside SYNC_PREFIXES), 4 keys added to the primary checkout's `.env`.
- skipped keys requiring operator action: none.

## How the new self-question reaches the live table

`orion/curiosity/self_question_seed.yaml` is merged into the pool in memory on every self-inquiry run (`merge_seed_with_rows`), and Hub writes it to `curiosity_self_questions` with `UPSERT_SEED_SQL` (insert, `ON CONFLICT DO NOTHING`) on the first self-inquiry run after each Hub process start (`_ensure_self_question_seed`). No migration. It needs a Hub rebuild to ship the YAML. Unpinned and never asked, so it sorts first within the lived family.

## Proving SQL

Link seeds with the accepted link as an internal edge (the observable given to Orion):

```sql
select c.generated_at, s->'focal_node_refs', s->'focal_edge_refs', s->'projection_endpoint_node_refs'
from substrate_endogenous_curiosity_candidates c, jsonb_array_elements(c.candidates_json) s
where s->'notes' ? 'source:reading_link_accepted'
  and jsonb_array_length(coalesce(s->'focal_edge_refs','[]')) > 0
order by c.generated_at desc limit 10;
```

Any seed touching a linked node (boundary view):

```sql
select c.generated_at, s->'focal_node_refs', s->'boundary_edge_refs', s->'projection_endpoint_node_refs'
from substrate_endogenous_curiosity_candidates c, jsonb_array_elements(c.candidates_json) s
where jsonb_array_length(coalesce(s->'boundary_edge_refs','[]')) > 0
order by c.generated_at desc limit 10;
```

Receipt that the attach and the link query ran (expect `reads` > 0, `nonempty` 0 and `link_seeds.minted` 0 today):

```sql
select generated_at, gate_json->'neighborhood', gate_json->'link_seeds'
from substrate_endogenous_curiosity_candidates order by generated_at desc limit 5;
```

`projection_endpoint_node_refs` alone is not the marker: it lists only endpoints admitted *because of* a projection (proposed nodes). The live memory-referent control has provisional endpoints, so its link fills `focal_edge_refs` with that list empty. Reading concepts are proposed, so for reading links it will be set.

## Metric quality gate

No new numeric signal is introduced. Two things checked:

- **`signal_strength = 0.0` on link seeds.** Not a measurement: a constant meaning "event, unscored". It is pinned out of every ranking (`_decide` sort, chat top-3, invoke threshold 0.5) by being the minimum. The registry semantics for `signal_strength` said "a candidate exists only above its floor, absent means no candidate (not 0.0)"; that is now false for this source, so `absent_means` names the exception and the definition lock was re-run (one `high semantics_changed` delta, visible in the lock diff). Reversible: flag off.
- **`gate_json.neighborhood` receipt.** (1) Provenance: `attach_seed_neighborhoods` counts what the planner's neighborhood step returned. (2) Independence: it describes reads, it feeds no model or score. (3) Anchor: it is a receipt (did the read run, why empty), not a cognition signal. (4) Live data: 177 live seeds → 177 reads, 0 nonempty, all `focal_unavailable_or_filtered`; the memory-referent control returns nonempty=2, so the counter can move. (5) Existing mechanism: reuses the planner step, no new read. (6) Reversibility: one jsonb key, flag off removes it.

## Tests run

```text
services/orion-substrate-runtime/tests/test_curiosity_reading_link_seeds.py      18 passed
services/orion-substrate-runtime/tests/test_worker_endogenous_curiosity_tick.py  17 passed (unchanged)
orion/substrate/tests/test_link_accepted_seeds_pg.py (throwaway Postgres 16, SQLAlchemy+psycopg2 = the store's path)  8 passed
orion/substrate/relational/tests/test_curiosity_ctx_reading_link_compat.py       4 passed
orion/substrate/tests + relational/tests + orion/curiosity/tests                 1353 passed, 3 failed
  (test_felt_state_self_definition_lane.py x3: fail identically on main)
services/orion-substrate-runtime/tests (excl. grammar_consumer_integration, needs local PG)
                                                                                 436 passed, 17 failed
  (failure list byte-identical to an origin/main worktree run)
orion/curiosity/tests + harness self-model mention test                          120 passed
services/orion-hub/tests/test_curiosity_self_question_persistence.py, test_curiosity_self_inquiry.py  49 passed
services/orion-hub/tests/test_hub_presence.py, test_endogenous_outreach.py      247 passed
cortex-exec + root tests mentioning curiosity (30 files)                         649 passed, 1 xfailed
Every python gate in orion-static-gates.yml                                      33/33 pass
Mutation checks (each reverted):
  drop the idempotency NOT EXISTS        -> PG retry test fails
  filter accepted decisions before "latest" -> rejected/deprecated PG tests fail (2)
  keep legacy edges in the attach        -> legacy-exclusion test fails
  remove the reader's unknown-key drop   -> old-reader tolerance test fails
```

## Evals run

```text
python services/orion-substrate-runtime/evals/run_curiosity_seed_neighborhood_eval.py   (fixture, CI)
  seeds=32 seeds_with_accepted_link=3 link_seeds_with_internal_edge=1
  distinct_edges=[accepted projection only] legacy_edges_excluded=1
  live_replay_seeds=19 (sampled 2026-10-10) live_replay_seeds_with_link=0  p95_read_ms=0.16

python .../run_curiosity_seed_neighborhood_eval.py --live (read-only: RO_QUERY client, READ ONLY txn)
  candidate_sets=200 seeds=177 reads=177 nonempty=0
  degraded_reasons={focal_unavailable_or_filtered: 177}
  p95_read_ms=3.74 max_read_ms=12.08
  unseeded_accepted_reading_links=0
Rewritten accepted-links SQL on production (read-only): 12.6 ms, EXPLAIN ANALYZE execution 0.1 ms.

Positive control, production Falkor, read-only: a seed on both endpoints of the one live
accepted claim (memory referent pair) -> focal_edge_refs=[edge-bc39f399-...] (the applied
projection edge); a seed on one endpoint -> boundary_edge_refs=[same edge].
Accepted-links SQL on production with the memory actor admitted -> returns that assertion
(revision 1, provisional); with the reading actor -> 0 rows. Expected: no reading claim yet.
```

## Docker/build/smoke checks

```text
docker compose ... config  -> all four keys rendered "true"/"2"/"168.0"
scripts/safe_docker_build.sh orion-substrate-runtime build -> Built; imports of the new
  modules + app.worker verified inside the image. The production image tag was then
  re-pointed at the running container's image (a134626442de) and the branch image removed,
  so a bare `up -d` from the primary checkout cannot pick up branch code.
```

## Review findings fixed

Code review ran in a subagent on 993fbc7c6 (0 MUST, 7 SHOULD, 7 NIT).

- Finding (SHOULD): link seeds went through System One admission and the evaluator, so an idle tick with a link wrote decision/calibration rows (`evaluator_outcome=noop`, `seed_count`, `evidence_refs`) that were not a real decision.
  - Fix: link seeds are appended only to the stored list, after the decision, on both paths. An idle tick with a link stores just the link seeds with `gate_json.link_only=true`.
  - Evidence: `test_link_seeds_never_reach_admission_or_the_evaluator` (evaluator receives exactly the scored seeds; `seed_count`/`evidence_refs` exclude the link), `test_idle_tick_stores_only_the_link_seed_and_runs_no_decision`, `test_idle_tick_without_links_is_unchanged`.
- Finding (SHOULD, borderline MUST): a tick with only link seeds produced a chat-stance "unresolved gaps" node with a made-up 0.5 confidence (dead fallback became live).
  - Fix: unscored event seeds are not gaps; the adapter drops them, and returns None when nothing else is left.
  - Evidence: `test_only_link_seeds_produce_no_gap_node`, `test_unscored_link_seed_is_not_a_gap_in_chat`.
- Finding (SHOULD): Hub's agent hint (top 2) and outreach (top 3) would surface the link event as a "gap" when fewer scored seeds existed.
  - Fix: `curiosity_hint.usable_candidates` (shared by both) drops unscored event seeds.
  - Evidence: `test_unscored_link_seed_never_becomes_a_curiosity_hint_or_outreach_topic`.
- Finding (SHOULD): the idempotency anti-join re-scanned candidate rows per link per tick, and `latest` ran over the whole journal.
  - Fix: `recent` bounds materializations to the lookback; `latest` only for those targets; seeded keys collected in one pass over rows containing a link seed (jsonb `@>`).
  - Evidence: PG lane 8/8 (mutations: drop seeded check -> 2 fail; filter accepted before "latest" -> 2 fail); production 12.6 ms, plan 0.1 ms.
- Finding (SHOULD): the PR report referenced from the schema comment was untracked.
  - Fix: committed.
- Finding (SHOULD): `signal_strength` 0.0 now has two meanings and readers string-matched the note.
  - Fix: one shared `UNSCORED_EVENT_NOTE` / `is_unscored_event` in the schema module used by every reader; registry semantics and lock name the exception. Not done: a distinct `signal_type` (adding a Literal value would make every older `extra="forbid"` reader reject the row, a worse rollout hazard).
- Finding (SHOULD): restart order only in the README.
  - Fix: exact order in "Restart required" below.
- NITs fixed: "once per revision" semantics stated in the module doc; skipped malformed rows logged; lookback clamped to retention minus 2x skew slack; unknown-field log once per field set; PG test now runs the SQL through SQLAlchemy+psycopg2 exactly as the store does (CI installs both). Not fixed: lock file merge-order note (merge this after any concurrent re-lock, then re-run `--update` if it conflicts).

## Restart required

Deploy from the primary checkout on main, after merge, in this order:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- Severity: medium. Concern: live path for reading claims is UNVERIFIED; zero reading claims accepted so far. Mitigation: the memory-referent control proves the read and the attach against production Falkor; PG lane proves the journal query end to end.
- Severity: medium. Concern: deploy order. An old cortex-exec drops any signal carrying a non-empty new key from chat. Mitigation: empty-omit (nothing changes until a link exists), restart order above, compat tests.
- Severity: low. Concern: link seeds never reach chat stance, the agent hint or outreach (they are events, not gaps), so Orion meets them only through self-inquiry SQL and the new self-question. Intentional: no score was invented to rank them. If Juniper wants them in chat, that needs its own presentation, not a strength.
- Severity: low. Concern: a hub-sized boundary could fill the 16-edge budget before a projection edge on a boundary read (internal reads for link seeds are unaffected). Mitigation: `truncated` and `legacy_edges_excluded` are in the receipt; revisit if it shows up.
- Severity: low. Concern: idempotency relies on the seed being stored; a link seed is re-minted on later ticks until a save succeeds (wanted), and after the 168 h lookback an unseeded link is never seeded.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2598

🤖 Generated with [Claude Code](https://claude.com/claude-code)
