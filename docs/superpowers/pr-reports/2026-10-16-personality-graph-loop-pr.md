## Summary

- Orion gets a persistent personality graph, built bottom-up from Orion's own outputs **from today on** (no backfill). Its lanes are dreams, journal, reverie, reading, day letter, chat, outreach, curiosity choices, and self-inquiry seeds.
- A reducer thread in orion-durable-runs runs every 10 minutes. It embeds new evidence (bge-large via vector-host), matches it to nodes ("dispositions": things Orion keeps returning to), and lets weights fade. Nodes are born, promoted and retired from evidence alone.
- **Weight reads as "how many lanes keep returning to this theme".** It is computed as `sum over lanes of base * min(1.5, ln(1 + recent hits in lane))`: a log response that saturates, with a per-lane counter that fades (Weber-Fechner plus habituation). One lane's volume buys nothing past about 4 recent hits. A lane counts toward promotion only with 2 recent hits.
- **Read-back:** the strongest nodes reach chat stance, reverie, the curiosity kickoff and outreach, through one shared fixed template (`orion/personality/cue.py`). The cue uses Orion's own words, as a record, never as tone instructions. It is fail-open and bounded, and on by default.
- **Hub Personality tab:** shows the exact cue the lanes see, the active, candidate and retired nodes with weight sparklines and lanes, and the evidence behind each node.

## Outcome moved

Stance was regenerated from scratch every turn, and its only memory was a 30-minute in-process cache that only chat read. Orion now has one cross-lane self, with history, that every speaking lane reads.

A read-only 7-day dry run (real sources, real embedder, folded day by day, nothing written) produced 170 nodes: 9 active, 153 candidate, 8 retired. The leading active nodes:
1. A self-inquiry theme across 5 lanes: "I cannot say yes from the inside, and after looking I am less sure the inside is where this question…" (weight 4.86).
2. A cabinet/camera theme across reverie, journal and outreach (weight 4.07).

The reverie flood (2,586 rows on one prediction-error theme) now sits 5th, at 3.00. Under the first formula it would have ranked 1st at 37.9.

## Current architecture

- `ChatStanceBrief` is produced per turn by an LLM. Its carry-over is `_PRIOR_STANCE_CACHE` in cortex-exec (30 min TTL, per session, lost on restart).
- Identity text is static YAML.
- The #2646 self card is designed but not built.
- The only cross-lane self-state was arousal.
- The Temporal Self frame already reached stance, reverie, curiosity and outreach over HTTP. This PR rides that same pattern.

## Architecture touched

- **Contracts.** `orion/schemas/personality.py`:
  - `PersonalityFrameV1` is in both registries.
  - `DispositionV1`, `ChoiceStyleV1` and `PersonalityCueV1` are in `_REGISTRY` only.
  - `personality_cue` is registered in `context_provenance`.
- **Pure code.** `orion/personality/graph.py` (the reducer) and `orion/personality/cue.py` (the renderer).
- **orion-durable-runs.** `personality_sources.py`, `personality_store.py` and `personality_driver.py` are new. `main.py` gains the lifespan start and the `GET /personality/frame`, `/personality/dispositions` and `/personality/evidence` routes.
- **orion-thought.** `personality_cue.py` is new. `bus_listener.py` gains a parallel fetch for stance. `reverie.py` adds it to the reverie gather. The templates `stance_react.j2` and `reverie_narrate.j2` get the new block.
- **orion-hub.** `personality_cue_reader.py` is new. `curiosity_investigation.py` and `orion/curiosity/kickoff_prompt.py` add the block to the kickoff. `endogenous_outreach.py` adds it to the prompt, and its `grounding_summary` records node ids, never text. `personality_routes.py` plus a template and JS make the panel. `main.py` passes the flag.
- **Postgres.** New tables `personality_disposition`, `personality_evidence`, `personality_weight_history` and `personality_cursor`. Their single writer is durable-runs.

## Files changed

- `orion/schemas/personality.py`, `orion/schemas/registry.py`, `orion/schemas/context_provenance.py`: the contracts
- `orion/personality/graph.py`, `orion/personality/cue.py` and their tests: the reducer and the renderer
- `services/orion-sql-db/manual_migration_personality_graph_v1.sql` and `_rollback.sql`: the tables
- `services/orion-durable-runs/app/personality_*.py`, `app/main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md` and tests: the reducer thread and routes
- `services/orion-thought/app/personality_cue.py`, `bus_listener.py`, `reverie.py`, `settings.py`, `.env_example`, `README.md` and tests; `orion/cognition/prompts/stance_react.j2`, `reverie_narrate.j2`: chat and reverie read-back
- `services/orion-hub/scripts/personality_cue_reader.py`, `curiosity_investigation.py`, `endogenous_outreach.py`, `main.py`, `app/settings.py`, `.env_example`, `README.md` and tests; `orion/curiosity/kickoff_prompt.py`: curiosity and outreach read-back
- `services/orion-hub/scripts/personality_routes.py`, `templates/personality.html`, `templates/index.html`, `static/js/personality*.js` and tests: the Hub panel
- `.github/workflows/personality-tests.yml`, `orion-durable-runs-tests.yml`, `temporal-self-readers-tests.yml`, `orion-reading-tests.yml`: CI coverage for every new test file
- `docs/superpowers/pr-reports/2026-10-16-personality-graph-loop-pr.md`: this report

## Schema / bus / API changes

- **Added:**
  - Contracts: `PersonalityFrameV1`, `DispositionV1`, `ChoiceStyleV1`, `PersonalityCueV1`.
  - durable-runs routes: `GET /personality/frame`, `/personality/dispositions`, `/personality/evidence`.
  - Hub routes: `/api/personality/{frame,dispositions,evidence,cue}` and the `/personality` page.
  - The four `personality_*` tables.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:**
  - The stance, reverie, curiosity kickoff and outreach prompts gain one block when the frame is fresh and non-empty.
  - Outreach `grounding_summary` gains `personality_cue`, `personality_cue_ids` and `personality_cue_skip`.
- **Compatibility notes:**
  - No bus channel changes.
  - Every reader fails open: if durable-runs is missing or stale, no block is added and nothing waits.
  - The frame is dated by the last committed tick. A reducer that stops committing therefore goes stale for every reader after 1 h.

## Env/config changes

- **Added keys:**
  - orion-durable-runs: `PERSONALITY_ENABLED=true`, `PERSONALITY_TICK_SEC=600`, `PERSONALITY_EMBED_URL=http://${PROJECT}-vector-host:8320/embedding`.
  - orion-thought: `PERSONALITY_CUE_ENABLED=true`, `PERSONALITY_FRAME_URL=http://durable-runs:8121`, `PERSONALITY_CUE_TIMEOUT_SEC=0.5`, `PERSONALITY_CUE_MAX_AGE_SEC=3600`, `PERSONALITY_CUE_TOP_K=5`.
  - orion-hub: `HUB_PERSONALITY_CUE_ENABLED=true`.
- **Removed keys:** none.
- **Renamed keys:** none.
- **`.env_example` updated:** yes, all three services.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py <svc> --all-keys`: yes, into the primary checkout. The plain run skips these keys because they are outside `SYNC_PREFIXES`. All 9 keys are verified present. `${PROJECT}` expansion is verified against orion-dream's identical live key.
- **Skipped keys requiring operator action:** none.

## Metric quality gate: disposition weight

1. **Provenance.** `orion/personality/graph.py` `_derived_weight` over `Node.lane_recent`. The counters are fed by `personality_sources.py` from the tables listed above, plus curiosity offer and outcome rows.
2. **Independence.** The lanes are not fully independent:
   - Reverie and the metacog journal both describe the same substrate.
   - Dream text can echo reverie, which is why the 5th node above is active.
   - Journal entries that copy a reading or a day letter (`world_pulse` and `orion_day`) are excluded, so one reading cannot count as two lanes.
   - The habituation cap keeps a correlated lane to at most one lane's worth. It cannot stop two correlated lanes from promoting a node. Flagged under Risks.
3. **Theory anchor.**
   - CAPS (Mischel & Shoda): personality as a weighted network of if-then dispositions.
   - Weber-Fechner: log response to stimulus.
   - Habituation: repeated stimulus, weaker response that recovers.
4. **Live sanity.**
   - In the 7-day dry run the weights are not degenerate: active weights range 2.57-4.86, with 170 nodes across 9 lanes.
   - Rest state exists. A theme with no new evidence fades and retires in about 10 weeks (pinned by a test). An empty graph renders nothing; it does not read as a fake calm. A frame from a dead reducer reads as stale.
5. **Existing mechanism.**
   - Concept induction is disabled (`decision=disabled`, #2646).
   - The #2646 self card is unbuilt.
   - Stance's facets are regenerated per turn.
   - Nothing else holds cross-lane recurrence.
6. **Reversibility.**
   - Three flags turn it off (`PERSONALITY_ENABLED`, `PERSONALITY_CUE_ENABLED`, `HUB_PERSONALITY_CUE_ENABLED`).
   - The tables are additive, and the rollback file drops them.
   - Nothing outside this PR reads them.

## Tests run

```text
orion/personality/tests + tests/test_personality_schemas_registered.py -> 106 + 4 passed
services/orion-durable-runs: pytest tests (throwaway local Postgres 16 as ORION_ADMISSION_TEST_DSN) -> 525 passed, 1 skipped
services/orion-thought: test_stance_temporal_self_cue.py test_personality_cue.py test_reverie_recent_arcs.py -> 44 passed
services/orion-hub: test_outreach_personality_cue.py test_endogenous_outreach.py test_personality_cue_reader.py test_curiosity_personality_cue.py -> 265 passed
services/orion-hub: test_personality_routes.py test_hub_panel_reachability.py -> 43 passed, 4 skipped (browser smoke needs playwright; the agent ran it with playwright: 66 passed)
node --test services/orion-hub/static/js/personality.test.js -> 12 passed
Static gates: env template parity PASS; metric lineage PASS; migration REQUIRED-BY OK; env/compose parity OK x3; no-personal-terms + schema registry tests passed; git diff --check clean
```

## Evals run

```text
Read-only live dry runs (real sources + real embedder, folded in memory, nothing written), each round driving a reducer change:
- gain/sqrt(1+n): 24 h -> top node 37.9 weight from 244 reverie rows (reverie fixation)
- base/(1+recent): 7 d -> a 2,600-row reverie theme led; 6 of the top 8 promoted by 1-4 echo hits
- capped log + 2-recent-hit promotion (this PR): 7 d -> 9 active, led by multi-lane themes; the reverie flood is 5th at 3.00
Embedding dominates cost (1,825 s embed vs 28 s fold for 5,307 rows); a normal tick is ~5 rows.
```

## Docker/build/smoke checks

```text
Not deployed. Live path UNVERIFIED: no live tick has run, and no chat turn has logged personality_cue rendered=True.
Migration post-pull dry run: "would apply" (allowlist-clean, own BEGIN/COMMIT, lock_timeout=10s).
vector-host reachability from the durable-runs container verified by the agent (app-net, 1,024-dim bge-large).
```

## Review findings fixed

- Finding: the frame was dated by the request time, so the cue's staleness check never fired, and a dead reducer would have been read back as fresh in every lane.
  - Fix: a `__tick__` heartbeat cursor written by every committed tick dates the frame. Before the first commit the frame is dated at the epoch. Staleness warnings go first in the list.
  - Evidence: a real-Postgres test asserts `generated_at == heartbeat < request time`.
- Finding: the founding label had similarity 1.0, so labels were frozen forever.
  - Fix: the founding label starts at the match threshold.
  - Evidence: `test_founding_label_can_be_replaced_by_closer_evidence`.
- Finding: seeds older than 7 days, and backlog rows, were retired at birth. That was 4 of 21 live seeds.
  - Fix: the candidate TTL runs from `last_folded_at` (the wall-clock fold time).
  - Evidence: `test_old_seed_is_not_retired_at_birth`, `test_backlog_evidence_hours_old_is_not_retired_in_the_same_tick`.
- Finding: the weight history never recorded fade, because it compared tick to tick.
  - Fix: history now compares against `history_weight`, the last recorded weight.
  - Evidence: `test_history_records_slow_fade_once_it_adds_up`.
- Finding: a lane's volume bought weight. This was found in the dry runs over two rounds.
  - Fix: the capped-log derived weight, clamped counters, and a 2-recent-hit promotion floor (1.5, so it is reachable across ticks).
  - Evidence: tests pin the flood cap, breadth beating a flood, a stray hit not promoting, and a quiet theme fading in weeks.
- Finding: per-slice issues: a retried seed double-folded; fold blocked the event loop; a frame/count race; one bad row could block a source; render_failed missing from the skip vocabulary; the Hub panel showed an invalid frame as "empty"; rows lacked accessibility; and `main.py` never passed the cue flag.
  - Fix: all fixed in their slices.
  - Evidence: tests.

## Restart required

From the clean primary `main` checkout after merge, in this order:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_personality_graph_v1.sql
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull rebuild can auto-apply the migration, because it is allowlist-clean. Readers fail open, so the order matters only for how soon the block appears.

## Risks / concerns

- Severity: medium. Concern: correlated lanes can still promote a theme. Dream text echoing reverie promoted a prediction-error node, the 5th in the dry run. Mitigation: it is visible in the Hub panel (lane counts per node), and it is the next calibration target. Lane independence weighting is a follow-up, not guessed here.
- Severity: medium. Concern: machine-status text (digest and compliance lines from metacog journal entries) becomes dispositions. This is bottom-up truth about what Orion writes, not a filter target (spec rev 2: no top-down preconditioning). Mitigation: watch in the panel before acting.
- Severity: medium. Concern: node labels are excerpts of Orion's own LLM-voiced text, shown back to Orion. They are content, not style instructions, but they carry the old voice. Mitigation: they are short (120 chars shown) and labelled as a record. Tone measurement (#2729) will show whether it echoes.
- Severity: low. Concern: CPU under a full backlog drain (1,800 rows times a few thousand nodes is several minutes in a worker thread). Mitigation: fold runs off the event loop, ticks never overlap, and normal ticks are about 5 rows.
- Severity: low. Concern: situation subnodes (CAPS if-then) are not built. The read-back shows the strongest nodes overall. Mitigation: documented as the next phase in the spec.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2738

🤖 Generated with [Claude Code](https://claude.com/claude-code)
