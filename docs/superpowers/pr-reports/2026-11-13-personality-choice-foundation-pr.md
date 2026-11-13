## Summary

Orion's personality graph now learns only from choices: what Orion picked, and what Orion turned down. The old graph grouped Orion's own sentences and ended up with job boilerplate; that path is gone.

- **One contract for every choice point.** `PersonalityChoiceV1` (orion/schemas/personality.py) now carries the picked option, every option turned down with its market rank, a "declined all" flag for "no move" / "say nothing", the runaway-guard flag `care_promoted`, a small situation dict, and the act and correlation ids. It rides the new bus channel `orion:personality:choice`.
- **One writer, one table.** orion-durable-runs writes every choice into the new `personality_choice` table (insert-only, `choice_id` primary key). It takes choices from the bus, and it turns the curiosity lane's stored offers and outcomes into choice rows itself, including a one-time backfill of the 296 curiosity runs already stored.
- **The reducer folds only choices.** Each choice becomes one direction: the picked option's meaning minus the average meaning of what was turned down. Following the market's first option counts 0.25, departing from it counts 1.0. Labels are computed ("people over pipeline, 11 of 14"), never quoted from prose, and no model writes them.
- **Prose sources deleted, not hidden.** reverie, journal, reading text, day letter, chat and outreach text, and self-inquiry seeds no longer feed anything. Rev 2 nodes are never loaded, counted or shown. Two analysis scripts that measured the dead path are removed.
- **Retire job written, not run.** `scripts/personality_rev3_retire_prose_nodes.py` snapshots and then retires the 180 rev 2 nodes (27 active + 153 candidate). Dry run by default; `--apply` is a production write that Juniper approves separately.

## Outcome moved

- Before: 9,558 evidence rows, 59% reverie, 1.3% real choices. The 27 "active dispositions" were repeated job sentences.
- After (read-only replay of the live curiosity backlog through the new reducer, `evals/personality_choice_replay.py`): 215 choices from 156 runs, 73% departed from the market's order, 117 tendencies, 1 active. The top tendency is "turn start context is stale snapshot over camera layer mediation not closeness (+13 more), 21 of 21". Every label is a computed "X over Y, n of m".
- Prime and outreach agents now have a fixed contract and channel to publish against.

## Current architecture

- `orion/personality/graph.py` folded embeddings of Orion's own output from 9 prose lanes, labelled nodes with an excerpt, and promoted a node when 2 lanes repeated it.
- `services/orion-durable-runs/app/personality_sources.py` read 9 tables by keyset cursors, no backfill. Curiosity picks were one lane among them.
- `PersonalityChoiceV1` existed as a P0 read-only model (two lanes, one of them a formula) with no table, no channel and no writer.

## Architecture touched

- Contract: `orion/schemas/personality.py` (`PersonalityChoiceV1`, `ChoiceOptionV1`, `ChoiceLane`, `DispositionOrigin` adds `choice`), both registries, `orion/bus/channels.yaml`.
- Postgres: `personality_choice` (new), `personality_disposition.opposed_count` (new column).
- orion-durable-runs: bus intake in `app/main.py`; `personality_sources.py`, `personality_driver.py`, `personality_store.py` rewritten for choices.
- Pure: `orion/personality/graph.py` (choice reducer), `orion/personality/choices.py` (curiosity rows -> choices), `orion/personality/naming.py` (no prose source keys).

## Files changed

- `orion/schemas/personality.py`: rev 4 contract and validation.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: registry comments; new channel `orion:personality:choice`.
- `config/metrics/metric_definitions.lock.json`: definition-drift lock for the new channel.
- `orion/personality/graph.py`: choice-only reducer (direction, weight, opposition, computed labels). Prose lanes, seeds, source keys and the rekey are deleted.
- `orion/personality/choices.py`: curiosity offer/outcome -> one `PersonalityChoiceV1` per tested prior. The formula-chosen attention lane is deleted (a formula is not a choice).
- `orion/personality/naming.py`: the "new source" re-ask rule reads choice lanes.
- `services/orion-durable-runs/app/personality_sources.py`: two reads only (curiosity sync, `personality_choice`) plus naming.
- `services/orion-durable-runs/app/personality_driver.py`: curiosity sync, bus intake `offer_choice`, choice -> evidence, option-vector cache.
- `services/orion-durable-runs/app/personality_store.py`: `insert_choices`, `opposed_count`, origin filter on every read.
- `services/orion-durable-runs/app/main.py`: subscribes to `orion:personality:choice` when the loop is on.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `README.md`: comments and docs (no key change).
- `services/orion-durable-runs/scripts/personality_unmerge.py`: no lane cap any more.
- `services/orion-durable-runs/evals/personality_choice_replay.py`: read-only live replay with degenerate-data checks.
- `services/orion-sql-db/manual_migration_personality_choice_v1.sql` (+ rollback): additive DDL, `REQUIRED-BY: orion-durable-runs`.
- `scripts/personality_rev3_retire_prose_nodes.py`: the retire job (not run).
- Deleted: `scripts/analysis/measure_personality_choices.py`, `measure_personality_lane_independence.py` and their tests (they measured the dead path).
- Tests: `orion/personality/tests/test_{graph,choices,naming}.py`, `services/orion-durable-runs/tests/test_personality_{loop,naming,postgres,choice_replay}.py`, `tests/test_personality_schemas_registered.py`, `tests/test_personality_rev3_retire_prose_nodes.py`.
- CI: `.github/workflows/personality-tests.yml`, `orion-durable-runs-tests.yml` paths and commands.

## Schema / bus / API changes

- Added: bus channel `orion:personality:choice` (kind `personality.choice.v1`, schema `PersonalityChoiceV1`, consumer orion-durable-runs, producers orion-mind and orion-hub once their patches land). Table `personality_choice`. Column `personality_disposition.opposed_count`.
- Changed: `PersonalityChoiceV1` / `ChoiceOptionV1` reshaped to the rev 4 contract. The P0 model had no channel, table or external consumer; its only users (the P0 analysis script and `choices.py`) are updated or deleted.
- Removed: prose sources; `voluntary_attention_override` choice lane.
- Behavior changed: `/personality/frame`, `/personality/dispositions`, `/health.personality` show choice nodes only. `ChoiceStyleV1` now counts every choice lane by decision.
- Compatibility: `DispositionV1` and `PersonalityFrameV1` are unchanged in shape (readers in hub and thought keep validating). `DispositionOrigin` gains `choice`, so a reader running an older image would refuse a frame that holds a choice node; the cue readers are off (R0, deployed 11-13), so nothing reads the frame today.

### The contract (for Prime / outreach producers)

```text
PersonalityChoiceV1
  schema_version      "personality.choice.v1"
  choice_id           str, deterministic per decision (and per pick), e.g. "prime:<thought_id>"
  lane                "prime" | "outreach" | "curiosity" | "reading" | "chat"
  occurred_at         aware datetime
  chosen              ChoiceOptionV1 | None   (None only when declined_all)
  alternatives        [ChoiceOptionV1], >= 1, <= 50   (every other option shown)
  chosen_market_rank  int >= 1 | None   (must equal chosen.market_rank)
  declined_all        bool = False   ("no move" / "say nothing")
  care_promoted       bool = False
  picks_in_decision   int >= 1 = 1
  situation           dict, <= 16 keys
  act_id              str | None
  correlation_id      str | None
ChoiceOptionV1 = {key: str, text: str (<= 400, what Orion saw), market_rank: int >= 1 | None}
Refused: no alternatives; chosen missing without declined_all; chosen given with declined_all;
duplicate option keys; rank mismatch; extra fields; naive time.
```

Publish: `BaseEnvelope(kind=PERSONALITY_CHOICE_KIND, payload=choice.model_dump(mode="json"))` on `PERSONALITY_CHOICE_CHANNEL`, from the act that already made the model call. Re-publishing the same `choice_id` is a no-op.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: comment only (`services/orion-durable-runs/.env_example`, PERSONALITY_ENABLED block).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (no key changes).
- skipped keys requiring operator action: none.

## Metric quality gate (choice direction and weight)

1. **Provenance.** Direction: `orion/personality/graph.py` `choice_direction`, from option texts embedded by vector-host (`app/personality_driver.py` `_embed_texts`). Weight: `graph.evidence_weight`, from `chosen_market_rank`, which the producer copies from the market order it showed. Curiosity ranks come from `curiosity_offer_decisions.offered[].rank` (ordered head only).
2. **Independence.** Direction and weight share one choice row but measure different things: what was preferred (meaning) versus how much the pick departed from the offered order (rank). Weight is not derived from the embedding. Opposition reuses the direction (a monotonic reading of the same cosine), so it is not new signal; it only feeds the label's "m".
3. **Theory anchor.** Revealed preference (Samuelson 1938): a choice among available alternatives reveals preference, and following the default reveals little. Chosen-minus-declined directions isolate the preference from what the options share.
4. **Live-data sanity** (read-only replay over the 296 live curiosity runs, 2026-11-13): 215 choices; 73% depart from rank 1 (not saturated at "always top"). Pairwise cosine of directions: median 0.024, p90 0.17 (rest point near 0, so unrelated choices really read as unrelated), p99 0.96 (repeated picks of one prior against a similar set). Raw option embeddings sit at median 0.56, which is why rev 2's 0.82 threshold cannot apply to directions; `match_threshold` is 0.5. Opposition at 0.5 never fired on curiosity (15-option decline sets dilute a reversal); at 0.25 it fired 38 times and at 0.2 122 times, which is noise (p01 of pairwise cosine is -0.22), so it stays at 0.5 and fires only on clean reversals from small option sets (Prime, outreach). Backlog choices are faded by their own age, so the backfill yields 1 active tendency, not a fake burst of 4.
5. **Existing mechanism.** Reuses the disposition tables, the vector-host embedder, the naming loop and the Temporal Self situation stamp. No new service.
6. **Reversibility.** `PERSONALITY_ENABLED=false` stops the loop and drops bus choices. The migration is additive with a rollback file. The rev 2 nodes are untouched until the retire job runs, and its `before.csv` restores them.

## Tests run

```text
orion/personality/tests tests/test_personality_schemas_registered.py tests/test_personality_rev3_retire_prose_nodes.py
  scripts/analysis/tests/test_measure_personality_situation_specificity.py scripts/analysis/tests/test_eval_personality_naming_grounding.py
  -> all passed
services/orion-durable-runs: pytest tests (full suite, CI env, disposable Postgres 16 on :55499)
  -> 1026 passed, 1 skipped (before the replay test was added); personality files re-run after: all passed
tests/test_personality_postgres.py + test_personality_rev3_retire_prose_nodes.py (DB part) on disposable Postgres -> passed
static gates: test_no_personal_terms, test_agent_trace_schema_registry, test_deploy_migration_gate,
  test_grammar_event_producer_catalog, test_no_private_eval_artifacts, check_migration_required_by,
  check_metric_lineage --gate, check_definition_drift --gate (after --update), check_inner_state_registry,
  check_env_key_single_source, check_async_routes_not_blocking, check_sentience_instruments --static-only -> pass
migration allowlist: orion.sql_migration_drift.auto_apply_refusals(...) -> [] (auto-applies on pull)
```

## Evals run

```text
python3 services/orion-durable-runs/evals/personality_choice_replay.py --dsn <live, read-only> --embed-url http://localhost:8320/embedding
choices: 215  decisions: 156  rank1: 58  departed: 157 (73%)
pairwise_cos_median: 0.024  p99: 0.964  nodes: 117  active: 1  opposed_nodes: 0
checks: has_choices, has_departures, directions_not_degenerate, labels_computed -> all True
```

## Docker/build/smoke checks

```text
No Docker build or deploy run (task scope: no `up`). The change is Python + SQL inside an existing image;
requirements unchanged. Live bus delivery of a published choice is UNVERIFIED until deploy.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Deploy order (from the clean primary main checkout after merge):

```bash
git pull   # post-pull applies manual_migration_personality_choice_v1.sql (allowlisted additive DDL) before rebuilding orion-durable-runs
```

If the auto-apply is off or refused, apply by hand first, then rebuild:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_personality_choice_v1.sql
```

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then (Juniper approves separately) the retire job:

```bash
POSTGRES_URI=<live conjourney DSN> python3 scripts/personality_rev3_retire_prose_nodes.py           # dry run
POSTGRES_URI=<live conjourney DSN> python3 scripts/personality_rev3_retire_prose_nodes.py --apply   # write
```

## Risks / concerns

- Severity: medium. Concern: `orion:personality:choice` is Redis pub/sub; a choice published while durable-runs is down or restarting is lost. Mitigation: producers make one choice per model call they already make; loss is bounded to restart windows. A durable stream is a follow-up if M1 counts come up short.
- Severity: low. Concern: the label's "n of m" opposition will read "n of n" for curiosity (15-option decline sets rarely produce a clean reversal). Mitigation: measured above; Prime and outreach (3 to 5 options) are where opposition can fire.
- Severity: low. Concern: curiosity option text is the prior's slug made readable, not the prior's full statement (the statement lives in FalkorDB, another service's store). Mitigation: slugs are descriptive; the label says what was compared.
- Severity: low. Concern: the separate cue readers (hub, thought) still exist with their flags off. Mitigation: they are retired by the Prime `tending` line patch (rev 4 week 2), not here.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
