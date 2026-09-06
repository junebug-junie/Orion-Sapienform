# PR report: attention schema surface -- one shape, four attending processes

**Date:** 2026-09-06
**Branch:** `feat/attention-schema-impl`
**Implements:** `docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md` (PR #2092)
**Program:** Sentience Striving Program, Objective 3

## Summary

- Orion now has one shared shape for "what am I attending to, and why" -- `AttentionSchemaV1`
  -- and every process that attends and selects writes to it: the substrate attention tick,
  reverie, curiosity, and the cortex chat turn.
- Each lane is a small pure projection of that lane's own existing artifact. No base class,
  no registry, no shared reason taxonomy: each keeps its own `attention_reason` words.
- All four publish on one bus channel, `orion:attention:schema`; `orion-sql-writer` is the
  single writer into a new table, `substrate_attention_schema`, with 90-day retention.
- Missing Question 1 is settled from a real run: *what* a curiosity run attended is
  recoverable from Orion's own graph stamps; *why it chose it* is recorded nowhere and needs a
  kickoff-prompt change, which is a separate proposal.
- Per Juniper: cortex is a producer on the surface and remains the kickoff point for any later
  sequencing. Nothing routes, gates, or budgets off the surface in this arc.
- Two lanes live-verified within minutes of deploy; the other two need a real chat turn and a
  real curiosity run, which happen on their own schedule (see below).

## Outcome moved

O3's blind-rater test now has real alternatives. Before, the AST/HOT reducer could only be
rated against a straw man (its own output vs noise). Now four processes emit the same four
facts, so the question becomes "which of these is deliberating, which drifting, which
reacting?", and it can be asked with one stratified query:

```sql
SELECT process, count(*), count(DISTINCT reason_narrative)
FROM substrate_attention_schema GROUP BY 1;
```

## Current architecture

Before this patch, three (really four) attention self-models existed with no shared surface:

- `substrate_attention_self_model` rows every ~30s, `attention_reason` + `reason_narrative`
  write-only; its scalar side-channel had a consumer, its self-model content had none.
- Reverie chains in `substrate_reverie_chain` with a `trigger` field that is NULL on all
  24,140 live rows (`chain.py` never sets it), plus LLM-written `interpretation`s per thought.
- Curiosity runs that stamp priors in Orion's own FalkorDB graph and write a `TurnOutcome`,
  with no field for why a prior was chosen.
- Cortex chat turns building a full `AttentionFrameV1` (selection, override, suppressions)
  and persisting only a salience trace of it.

## Architecture touched

```
substrate tick  --to_attention_schema(self_model)-->  \
reverie chain   --to_attention_schema(chain, thoughts)-> orion:attention:schema --> orion-sql-writer --> substrate_attention_schema
curiosity run   --to_attention_schema(run, outcome, priors)->  /
cortex turn     --to_attention_schema(frame)------->  /
```

- Contract: `orion/schemas/attention_schema.py`, registered in both registry maps,
  `orion/bus/channels.yaml`, `orion/inner_state_registry.py`, metric definition lock,
  `orion/sentience_striving_program/instruments.yaml`.
- Producers: `orion-substrate-runtime` (async loop publishes what the sync tick projected),
  `orion-thought` (after the chain's own publish + persist), `orion-hub` (after the run's
  graph read, before the journal), `orion-cortex-exec` (after the salience trace).
- Consumer/writer: `orion-sql-writer` route + model + subscribe guard + retention entry.

## Files changed

- `orion/schemas/attention_schema.py`: the schema, channel/kind constants, `clip()`.
- `orion/schemas/registry.py`: registered in `_REGISTRY` and `SCHEMA_REGISTRY` (verified via `resolve()`).
- `orion/bus/channels.yaml`: `orion:attention:schema`, four producers, one consumer.
- `orion/inner_state_registry.py`: `attention_schema.v1` entry (REHEARSAL -- honest: no cognition consumer yet).
- `config/metrics/metric_definitions.lock.json`: re-locked for the new channel + inner-state entry.
- `orion/sentience_striving_program/instruments.yaml`: `attention_schema_surface` instrument, two claims.
- `orion/substrate/attention_self_model.py`: `to_attention_schema()` (reducer untouched).
- `orion/substrate/attention_frame.py`: `to_attention_schema()` for the chat-turn frame.
- `orion/reverie/attention_schema.py`: reverie adapter.
- `orion/curiosity/attention_schema.py`: curiosity adapter + the stamped-priors Cypher read.
- `services/orion-substrate-runtime/app/worker.py`: one-slot handoff from the sync tick to the async loop; publish.
- `services/orion-thought/app/chain.py`: keep the thoughts, publish after persist.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_publish_attention_schema()` after `_read_turn_result()`.
- `services/orion-cortex-exec/app/attention_schema_publish.py` (+ `chat_stance.py`, `main.py`): module-bound bus publish, fail-open.
- `services/orion-sql-writer/app/models/attention_schema.py` (+ `models/__init__.py`, `worker.py`, `settings.py`, `grammar_truth.py`, `grammar_retention_loop.py`, `.env_example`, `README.md`): table, route, subscribe guard, retention.
- Tests: `tests/test_attention_schema_surface.py`, `services/orion-thought/tests/test_reverie_attention_schema_publish.py`, `services/orion-sql-writer/tests/test_attention_schema_sql_shape.py`, `services/orion-cortex-exec/tests/test_attention_schema_publish.py`, appended sections in the hub and substrate-runtime suites.
- `docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md`: "What shipped", MQ1/MQ2 answers, reverie correction, "Cortex is the kickoff".

## Schema / bus / API changes

- Added: `AttentionSchemaV1` (`attention.schema.v1`); channel `orion:attention:schema`; table `substrate_attention_schema`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for any existing consumer. Every adapter is write-only and best-effort; the reverie chain, curiosity journal, and chat turn all complete even if the surface publish raises (tested).
- Compatibility notes: `narrative_kind` is on the schema so the blind-rater control arm can stratify computed vs self-report rows. `attention_reason` is free text by design and must never become a shared enum.

## Env/config changes

- Added keys: `SUBSTRATE_ATTENTION_SCHEMA_RETENTION_DAYS=90` (orion-sql-writer); `orion:attention:schema` appended to `SQL_WRITER_SUBSCRIBE_CHANNELS` and `attention.schema.v1` to `SQL_WRITER_ROUTE_MAP_JSON` in `.env_example`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (orion-sql-writer only).
- local `.env` synced: `python scripts/sync_local_env_from_example.py orion-sql-writer --all-keys` -> `+SUBSTRATE_ATTENTION_SCHEMA_RETENTION_DAYS='90'` in the primary checkout; copied into the worktree for the deploy. Note the default pass reported "No changes needed" because `SUBSTRATE_` is outside the sync prefix list -- `--all-keys` was required.
- skipped keys requiring operator action: the live `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` lists were already diverged from the example before this patch and were left alone. Both are covered by code: `effective_subscribe_channels` appends the channel and `route_map` merges over the defaults. Confirmed live: the deployed sql-writer subscribed to the channel and routed the kind.

## Tests run

```text
tests/test_attention_schema_surface.py                                20 passed
services/orion-thought/tests/test_reverie_attention_schema_publish.py + test_reverie_chain.py   21 passed
services/orion-sql-writer/tests/test_attention_schema_sql_shape.py + test_route_map_completeness.py   9 passed
services/orion-cortex-exec/tests/test_attention_schema_publish.py + attention frame/trace suites   31 passed
services/orion-hub/tests/test_curiosity_investigation.py            120 passed (3 new)
services/orion-substrate-runtime/tests/test_worker_attention_self_model_tick.py + store   24 passed (4 new)
tests/ registry + channel-catalog + curiosity worldview + inner-state + drift   green
  (3 tests already red on main before this branch, unrelated: channel_prefix_guardrail,
   autonomy_goals_bus_catalog schema, exec_result_channel specificity)

Static gates: check_inner_state_registry OK, check_definition_drift OK (re-locked),
check_env_key_single_source OK, check_service_env_compose_parity orion-sql-writer N/A (env_file),
check_journal_dispatch_registry OK, check_metric_lineage OK, check_system_health_producers OK,
check_sentience_instruments --static-only OK, git diff --check OK.
```

## Evals run

```text
No eval harness exists for this seam. The acceptance checks in the design doc are the eval:
  1. producers_live   -- SQL claim in instruments.yaml (re-run by check_sentience_instruments with a DB)
  2. blind-rater      -- MANUAL, needs a 24h stratified window; pre-registered threshold TBD in that run
  4. narrative_diversity_per_lane -- MANUAL claim, per-lane query recorded in instruments.yaml
Acceptance Check 5 (nothing downstream changed) is covered by the existing suites above staying green.
```

## Docker/build/smoke checks

Deployed from the worktree via `scripts/safe_docker_build.sh <svc> up -d --build`.

Wave 1 (00:44-00:46Z): orion-sql-writer, orion-substrate-runtime, orion-thought.

```text
sql-writer:   subscribed to orion:attention:schema; route attention.schema.v1 -> AttentionSchemaSQL;
              retention loop days={... 'substrate_attention_schema': 90}; table created on boot.
first row:    00:46:05Z, 9s after substrate-runtime came up, process=substrate_attention,
              attention_reason=top_down_override, a real override with real numbers:
              "loop 'open-loop-a2bcad525fe2' (bottom_up=0.58) beat 'open-loop-968fda08910c' (0.75)
               via applied_bias=1.00"
reverie row:  00:51Z (first chain after restart), process=reverie, attention_reason=coalition_broadcast,
              narrative_kind=self_report, confidence=0.547 (ema over 4 thoughts, ended max_steps),
              narrative = Orion's own interpretation ("The coalition is focused on the harness
              closure prediction error, which has decayed...").
code-in-container: sql-writer imports the model and reports the route/retention; substrate-runtime
              and thought containers grep the new symbols. Not "Image Built" -- the actual files.
```

After 4.5h live (00:46Z -> 05:11Z), zero producer/consumer errors in any log:

```text
process              rows  distinct narratives  distinct reasons
substrate_attention   508   116                  4  (goal_target_already_winning 240, goal_matched_no_loop 112,
                                                     top_down_override 94, no_open_loops 62)
reverie                39    36                  2  (coalition_broadcast 25, no_coalition 14)
```

Wave 2: see the section below (filled after review).

## Review findings fixed

Code review ran twice (the first run was killed by the API rate limit mid-flight); the
second's verify pass confirmed four findings, all fixed.

- Finding: every producer built its `BaseEnvelope` without `correlation_id=`, and `orion-sql-writer` stamps the persisted row's `correlation_id` column from the *envelope* (a fresh uuid4 by default), after payload validation. So the reverie thought id, the cortex frame id and curiosity's uuid5 run id were all silently replaced in the table -- every join back to the originating artifact dead on arrival.
  - Fix: `bind_correlation(row)` in `orion/schemas/attention_schema.py` returns the row and a UUID guaranteed equal; all four producers pass it on the envelope. A row with no usable id gets one minted so payload and column still agree (the same reason `chat_stance_belief_bus.py` passes `correlation_id=`).
  - Evidence: every producer test now asserts `str(envelope.correlation_id) == payload["correlation_id"]`; the hub test additionally asserts it equals the journal entry's correlation id. Wave-2 live check below.
- Finding: a unified turn runs at least two brain-mode cortex legs (`harness_finalize_reflect`, `orion_voice_finalize`) under one correlation id and no `turn_id`, so both produced `entry_id=cortex-<corr>`; the writer kept whichever arrived first, the richer finalize frame was dropped with only an INFO log.
  - Fix: `to_attention_schema(frame, leg=ctx["verb"])` puts the leg in the key (`cortex-<corr>-<verb>`), falling back to a generated_at stamp when no leg is known. Both legs persist, joined by `correlation_id`.
  - Evidence: `test_cortex_two_legs_of_one_turn_do_not_collide`.
- Finding: curiosity / endogenous-outreach / journal turns also go through the cortex brain-mode legs and were landing as `cortex_turn`, while the channel and docs described that lane as "every real chat turn".
  - Fix: corrected the claim rather than adding a session-name heuristic. `cortex_turn` is defined as every unified turn cortex built a stance for, human or self-initiated; a self-initiated turn's row joins to its originating lane's row by `correlation_id` (made real by the first fix). Live check that motivated this: the previous two days of `chat_history_log` were entirely `orion_journal` / `orion_outreach` sessions -- filtering to human-only would have emptied the lane.
  - Evidence: docstrings on the adapter and schema, the channel description, and the design doc's "What shipped" say so.
- Finding: the curiosity graph read (`read_attended_priors`, in `to_thread`) sat before the journal write with no bound; a stalled FalkorDB (5s connect + 5s read socket timeouts) would hold this loop's sole persistence path ~10s per run.
  - Fix: `asyncio.wait_for(..., timeout=self.attention_schema_graph_read_timeout_sec)` (6.0s); a timeout is reported as `graph_unreadable` and the journal proceeds.
  - Evidence: `test_attention_surface_a_hung_graph_read_does_not_hold_the_journal` -- a reader that sleeps 3s returns in <2s with a `graph_unreadable` row.

Reviewer-verified wiring with no finding: sql-writer's validate/filter/insert path (duplicate PK is a logged no-op, not a poisoned session), the substrate bus/`_service_ref` lifecycle, `publish_with_reconnect`'s positional call, the thought chain's `publish` gate, and all static gates green.

## Restart required

Wave 1 already restarted. For the remaining producers:

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  Concern: the curiosity lane's `reason_narrative` is computed from graph facts (which prior, how far confidence moved). It varies per run but it is not Orion's reason for choosing that prior -- that is not recorded anywhere. Acceptance Check 3 is met for the *what*, open for the *why*.
  Mitigation: a kickoff-prompt change asking Orion to write one `why_chosen` line onto the `TurnOutcome` node. Cognition-loop change, proposal mode; not done here.
- Severity: low
  Concern: `ReverieChainV1.trigger` is NULL on every live chain, so reverie's reason is `coalition_broadcast` on ~100% of rows -- template-thin at the reason level (the narrative is not).
  Mitigation: reported per lane, never aggregated (instruments.yaml claim is MANUAL and per-lane for this reason). Wiring a real trigger into `chain.py` is a separate, small patch.
- Severity: low
  Concern: `producers_live` claim is recorded as 0 at PR time and will read DRIFTED as lanes come online.
  Mitigation: that is the claim doing its job; re-record after 24h.
- Severity: low
  Concern: `scripts/safe_graphify_update.sh` refused the incremental graph update (known bug) and auto-restored; the committed graph does not yet include the new modules.
  Mitigation: per CLAUDE.md, not re-run; a full re-extraction is the safe path when someone next does one.

## PR link

(filled on push)
