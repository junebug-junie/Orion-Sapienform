# Ladder retirement PR 6: retire the proposal (L7), policy (L8) and dispatch (L9) rungs, with no orphans

## Summary

- **Three ladder services go:** orion-proposal-runtime, orion-policy-runtime and
  orion-execution-dispatch-runtime, with their packages, schemas, config, tests, smoke scripts and
  analysis scripts. About 33,700 lines are deleted.
- **Nothing real stops.** Every real act already starts itself: the cooling act (#3131), painting
  (#3134, #3137), retro play (#3140), AI Town (#3143) and image pruning (#3153). In the last 24 hours
  the three services wrote about 40,900 frames each and sent nothing.
- **Every reader outside the ladder is cut or repointed** (the nine couplings below). The two that
  needed real work: the field digester now computes `action_warrant` itself, and the painting's
  self-study seed now reads curiosity's answers.
- **The Hub lattice page and the ladder check show L7–L9 as "retired".** The sensing half stays
  checked, and the ladder check is GREEN on live data.
- **The tables stay until PR 7.** Their drop is a separate draft PR (#DROP_PR), to merge when it is
  applied, at least 2 days after this deploys.

Spec: `docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md` ("PR 6", the
coupling list, "No replacement needed"). Pattern: #3115 (feedback and consolidation).

## Outcome moved

- **About 122,700 ladder frames a day stop** (three tables, 21 GB together, two of them never pruned).
- **Three always-running containers go.** Their only output in the last 24 h was frames: dispatch's
  allocator logged `sending=0` on every tick and its motor budget spent 0.0 s.
- **Nothing Orion uses changes:** the act ledger still grades every act, the cooling act still
  learns, painting still gets a self-study seed, and the anomaly scorer still gets `action_warrant`
  (300 of 300 live ticks recompute to the exact stored value).

## What the three services actually did in the last 24 h (live, 2026-11-07 ~04:00Z)

| Service | Rows written | Real output |
|---|---|---|
| proposal | 40,895 frames, 135,410 candidates (inspect 60k, maintain 34k, summarize 25k, observe 13k) | none: no template carries a real act any more |
| policy | 40,895 frames, 127,466 approvals | none: nothing downstream acts on an approval |
| dispatch | 40,884 frames: 30,051 `dispatch_read_only`, 10,834 `dry_run`; **0** with a dispatch | **0** `substrate_dispatch_results` rows (the last one is 2026-11-05 16:52Z); 0 RPCs; motor budget spent 0.0 s; the allocator logged `sending=0` on every tick |

Also: `substrate_execution_dispatch_inflight` holds 0 rows. `substrate_act_episodes` got its last row
on 2026-11-05 04:13Z, and dispatch was its only writer. No ladder-kind outcome row (summarize,
maintain, inspect) has been written in 48 h.

## The couplings, and how each was resolved

1. **sql-writer retention floor.** The `policy_pending` and `dispatch_pending` probes, the floor, and
   the proposal-frame retention job are removed, with `SUBSTRATE_PROPOSAL_FRAMES_RETENTION_DAYS`.
   The generic `floor_resolver` seam stays, and its tests stay too. A test now proves that no probe
   or marker SQL remains in code.
2. **The act ledger and `self_shed`.** `AttentionWinnerRefV1` moves into `orion/autonomy/self_shed.py`,
   because readers remain (the act ledger, the cooling act, and hardware-watch's rules via `self_shed`).
   **The act ledger's workspace binding is kept, on purpose.** The brief said "0 winners bound in
   24 h", but live data disagrees: 51 Orion `acted` loop verdicts on `hub.chat_reply` since 10-15,
   2 in the last 24 h, the newest at 2026-11-07 03:19Z. The binding does not depend on the ladder.
   The "0" was the proposal layer's own winner binding, which retires here.
3. **The field digester's import of `orion.proposals.scoring`.** `PRESSURE_DIMENSIONS`, the precision
   floors, alpha and min-samples move to `orion/field/pressure_dimensions.py`. The comments that
   record each metric gate move with them, verbatim. This also removes `action_warrant`'s lazy
   import cycle.
4. **mood_arc's `action_warrant`.** The digester computes `action_warrant(state)` on the tick's final
   state and writes it into the corpus row, so the anomaly scorer and the JSONL corpus both get it.
   `corpus_enrichment` no longer reads `substrate_proposal_frames`.
   **Parity, live:** for 300 recent ticks, recomputing from the stored field state gives the value
   the proposal runtime stored, exactly (max difference 0.0). The value is absent (never 0.0) when
   nothing can be scored.
5. **Temporal Self.** The `action_outcome` source reads act rows only (`act:%` and the cooling act's
   `self_regulate`). A Postgres test pins it, and a mutation (filter removed) makes that test fail.
6. **The dream compaction gate.** The `PolicyDecisionFrameV1` gate is replaced by an injected
   `consent` callable. It fails closed: there is no consent, a non-True answer is no, and an
   exception is no. Apply stays off (`ORION_DREAM_COMPACTION_APPLY_ENABLED=false`, Juniper's
   decision 6). The market's consent gate plugs in here later.
7. **The reading pull.** These are removed:
   - the `orion:world_pulse_read:pull:*` RPC, its schemas, and the Hub listener;
   - the pull handling in `world_pulse_read_pipeline.py`;
   - the gate module `orion/world_pulse_read/pull.py`;
   - the three `HUB_WORLD_PULSE_READ_PULL_*` keys.

   Dispatch was the only sender, and its last pull was on 10-30. Hub's own lane still reads, and
   its pipeline tests pass. The `world_pulse_read_seed` pull columns stay as history.
8. **Self-study context. The check FAILED, so the read is repointed.** The visual chain's self-study
   seed read four analysis producers in `journal_entries`. Their only trigger was dispatch's
   `analyze_self_study_source`, and the last row was written 2026-11-05 16:24Z. Curiosity does not
   write those rows. The seed now reads curiosity's self-inquiry answers
   (`self_concept_history`, `produced_by='curiosity_self_inquiry'`, about 6 a day). These are clean
   first-person answers. Curiosity's `journal_entries` bodies were not used, because they open with
   bookkeeping. Over 7 days, a 6 h window holds a fresh answer 78.7% of the time; the old source
   had 70.7% over its last 12 days. So `ORION_SELF_STUDY_CONTEXT_MAX_AGE_SEC` still fits.
   The old prefix allowlist was a privacy guard. It goes with its source, because Juniper's standing
   rule is that there is no privacy boundary between her and Orion (2026-10-07).
9. **Reverie → action.** `orion/reverie/proposal.py` and its tests are deleted.
   `ORION_REVERIE_PROPOSE_ENABLED` went with the proposal service's `.env_example`. 0 dispatches in
   14 days.
10. **Shared tables stay:** `substrate_action_outcomes`, `substrate_action_effect_posterior`,
    `substrate_signal_control_cells`, `substrate_world_action_episodes` and `gpu_pool_orion_shed`.
    These modules stay: `orion/autonomy/{prediction,contrast,world_episodes,world_settlement*,shed_act,shed_settlement}.py`.
    The allocator is deleted: nothing outside dispatch and its evals called it.
11. **The rest:**
    - **Hub:** the routes `/api/substrate/{proposals,policy,execution-dispatch}/latest` are gone.
      The lattice shows L7–L9 as `retired` and never queries their tables. The `action_ceiling`
      gate reads `retired`, not a defaulted `dry_run`.
    - **Ladder liveness:** `RETIRED_RUNGS` now has five entries.
    - **Metrics liveness:** the `l7_l11_ladder` source and its row-count helpers are removed.
    - **Registry rows:** the inner-state `l7_l11_ladder` row is retired, as are the schema registry
      entries and the channels.
    - **Config and CI:** the physiology contract, `CODEOWNERS`, the env-sync prefixes and
      `DEFAULT_SERVICES`, and CI (`thinking-rung3`, `attend-act-loop`, `visual-baseline`,
      `self-monitor-tests`, `orion-hardware-watch-tests`) are updated. A new
      `orion-field-digester-tests` workflow runs because no workflow ran the digester's tests before.
    - **Scripts:** `Makefile`, the smoke scripts, and the analysis scripts.
    - **Audits and evals:** the agency-episode audit no longer reads the dispatch tables; it reports
      them as `retired`. `self_monitor/replay.py` loses the `allocator` seam.
    - **Migration headers:** the three migrations whose `REQUIRED-BY` named dispatch are re-headed to
      `none <reason>`. One of them is `manual_migration_feedback_control_reduction.sql`.

Also found and fixed in the sweep:
- `visual_baseline.v1.yaml` lived under `config/proposals/`. It moves to `config/reverie/`, with the
  thought and durable-runs Dockerfiles. `load_baseline_policy()` silently falls back to
  `enabled=False` when its file is missing, so a wrong path would have stopped every waking
  painting. A test pins the file and both `COPY` lines. In-image checks show `enabled True` in both
  images.
- `orion/substrate/pending_marker_reconcile.py` was used only by policy and dispatch. It is deleted,
  with the `POLICY_RECONCILE_`, `DISPATCH_RECONCILE_` and `TRANSPORT_PROPOSAL_` sync prefixes.
- The artifact-rating path (`scripts/rate_artifact.py`, `scripts/score_artifact_ratings.py`,
  `orion/autonomy/rating.py`) rated dispatch results and read the dispatch tables. It had scored
  0 rows. It is deleted. `ChatResponseFeedbackV1.target_artifact_ref` stays, because it is a bus
  contract.
- `scripts/test_check_metric_lineage_liveness_wiring.py` had 2 tests failing on main: the fake node
  had no `.surface`. Fixed.

## Ladder check after the change (live, read-only, `scripts/check_substrate_ladder_liveness.py`)

```text
ok  grammar:orion-bus / orion-biometrics / orion-cortex-exec   (3-18 s old)
ok  receipts:transport_bus_reducer / biometrics_node_reducer / substrate.bus_synaptic /
    substrate.perception / substrate.perception_walkway / vision_organ_reducer
ok  camera_input:cam0 / walkway
ok  field_state (M4): 2s old, limit 15m
ok  attention (M5): 1s old, limit 15m
retired proposal (2026-11-07), policy (2026-11-07), dispatch (2026-11-07),
retired feedback (2026-11-04), consolidation (2026-11-04)
GREEN
```

## Repo-wide grep: no reader remains outside history and migrations

```text
imports of a deleted module (code):                                   none
SQL FROM/JOIN/INTO/UPDATE on the six retired tables, outside migrations:
  three historical replay evals, --live/--export mode only, each now documented as needing a
  pg_dump restore once PR 7 drops the table (town_self_start_replay degrades with a stderr note):
    services/orion-hub/evals/retro_leisure_self_start_replay.py
    services/orion-hardware-watch/evals/run_image_prune_replay_eval.py (--export; CI uses the fixture)
    services/orion-embodiment/evals/town_self_start_replay.py
  plus test docstrings/fakes that assert the table is NOT read
bus channel orion:world_pulse_read:pull:*:                            a tombstone comment in channels.yaml
```

Everything else that names the retired services is in a docstring or comment that records history.
Other remaining names: the historical skew fixtures (`tests/fixtures/schema_skew/proposal_frame.*`,
which pin the nested-forbid-model mechanism) and the body-atlas fixtures.

## Current architecture (before)

Field state (M4) -> attention (M5) -> proposal (L7) -> policy (L8) -> dispatch (L9). Each rung polled
the previous rung's table and wrote a frame. Dispatch's allocator decided what to send to cortex-exec,
Hub, the GPU pool and embodiment. After PRs 1-5 it had no real route left: it refused everything on
every tick.

## Architecture touched

The three services are deleted. The contracts change: `orion/bus/channels.yaml`,
`orion/schemas/registry.py`, `orion/inner_state_registry.py` and `config/physiology/contract.yaml`.
These services change: sql-writer, field-digester, hub, thought, durable-runs and dream.
The new shared module is `orion/field/pressure_dimensions.py`. The config move is
`config/reverie/visual_baseline.v1.yaml`.

## Files changed

- **Deleted (142 files):**
  - `services/orion-{proposal,policy,execution-dispatch}-runtime/`, `orion/{proposals,policy,execution_dispatch}/`;
  - `orion/schemas/{proposal_frame,policy_decision_frame,execution_dispatch_frame,world_pulse_read_pull}.py`;
  - `config/{proposals,policy,execution_dispatch}/`;
  - `orion/autonomy/{allocator,act_candidates,act_episodes,rating}.py`, `orion/reverie/proposal.py`,
    `orion/world_pulse_read/pull.py`, `orion/substrate/pending_marker_reconcile.py`;
  - Hub `substrate_{proposal,policy,execution_dispatch}_routes.py` and `world_pulse_read_pull_listener.py`;
  - `scripts/smoke_{proposal_frame,policy_decision_frame,execution_dispatch}_v1.sh`;
  - `scripts/{rate_artifact,score_artifact_ratings}.py` and three ladder-only analysis scripts;
  - the rung-3 / reading-pull / allocator evals;
  - 46 ladder-only tests, and the sql-writer retention-integrity eval.
- **New:**
  - `orion/field/pressure_dimensions.py`;
  - `tests/test_field_pressure_dimensions.py`;
  - `orion/reverie/tests/test_baseline_policy_file.py`;
  - `.github/workflows/orion-field-digester-tests.yml`;
  - `config/reverie/visual_baseline.v1.yaml` (moved).
- **Changed:** the couplings above, plus the comment and README updates that pointed at deleted code.

## Schema / bus / API changes

- **Removed:**
  - the schemas `ProposalFrameV1`, `ProposalCandidateV1`, `PolicyDecisionFrameV1`, `PolicyDecisionV1`,
    `ExecutionDispatchFrameV1`, `ExecutionDispatchCandidateV1`, `WorldPulseReadPullRequestV1` and
    `WorldPulseReadPullResultV1`;
  - the channels `orion:world_pulse_read:pull:request` and `orion:world_pulse_read:pull:reply:*`;
  - the Hub routes `/api/substrate/proposals/latest`, `/api/substrate/policy/latest` and
    `/api/substrate/execution-dispatch/latest`.
- **Producer lists:** orion-execution-dispatch-runtime leaves `orion:cortex:exec:request:background`
  and `orion:autonomy:action:outcome`.
- **Moved:** `AttentionWinnerRefV1` (now in `orion/autonomy/self_shed.py`; it is never on the bus).
- **Behavior changed:**
  - the field corpus row carries `action_warrant`;
  - the Temporal Self outcome source is act rows only;
  - the dream applier takes `consent`;
  - the painting's self-study seed source changes;
  - the lattice L7–L9 are `retired`.
- **Compatibility:** no remaining `extra="forbid"` model gained or lost a field.

## Env/config changes

- **Removed:**
  - `SUBSTRATE_PROPOSAL_FRAMES_RETENTION_DAYS` (sql-writer);
  - `HUB_WORLD_PULSE_READ_PULL_ENABLED`, `HUB_WORLD_PULSE_READ_PULL_DAILY_CAP` and
    `HUB_WORLD_PULSE_READ_PULL_AROUSAL_ENABLED` (hub);
  - every key of the three deleted services' `.env_example` (16 + 17 + 49).

  `git grep` confirms that no code or compose file reads any of them. Both settings classes ignore
  unknown variables.
- **Added:** none.
- **Sync script:** the three services leave `DEFAULT_SERVICES`. The prefixes `POLICY_RECONCILE_`,
  `DISPATCH_RECONCILE_` and `TRANSPORT_PROPOSAL_` are dropped, and a test pins both.
- **Local `.env`:** no key is added. The stale keys are removed after deploy (see Restart required).

## Tests run

```text
TESTS_PLACEHOLDER
```

## Evals run

```text
EVALS_PLACEHOLDER
```

## Docker/build/smoke checks

```text
DOCKER_PLACEHOLDER
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Deploy from the clean primary `main` after merge, when no chat turn is running. Order matters only in
one place: the field digester must be on the new image before the proposal runtime stops, or the
anomaly scorer loses `action_warrant` until it is.

1. Run the post-pull rebuild as usual. These rebuild: sql-writer, field-digester, hub, thought,
   durable-runs and dream. The deleted service dirs drop out of the rebuild on their own.
2. Stop and remove the three retired containers. No compose file is left for them. Stop dispatch
   first:
```bash
docker stop orion-athena-execution-dispatch-runtime orion-athena-policy-runtime orion-athena-proposal-runtime && docker rm orion-athena-execution-dispatch-runtime orion-athena-policy-runtime orion-athena-proposal-runtime
```
3. Remove the stale keys from the local `.env` files:
```bash
sed -i '/^SUBSTRATE_PROPOSAL_FRAMES_RETENTION_DAYS=/d' services/orion-sql-writer/.env
```
```bash
sed -i '/^HUB_WORLD_PULSE_READ_PULL_\(ENABLED\|DAILY_CAP\|AROUSAL_ENABLED\)=/d' services/orion-hub/.env
```
4. Delete the orphaned local `.env` files that the deleted service dirs leave behind. This needs
   Juniper's OK:
```bash
rm services/orion-proposal-runtime/.env services/orion-policy-runtime/.env services/orion-execution-dispatch-runtime/.env && rmdir services/orion-proposal-runtime services/orion-policy-runtime services/orion-execution-dispatch-runtime
```
5. In 2 or more days: the drop PR (#DROP_PR). First apply #3116 (feedback and consolidation) if it
   has not run. Then follow the drop PR's header: check that nothing has written, take the backup,
   apply, and merge in the same sitting.

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
