# Retire the feedback rung (L10) and the consolidation rung (L11)

Date: 2026-11-04. Status: proposal (docs only). It is the first demolition step in
`2026-11-04-attention-as-want-competition-design.md` (PR #3089), "The ladder: revive or kill?".
Juniper decided on 11-04 that consolidation is killed outright, with its own proposal first. This
is that proposal, together with feedback, because consolidation's main input is feedback.

Every number below was read live on 2026-11-04 between 06:00 and 06:30 UTC.

## In one paragraph

The feedback rung grades what the ladder's dispatch rung did. It writes about 43,000 frames a
day, and the table holds 9.7 GB with no retention. Almost all of that work grades nothing: the
dispatches it grades are either empty or LLM boilerplate. The act ledger already grades every
real act Orion does: 2,506 outcome rows in 14 days, against feedback's 390. The consolidation rung
learns patterns from those feedback frames and writes 23 frames a day. Its one live reader is the
reverie, which attaches frame ids as references. Both rungs can go. **But they are wired into
three things outside the ladder.** If any of the three is missed, a silent failure follows.
They are a retention floor in sql-writer, the dispatch rung's learning, and Spark's intake. Each
one gets a fix in the same PR.

## Arsonist summary

- **The dangerous one: sql-writer's retention floor.**
  - Where: `services/orion-sql-writer/app/grammar_truth.py:_substrate_chain_floor`.
  - What it does: it refuses to prune any proposal frame that a later stage still owes work on.
    It reads the dispatch frames' `feedback_pending` marker.
  - How it breaks: the marker is a column default (`TRUE`) on every new dispatch frame, and only
    the feedback runtime clears it. Stop feedback alone, and the oldest marker pins the floor forever.
  - What follows: `substrate_proposal_frames` (2.3 GB, 10-day retention) stops pruning and grows
    by about 40k rows a day. Nothing errors. The code's own comment names this hazard ("A LEAKED
    MARKER pins retention forever").
- **Dispatch learns from feedback.**
  - What dispatch reads: the dispatch store (`load_effect_posteriors`, `store.py:1325`) reads
    `substrate_action_effect_posterior` to choose among its own action kinds.
  - Who writes it for those kinds: only feedback, for 63 of the 86 cells (maintain 25, inspect 24,
    summarize 9, self_regulate 5).
  - What happens: once feedback stops, those cells freeze. The act-ledger kinds (`act:*`, 23
    cells) are written by the act ledger itself and are unaffected.
  - Verdict: acceptable, because dispatch retires next. But it is a real change in dispatch's
    behavior and must be named.
- **The dispatches being graded are not worth grading.** In 14 days:
  - summarize: 516 LLM calls that return the same boilerplate ("The target is in a stable state
    with low risk and moderate priority…");
  - express: 2,020 "succeeded" with `raw_len` 0;
  - maintain: 155;
  - inspect: 37.

  None of these opens a grammar act (0 of 390 outcome rows join to `grammar_acts`). This is the
  case for retiring dispatch next. It is not part of this proposal.
- **Spark turns every feedback frame into two empty artifacts.** `orion-spark-concept-induction`
  subscribes to `orion:feedback:frame`. For each frame it publishes an identity snapshot and a debug
  dossier, then skips concept induction ("carry no text"). Those are bus-only (nothing stores
  them), so about 86k empty artifacts a day simply stop.

## Current architecture

```text
proposal (L7) -> policy (L8) -> dispatch (L9) -> feedback (L10) -> consolidation (L11)
                                    |                |
                                    |                +-> substrate_action_outcomes (ladder kinds only)
                                    |                +-> substrate_action_effect_posterior (63 cells)
                                    |                +-> substrate_signal_control_cells (control arm)
                                    |                +-> substrate_world_action_episodes scoring
                                    |                +-> attention_loop_outcome ('acted', via world episodes)
                                    +-- feedback_pending (column default TRUE); sql-writer floors proposal retention on it

act ledger (orion/grammar/act_ledger.py, sql-writer): every real act -> substrate_action_outcomes
  (act:* kinds) + posterior. Independent of the feedback runtime (it only mentions it in comments).
```

| Output of the feedback runtime | Live volume | Who reads it |
|---|---|---|
| `substrate_feedback_frames` | 42,881 / day; 3.08M rows, 9.7 GB, no retention | consolidation; Hub (feedback route, lattice page); liveness (`substrate_ladder_liveness`, `metrics/liveness`); `agency_episode_reader` (a report); analysis scripts; smoke scripts |
| bus `orion:feedback:frame` | same | Spark concept induction (empty artifacts, see above) |
| `substrate_action_outcomes` (ladder kinds) | 390 / 14 d | the dispatch posterior only; `report_action_value.py` |
| `substrate_action_effect_posterior` (63 cells) | updated on each graded dispatch | **dispatch's choice of action** (`store.py:1325`) |
| `substrate_signal_control_cells` | 71 rows | `report_action_value.py` (analysis) |
| `substrate_feedback_control_reductions` | 0 / day | none |
| world-episode scoring | 4 episodes in 14 d, last 10-25 | rung-3 shed (`self_shed`), dead since 10-25 |
| clears `feedback_pending` on dispatch frames | each frame | **sql-writer's retention floor** |

| Output of the consolidation runtime | Live volume | Who reads it |
|---|---|---|
| `substrate_consolidation_frames` | 23 / day; 98 MB | **reverie grounding** (`services/orion-thought/app/grounding.py`, flag `reverie_ground_consolidation`: attaches frame ids as `motif_refs`); Hub consolidation routes and lattice page; liveness |
| `substrate_tensor_slices` | 46 / day | Hub (`/tensor-slices/latest`) |
| `substrate_expectations`, `substrate_schema_candidates` | 6 and 2 rows total | Hub routes; `consolidation_frame.py` schema |

## The plan: one PR per rung, feedback first

### PR 1: feedback

The kill:
1. Stop the service. Remove `orion-athena-feedback-runtime` from the deploy and from the post-pull
   rebuild lists, and delete `services/orion-feedback-runtime/`.
2. Remove the floor probe. Delete the `"dispatch->feedback"` probe from
   `_SUBSTRATE_CHAIN_PENDING`, so the floor never reads a marker nobody clears. That one change is
   enough. The marker is a column default (`feedback_pending DEFAULT TRUE`), not dispatch code, so
   dispatch needs no change and no migration. The column goes dead and is dropped later with the
   tables.

   A test gives the floor a dispatch frame with `feedback_pending = true` and no feedback runtime.
   It must not pin the floor. Also update
   `services/orion-sql-writer/evals/test_substrate_retention_integrity_eval.py`, which probes the
   marker.
3. Old markers. Rows already marked `true` are harmless once the probe is gone. They are left as
   they are: no production write.

Readers, each cut:

4. **Spark.** Remove `orion:feedback:frame` from its intake list. It appears in four places:
   - `orion/spark/concept_induction/settings.py`;
   - the compose default;
   - `.env_example`;
   - the local `.env`, synced with `scripts/sync_local_env_from_example.py`.

   Also remove the `feedback.frame.v1` branches in `bus_worker.py`.
5. **Hub.** Delete `substrate_feedback_routes.py` and its router include. The lattice page shows
   L10 as "retired": the row stays visible with a plain label. It does not show as "stale" or
   "down".
6. **Liveness.** Remove the feedback rung from `orion/substrate_ladder_liveness.py` and
   `orion/metrics/liveness.py`, so `make substrate-ladder-check` stays green for the right reason.
   That reason is "retired", not "not checked".
7. **Contracts.** Remove these, then run `check_inner_state_registry.py` and the metric lock gate:
   - the `orion:feedback:frame` channel in `orion/bus/channels.yaml`;
   - the `FeedbackFrameV1` registry entry;
   - the `metric_definitions.lock.json` entries;
   - the `config/physiology/contract.yaml` module lines.
8. **Package.** Delete `orion/feedback/` and `orion/schemas/feedback_frame.py`, apart from one
   piece. `orion/field/credit_integrity.py` (report-only, used by three scripts) reads
   `config/feedback/feedback_policy.v1.yaml`. Move that read into `credit_integrity.py`, or delete
   it with its scripts. PR 1 decides once it has read those three scripts.
9. **Evals, analysis, smoke, CI.** Delete or rewrite:
    - `services/orion-proposal-runtime/evals/test_baseline_replay.py`;
    - `orion/autonomy/evals/eval_action_value_contrast.py` (it imports one constant);
    - `scripts/smoke_feedback_frame_v1.sh`;
    - `measure_proposal_feedback_correlation.py`;
    - the feedback steps in `.github/workflows/attend-act-loop.yml` and `visual-baseline.yml`;
    - the `FEEDBACK_*` prefixes in `sync_local_env_from_example.py`.

    `agency_episode_reader.py` and `report_action_value.py` keep working on history. A missing
    new feedback frame reads as "none since 11-xx".
10. **Dispatch's frozen learning.** This is named, not fixed. Dispatch keeps its last posteriors
    for its four kinds. It is the next rung to retire.
11. **World episodes and control cells.** Their only writer goes. The rung-3 shed that creates
    world episodes has been dead since 10-25. It retires with dispatch / rung-3 binding, and its
    unscored rows stay as history.

### PR 2: consolidation

1. Stop and delete `orion-athena-consolidation-runtime`, `services/orion-consolidation-runtime/`,
   `orion/consolidation/` and `orion/schemas/consolidation_frame.py`.
2. **Reverie.** Turn off `reverie_ground_consolidation` and delete the motif loader. Reverie
   loses `motif_refs` (frame ids it attached). Before cutting, PR 2 checks whether anything reads
   `motif_refs` downstream of reverie. Episode refs (`episode_summary_refs`) come from a different
   loader and stay.
3. Hub: delete `substrate_consolidation_routes.py` and its router include. On the lattice page,
   L11 shows as "retired".
4. Remove consolidation from liveness, channels, the registry, the metric lock, CI and the smoke
   scripts. That includes `smoke_consolidation_v1.sh` and the step in
   `smoke_orion_bus_transport_full_stack.sh`.

### Not in either PR

- **No table drops.** The feedback frames (9.7 GB), consolidation, tensor-slice, expectation and
  schema-candidate tables stay as read-only history. Dropping them is destructive (`DROP TABLE`),
  so it is Juniper's call, as its own step.
- No change to the act ledger, the want ledger, or the allocator.
- No change to proposal, policy or dispatch code.

## Missing questions (defaults in brackets)

1. **Drop the old tables later?** [Keep them 30 days after PR 2, then a separate drop PR with a
   `pg_dump` of each table to `/mnt/storage-warm` first.]
2. **Dispatch next?** The audit found that dispatch's LLM work (516 summarize calls in 14 days)
   produces boilerplate. [Yes: dispatch is the next proposal, ahead of proposal/policy.]
3. **Hub lattice page:** show retired rungs as "retired", or drop the rows? [Show "retired" until
   the whole ladder is gone, so the page tells the story.]

## Files likely to touch

PR 1:
- **Feedback rung itself:** `services/orion-feedback-runtime/` (deleted), `orion/feedback/`,
  `orion/schemas/feedback_frame.py`.
- **Retention floor:** `services/orion-sql-writer/app/grammar_truth.py` and its retention eval.
- **Spark:** `orion/spark/concept_induction/{settings,bus_worker}.py`,
  `services/orion-spark-concept-induction/{docker-compose.yml,.env_example}`.
- **Hub:** `services/orion-hub/scripts/{substrate_feedback_routes,substrate_lattice_routes,api_routes}.py`.
- **Liveness:** `orion/substrate_ladder_liveness.py`, `orion/metrics/liveness.py`.
- **Contracts:** `orion/bus/channels.yaml`, `orion/schemas/registry.py`,
  `config/metrics/metric_definitions.lock.json`, `config/physiology/contract.yaml`,
  `config/feedback/`.
- **Report module:** `orion/field/credit_integrity.py`.
- **Scripts and CI:** the scripts and workflows named above, plus the rebuild/deploy service
  lists.

PR 2: `services/orion-consolidation-runtime/`, `orion/consolidation/`,
`orion/schemas/consolidation_frame.py`, `services/orion-thought/app/{grounding,reverie}.py`, and
the Hub consolidation and lattice routes. It also touches the same liveness, contract, CI and
smoke files as PR 1.

## Proposal-mode checklist

- **Capability that changes:**
  - The ladder no longer grades its own dispatches.
  - Dispatch stops learning for four action kinds.
  - Reverie stops attaching consolidation frame ids.
  - Orion keeps grading every real act through the act ledger.
- **Data touched:**
  - Two runtimes stop writing.
  - No rows are changed or deleted.
- **Privacy boundary:** none.
- **Trace that proves it worked:**
  - Feedback and consolidation frames stop growing.
  - Proposal-frame retention keeps pruning: the oldest proposal row stays about 10 days old,
    checked the day after deploy and again a week later.
  - The act ledger's outcome rows per day are unchanged (about 180 a day).
  - `make substrate-ladder-check` is green and lists L10 and L11 as retired.
  - Spark's intake rate drops by the feedback share.
- **Dangerous failure modes:**
  1. The retention floor pins, and the proposal table grows silently. Mitigation: the probe is
     removed in the same PR that stops the feedback runtime, a test proves it, and the floor's
     own WARNING log is checked after deploy.
  2. A hidden reader breaks. Mitigation: this inventory, plus a repo-wide grep gate in PR 1 for the
     table, channel and schema names, outside history and migrations.
  3. Reverie loses grounding it used. Mitigation: PR 2 checks the downstream readers of
     `motif_refs` first.
- **Disable or roll back:** revert the PR and redeploy the two services. The tables are untouched,
  so nothing is lost. Old `feedback_pending` markers would be picked up again, oldest first.

## Acceptance checks

1. A repo-wide search for `substrate_feedback_frames`, `orion:feedback:frame`, `FeedbackFrameV1`
   and `feedback_pending` finds only migrations, history docs and the sql-writer retention test.
2. Sql-writer's test: a pending marker with no feedback stage does not pin the floor.
3. Live, the day after deploy:
   - feedback frames written since deploy = 0;
   - the oldest `substrate_proposal_frames.created_at` is within 10 days + 1;
   - no `substrate_chain_floor` WARNING;
   - act-ledger outcomes per day within ±25% of the prior week.
4. `make substrate-ladder-check` is GREEN, with L10 (and, after PR 2, L11) shown as retired.
5. Spark processes no `feedback.frame.v1` envelopes (its logs, and its intake list).

## Recommended next patch

PR 1 (feedback), exactly as above. Then PR 2 (consolidation). Then a dispatch-retirement
proposal, using this audit's finding that dispatch's LLM work is boilerplate.
