## Summary

- Every durable run Orion starts for itself now writes an act to `grammar_acts`. The act opens when the run first gets its GPU pool grant, and it closes when the run ends.
- Four families are covered. Curiosity investigate maps to `infer`, self-inquiry to `ask`, self-sense eval to `sense`, and journal compose to `make`.
- The outcome comes from the run status: `completed` = 1, `failed` / `cancelled` = 0. The producer sends only the wall seconds (ActCostV1). The ledger fills GPU slot-seconds itself.
- Opening happens exactly once per run. Only the call that inserted the run's first grant row publishes the open, so a restart, a resume or a re-grant after a lost hold does not open it again.
- Flag `ORION_DURABLE_ACT_EMIT_ENABLED` ships ON.

## Outcome moved

Before this patch, these four want families had zero graded acts, so the body economy could not grade or price them. A live replay of the last 7 days (read-only) gives these numbers:

The replay applies the ledger's own cost window. A lease counts as joined only if its release lands in [first grant - 3600 s, terminal + 120 s] and its row was already written 5 s after the terminal.

| family | actor / verb | runs | acts opened | closed 1 / 0 | still open | acts/day | median s | slot-h the ledger would fill | closes missing a lease |
|---|---|---|---|---|---|---|---|---|---|
| curiosity investigate | `durable.curiosity_investigate` / infer | 13 | 13 | 12 / 0 | 1 | 1.86 | 3494 | 8.48 | 1 (2.2 h, Door-A) |
| self-inquiry | `durable.self_inquiry` / ask | 16 | 15 | 14 / 0 | 1 | 2.14 | 1466 | 8.38 | 0 |
| self-sense eval | `durable.self_sense_eval` / sense | 49 | 48 | 42 / 5 | 1 | 6.86 | 2059 | 40.77 | 0 |
| journal compose | `durable.journal_compose` / make | 9 | 9 | 9 / 0 | 0 | 1.29 | 10 | 0.02 | 0 |

That is about 12 acts/day. Self-sense eval reaches the 30-graded-close cutover bar in about 5 days. The other three families need about 2 to 3 weeks.

Runs that never got a grant open nothing: 1 investigate run and 1 self-inquiry run.

## Current architecture

- Only Hub (chat reply, outreach, reading, retro) and orion-thought (painting) produced acts. The dream hops act came from another producer.
- Durable-runs drives admitted runs on GPU pool holds (`app/admission_runtime.py`). It records `run.resource_granted` in `lease()` and projects the terminal state in `_terminal()`. None of this touched the act ledger.

## Architecture touched

- `services/orion-durable-runs`: a new producer module, wired at the two existing seams (the first grant and the terminal).
- Contract: `orion:grammar:event` gets a new producer identity, `orion-durable-runs`. The grammar schema is unchanged. Act-ledger semantics are unchanged.

## Files changed

- `services/orion-durable-runs/app/run_acts.py`: new. Holds the family map, measured horizons, event builders (the existing `orion/grammar/acts.py` helpers) and `RunActEmitter`, which is best-effort and never raises.
- `services/orion-durable-runs/app/admission_runtime.py`: calls the emitter on a grant and on a terminal.
- `orion/durable_runs/registry_store.py`: adds `first_event()`, which is additive and returns the entry id and time of the first grant.
- `orion/grammar/acts.py`: adds an `orion-durable-runs` provenance branch. This is additive and is required by the producer catalog gate.
- `orion/bus/channels.yaml`: adds `orion-durable-runs` to the `orion:grammar:event` producer_services list.
- `orion/schemas/grammar.py`: comment only (the verb producer list).
- `config/metrics/metric_definitions.lock.json`: re-locked, because the channel's producer set changed (definition drift gate).
- `services/orion-durable-runs/{settings.py,.env_example,docker-compose.yml,README.md}`: the flag and the docs.
- `services/orion-durable-runs/tests/test_run_acts.py`: unit tests and real-Postgres + real-pool tests.
- `services/orion-durable-runs/evals/run_acts_live_replay.py`: a read-only live replay.

## Schema / bus / API changes

- Added: producer `orion-durable-runs` on `orion:grammar:event` (act_opened / act_closed, `GrammarEventV1`). New actors are `durable.curiosity_investigate`, `durable.self_inquiry`, `durable.self_sense_eval` and `durable.journal_compose`.
- Removed / renamed: none.
- Behavior changed: these runs now produce act rows and L10 grades (posterior key `act:<verb>`, actor, `artifact_produced`).
- Compatibility notes: the cost is V1 only (`seconds`), so readers running older code still parse it. No ActCostV2 field crosses the bus.

## Design decisions (and where the brief conflicted)

- **`expect` is set on every act** (`artifact_produced` by a horizon). The brief says to leave it unset when there is no gradeable claim. However, it also says that no act may stay open forever, and the ledger expiry only closes acts that have a horizon. In addition, `ActV1` refuses a `deliberate` act that has no expect. So the claim is the one every artifact act already makes: "this run produces its result by the deadline it gives itself." A curiosity prior cannot be expressed. `ActSignal` is a closed set {`juniper_replied`, `artifact_produced`}, and curiosity briefs carry no structured prior: the prior is free text in the prompt. So nothing extra was invented.
- **Horizons were measured, not guessed.** They are based on first grant -> terminal over 30 days of non-urgent completed runs. Investigate: p99 55648 s, max 74107 s, giving a 24 h horizon. Self-inquiry: p99 49700 s, max 52130 s, giving 24 h. Self-sense: p99 14843 s, max 25905 s, giving 12 h. Journal: max 27 s, giving 1 h, or the run's own deadline + 600 s when that is later. The horizon is sized to the slowest run because a success reported after the horizon grades 0.
- **Open at the first grant, not at submit.** This is the same rule the reading act uses: the GPU queue wait (p90 up to about 10 h) is the scheduler's time, not the act's.
- **Included:** the four families above. **Excluded:** urgent curiosity incident runs (a sensor alarm forced them, so they are a reflex, not a want). Also excluded: reading.turn and reverie.visual (Hub and thought already emit acts for them), dream.carry (the dream producer covers it), and the upkeep workflows memory.episode_distill, curiosity.compaction, compactor.digest, orion_day.letter and self_study.reflect. temporal_self, physiology, personality and situation are timer threads, not durable runs. The legacy non-admitted `runner.start_run` path is also excluded: it had 0 runs in 7 days, and every live run is admitted.

## Correlation join

- The act's `correlation_id` is the run's request `correlation_id`. This is exactly the `turn_correlation_id` on the run's pool **hold** lease. I checked it live on one run per family, and over 7 days 82 of 83 closed acts join their hold inside the ledger's window. No key fix was needed.
- **Ordering.** The hold is released by the graph node that ends the run, before `_terminal` publishes the close. Live data shows 83 of 84 hold-release rows were already in Postgres before the run's terminal row was written. So the sql-writer cost fill finds them.
- **Known gap: Door-A outreach.** A completed curiosity run can keep its hold for Hub's outreach composition, and that hold is released minutes later. The ledger fills cost once at close, with a 120 s grace, so that hold time is not charged. Live this happened to 1 act in 7 days (2.2 slot-h). Fixing it needs a ledger-side late cost refill. That is a change to `act_ledger` semantics, so it is out of scope here and is the proposed follow-up.
- Child calls inside a curiosity harness turn (cortex-exec, orion-mind) carry the turn's own fenced id (`app/graph.py turn_correlation_id`), so they are not joined. Over 7 days this is about 2.0 slot-h, compared with about 66 slot-h on the holds. The children on the hold's own lane run inside the hold's slot, so leaving them out avoids a double count. The off-lane children (metacog about 0.1 h, chat about 0.3 h) are a small known undercount.

## Env/config changes

- Added keys: `ORION_DURABLE_ACT_EMIT_ENABLED=true` (orion-durable-runs).
- `.env_example` updated: yes. The compose entry and settings default are also `true`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-durable-runs`: yes (`+ORION_DURABLE_ACT_EMIT_ENABLED='true'`).
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-durable-runs && pytest tests (real Postgres 16 + in-process pool) -> 611 passed, 1 skipped
  of which tests/test_run_acts.py -> 32 passed (open-once/close-once for completed, failed, cancelled;
  restart mid-turn does not reopen; lost-hold re-grant does not reopen; refused-before-grant emits nothing;
  flag off / non-want emit nothing; act correlation_id == hold lease turn_correlation_id)
tests/test_grammar_event_producer_catalog.py tests/test_act_ledger_contract.py: 21 passed
```

## Evals run

```text
python services/orion-durable-runs/evals/run_acts_live_replay.py --dsn <live, read-only> --days 7  -> exit 0 (table above)
```

## Docker/build/smoke checks

```text
make check-body-economy-contract                     -> OK (66 state-reading modules, all registered)
scripts/check_env_template_parity.py                 -> PASS (95 services)
scripts/check_service_env_compose_parity.py orion-durable-runs -> OK (102 keys)
scripts/check_env_key_single_source.py               -> OK
scripts/check_definition_drift.py --gate             -> PASS (after re-lock)
scripts/check_metric_lineage.py --gate               -> PASS
scripts/check_compose_no_relative_mounts.py / check_service_hostname_refs.py -> PASS
```

No Docker build was run. The image already copies `orion/`, and no dependency changed.

## Review findings fixed

- Finding: the GPU cost join could race the hold's release row, and the eval joined over the whole day, so it overstated what the ledger fills.
  - Fix: the eval now applies the ledger's exact window plus the row-insert time. It reports `closes missing a lease` and fails only when fewer than 80% of a family's closes join (a broken key). The Door-A gap is documented above as a follow-up.
  - Evidence: live, 83 of 84 hold releases were persisted before the terminal row. The replay exits 0 and shows 1 Door-A miss.
- Finding: a later step in `_terminal` raising, or a crash after `finish_projection`, could lose the close. A completed run would then expire at 0.
  - Fix: the close is now published directly after `finish_projection` returns, before any step that can raise. `on_terminal` never raises.
  - Evidence: the full service suite passes (611 passed).
- Finding (my own, from the full suite): some tests build `AdmissionRuntime` without `__init__`, which raised `AttributeError: acts`.
  - Fix: both call sites now use `getattr(self, "acts", None)`.
  - Evidence: 15 failures went to 0.
- Not fixed (low): the crash window between inserting the first grant and publishing the open. The close-before-open path still records and grades the act. This is listed under Risks.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run this from the clean primary `main` checkout after merge. No migration is needed. orion-sql-writer already persists act events from any producer.

## Risks / concerns

- Severity: low. Concern: if the process crashes between inserting the first grant row and publishing the open, the open is lost. Mitigation: the close then creates the row (the ledger's close-before-open path), so the act is still graded, but it has no `opened_at` and only a partial GPU cost window.
- Severity: low. Concern: if the process crashes in the small window after `finish_projection` but before the close is published, the act is expired at 0 at its horizon, even if the run completed. The designed backstop catches it, and a reconcile-side re-send is a possible follow-up.
- Severity: low. Concern: a Door-A outreach hold is not charged to the act (1 act / 2.2 slot-h in 7 days). Follow-up: a ledger-side late cost refill.
- Severity: low. Concern: the outcome is status-based, so a `completed` run with thin content still grades 1. Finer grading (for example, did a self-inquiry write a `:SelfDefinition`) is a follow-up once a second signal exists.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2887

🤖 Generated with [Claude Code](https://claude.com/claude-code)
