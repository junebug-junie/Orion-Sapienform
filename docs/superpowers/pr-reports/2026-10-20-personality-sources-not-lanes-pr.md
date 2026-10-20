## Summary

Orion's personality graph promoted three "prediction error" themes. Each was backed by two lanes, reverie and the metacog journal, but both lanes narrate the same substrate signal. So one source was counted twice. This patch counts sources, not lanes.

- The reducer keys its per-lane hit counter (`lane_recent`) by source (`source_key`). Reverie and the metacog journal share one key, `substrate_narration`. Every other lane keeps its own key. The per-lane counts that the panel shows do not change.
- The first durable-runs tick after deploy rebuilds the stored nodes once from their evidence (`rekey_sources`, marker cursor `rekey:source_v1`). It demotes an active node to candidate when it no longer has two sources. The rebuild commits in the same transaction as that tick.
- New read-only script `scripts/analysis/measure_personality_lane_independence.py`. It measures which evidence streams fire together on the same node. Run it again when a lane or a journal kind is added.

Handoff step 1 (#2770) is also now verified live: orion-thought logs `personality_cue rendered=True` with all 4 ids at 732 characters (10-20 04:56–04:59 UTC).

## Outcome moved

Live dry run of the rebuild against today's graph (read-only; the replay reproduces the stored counters to within 0.002):

| Node | Before | After |
|---|---|---|
| Biometrics prediction error | active, 4.93 | active, 3.45. Dream is a real second source. |
| Cabinet prediction error | active, 3.57 | **candidate**, 2.14 |
| Bus synaptic prediction error | active, 2.94 | **candidate**, 1.49 |
| Self-inquiry ("seventy-fourth answer") | active, 3.49 | active, 4.76. Its 3 metacog journal hits now count as their own source. |
| Telemetry anomaly / prediction error cluster | candidate, 2.78 | candidate 3.60. It should be promoted on the next tick, from 254 metacog journal hits plus 4 non-metacog journal hits. 3 of those 4 are curiosity self-study write-ups in which Orion tests its own prediction-error priors (for example "The strain reverie, tested"). |

After the rebuild, 3 nodes are active and 2 of them are prediction-error narration. Before, it was 3 of 4.

## Current architecture

`orion/personality/graph.py` weight = sum over `lane_recent` keys of `base × min(1.5, ln(1 + recent))`. A node is promoted when at least 2 non-seed keys have recent ≥ 1.5. Before this patch the keys were lane names. The durable-runs `PersonalityDriver` is the only writer. It loads non-retired nodes, folds evidence, and commits each tick in one transaction.

## Architecture touched

- Reducer: `orion/personality/graph.py` (`source_key`, `SUBSTRATE_NARRATION`, `rekey_sources`, `REKEY_LANES`).
- Driver/store: `services/orion-durable-runs/app/personality_{driver,store,sources}.py`. Adds the one-shot rekey, `rekey_hits` (a read of `personality_evidence`) and the `REKEY_CURSOR` marker.
- No schema, bus, frame or API change. `lane_recent` is internal: only the store reads it.

## Metric gate (the new source key)

1. **Provenance.** Reverie interpretations come from `substrate_reverie_thought`. Chains start from the substrate broadcast coalition (`orion/reverie/attention_schema.py`). Metacog journal entries come from `MetacogTriggerV1`. That trigger is opened by `services/orion-equilibrium-service/app/self_signal_episode_gate.py`, which reads `substrate_node_prediction_error_history` through the same attention eligibility rule (`world_first.node_candidate`/`judge_candidate`).
2. **Independence.** They are not independent. That is the finding. Both read the same prediction-error history.
3. **Theory anchor.** Promotion means that "more than one independent process returns to this theme", which is the spec's CAPS cross-situational consistency. Two narrations of one signal are one observation.
4. **Live data.** Across the 3 prediction-error nodes, 16 of 27 metacog hits had a reverie hit on the same node within 10 minutes. At a +6 h or −3 h shift, it was 1 of 27. On the self-inquiry node it was 0 of 3. The script over all nodes gives lift 9.0 and 7.8 for the pair, and 1.0 or less for every other pair except self-study journal ↔ self-inquiry seed (lift 5.0; see Risks).
5. **Existing mechanism.** The handoff named this risk in #2738. No independence handling existed.
6. **Reversibility.** It is cheap to reverse. Remove the `source_key` mapping and delete cursor `rekey:source_v1`, and a rekey with the identity mapping would rebuild lane keys. No schema change.

## Files changed

- `orion/personality/graph.py`: source keys, one-time rekey, docstring with the measurement.
- `orion/personality/tests/test_graph.py`: source key and rekey tests. Two flood tests now expect the `substrate_narration` key for reverie.
- `services/orion-durable-runs/app/personality_driver.py`: runs the rekey once, before seed and fold, and merges it into the tick's write.
- `services/orion-durable-runs/app/personality_store.py`: `rekey_hits`.
- `services/orion-durable-runs/app/personality_sources.py`: `REKEY_CURSOR`.
- `services/orion-durable-runs/tests/test_personality_loop.py`, `test_personality_postgres.py`: driver rekey test, real-Postgres `rekey_hits` test, cursor set, metacog key.
- `scripts/analysis/measure_personality_lane_independence.py` + test, `.github/workflows/personality-tests.yml`: co-occurrence measurement in CI.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `personality_disposition.lane_recent` keys are sources (`substrate_narration` replaces `reverie`, and replaces `journal` for metacog hits). One new row in `personality_cursor`: `rekey:source_v1`.
- Compatibility notes: readers (thought, Hub) use the frame, which carries `lane_counts` and not `lane_recent`. No consumer-first deploy is needed.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/personality/tests scripts/analysis/tests -q                         -> 592 passed
services/orion-durable-runs: pytest tests -q (PYTHONPATH=../..)                  -> 452 passed, 91 skipped (personality loop 34; real-Postgres 4 run separately)
test_personality_postgres.py on throwaway pg16 (ORION_ADMISSION_TEST_DSN)        -> 4 passed
git diff --check                                                                  -> clean
```

## Evals run

```text
Read-only live dry run of rekey_sources on 92 non-retired nodes -> 32 rekeyed, 2 demoted (table above);
  fidelity: replay with the old lane keys reproduces 38 stored counters, max abs error 0.002
measure_personality_lane_independence.py (live, 964 rows) -> reverie/metacog lift 9.0 and 7.8; table in PR
```

## Docker/build/smoke checks

```text
UNVERIFIED until deploy: after the rebuild, durable-runs logs one `personality_rekey nodes=… demoted=[…]` line
(expected: the cabinet and bus-synaptic node ids), and
`select status,count(*) from personality_disposition group by 1` shows active 3 (then 4 once the
telemetry cluster promotes).
```

## Review findings fixed

Code review ran in a subagent. It found no correctness bugs. It checked: rekey then fold in the same tick, merge precedence, the history primary key, the light vs full upsert path, idempotence, and the first boot. Its own dry run matched ours: 32 nodes changed and the same 2 were demoted.

- Finding: if a demoted node was last reinforced more than 7 days ago, the candidate time limit would retire it permanently in the same tick.
  - Fix: a demotion restarts the candidate window (`last_folded_at = now`).
  - Evidence: `test_rekey_demotion_restarts_the_candidate_ttl_instead_of_retiring`.
- Finding: no test covered splitting a mixed metacog + self-study journal key, which is what promotes the telemetry cluster.
  - Fix: added `test_rekey_splits_a_mixed_journal_key_into_its_sources`.
  - Evidence: the test passes, and the Outcome table names the promotion.
- Finding: are `notify` and `self_study` journals independent of metacog?
  - Fix: none needed.
  - Evidence: the measurement shows notify ↔ metacog at 0 near hits and self_study ↔ metacog at lift 0.5–1.0.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

(Run it from the clean primary `main` checkout after merge.)

## Risks / concerns

- Severity: low. Concern: demoting 2 active nodes changes what the cue shows to chat, reverie, curiosity and outreach. Mitigation: that is the intended fix, and the candidates keep their evidence. They are promoted again if an independent source returns to them.
- Severity: low. Concern: self-study journal ↔ self-inquiry seed fire together (lift 5.0), because a curiosity run writes both its concept and its write-up. Mitigation: seeds never count toward promotion and weigh 0.5. Re-run the script if that changes.
- Severity: low. Concern: the replay fades between evidence times, but the live reducer fades at tick times, so hits in one tick are not faded against each other. Mitigation: the replay matches the stored values to within 0.002.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2778

🤖 Generated with [Claude Code](https://claude.com/claude-code)
