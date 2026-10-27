## Summary

- **Practice mode (`services/orion-retro/evals/zelda_dojo.py`).** Orion's live Zelda play made 223 finished goals in total, too few to train any world model. The dojo makes experience instead:
  - it plays headless, with the real core, reader and goal motor, and no LLM;
  - it keeps an archive of game states, restores the rarely chosen ones (0.04 ms), and runs every offered goal from each one (Go-Explore, Ecoffet et al. 2019).
- **Action-conditioned world model (`services/orion-world-model/app/transition.py`).** The service's parked `WorldModel` has no action input. The new `TransitionModel` answers "if the body does this here, what happens?":
  - inputs are body-level facts in the `orion/embodiment/skills.py` vocabulary;
  - Zelda reaches it through an adapter.
- **Gate (`scripts/train_transition.py`).** Each prediction is checked against a count baseline shaped like `action_beta_backoff`. The test uses only states training never saw, with a bootstrap over (worker, room) clusters.

## Outcome moved

- **Experience:** 16 dojo workers x 20 min gave 50,850 goals and 97 distinct places (39,736 after deduplication). Live play had 223 goals in total.
- **World model, on 1,431 held-out goals from states training never saw (98 clusters):**

| Prediction | Rate | Model NLL | Count NLL | Gain 95% CI | Verdict |
|---|---|---|---|---|---|
| place changed | 0.391 | 0.248 | 0.278 | [+0.010, +0.050] | PASS |
| died | 0.074 | 0.140 | 0.166 | [+0.007, +0.046] | PASS |
| health lost | 0.291 | 0.439 | 0.426 | [-0.035, +0.011] | FAIL |
| item gained | 0.128 | 0.245 | 0.259 | [-0.000, +0.031] | FAIL |

NLL is log loss; lower is better. The gain is count NLL minus model NLL.

## Current architecture

- `orion-world-model` is parked. It holds an untrained Transformer with no action input and no consumer.
- The Hub chooses Zelda moves from orion-prediction's count model (`action_beta_backoff`) and a hand-set value table.

## Architecture touched

- orion-retro: a new eval tool. It imports only the eval harness and retro app code.
- orion-world-model: a new module and script. They read dojo JSONL only.
- No service runtime, contract, bus or env changes.

## Files changed

- `services/orion-retro/evals/zelda_dojo.py`: the Go-Explore dojo.
- `services/orion-retro/tests/test_zelda_dojo.py`: tests for goal options and cell keys.
- `services/orion-retro/README.md`: the dojo section.
- `services/orion-world-model/app/transition.py`: the model, the features and labels, and the Zelda adapter.
- `services/orion-world-model/scripts/train_transition.py`: training and the gate.
- `services/orion-world-model/tests/test_transition.py`:
  - labels (cave exit, an item that was unread, death);
  - the feature and count-key layout;
  - baseline backoff;
  - the cluster interval;
  - model heads.
- `services/orion-world-model/README.md`: the transition model section.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
services/orion-retro/tests/test_zelda_dojo.py      -> 2 passed
services/orion-world-model tests (in its image)    -> 68 passed
```

## Evals run

```text
dojo run1 (before review fixes): 16 x 20 min, 77,778 goals; first gate PASS 3/4 -- invalid (see review H1-H3)
dojo run2 (after fixes):         16 x 20 min, 50,850 goals, 97 places; gate on unseen states: PASS place_changed, died
training + gate: 44 s on 4 CPUs (torch thread cap)
```

## Docker/build/smoke checks

```text
dojo ran in orion-retro-retro (read-only repo + catalog mounts); training in orion-world-model-world-model on CPU
```

## Review findings fixed

- **H1 (high):** a restore replayed identical rows, so the bootstrap counted copies as evidence.
  - Fix: a random 0-60 frame idle after each restore, plus deduplication on (before, goal).
  - Evidence: smoke run, 62 unique of 74 rows.
- **H2 (high):** held-out workers shared states with training. 86% of held-out rows had been seen.
  - Fix: the gate uses only test rows from unseen states, and the report gives the share.
  - Evidence: `held_out_share_seen_in_train: 0.858`.
- **H3 (high):** the bootstrap was per row.
  - Fix: a bootstrap over (seed, room) clusters.
  - Evidence: `test_cluster_interval_is_wider_than_one_cluster_pretending_to_be_many`.
- **M1:** the atlas carried the last frame across restores.
  - Fix: `on_load` on the same lineage after each restore. `open_sides` and `before` now come from the offer the goal committed against.
- **M2:** a scroll or stairs ending counted as a death.
  - Fix: `died` = outcome `death` or after-mode death/menu.
- **M3:** timeout rows were kept.
  - Fix: `dojo:*` rows have no `after` and are dropped from training.
- **M4:** a goal that Retro refused crashed the worker.
  - Fix: the refusal is recorded as `dojo:rejected:*`.
- **M5:** a cave exit was labelled "same place".
  - Fix: the place includes mode, and cave and dungeon features were added.
- **M6:** the baseline lacked health and backoff.
  - Fix: the baseline is skill x side x threat x health, backing off to skill, then to the global rate.
  - Effect: the first run's "health lost PASS" disappeared, which was expected.
- **L1:** an item going from `unknown` to a number counted as a gain.
  - Fix: items compare only when both readings are numbers.
- **L2:** rupees split the archive into new cells.
  - Fix: cells use the set of items held.
- **L3:** duration included time spent waiting for the offer.
  - Fix: duration comes from the goal's `start_frame`/`end_frame`.
- **L4:** rows piled up.
  - Fix: rows are cleared on rewind.
- **L5:** the skill list was copied.
  - Fix: it is imported from `orion.embodiment.skills`.
- READMEs and tests were added.

## Restart required

```text
No restart required (offline tools; nothing deployed reads them).
```

## Risks / concerns

- Severity: medium
  - Concern: no live path uses the model yet. Health and item predictions do not beat counts.
  - Mitigation: next PR, the model runs in orion-prediction next to `action_beta_backoff`, scored on live bets, before it can steer.
- Severity: medium
  - Concern: dojo runs load the host CPUs (16 cores for 20 min). Heat is not checked.
  - Mitigation: manual runs only. A nightly job needs Juniper's OK and a heat gate.
- Severity: low
  - Concern: every worker starts from one fixed opening, so coverage is near the start (97 places).
  - Mitigation: start from saved live states, and add enter/cave options.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
