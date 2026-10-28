## Summary

**Merge only after #2938 is deployed.** #2938 makes orion-prediction ignore columns it does not know. This PR's migration adds a column before the service rebuilds.

- The trained world model (`TransitionModel`, from the dojo) now runs as a never-served shadow in orion-prediction's action bets. Every Zelda bet gets the world model's chance next to the served chance, and `shadow_verdict` grades the two on the same settled bets, with the same per-session paired test as the existing gate.
- One shared contract, `orion/embodiment/transition.py`, is used by every side:
  - the dojo writes facts with it;
  - the trainer labels and featurizes with it;
  - the Hub sends features with it;
  - orion-prediction runs the exported weights with it, in numpy.
- The Hub's settle rules (`start_facts`, `place_changed`, `gained`) now live in that module, so a world-model label means exactly what a bet settles on.
- The trainer exports numpy weights. `--publish` replaces the served file atomically, and `--keep` samples a multi-million-row dojo run.

## Outcome moved

- **Before:** the world model could not affect or be judged on live play.
- **After:** every traverse, acquire and hold bet carries its shadow p, and the verdict will say better, worse or unclear after 20 sessions.
- Published weights (`zelda-body-v1:581932rows:20261028T221311Z`): 774,062 deduplicated dojo goals. On 7,695 held-out goals from unseen states (3,468 clusters), against counts:
  - PASS: item gained, died;
  - FAIL: place changed, health lost (95% intervals cross 0 barely).

## Current architecture

Action bets served `action_beta_backoff` (counts) against the `action_skill_rate` baseline. The world model was offline only.

## Architecture touched

- Contract: `ActionBetPlaceV1` gains optional `features` and `feature_version`; `ActionBetV1` gains `shadow`.
- orion-prediction: `WorldModelShadow`, `shadow_verdict`, a new column, settings, and a compose read-only mount.
- Hub: `transition_features` on place; the settle rules are imported from the shared module.
- World model: the shared module, the export, and numpy in requirements.
- Dojo: facts come from the shared module.

## Files changed

- `orion/embodiment/transition.py` (new): features, labels, settle rules, `success_p`, `TransitionWeights` (lazy numpy).
- `orion/schemas/prediction.py`: the optional fields above.
- `services/orion-prediction/app/action_bets.py`: `WorldModelShadow` (reloads on file change; a missing or bad file means no shadow), the shadow on place, `shadow_verdict`, and the status in `/health` and `/live`.
- `services/orion-prediction/app/store.py`: the `shadow` column (SQL NULL when absent), the boot check, and the settled query.
- `services/orion-prediction/app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the wiring, the keys and the mount.
- `services/orion-sql-db/manual_migration_prediction_action_bet_shadow_v1.sql`: a nullable column (additive, auto-applied).
- `services/orion-hub/scripts/retro_experience.py`: features on place; the settle rules come from the shared module.
- `services/orion-world-model/app/transition.py`, `scripts/train_transition.py`, `requirements.txt`, `README.md`: the shared contract, the export, `--publish` and `--keep`, and numpy.
- `services/orion-retro/evals/zelda_dojo.py`: facts from the shared module.
- Tests:
  - prediction: shadow scoring, version mismatch, missing or bad file, reload, verdict, `success_p`;
  - Hub: a bet carries features;
  - world model: numpy matches torch, and the labels follow the settle rules;
  - retro: imports without numpy.

## Schema / bus / API changes

- Added: `ActionBetPlaceV1.features`, `ActionBetPlaceV1.feature_version`, `ActionBetV1.shadow`; `/health` `action_bets.world_model_shadow`; `/action-bets/live` `shadow_verdict` and `world_model_shadow`.
- Compatibility notes:
  - The fields are optional.
  - A new Hub sending to an old prediction gets a 422 on place (the bet is not placed) until prediction rebuilds, which is seconds in the same pull.
  - An old Hub ignores `shadow`.
  - After #2938, a migration-first apply cannot break the old prediction.

## Env/config changes

- Added keys: `PREDICTION_WORLD_MODEL_SHADOW_ENABLED=true`, `PREDICTION_WORLD_MODEL_WEIGHTS=/models/transition.npz`, and `PREDICTION_WORLD_MODEL_DIR=/mnt/storage-warm/orion-dojo/models` (compose-only host path).
- `.env_example` updated: yes. Local `.env`: run `python scripts/sync_local_env_from_example.py` after merge (compose defaults cover it until then).

## Tests run

```text
orion-prediction tests/test_action_bets.py tests/test_schemas.py -> 28 passed
  (8 unrelated failures elsewhere in the suite locally: missing forecasting libs; same on main; CI installs them)
orion-hub tests/retro                                            -> 194 passed (settle-table tests unchanged)
orion-world-model tests (in its image)                           -> 70 passed (numpy == torch within 1e-5)
tests/retro + orion-retro tests                                  -> 164 passed
Mutations: top-level numpy import -> no-numpy test fails
```

## Evals run

```text
train_transition.py on the live dojo (7.4M rows, keep 0.15, 8 epochs, 8 CPUs, held-out workers, unseen-state gate):
  place_changed FAIL [-0.0012, 0.0052]   health_lost FAIL [-0.0001, 0.0086]
  item_gained   PASS [+0.0162, 0.0261]   died        PASS [+0.0021, 0.0096]
```

## Docker/build/smoke checks

```text
Training and world-model tests ran in orion-world-model-world-model. Prediction is not yet rebuilt (post-pull).
```

## Review findings fixed

- Finding (high): the dojo would crash. The live workers import the shared module in the retro image, which has no numpy.
  - Fix: numpy is imported only to run weights.
  - Evidence: the retro test imports with numpy blocked; the mutation fails.
- Finding (high): a migration-first deploy breaks reads in the old prediction service.
  - Fix: #2938, shipped first, plus `shadow` in the boot check.
- Finding (medium): labels did not match the settle rules. A death read as "place changed", and a heart pickup was not a gain.
  - Fix: the shared `start_facts`, `place_changed` and `gained` are used by the Hub and the labels. Hold is `1 - health_lost`; enter has no shadow.
  - Evidence: `test_labels_follow_the_hubs_settle_rules`; the Hub's 194 settle tests still pass.
- Finding (low): the comparator was the counts model, not the served p.
  - Fix: `shadow_verdict` uses each bet's stored `p`.
- Finding (low): a missing shadow was stored as jsonb `null`.
  - Fix: it is now SQL NULL.

## Restart required

```bash
git pull   # applies the migration, rebuilds orion-prediction + orion-hub (+ gateway)
python scripts/sync_local_env_from_example.py
```

## Risks / concerns

- Severity: medium
  - Concern: two of four heads do not beat counts on unseen states. The unseen-state test slice is unusual (96% of held-out states were seen in training).
  - Mitigation: shadow only. The live verdict decides, and nothing steers on it.
- Severity: low
  - Concern: the shadow_verdict needs 20 sessions. Fights, retreats and caves get no shadow.
  - Mitigation: stated in `/live`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2939

🤖 Generated with [Claude Code](https://claude.com/claude-code)
