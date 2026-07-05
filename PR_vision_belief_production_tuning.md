# PR: fix(vision-window): tighten belief debounce to stop package ping-pong

**Branch:** `fix/vision-belief-production-tuning`  
**Base:** `main`  
**Create:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/vision-belief-production-tuning?expand=1

---

## Summary

- Raise default `WINDOW_BELIEF_ENTER_VOTES` from 2 → **3** and `WINDOW_BELIEF_EXIT_VOTES` from 1 → **0** after live deploy evidence.
- Add regression test for intermittent marginal labels (package) never entering belief.
- Align `SceneBeliefTracker` / `SceneBeliefRegistry` constructor defaults with settings.
- Document production-tuned defaults in window README; clarify council refresh comment in `.env_example`.

## Outcome moved

Post-PR #811 deploy fixed observed-label flicker (`door`/`screen` stable_scene) but **package** still ping-ponged in belief every ~10–20s → ~14% council interpret rate vs ≤2% target. Stricter debounce blocks intermittent detections from entering belief.

## Tests run

```text
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests -q  → 22 passed
```

## Review findings fixed

- `SceneBeliefTracker` class defaults aligned to 3/0
- README documents production defaults
- Local `.env` synced (`ENTER=3`, `EXIT=0`; council `REFRESH_SEC=0`)

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-vision-window/.env \
  -f services/orion-vision-window/docker-compose.yml up -d --build
```

Verify:

```bash
docker exec orion-athena-vision-window printenv WINDOW_BELIEF_ENTER_VOTES WINDOW_BELIEF_EXIT_VOTES
# expect: 3, 0
```

## Test plan

- [ ] Merge + redeploy vision-window
- [ ] 15 min cam0: no `package` belief_transition ping-pong
- [ ] Council interpret rate drops toward ≤2%
