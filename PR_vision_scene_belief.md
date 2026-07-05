# PR: feat(vision): scene belief habituation to stop council metacog flicker

**Branch:** `feat/vision-scene-belief`  
**Base:** `main`  
**Create:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/vision-scene-belief?expand=1

---

## Summary

- Add `SceneBeliefTracker` in `orion-vision-window` with per-stream vote-based habituation (`observed` → `believed` tier).
- Window flush emits `summary.evidence.believed_hard_labels` + `belief` metadata on every publish.
- Council `evidence_transition` gates metacog on **believed** labels only; rename `labels_changed` → `salient_labels_changed`.
- Disable `COUNCIL_TRANSITION_REFRESH_SEC` by default (`0`) — contract A silence on stable scenes.
- Post-review fix: wire `WINDOW_BELIEF_EXIT_VOTES` with split enter/exit counting and rollback tests.

## Outcome moved

Stable office scenes should stop triggering council metacog on single-frame `hard_labels` flicker. Target: ≥98% `stable_scene` skip; zero `refresh_ttl` interprets at default config.

## Current architecture

Before: window emitted instantaneous `hard_labels` from each 5s batch; council compared observed labels and forced refresh every 120s → ~26% metacog calls, mostly `labels_changed` flicker.

## Architecture touched

| Layer | Service | Seam |
|-------|---------|------|
| Producer | `orion-vision-window` | `scene_belief.py`, `_flush_and_publish()` |
| Consumer | `orion-vision-council` | `evidence_transition.py` `_labels_for_gate()` |
| Config | both services | `WINDOW_BELIEF_*`, `COUNCIL_TRANSITION_REFRESH_SEC=0` |

## Files changed

- `services/orion-vision-window/app/scene_belief.py`: vote ring, enter/exit habituation, registry
- `services/orion-vision-window/app/main.py`: belief registry wiring on flush
- `services/orion-vision-window/app/settings.py`, `.env_example`, `docker-compose.yml`: belief env keys
- `services/orion-vision-council/app/evidence_transition.py`: believed-label gate + reason rename
- `services/orion-vision-council/app/settings.py`, `.env_example`, `docker-compose.yml`: refresh TTL default 0
- Tests: `test_scene_belief.py`, `test_belief_flush_wiring.py` (window); `test_scene_belief.py` (council)
- Docs: service READMEs, `docs/vision_services.md`

## Schema / bus / API changes

- Added: `summary.evidence.believed_hard_labels`, `summary.evidence.belief` (`schema: scene_belief.v1`)
- Removed: nothing
- Renamed: council transition reason `labels_changed` → `salient_labels_changed`
- Behavior changed: council metacog gate reads belief tier when present; refresh TTL off by default
- Compatibility: council falls back to `hard_labels` when belief tier absent (safe window-first deploy)

## Env/config changes

- Added keys: `WINDOW_BELIEF_ENABLED`, `WINDOW_BELIEF_VOTE_N`, `WINDOW_BELIEF_ENTER_VOTES`, `WINDOW_BELIEF_EXIT_VOTES`
- Changed: `COUNCIL_TRANSITION_REFRESH_SEC` default `120` → `0`
- `.env_example` updated: both services
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-vision-window --all-keys` (where `.env` exists)

## Tests run

```text
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests -q  → 21 passed
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q  → 61 passed
docker compose config (window + council) → OK
```

## Evals run

```text
No dedicated eval harness for vision-window/council belief path; covered by unit + integration tests above.
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-vision-window/.env \
  -f services/orion-vision-window/docker-compose.yml config → OK
docker compose --env-file .env --env-file services/orion-vision-council/.env \
  -f services/orion-vision-council/docker-compose.yml config → OK
```

## Review findings fixed

- Finding: `WINDOW_BELIEF_EXIT_VOTES` stored but unused (`count==0` hardcoded)
  - Fix: exit uses raw ring counts with `count <= exit_votes`; enter uses effective counts with empty carry-forward
  - Evidence: `test_exit_votes_respected`, `test_belief_requires_exit_votes`
- Finding: `enrich_evidence` emitted vote-threshold labels diverging from `_believed`
  - Fix: commit `bd84b8c2` — emit `sorted(self._believed)`
  - Evidence: council stable-scene tests pass on flicker
- Finding: no rollback test for `WINDOW_BELIEF_ENABLED=false`
  - Fix: `test_belief_disabled_omits_belief_tier`
  - Evidence: 21 window tests pass

## Restart required

Deploy **window first**, then council:

```bash
docker compose --env-file .env --env-file services/orion-vision-window/.env \
  -f services/orion-vision-window/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-vision-council/.env \
  -f services/orion-vision-council/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: ~15s warmup before first belief promotion (full ring gate); council may use observed `hard_labels` fallback briefly
- Mitigation: empty carry-forward accelerates enter; monitor `belief_transition` logs post-deploy

- Severity: low
- Concern: runtime acceptance (30-min cam0 sample) not yet measured on live rail
- Mitigation: acceptance checks in spec §10 — interpret ≤2% of window count, zero `refresh_ttl`

## Test plan

- [ ] Deploy vision-window; confirm `/current` payloads include `believed_hard_labels`
- [ ] Deploy vision-council; confirm stable cam0 scene logs `stable_scene` not `salient_labels_changed`
- [ ] Confirm no `refresh_ttl` interpret lines with default env
- [ ] Introduce persistent object; confirm single `salient_labels_changed` metacog call
