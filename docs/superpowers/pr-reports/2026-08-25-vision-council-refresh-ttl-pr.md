# PR report: give the stable-scene refresh gate a real ceiling

## Summary

- Root-caused a real, live incident: `orion:vision:events`/`vision_events` had been silent for 44+ continuous hours (last row 2026-08-23 09:22 UTC) with zero errors anywhere in the pipeline.
- Wrongly suspected `orion-vision-scribe`'s bus consumer first (an `iter_messages()`/`pubsub.listen()` unbounded-block hang was a real, separate, latent risk shared by 67 consumers across the repo) — restarted it live, confirmed that fixed nothing, and traced further upstream.
- Actual cause: `orion-vision-council`'s "stable scene" transition gate (`evidence_transition.py`) skips its interpretation LLM call when the camera scene's coarse `hard_labels` haven't changed since the last interpretation. `COUNCIL_TRANSITION_REFRESH_SEC` (meant to force a periodic re-interpretation even on a stable scene) defaulted to `0` ("never force"). A home-office desk scene's coarse labels (chair/clothing/desk/door/person/table) genuinely never change turn to turn, so the gate correctly and deterministically read `stable_scene` on every single window for 44+ hours — vision-host kept doing GPU inference the whole time, council kept ticking the whole time, nothing crashed.
- Fix: default `COUNCIL_TRANSITION_REFRESH_SEC` to `600` (10 minutes) everywhere it's declared (Settings, `.env_example`, `docker-compose.yml`'s own separate `${VAR:-N}` fallback), with the incident and tradeoffs documented in-line.
- Live production `.env` for this service was hand-edited to `600` in the same session so the fix is already in effect pending a container rebuild/restart.

## Outcome moved

`orion:vision:events` can no longer go silently, permanently dark on a scene whose coarse object labels don't change — the worst case is now bounded to 600s instead of unbounded. This restores the eventual data source for: `orion/situational/context.py`'s perception-context slot of the unified-turn `Situation:` prompt fragment (currently disabled for the hub call site, but live on cortex-exec's legacy path), `orion-thought`'s reverie perception narration, and `node:substrate.vision`'s `perception_staleness`/`prediction_error` metrics in the attention/substrate graph.

## Current architecture

`orion-vision-host` → (raw detections) → `orion-vision-council` (`evidence_transition.py` gates on coarse `hard_labels` delta + person-presence delta + a refresh TTL) → `orion:vision:events` (bus) → `orion-vision-scribe` (sole consumer, writes to Postgres `vision_events`) → several downstream readers (situational-brief perception context, reverie's vision_reader.py, substrate `perception_prediction_error()`, sql-writer's object-permanence/scene-inventory tables).

The refresh-TTL escape valve already existed in `EvidenceTransitionTracker.evaluate()` (`if max_refresh_sec > 0 and (now - last_interpret_at) >= max_refresh_sec: interpret=True, reason="refresh_ttl"`) — it was correctly implemented and already tested (`test_tracker_refresh_ttl`). The bug was purely a config default with no ceiling.

## Architecture touched

`services/orion-vision-council` only: `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_transition_settings.py`, `tests/test_evidence_transition.py`, `tests/test_scene_belief.py`. Plus two historical design-doc "superseded" notes. No schema/bus/API changes.

## Files changed

- `services/orion-vision-council/app/settings.py`: `COUNCIL_TRANSITION_REFRESH_SEC` default `0.0` → `600.0`, with a full incident + reverie-tradeoff writeup as an in-line comment.
- `services/orion-vision-council/.env_example`: matching value + comment.
- `services/orion-vision-council/docker-compose.yml`: `${COUNCIL_TRANSITION_REFRESH_SEC:-0}` → `${COUNCIL_TRANSITION_REFRESH_SEC:-600}` (review caught this — a separate substitution point from `.env_example`/`settings.py` that would have silently regressed back to `0` on any fresh deploy/CI run that doesn't inherit today's hand-edited `.env`).
- `services/orion-vision-council/README.md`: documented the incident and the accepted reverie-freshness tradeoff.
- `services/orion-vision-council/tests/test_transition_settings.py`: renamed/updated `test_settings_refresh_ttl_default_zero` → `test_settings_refresh_ttl_default_is_nonzero`, asserting `600.0`.
- `services/orion-vision-council/tests/test_evidence_transition.py`: added `test_tracker_stable_scene_never_refreshes_when_ttl_disabled`, reproducing the exact incident shape (identical static label set, 44 simulated hours, `max_refresh_sec=0.0` explicit) directly against `EvidenceTransitionTracker`.
- `services/orion-vision-council/tests/test_scene_belief.py`: renamed `test_refresh_ttl_disabled_by_default` → `test_max_refresh_sec_zero_means_disabled` with a docstring clarifying it guards the tracker's own zero-contract, not the Settings default (review finding — it was passing `max_refresh_sec=0.0` explicitly, so unaffected by the Settings change, but its old name now read as if it asserted the (former) default).
- `docs/superpowers/specs/2026-07-04-vision-scene-belief-design.md`, `docs/superpowers/plans/2026-07-04-vision-scene-belief.md`: one-line "superseded 2026-08-25" notes so a future reader doesn't mistake the old `Contract A: default 0` decision for still-current guidance.

## Schema / bus / API changes

None. Config-value change only; no new/removed/renamed keys, no payload/schema changes.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Changed default: `COUNCIL_TRANSITION_REFRESH_SEC` `0` → `600` in `.env_example`, `docker-compose.yml`'s inline fallback, and `app/settings.py`'s Pydantic default.
- `.env_example` updated: yes.
- local `.env` synced: hand-edited directly in the primary checkout (`/mnt/scripts/Orion-Sapienform/services/orion-vision-council/.env`) to `600`, since `scripts/sync_local_env_from_example.py` reads `.env_example` from the primary checkout and can't see this worktree's edits until merge. Confirmed present.
- skipped keys requiring operator action: none.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-vision-council-refresh-ttl
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-vision-council/tests -q
  65 passed, 18 warnings (warnings are pre-existing, unrelated pydantic
  protected-namespace noise on TopicFoundry* models)
```

## Evals run

No dedicated eval harness exists for this service; covered by the gate tests above plus the live-incident evidence gathered before the fix (direct Postgres queries against `vision_events`, live container log inspection of `orion-vision-council`/`orion-vision-scribe`/`orion-vision-host`, live env inspection inside the running container).

## Docker/build/smoke checks

Not rebuilt in this session — config/test/doc change only, no dependency or Dockerfile change. The running container was not rebuilt; the live `.env` value is already correct for the next restart/rebuild to pick up.

## Review findings fixed

- Finding: `docker-compose.yml`'s inline `${COUNCIL_TRANSITION_REFRESH_SEC:-0}` fallback was not updated alongside `settings.py`/`.env_example`, and the repo's own `check_service_env_compose_parity.py` only checks key presence, not default-value agreement — a real landmine for a future fresh deploy/CI run.
  - Fix: changed to `${COUNCIL_TRANSITION_REFRESH_SEC:-600}`.
  - Evidence: `grep -n COUNCIL_TRANSITION_REFRESH_SEC services/orion-vision-council/docker-compose.yml` now shows `:-600`.
- Finding: 600s does not restore reverie's own 180s percept-freshness gate on a genuinely stable scene (reverie still blind ~70% of each cycle) — the tradeoff wasn't stated anywhere in the diff.
  - Fix: added an explicit "known, accepted gap" writeup in both `settings.py` and `README.md`, naming the exact numbers and why 180s wasn't chosen instead (would out-pace the natural rate of genuine scene changes).
  - Evidence: see the new comment blocks in both files.
- Finding: `test_scene_belief.py::test_refresh_ttl_disabled_by_default`'s name reads as if it documents the (former) Settings default, though it actually passes `max_refresh_sec=0.0` explicitly and is unaffected by the Settings change.
  - Fix: renamed to `test_max_refresh_sec_zero_means_disabled` with a clarifying docstring.
  - Evidence: `services/orion-vision-council/tests/test_scene_belief.py`.
- Finding: two historical design docs still describe `default 0` as the sanctioned decision, which could mislead a future reader grepping for rationale.
  - Fix: one-line "superseded 2026-08-25" notes added to both, without otherwise editing the historical record.
  - Evidence: `docs/superpowers/specs/2026-07-04-vision-scene-belief-design.md`, `docs/superpowers/plans/2026-07-04-vision-scene-belief.md`.
- Confirmed correct, no fix needed: Pydantic `AliasChoices`/legacy-alias behavior unaffected (only the `default=` value changed); no misprocessing risk for a genuinely unattended stream (a forced `refresh_ttl` interpretation runs the identical grounding/publish path as any real change — no synthetic "change" marker a consumer could key off of; presence-duration tracking is fed by a separate write path (`substrate_embodied_presence`) unaffected by this gate).

## Restart required

```bash
docker restart orion-athena-vision-council
```

(Config-only change to an already-running container reading `env_file: .env` — a plain restart picks up the new `.env` value already in place. No image rebuild needed since no dependency/Dockerfile changed. Use `scripts/safe_docker_build.sh` instead if a full rebuild is ever wanted for other reasons.)

`orion-athena-vision-scribe` was already restarted live during this investigation (harmless, and confirmed not the actual cause — included here for the record, not because this PR requires it).

## Risks / concerns

- Severity: low
  Concern: `600`s was chosen as a middle ground (bounds the hub/cortex-exec situational-awareness 900s staleness gate comfortably; does not fully satisfy reverie's tighter 180s gate).
  Mitigation: documented explicitly rather than left implicit; a follow-up could tune reverie's own gate or add a reverie-specific override if fuller freshness there is wanted later.
- Severity: low
  Concern: the same unbounded-block pattern (`pubsub.listen()` with no timeout) that I initially, incorrectly suspected as this incident's cause is real and still present in `orion/core/bus/async_service.py::iter_messages()`, shared by 67 consumers across the repo. Not touched by this PR (it wasn't the cause here), but worth a separate, carefully-scoped follow-up given the blast radius.
  Mitigation: none in this PR; flagged for a future ticket.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/vision-council-refresh-ttl

🤖 Generated with [Claude Code](https://claude.com/claude-code)
