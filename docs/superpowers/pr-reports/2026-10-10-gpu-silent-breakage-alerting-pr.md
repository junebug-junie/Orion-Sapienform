# fix: GPU swap breakage can no longer go silent

## Summary

- **The incident (2026-10-09 04:07 → 10-10 05:51 UTC, 25.7 h).** The circe GPU lane controller's
  image predated a new `gpu_pool.yaml` field, so it refused every card swap:
  155 × `config_unloadable:ValidationError`. Orion produced **no painting for 27.6 h**, and nothing
  told anyone.
- **The guardian now watches the GPU controllers in two ways:**
  - every 5 minutes it asks each controller whether it can read its config and matches the pool's
    commit;
  - it listens to the pool's own refusal events.
  - A Hub attention card goes out, critical for "can't read config", which also emails at once.
- **The controller's `/health`** answers 503 with the error when its config won't load. It was a
  constant `ok` through the whole incident.
- **The painter has a second check, "no real painting in 12 hours"**, in its own loop.
  - The old staleness check never ran in production: it lives in the legacy worker's watchdog,
    which is off because paintings now come through the durable path.
  - The new loop does not depend on that worker's flag.

## Outcome moved

- **Time to first alert, replaying the real incident:**
  - before: never (25.7 h, found by hand);
  - after: **0 s** from the first refused swap via the pool-event path, ≤ 295 s worst case via the
    probe alone.
- **Card count:** 5 over the incident instead of 155, with dedup at one card per problem per 6 h.
- **No-painting check on every painting with a receipt** (09-14 → 10-10, 26 days, 169 gaps): at
  12 h it alerts on 5 gaps: 247 h (09-14 → 09-25, **a 10-day stretch with no paintings that nobody
  noticed**), 27.6 h (this incident), 17.8 h, 16.0 h and 14.7 h. Only the 27.6 h gap has a known
  cause. The next-longest gap is 7.6 h. The threshold was picked on this same data.

## Current architecture

- **orion-mesh-guardian** (always on) raises Hub cards for crash loops, Redis and FalkorDB. It had
  no GPU checks.
- **The GPU pool** logs refusals and emits `actuate_refused` on `orion:gpu_pool:event`. sql-writer
  stores them and the Hub displays them. Nobody alerted on them.
- **The lane controller's `/health`** was constant `{"ok": true}`.
- **The painter's staleness monitor** checks the newest `reverie_visual_chain` row of any kind.

## Architecture touched

- `orion/gpu_pool/actuator_probe.py` (new, shared):
  - request builder, verdict classifier, async probe;
  - `scripts/gpu_pool_actuator_probe.py` is now a thin CLI over it (same output and exit codes).
- orion-mesh-guardian:
  - `app/gpu_watch.py` (new, pure: `ActiveProbeTracker`, `RefusalWatch`);
  - `service.py` `_gpu_loop` (own loop, 300 s), `_gpu_event_loop`, and a shared `_publish_alerts`;
  - `AlertGate` lets a higher severity through inside the window.
- orion-gpu-lane-controller: `/health` reads the config the same way actuation does.
- orion-thought:
  - `visual_painting_gap` check (severity error) in its own lifespan loop,
    `run_visual_painting_gap_watchdog`;
  - per-key edge state in `VisualChainHealthMonitor`;
  - `store.visual_last_painting_age_hours()`;
  - watchdog wiring.
- Contracts: `orion/bus/channels.yaml` adds guardian as a producer on `orion:gpu_pool:actuate:request`
  and a consumer on `actuate:result` and `orion:gpu_pool:event`; metric lock refreshed.

## Files changed

See `git diff --stat origin/main...HEAD`. The new tests and evals:

- `services/orion-mesh-guardian/tests/test_gpu_watch.py`
- `services/orion-mesh-guardian/evals/test_gpu_incident_replay.py`, with the real 155 refusal rows
- `tests/test_gpu_pool_actuator_probe.py`
- `services/orion-gpu-lane-controller/tests/test_api.py`
- `services/orion-thought/tests/test_visual_painting_gap_monitor.py`
- `services/orion-thought/evals/test_painting_gap_replay.py`, with the real 21-day gap series

## Schema / bus / API changes

- **Added:** guardian roles on the three GPU pool channels.
- **Behavior changed:** the lane controller's `/health` returns 503 when its config is unloadable.
  Nothing restarts on it: there is no compose healthcheck and it is not in the remediation roster.
- **Compatibility:**
  - The guardian sends only the `digest` probe, a load with a never-allowed profile. It is refused
    before any docker call or generation spend.
  - The pool drops results whose action_id it didn't issue.
  - The `status` probe is deliberately **not** sent periodically: the controller replays its last
    result under the pool's own action_id, which can flip the pool's belief about a card.

## Env/config changes

- **Added:**
  - orion-mesh-guardian: `MESH_GUARDIAN_GPU_WATCH_ENABLED=true`, `MESH_GUARDIAN_GPU_PROBE_INTERVAL_SEC=300`,
    `MESH_GUARDIAN_GPU_PROBE_WAIT_SEC=90`
  - orion-thought: `ORION_VISUAL_PAINTING_GAP_THRESHOLD_HOURS=12`,
    `ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED=true`, `ORION_VISUAL_PAINTING_GAP_CHECK_INTERVAL_SEC=600`
- **`.env_example`, settings, compose and README:** updated.
- **Local `.env` synced:** yes, `--all-keys` per service, run from the branch copies. No keys skipped.

## Tests run

```text
orion-mesh-guardian tests+evals          108 passed
orion-gpu-lane-controller                 81 passed
orion-thought tests+evals                558 passed, 1 failed
  pre-existing on main: test_settings_mind_enrichment default URL
tests/test_gpu_pool_actuator_probe.py     15 passed
gpu-pool e2e incl. probe                   5 passed
check_definition_drift --gate PASS · check_metric_lineage --gate PASS · env parity PASS · diff --check clean
Mutation checks, each one reverted and caught by a test:
  guardian 14/14; painting-gap 8/8 + 6/6 (loop not gated on the legacy flag, None not a reading);
  /health config read.
```

## Evals run

```text
services/orion-mesh-guardian/evals/test_gpu_incident_replay.py
  Replays the real 155 refusals (25.7 h):
    first card 0 s after the first refusal;
    probe-only worst case 295 s, swept across every phase;
    5 cards total; healthy baseline silent.
services/orion-thought/evals/test_painting_gap_replay.py
  Replays all 169 real gaps (09-14 → 10-10) at 10-minute ticks:
    at 12 h exactly 5 alerts and 5 recoveries;
    sensitivity: 6 h is noisy, 20 h catches only the 247 h and 27.6 h gaps.
  Calibrated in-sample on the same data, so it confirms the separation; it does not predict.
```

## Docker/build/smoke checks

```text
Not deployed: live path UNVERIFIED. After deploy:
  guardian log "gpu probe cycle {...}"
  orion:gpu_pool:actuate:request traffic from orion-mesh-guardian:gpu-watch
  curl circe:8090/health -> 200 config_loadable=true
  thought log "visual painting gap watchdog started"
```

## Review findings fixed

- **Guardian (its own review):**
  - Blocker: the periodic `status` probe could replay an old result into the pool.
    - Fix: digest-only probing, pinned by a test.
  - Persistent non-transient refusals were silent.
    - Fix: a refusal streak alerts.
  - An early error card could mask a later critical on the same key.
    - Fix: AlertGate severity escalation.
  - Probe-side mismatch wording always blamed the controller.
    - Fix: the wording now says either side may be stale.
  - The guardian re-read an unparseable config on every event.
    - Fix: throttled.
- **High: the painting-gap check never ran in production.** It sat inside the legacy watchdog,
  which is off live (`visual chain disabled; watchdog not started`).
  - Fix: its own loop, started in the lifespan, gated only on its own flag.
  - Evidence: tests with the legacy flag and the bus off; mutation-checked.
- **Medium: wrong rationale.** The docs said deferral rows kept the staleness check green. Live,
  the chain table held only 2 real paintings in the window; the check was silent because it never
  ran.
  - Fix: corrected in all five places.
- **Medium: a failed DB read mid-gap sent a false "recovered", then re-alerted.**
  - Fix: None is no observation.
  - Evidence: an unhealthy → None → unhealthy test.
- **Medium: the eval overstated its window and its causes.**
  - Fix: the full receipt history (26 days) with in-sample calibration stated and unknown causes
    marked. This surfaced the 10-day 09-14 → 09-25 gap.
- **Low: the "rebuild" hint was given for every config failure.**
  - Fix: `config_fix_hint` by exception type (rebuild only for `ValidationError`; mount/path;
    YAML syntax), shared by `/health` and the guardian card.
- **Low: stale docs** (guardian docstring, controller README).
  - Fix: updated.

## Restart required

From the primary checkout on main after merge. The controller runs on circe:

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-mesh-guardian up -d --build && scripts/safe_docker_build.sh orion-thought up -d --build && ssh circe@circe "cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build"
```

## Risks / concerns

- Severity: low
  - Concern: the guardian compares controllers against the live main checkout, not the pool's
    baked config. If the pool image lags main, a probe-side mismatch card can appear when the pool
    and controller actually agree.
  - Mitigation: the card wording says either side may be stale. Only the pool-event path claims a
    pool-vs-controller mismatch.
- Severity: low
  - Concern: the passive refusal window is in memory, so a guardian restart delays a burst card.
  - Mitigation: immediate-critical reasons are unaffected, and the probe re-detects within 300 s.
- Severity: note
  - Concern: one incident can raise several cards: guardian critical, then the painting gap 12 h
    later.
  - Mitigation: each names its own layer (controller vs painter output), and the painting-gap card
    points at the guardian card first.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2595

🤖 Generated with [Claude Code](https://claude.com/claude-code)
