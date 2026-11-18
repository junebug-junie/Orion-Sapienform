## Summary

- Heartbeat organ occupancy (`dark_seats`, `organ_fire_counts`, `organ_distinctness`) now uses a wall-clock window (default 300 s, `HEARTBEAT_ORGAN_FIRE_WINDOW_SEC`) instead of the last 64 absorbed events.
- Empty window, or the first full window after a restart (warm-up), reads unknown (`dark_seats=[]`, `organ_fire_counts={}`, `organ_distinctness=null`), never all-dark.
- Pruning happens on every read by clock, so silence is detected with no incoming events.
- New additive /h1 fields: `organ_last_fired_at`, `organ_seconds_since_last_fire`, `fire_window_sec` so a consumer can tell dark from merely rare. Last-fire survives pruning; null = never seen since boot.
- Memory bounded: 50,000 timestamps per organ max.

## Outcome moved

A rare organ (cortex-orch ~26/h) no longer reads "dark" because biometrics (~11k/h) flooded a 64-event count window.

## Current architecture

`OrganFireWindow` was a `deque(maxlen=64)` of organ names, `compute_proprioception` derived dark seats from counts; an all-zero window produced all-dark.

## Architecture touched

orion-heartbeat only: `app/substrate/proprioception.py`, `ensemble.py`, `reconstruction.py`, `service.py`, `settings.py`, compose, README, `.env_example`. No bus/schema registry change (/h1 is an HTTP debug surface; fields additive).

## Files changed

- `app/substrate/proprioception.py`: time window, snapshot, empty-is-unknown
- `app/substrate/ensemble.py`, `reconstruction.py`: carry recency fields
- `app/service.py`: window from settings; H1 loop passes snapshot
- `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: new key + docs
- `tests/test_proprioception.py`: regression tests

## Schema / bus / API changes

- Added: /h1 `organ_last_fired_at`, `organ_seconds_since_last_fire`, `fire_window_sec`
- Behavior changed: all-zero window now yields `dark_seats=[]` (was all five organs); `FIRE_WINDOW` constant replaced by `FIRE_WINDOW_SEC`
- Compatibility: existing field names unchanged. Recency values are computed at H1 tick time (see `generated_at`, 30 s cadence).

## Env/config changes

- Added keys: `HEARTBEAT_ORGAN_FIRE_WINDOW_SEC=300.0`
- `.env_example` updated: yes
- local `.env` synced: sync script ran but writes to primary checkout and does not pick up worktree edits; key appended by hand to the primary `services/orion-heartbeat/.env`
- skipped keys: none

## Tests run

```text
PYTHONPATH=. pytest services/orion-heartbeat/tests -q -> 126 passed
python3 scripts/check_env_template_parity.py -> PASS
```

New tests: rare organ not dark inside window; dark after window with recency; empty window unknown; wall-clock expiry with no events; bounded memory.

## Evals run

No eval harness for heartbeat proprioception; not claimed. Live check UNVERIFIED (container not restarted by instruction).

## Docker/build/smoke checks

Not run (no restart permitted).

## Review findings fixed

- Finding (medium): empty-window branch dropped recency fields, so "silent 40 min" looked like "just booted".
  - Fix: recency is reported on the unknown reading too (null after fresh boot).
  - Evidence: `test_wall_clock_expiry_with_no_incoming_events`.
- Finding (medium): after restart one organ firing made the other four read dark.
  - Fix: `warm` flag; unknown until process uptime >= window.
  - Evidence: `test_warmup_after_restart_reads_unknown_not_dark`.
- Finding (low): wall clock can step backward.
  - Fix: window uses `time.monotonic`; `time.time` only for `last_fired_at`.
- Finding (low, not fixed): attention_self_model and hub ignore the recency fields. Follow-up: have them surface `organ_seconds_since_last_fire`.

## Restart required

```bash
scripts/safe_docker_build.sh heartbeat up -d --build   # run from the deploying worktree; then curl -s localhost:7251/h1
```

## Risks / concerns

- Severity: low. Concern: at ~26/h cortex-orch still reads dark about 11% of 5-min windows by chance (Poisson, mean 2.2 per window); use `organ_seconds_since_last_fire`. Mitigation: consumers should use recency, or raise the window.
- A restart reads unknown for the first window (5 min).

## PR link

(see PR)
