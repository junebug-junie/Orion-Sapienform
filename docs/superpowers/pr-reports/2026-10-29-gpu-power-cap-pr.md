## Summary

- circe's gpu2 is now held at 150 W from a checked-in setting, `config/gpu_power_limits.json`. Juniper chose 150 W after the long-context bench (#2963): energy per written token 0.80x, write speed 0.86x.
- `scripts/ops/gpu_power_limits.py` is a stdlib root job, installed like circe's LLM port gate. It applies the limits at boot and re-checks every 5 minutes, logging any drift it puts back.
- Rollback cannot leave a card capped. Stopping the service resets every listed card and stops the checks. `uninstall` resets explicitly before deleting anything, and stops loudly if a reset fails.
- The GPU sampler gains a `power_limit_watts` column, so biometrics shows each card's limit next to its draw.
- The power bench now locks the card before reading its limit, and it expects the configured limit instead of the default.
- Includes the design (#2964), with a section on how the body economy reads the cap, and a runbook.

## Outcome moved

The 150 W decision no longer depends on a hand-typed `nvidia-smi -pl 150` that vanishes on reboot, and nothing silently undoes it. Proof that the cap holds will be live, once installed: acceptance checks 1–7 in the spec, then the one-week outcome check (#8) around 2026-11-05.

## Current architecture

Nothing set GPU power limits except the bench, which always restored the starting limit. circe's lane controller has no GPU access (its only route to the host is the Docker socket, through an allowlist), so it cannot hold a limit. orion-biometrics sampled `power.draw` per card but not `power.limit`.

## Architecture touched

- A new host-level systemd job on circe (root): `/usr/local/sbin/orion-gpu-power-limits` plus three units, installed from `scripts/ops/`.
- A new config file.
- One CSV column in `orion/sensors/gpu_host_stats.sh`. biometrics reads it by header name, through a directory bind mount, so it needs no rebuild.
- The bench's start check.
- No service code, bus, schema, or pool change.

## Files changed

- `config/gpu_power_limits.json`: new. gpu2 = 150 W. `"watts": "default"` un-caps a card.
- `scripts/ops/gpu_power_limits.py`: new. Verbs: apply, remove, status, install, uninstall.
- `scripts/ops/orion-gpu-power-limits.service`, `orion-gpu-power-limits-check.service`, `orion-gpu-power-limits.timer`: new.
- `orion/sensors/gpu_host_stats.sh`: `power_limit_watts` as the last column.
- `orion/telemetry/biometrics_pipeline.py`: a comment that called temperature "the last column".
- `scripts/gpu_power_bench.py`: takes the lock before reading the limit; adds `expected_limit()` and `--limits-config`.
- `docs/superpowers/specs/2026-10-29-gpu-power-cap-design.md`: the design from #2964, plus the economy section and gpu1 status.
- `docs/runbooks/2026-10-29-circe-gpu-power-limits.md`: new. Install, check, change, rollback, bench.
- `tests/scripts/test_gpu_power_limits.py`: new, 24 tests against a fake nvidia-smi.
- `tests/test_gpu_power_bench.py`: 3 new tests.
- `services/orion-biometrics/tests/test_gpu_collector.py`: the real-script test now carries the new column.

## Schema / bus / API changes

- Added: the CSV column `power_limit_watts` (last; consumers read by header name). It appears as `gpus[].power_limit_watts` in biometrics samples.
- Removed: none
- Renamed: none
- Behavior changed: the power bench starts only if the card is at its configured limit (or its default when not listed), and it restores that limit.
- Compatibility notes: on a card where `power.limit` reads `[N/A]`, the value stays a harmless string; nothing parses it numerically.

## Env/config changes

- Added keys: none in any `.env_example`. The root job reads `ORION_GPU_POWER_LIMITS_CONFIG` and `ORION_GPU_POWER_LIMITS_LOCK_DIR` only as optional overrides; the defaults are the circe paths.
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest tests/scripts/test_gpu_power_limits.py -q   -> 24 passed
PYTHONPATH=. .venv/bin/python -m pytest tests/test_gpu_power_bench.py -q            -> 51 passed
(services/orion-biometrics) pytest tests/test_gpu_collector.py -q                    -> 4 passed
git diff --check                                                                     -> clean
```

## Evals run

```text
None: this is an actuator with deterministic tests. The outcome check is acceptance check 8 in the spec
(one week of live data, about 2026-11-05).
```

## Docker/build/smoke checks

```text
Live on circe, read-only (no root, no limit changed):
  gpu_power_limits.py status  -> {"gpu": 2, "target_w": 150, "current_w": 250.0, "default_w": 250.0, "at_target": false}
  DRY_RUN=1 ... install       -> install -m 0755 ... /usr/local/sbin/orion-gpu-power-limits, 3 units, enable, restart
  gpu_host_stats.sh (branch)  -> header ends ...,temperature_gpu_c,power_limit_watts; rows end "250.00"
  hostname -> circe (matches config "host"); nvidia-persistenced active; python3 3.12.3
Note: one ordinary sample from the current main sampler was written to circe's
/mnt/telemetry/gpu_stats during the smoke (same kind biometrics writes every 31 s).
```

## Review findings fixed

- Finding (medium): `uninstall` relied on the service's ExecStop. That never runs when the boot `apply` failed (the oneshot is then inactive, while the timer keeps the card capped), so uninstall deleted everything and left gpu2 at 150 W.
  - Fix: uninstall disables the timer, stops any running check, runs `remove` itself, and only then deletes. If a reset fails, it keeps the units and exits 1.
  - Evidence: `test_uninstall_resets_explicitly_before_deleting`, `test_uninstall_keeps_units_when_the_reset_fails`.
- Finding (medium): stopping only the service was undone by the next 5-minute check.
  - Fix: the timer is `PartOf=orion-gpu-power-limits.service`.
  - Evidence: `test_timer_stops_with_the_service`.
- Finding (low): `remove` skipped a card whose lock was held (by a running check, or a bench) and reported success.
  - Fix: `remove` waits up to 120 s for the lock and fails loudly with "NOT reset to default". `apply` still skips a held card, because the next check catches up.
  - Evidence: `test_remove_waits_for_a_held_card_and_fails_loudly_instead_of_skipping`.
- Finding (low): deleting a card from the config left it capped, with no way to un-cap through a PR alone.
  - Fix: `"watts": "default"`.
  - Evidence: `test_default_sentinel_uncaps_a_card_through_the_config`.
- Finding (low): a non-object config, a duplicate index (`"2"` and `"02"`), and bool or float watts were accepted or crashed.
  - Fix: each is refused as a ConfigError, and no card is touched.
  - Evidence: `test_ambiguous_configs_are_refused`.
- Finding (self): `load_config` froze its path as a default argument at import, so the module setting could never change it.
  - Fix: the path is looked up per call.
  - Evidence: the 15 original enforcer tests failed until this was fixed.
- Not changed:
  - The bench refuses gpu2 between circe's pull and the install. That is fail-safe, and the runbook says to install right after the pull.
  - Install does no host check. On another host the units just refuse every run.
  - `remove` at shutdown is harmless extra work, since limits reset at reboot anyway.

## Restart required

```bash
# after merge, on circe (Juniper):
ssh circe@circe git -C /mnt/scripts/Orion-Sapienform pull --ff-only
ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/gpu_power_limits.py install
```

No container restart is needed: biometrics reads the sampler through a directory bind mount.

## Risks / concerns

- Severity: low
  - Concern: the agent lane on gpu2 writes about 14% slower, and queueing could amplify that on a busy lane.
  - Mitigation: acceptance check 8 compares agent-lane stance p50 against the 2026-10-29 baseline (agent-gpu2 147 s). Rollback is one command.
- Severity: low
  - Concern: root runs a job that reads its config from the repo checkout.
  - Mitigation: the code runs from the installed copy. The config can only choose a wattage inside the card's own [min, max] range, and only for the host it names.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
