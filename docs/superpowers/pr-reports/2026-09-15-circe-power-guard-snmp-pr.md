# Circe power-guard SNMP backend — PR report

## Summary

- Add `POWER_GUARD_UPS_BACKEND=nis|snmp` so Circe can poll its AP9631 over
  SNMP while Athena stays on USB/`apcupsd` NIS (default unchanged).
- Document Circe multi-homed NMC Access Control + arming order.
- Add `orion-power-guard` to Circe's mesh bring-up allowlist.
- Fill `ORION_BUS_VELOCITY_TRACKING_ENABLED` in `.env_example` so compose
  stop warning on blank.

## Outcome moved

Circe has a live, armed graceful-shutdown watcher for its own SRT5K UPS
(network card), independent of Athena's USB UPS path.

## Current architecture

Athena: USB Smart-UPS → host `apcupsd` → NIS → `orion-athena-power-guard`.
Circe previously had a dead container (no apcupsd / no SNMP wiring). The
SNMP client existed but `main.py` was hard-wired to NIS.

## Architecture touched

- `services/orion-power-guard` — backend toggle, docs, env template, tests
- `mesh-utilities/common/include_services_circe.txt` — Circe allowlist

## Files changed

- `services/orion-power-guard/app/main.py` — `build_ups_client()` NIS/SNMP
- `services/orion-power-guard/app/settings.py` — `POWER_GUARD_UPS_BACKEND`
- `services/orion-power-guard/docker-compose.yml` — pass backend env
- `services/orion-power-guard/.env_example` — backend + velocity key
- `services/orion-power-guard/README.md` — dual-backend + Circe notes
- `services/orion-power-guard/tests/test_ups_backend.py` — selection tests
- `mesh-utilities/common/include_services_circe.txt` — include power-guard

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: Circe can select SNMP; Athena default remains NIS
- Compatibility notes: unset `POWER_GUARD_UPS_BACKEND` → `nis`

## Env/config changes

- Added keys: `POWER_GUARD_UPS_BACKEND`; `ORION_BUS_VELOCITY_TRACKING_ENABLED`
  (template only; compose already referenced it)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes
- skipped keys requiring operator action: none for this service

Circe live `.env` (gitignored, not in PR):

```text
POWER_GUARD_UPS_BACKEND=snmp
POWER_GUARD_UPS_HOST=192.168.1.41
POWER_GUARD_SNMP_COMMUNITY=public-2
POWER_GUARD_ONBATTERY_GRACE_SEC=300.0
POWER_GUARD_ENABLE_SHUTDOWN=true
```

## Tests run

```text
PYTHONPATH=services/orion-power-guard:. pytest services/orion-power-guard/tests -q
11 passed
```

## Evals run

```text
No evals/ for this service (poll loop + shutdown trigger; gate tests only).
```

## Docker/build/smoke checks

```text
Circe: docker compose up -d --build orion-circe-power-guard
Live polls: raw=ONLINE charge=100% volts~247 (input V, not watts)
Shutdown path: docker exec … SSH_OK / hostname circe
Athena: orion-athena-power-guard left running; config not modified
```

## Review findings fixed

- Finding: SNMP `output_status==3` set `raw_status=ONBATT` but left
  `on_battery=False` whenever line voltage stayed above 80V — Circe would
  never start the grace timer / shutdown.
 - Fix: prefer APC output status for on-battery; add regression tests.
 - Evidence: `ups_snmp_client.py` + `tests/test_snmp_on_battery.py`

## Restart required

Circe (already applied for live smoke; re-run after merge if image drifts):

```bash
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && \
  docker compose --env-file .env --env-file services/orion-power-guard/.env \
  -f services/orion-power-guard/docker-compose.yml up -d --build'
```

Athena: no restart required for this change (default remains `nis`).

## Risks / concerns

- Severity: medium
- Concern: Circe multi-homed source IP must stay allowed on the NMC ACL;
  changing default route NIC would break SNMP until ACL is updated.
- Mitigation: README Circe notes; live ACL row for `.24` / `public-2`.

- Severity: medium
- Concern: Armed shutdown is real (`shutdown -h now` after 5 min on battery).
- Mitigation: Verified SSH path with harmless probe; grace matches Athena.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2231
