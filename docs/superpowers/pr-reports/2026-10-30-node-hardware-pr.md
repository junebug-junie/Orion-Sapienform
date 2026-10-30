## Summary

- The machine list Orion's body atlas reads (`config/biometrics/node_catalog.yaml`) now records each machine's real hardware, read from the box itself on 2026-10-30, not from memory.
- circe: Gigabyte G481-HA0, 2x Xeon Gold 6150, 251 GB, 4 GPUs (was "model not recorded").
- athena: HPE DL380 Gen10, 2x Xeon Platinum 8168, 377 GB, **Tesla P4 + Tesla T10** (the README said "no GPU").
- hecate (offline since 10-15): Inspur NF5288M5 AGX-2, 8 SXM2 slots, 1 V100-SXM2-32GB installed, 80 threads, from its last biometrics row.
- carbon-x1: ThinkPad X1 Carbon Gen 12, Core Ultra 7 155U, 32 GB. janus: HP ProDesk 600 G2 mini PC, i7-6700T, 16 GB, a desktop with a local Postgres 16; role `desktop_workstation`.
- pan-1: hardware still unknown. No ssh login reaches it from athena; the comment says so.

## Outcome moved

`python -P -m orion.dig body` shows true hardware per machine. Before: circe "model not recorded", janus "unknown". The README mesh table and hardware section were stale (circe GPU list, athena "no GPU") and are corrected.

## Current architecture

`hardware:` is free text read only by the body atlas (`orion/body_atlas/sources.py`). NodeCatalog ignores it; `unmonitored_nodes` are ignored by biometrics.

## Architecture touched

Config text and README only. No code, schema, bus or env change.

## Files changed

- `config/biometrics/node_catalog.yaml`: hardware text + provenance comments for athena, circe, hecate, carbon-x1, janus; pan-1 gap comment; janus role.
- `README.md`: mesh table (athena GPUs, circe GPUs, hecate row) and hardware section.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest orion/body_atlas/tests services/orion-biometrics/tests tests/test_static_body_descriptions.py
  -> 259 passed, 1 failed
```

The 1 failure (`test_circe_node_availability_reflects_expected_offline`) fails identically on clean `main`; it predates this patch.

## Evals run

None; text-only change.

## Docker/build/smoke checks

```text
ssh circe@circe / juniper@carbon-x1 / janus@janus: dmi, lscpu, free, nvidia-smi
athena: local dmi, lscpu, nvidia-smi
hecate: last orion_biometrics row 2026-10-15T06:31 (Tesla V100-SXM2-32GB, cores 80)
pan-1: ssh denied for juniper/athena/pi/ubuntu/root
```

## Review findings fixed

Text-only config/docs change; no code review subagent run.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

(The harness reads the catalog from its repo clone; a rebuild or the next post-pull run picks it up.)

## Risks / concerns

- Severity: low. Concern: pan-1 hardware unknown. Mitigation: stated in the catalog; needs a login.
- Severity: low. Concern: pre-existing biometrics test failure on main. Mitigation: not touched here; flagged.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
