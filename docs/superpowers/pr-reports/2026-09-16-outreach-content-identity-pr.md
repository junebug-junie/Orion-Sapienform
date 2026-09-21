# PR #2236 — Outreach content identity ledger

https://github.com/junebug-junie/Orion-Sapienform/pull/2236

## Summary

- Persist `prior_ids` + content-stable `curiosity_content_ids` on outreach
  `grounding` (no motive taxonomy).
- Read-only anti-drives-2.0 measure script with 7d/14d windows.
- Design doc records the stress-test verdict: do not mint six kinds yet.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Status

DONE — branch pushed, PR open, tests green. Live identity coverage
UNVERIFIED until hub rebuild + new sends.
