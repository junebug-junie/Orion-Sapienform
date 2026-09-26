# Raise Reading Daily Caps

## Summary

- Raise independent Stage 1 / Wallet A and Stage 2 / Wallet B defaults from 6 to 12.
- Align Hub settings, operator template and Stage 2's Wallet A reentry fallback.
- Preserve counts, cooldown floors, retry limits, backoff and GPU safeguards.

## Current architecture

- Service: orion-hub; workers wired by scripts/main.py.
- Config: app/settings.py and .env_example; docker-compose.yml loads local .env.
- Existing Redis wallet keys and Postgres reading queue remain unchanged.
- No bus channels, schemas, dependencies or migrations changed.
- Tests: route/default contracts and wallet A/B/refund tests.
- Evals: evals/test_reading_handoff_eval.py.
- Gap: six settled Stage 1 runs exhausted the old daily cap despite queued retries.

## Verification

- Focused route/wallet/refund tests plus reading handoff eval: 64 passed,
  including 11 handoff eval cases.
- Regression covers admission after 6 and 11 runs, blocking at 12, both wallets,
  default-window pacing (70 minutes), disabled-window floor (30 minutes),
  cooldown and refund-backoff preservation, and Stage 2 fallback consistency.
- Isolated Docker build: reading-caps-check-hub-app passed.
- Network-disabled image smoke: actual Settings resolved both defaults to 12.
- Local Hub .env updated only for these two caps and remains ignored.
- Ran sync_local_env_from_example.py orion-hub; preserved intentional local
  Stage 1 window 0/0 and curiosity elastic-activation overrides.
- No new metrics; metric quality gate not applicable.
- CI required refreshing the metric lock's generated previous-change summary:
  zero metric definitions changed; definition drift gate passes after regeneration.

## Review findings fixed

- Finding: unavailable-settings dashboard fallbacks still advertised 6.
- Fix: align all schedule/status fallback caps to 12.
- Evidence: regression covers absent fields and settings-provider exceptions.
- Independent reviewer found no critical or important issues.

## Rollout and runtime evidence

Production was not restarted. Live status still reports caps 6/6 with daily
counts 6/4; applying the new environment to the running process is UNVERIFIED.
From this worktree after merge/deployment, recreate Hub:

```sh
scripts/safe_docker_build.sh orion-hub up -d --build --force-recreate
curl -fsS http://127.0.0.1:8080/world-pulse-read/api/status
```

Require wallet_a.daily_cap=12 and wallet_b.daily_cap=12. Counts must not be
reset. Raising caps is not evidence that a reading completed or GPU admission
succeeded. Roll back by setting both local caps to 6 and recreating Hub.
