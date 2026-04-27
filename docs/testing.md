# Testing Contract (Global)

This repository uses a single pytest contract across Python services.

## Rules

- Use `python3 -m pytest` (or `<venv>/bin/python -m pytest`), not bare `pytest`.
- Prefer service-scoped test runs.
- Do not use ad-hoc runtime `pip install pytest` unless emergency debugging is explicitly required.
- Always report exact command and exit code in verification notes.

## Bootstrap test environments

Bootstrap dev + service requirements into `venv` and `orion_dev`:

```bash
./scripts/bootstrap_test_envs.sh --service orion-actions
```

If `--service` is omitted, only root dev requirements are installed.

## Shared service runner

Run tests for any service through one entrypoint:

```bash
./scripts/test_service.sh orion-actions
./scripts/test_service.sh orion-actions services/orion-actions/tests/test_daily_actions.py -q
```

Default behavior is service-scoped (`services/<service>/tests -q --tb=short`) when no pytest args are provided.

## Existing wrapper scripts

- Hub: `./scripts/test_hub.sh`
  - Default mode uses `hub-app` container.
  - Set `HUB_TEST_RUNNER_MODE=local` to run via shared local runner.
- Actions: `./scripts/test_orion_actions.sh`

## Make targets

```bash
make test SERVICE=orion-actions ARGS='services/orion-actions/tests/test_daily_actions.py -q'
make test-hub
make test-actions
```
