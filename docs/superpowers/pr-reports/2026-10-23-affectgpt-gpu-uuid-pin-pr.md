## Summary

- The AffectGPT worker's GPU pin is now a GPU UUID from an env key (`AFFECTGPT_GPU_DEVICE_ID`), not the bare index `"1"`.
- The checked-in value is **empty on purpose**. Docker compose refuses to start the worker until Juniper picks a card. No card is chosen in this PR.
- The compose comment, `.env_example`, README and one docstring now say what is true today: index 1 is the agent llama-server's card, the SXM2 is index 3, and no circe card has room right now.
- A gate test refuses a bare index pin and requires the key to be empty or a UUID.

## Outcome moved

The worker can no longer land on whatever card currently has index 1. From 2026-10-14 03:00Z to 2026-10-23 22:27Z it did exactly that: CUDA OOM at model load, 601 restarts, exit 3, ~144 W for 9 days. With an empty pin, a pull-triggered rebuild on circe fails loudly for this one service instead of starting a new crash loop.

## Current architecture

`services/orion-affectgpt-worker/docker-compose.yml` pinned `device_ids: ["1"]`. The comment said that was the V100-SXM2-32GB (true on 2026-08-25). circe's numbering later changed, so index 1 became the V100-PCIE-32GB that holds the agent llama-server (~24.3 GB).

## Architecture touched

One service's compose GPU reservation and its docs. No bus, schema or code-path change.

## Files changed

- `services/orion-affectgpt-worker/docker-compose.yml`: `device_ids: ["${AFFECTGPT_GPU_DEVICE_ID:?...}"]`, comment with the incident, the circe UUID table, and the open placement note.
- `services/orion-affectgpt-worker/.env_example`: new `AFFECTGPT_GPU_DEVICE_ID=` (empty), header no longer claims GPU1.
- `services/orion-affectgpt-worker/README.md`: placement is open; why.
- `services/orion-affectgpt-worker/app/transcribe.py`: docstring no longer claims GPU1.
- `services/orion-affectgpt-worker/tests/test_gpu_pin.py`: gate (required env key, no bare index, value empty or UUID).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: compose refuses to render/start the worker until `AFFECTGPT_GPU_DEVICE_ID` is set.
- Compatibility notes: same pattern as `KEV_GPU_DEVICE_ID`, `VISION_HOST_GPU_DEVICE_ID`, `WHISPER_TTS_GPU_DEVICE_ID`.

## Env/config changes

- Added keys: `AFFECTGPT_GPU_DEVICE_ID` (empty)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-affectgpt-worker`: yes (athena copy; also added the pre-existing missing `AFFECTGPT_TRANSCRIBE_MAX_NO_SPEECH_PROB`)
- skipped keys requiring operator action: **circe's** `services/orion-affectgpt-worker/.env` needs `AFFECTGPT_GPU_DEVICE_ID=<UUID>` once Juniper picks a card. The worker needs more than 7.3 GB free; the README's 2026-08 measurement says ~18.4 GB peak. On 2026-10-23 the SXM2 (index 3, `GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a`) had ~12 GB free.

## Tests run

```text
pytest services/orion-affectgpt-worker/tests -q        -> 44 passed
pytest tests/test_compose_logging_journald.py -q       -> 2 passed
```

## Evals run

```text
None. This is a config pin; the gate test is the check. The crash-loop detection eval for this incident is in the companion PR (fix/guardian-remote-crash-loop).
```

## Docker/build/smoke checks

```text
docker compose --env-file <.env_example copy> -f services/orion-affectgpt-worker/docker-compose.yml config
  -> rc=1 "required variable AFFECTGPT_GPU_DEVICE_ID is missing a value: ... must be set to a circe GPU UUID"
same with AFFECTGPT_GPU_DEVICE_ID=GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a
  -> renders device_ids: [GPU-599d16a7-8c7c-1e3a-ac94-a01e5cb5cf7a]
NVIDIA runtime accepting the UUID at `up` on circe: UNVERIFIED (no circe shell from athena; documented NVIDIA behaviour).
```

## Review findings fixed

- Reviewed together with the companion guardian PR's review subagent scope note: this PR is config/docs only; no material findings specific to it.

## Restart required

```text
No restart required. Do not start the worker until a card is chosen.
When Juniper picks one, on circe: set AFFECTGPT_GPU_DEVICE_ID in services/orion-affectgpt-worker/.env, then
scripts/safe_docker_build.sh orion-affectgpt-worker up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a circe pull that touches this service now fails its rebuild step for this one service until the key is set.
- Mitigation: that is intended (no crash loop); the error names the key and the file.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
