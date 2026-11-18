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
pytest services/orion-affectgpt-worker/tests -q        -> 43 passed
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

Review ran in a subagent (manual; no code-review skill available). No must-fix findings. It checked that a compose refusal for this one service does not abort `rebuild_services_from_git_diff.sh`, `up_all_services*.sh`, or any CI job.

- Finding: the memory need was stated three ways (">7.3 GB", "~15 GB", "~18.4 GB peak").
  - Fix: one figure everywhere: ~18.4 GB peak (2026-08 README measurement); 7.3 GB is "the OOM hit with 7.3 GB free".
- Finding: the empty pin also blocks `docker compose stop/down/logs` for this service and does not stop an existing container.
  - Fix: compose comment and this report name the working command (`docker rm -f orion-circe-affectgpt-worker`, or set the variable for one command).
- Finding: rebuild noise on circe is wider than "pulls touching this service" (shared `orion/` changes also select it).
  - Fix: stated under Risks.
- Finding (nit): the bare-index regex was easy to bypass.
  - Fix: the test parses the compose YAML and requires every `device_ids` entry to be the required env key.
- Finding (nit): the UUID table's role labels will go stale.
  - Fix: role labels dropped; the table is marked as a snapshot to re-check with `nvidia-smi -L`.
- Not changed: `orion-kev` still pins by bare index (`KEV_GPU_DEVICE_ID=3`) and can fail the same way. Follow-up, not this PR.

## Restart required

```text
No restart required. Do not start the worker until a card is chosen.
When Juniper picks one, on circe: set AFFECTGPT_GPU_DEVICE_ID in services/orion-affectgpt-worker/.env, then
scripts/safe_docker_build.sh orion-affectgpt-worker up -d --build
```

## Risks / concerns

- Severity: low
- Concern: until the key is set on circe, every circe post-pull rebuild that selects this worker prints `FAILED orion-affectgpt-worker` and exits 1 at the end. It is selected by its own files AND by shared `orion/bus/`, `orion/schemas/` changes, so this will be frequent noise that can hide a real failure. Other services still rebuild.
- Mitigation: intended (no crash loop); the error names the key and file. If the noise is unwanted before a card is picked, drop `orion-affectgpt-worker` from `mesh-utilities/common/include_services_circe.txt` and restore it with the UUID. Left for Juniper.
- Severity: medium
- Concern: the empty pin does not stop a container that already exists. How the worker was stopped at 22:27Z and whether its container still exists on circe is UNVERIFIED from here.
- Mitigation: on circe, `docker ps -a --filter name=affectgpt`; remove with `docker rm -f orion-circe-affectgpt-worker` (compose commands for this service refuse while the key is empty).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2894 (companion: https://github.com/junebug-junie/Orion-Sapienform/pull/2895)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
