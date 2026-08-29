# Whisper-TTS: pin the GPU by host index (circe P100 move, part 1)

Branch: `chore/whisper-tts-circe-p100`

## Summary

- Replaced `gpus: all` / `count: all` in this service's compose with an
  explicit `device_ids` pin read from a new `WHISPER_TTS_GPU_DEVICE_ID`.
- Prep for moving the service off athena's shared 8GB P4 onto circe's
  **Tesla P100-PCIE-16GB**. **No behavior change on athena** (default `0`).
- Documented the cutover, the one-time root prep on circe, and how to
  confirm a GPU pin correctly.
- **The move itself is NOT done in this PR** — it needs a root-owned
  directory created on circe. See "Remaining work".

## Outcome moved

Nothing at runtime yet. What changes is that the service *can* be placed on
a specific card at all: with `gpus: all`, torch picks `cuda:0`, which on
circe is a V100 already carrying other lanes — the P100 would never have
been used no matter what the deploy claimed.

## Current architecture

`services/orion-whisper-tts` runs XTTS-v2 (TTS) and Whisper (STT) in one
container on athena, container `orion-athena-whisper-tts`, port 7800.
It is bus-addressed on `orion:tts:intake` / `orion:stt:intake`; **no code
anywhere in the repo references port 7800 outside this service**, so its
host placement is not pinned by any consumer.

Compose declared both `gpus: all` and
`deploy.resources.reservations.devices` with `count: all`.

## Architecture touched

One compose file, one new env key, README. No Python, no bus channel, no
schema.

## Why one compose file, not `docker-compose.circe.yml`

`services/orion-vision-host/docker-compose.circe-qwen.yml` is the existing
precedent for a circe lane, and it is deliberately *not* followed here.
That file exists for a second, parallel, differently-configured instance
running **alongside** athena's, which is why it also needed its own
isolated bus channel.

This is a **move**. TTS intake dispatch is Redis **pub/sub**
(`app/tts_worker.py:232`, `bus.subscribe(settings.channel_tts_intake)`),
not a consumer group, so two live instances would both receive every
request, both synthesize, and both reply on the same reply channel with the
caller keeping whichever landed first. That is the same class of incident
already recorded for vision-host in PR #1859/#1860. Exactly one instance
runs; the only host-specific value is the GPU index; and a duplicated
~100-line compose file would drift — which this very service's README and
`.env_example` already did once (fixed in PR #1956).

## Files changed

- `services/orion-whisper-tts/docker-compose.yml`: `gpus: all` + `count:
  all` -> `device_ids: ["${WHISPER_TTS_GPU_DEVICE_ID:-0}"]`.
- `services/orion-whisper-tts/.env_example`: new key + why the value is a
  host index and must not be turned into `cuda:4`.
- `services/orion-whisper-tts/README.md`: "Running on circe (P100 lane)" —
  single-instance warning, verified card table, root prep, staging, cutover
  order, how to confirm a pin.
- `config/metrics/metric_definitions.lock.json`: routine per-branch re-lock
  (0 definitions changed; the header otherwise carries an inherited
  `_last_change` and fails the gate).

## Circe GPU inventory, verified live 2026-08-29

Prior notes in this environment said circe had **six** GPUs with the P100 at
index **3**. Live `nvidia-smi` says **seven**, P100 at index **4**:

```text
0 Tesla V100-PCIE-32GB    23272/32768 MiB
1 Tesla V100-SXM2-32GB    18447/32768 MiB
2 Tesla PG500-216         25046/32768 MiB
3 Tesla V100-PCIE-32GB     8530/32768 MiB
4 Tesla P100-PCIE-16GB     6663/16384 MiB  <- target, ~9.7GB free
5 Tesla V100-PCIE-16GB     8052/16384 MiB
6 Tesla V100-PCIE-16GB     7340/16384 MiB
```

GPU 4's current occupant is `orion-circe-circe-vision-host-qwen`. XTTS needs
~2.4GB, so it fits with room to spare. This is why the README says to verify
the card rather than trust any table, including its own.

## Schema / bus / API changes

None. Added/Removed/Renamed: none. Behavior changed: none.

## Env/config changes

- Added key: `WHISPER_TTS_GPU_DEVICE_ID` (default `0`)
- `.env_example` updated: yes
- local `.env` synced: **by hand**, see below
- skipped keys requiring operator action: none

The sync script silently skipped this key. `WHISPER_TTS_` is not in
`should_sync_key()`'s prefix allowlist and `orion-whisper-tts` is not in
`DEFAULT_SERVICES` on `main`, so `sync_local_env_from_example.py
orion-whisper-tts` printed "No changes needed" while adding nothing. **PR
#1956 fixes exactly this** and is not merged yet; this branch is cut from
`main`, so it does not have the fix. Second live occurrence of that
false-green in one session. Added to athena's `.env` by hand and confirmed
with `--all-keys`.

## Tests run

```text
cd services/orion-whisper-tts && PYTHONPATH=. python -m pytest tests -q
  57 passed
```

## Evals run

```text
none -- services/orion-whisper-tts still has no evals/ directory (carried
over from PR #1956; not introduced here).
```

## Docker/build/smoke checks

Both directions of the new pin, not just the default:

```text
scripts/safe_docker_build.sh orion-whisper-tts config
  -> device_ids: ["0"]

WHISPER_TTS_GPU_DEVICE_ID=4 scripts/safe_docker_build.sh orion-whisper-tts config
  -> device_ids: ["4"]
```

athena recreated under the new form, to prove rollback before moving off it:

```text
docker inspect ... HostConfig.DeviceRequests
  -> [{"Driver":"nvidia","DeviceIDs":["0"],"Capabilities":[["gpu"]]}]
/health -> ok, cuda_available true
live bus synthesis -> synthesis_ms=3693,
                      speaker_wav_basename=orion_reference_v2.wav
```

circe readiness probed live: bus reachable, port 7800 free, docker 29.4.1
with the nvidia runtime, repo checkout present, `/mnt/telemetry` a 388G
NVMe with 190G free.

## Remaining work (blocked on root)

`/mnt/telemetry/models` on circe is root-owned and circe's `sudo` requires a
password, so this cannot be done from an agent session:

```bash
sudo mkdir -p /mnt/telemetry/models/coqui/tts /mnt/telemetry/models/coqui/voices
sudo chown -R circe:circe /mnt/telemetry/models/coqui
```

`/mnt/telemetry` is chosen over `/mnt/storage-warm` (where circe's other
model dirs live) deliberately: `/mnt/storage-warm` is a directory on the
root LV at **91% full, 7.7G free**, and a 2GB model cache does not belong
there. `/mnt/telemetry` also matches athena's path exactly, so the compose
volume lines need no change at all.

After that: stage ~2GB of weights + the reference voice, set
`WHISPER_TTS_GPU_DEVICE_ID=4` in circe's service `.env`, build, then cut
over (stop athena, start circe) and verify.

## Restart required

Already applied on athena during verification. To roll back this PR's form:

```bash
scripts/safe_docker_build.sh orion-whisper-tts up -d --force-recreate --no-build
```

## Risks / concerns

- **Severity: low.** Concern: `device_ids` is a host-index pin, so if
  circe's enumeration changes again (it already has once) the service lands
  on the wrong card silently — torch will still report `cuda:0`. Mitigation:
  README requires confirming from the host via `docker inspect` DeviceIDs +
  `nvidia-smi`, never from a container log line. A UUID-based pin would be
  immune but is not supported by this compose schema field.
- **Severity: medium.** Concern: moving TTS also moves **STT** — one
  container serves both. Both are bus-addressed so nothing breaks by
  reference, but circe becomes a hard dependency for Orion hearing *and*
  speaking. Mitigation: rollback is starting athena's container again.
- **Severity: low.** Concern: this branch does not carry PR #1956's
  env-sync gate fix, so `WHISPER_TTS_GPU_DEVICE_ID` is not covered by the
  default sync until #1956 merges and `main` is merged back in.

## Status

DONE_WITH_CONCERNS for the config change; the relocation itself is
**BLOCKED** on the root command above.
