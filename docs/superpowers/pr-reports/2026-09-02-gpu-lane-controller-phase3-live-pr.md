# PR report: Qwen3.8-27B agent-lane profile + live GPU1 flip (Phase 3, follow-on to #2038)

## Summary

- New `config/llm_profiles.yaml` entry `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`
  (Q4_K_XL, single V100-32GB, thinking-mode defaults + `reasoning_effort=xhigh`
  per Unsloth's own Qwen3.8 docs), wired to `ATLAS_AGENT_PROFILE_NAME` as the
  new default agent-lane model.
- Fixed `orion-gpu-lane-controller`'s Dockerfile: Debian trixie's `docker.io`
  apt package doesn't ship `/usr/bin/docker` at all, and neither
  `docker-compose-plugin` nor `docker-compose-v2` exist as apt package names
  on this base image — switched to static binaries pinned to circe's exact
  host versions (docker 29.4.1, compose v5.1.3), with sha256 verification.
- **Live-verified end to end on circe**: flipped GPU1 from affect-gpt to the
  new agent lane via the deployed controller itself (not a manual docker
  compose call), confirmed real model download + load + a working
  `/v1/chat/completions` call with correctly-structured `reasoning_content`.
- Fixed a real production misconfiguration this surfaced: `ATLAS_AGENT_HOST_PORT`'s
  default (8014) collides with `orion-circe-diffusion-host`'s own port mapping
  on circe — moved to 8015, and updated `orion-llm-gateway`'s live route table
  (was pointing real production `agent` traffic at the wrong/colliding port).
- Addressed 5 code-review findings (see below) — most load-bearing: the port
  fix above only updated the live circe `.env`, not the checked-in
  `.env_example`/compose default, which would have recreated the same
  collision on a fresh deploy.

## Outcome moved

GPU1 now actually runs the intended default (agent-lane, Qwen3.8-27B) instead
of sitting idle-but-pinned to affect-gpt. The `agent` LLM-gateway route, which
was silently broken (pointing at a port collision, `orion-atlas-llamacpp-agent`
had been exited for 12+ days before this) is now live and gateway-verified
(`GET /routes` reports `agent` `status: "up"`).

## Current architecture (before this patch)

See PR #2038's report for the base `orion-gpu-lane-controller` service. As of
that PR: `ATLAS_AGENT_PROFILE_NAME` was empty on circe (Muse Glimmer's line
commented out), `ATLAS_AGENT_CUDA_VISIBLE_DEVICES=2` (stale — that card,
PG500-216, is shared with bursty diffusion-host, not a stable home), and the
agent worker container had been sitting exited since 2026-08-14.

## Architecture touched

- `config/llm_profiles.yaml` — new profile only, nothing else changed.
- `services/orion-gpu-lane-controller/Dockerfile` — build-time fix.
- `services/orion-llamacpp-host/.env_example`, `docker-compose.atlas-workers.yml`,
  `README.md` — port default + doc fixes.
- `services/orion-llm-gateway/.env_example`, `README.md` — port + doc fixes.
- circe's live `services/orion-llamacpp-host/.env` (gitignored, not in this
  diff): `ATLAS_AGENT_PROFILE_NAME`, `ATLAS_AGENT_CUDA_VISIBLE_DEVICES=1`,
  `ATLAS_AGENT_HOST_PORT=8015`.
- athena's live `services/orion-llm-gateway/.env` (gitignored, not in this
  diff): `agent` route port 8014 → 8015.

## Files changed

- `config/llm_profiles.yaml`: new `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex` profile
- `services/orion-gpu-lane-controller/Dockerfile`: static docker CLI + compose
  plugin binaries (apt packages don't exist on this base image), checksum-verified
- `services/orion-llamacpp-host/.env_example`: `ATLAS_AGENT_HOST_PORT` 8014→8015
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: same port default fix
- `services/orion-llamacpp-host/README.md`: port + `.env.atlas`→`.env` doc fixes
- `services/orion-llm-gateway/.env_example`: `agent` route port fix + stale comment fix
- `services/orion-llm-gateway/README.md`: 3 stale port-8014 references fixed

## Schema / bus / API changes

- Added: `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex` LLM profile.
- Removed: none.
- Renamed: none.
- Behavior changed: `agent` LLM-gateway route now points at :8015 (was :8014,
  colliding with diffusion-host); serves the new Qwen profile instead of
  Muse Glimmer (which is still defined in `llm_profiles.yaml`, just not the
  default anymore).
- Compatibility notes: any other config/doc/script hardcoding port 8014 for
  the agent lane specifically (not diffusion-host's own use of that port,
  which is unrelated and untouched) needs the same update — searched
  `.env_example`/`README.md`/`docker-compose*.yml` repo-wide for it, nothing
  else found.

## Env/config changes

- Added keys: none new.
- Changed defaults: `ATLAS_AGENT_HOST_PORT` 8014→8015 in both
  `services/orion-llamacpp-host/.env_example` and its compose file's fallback.
- `.env_example` updated: yes, both `orion-llamacpp-host` and `orion-llm-gateway`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  N/A for the live fixes (circe's and athena's `LLM_GATEWAY_ROUTE_TABLE_JSON`
  changes are per-host live values already applied by hand and live-verified
  — the sync script can't reach circe at all, and athena's own live `.env`
  was hand-edited directly and confirmed via `GET /routes`).
- Skipped keys requiring operator action: none new beyond what #2038 already
  named (the `GPU_LANE_CONTROLLER_TOKEN` setup).

## Tests run

```text
PYTHONPATH=/mnt/scripts/Orion-Sapienform-circe-gpu1-lane-flex \
  /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-gpu-lane-controller/tests -q
21 passed, 18 warnings (pre-existing, unrelated)

PYTHONPATH=. .venv/bin/python scripts/check_service_env_compose_parity.py orion-llamacpp-host
-> N/A (declares env_file:, no compose-level overrides to drift)

PYTHONPATH=. .venv/bin/python scripts/check_service_env_compose_parity.py orion-llm-gateway
-> N/A (same)

git diff --check -> clean
```

New profile was schema-validated against the real production
`LLMProfileRegistry` loader (not a hand-rolled check) before deploy:

```text
cd services/orion-llamacpp-host && LLM_PROFILES_CONFIG_PATH=.../config/llm_profiles.yaml \
  LLM_PROFILE_NAME=qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex \
  .venv/bin/python -c "from app.settings import Settings; Settings().load_profile_registry().get('qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex')"
-> parsed OK; 19/21 profiles loaded (2 non-llamacpp backends correctly skipped, pre-existing)
```

## Evals run

None — same as #2038, no eval harness exists for this control-plane service.
The new LLM profile's actual generation quality is exercised only by the live
smoke test below, not a formal eval; flagging as a gap if Juniper wants
ongoing quality tracking for this lane.

## Docker/build/smoke checks

**Full live deploy + flip on circe**, not simulated:

```text
POST http://localhost:8090/v1/gpu-lane/flip {"target":"agent"}
-> stopped orion-circe-affectgpt-worker (exit 0)
-> docker compose build atlas-agent (first real build: ~7GB ghcr.io/ggml-org/
   llama.cpp:server-cuda-b10398 pull, never cached before -- needed
   GPU_LANE_COMMAND_TIMEOUT_SEC raised from 900s to 1800s; two earlier
   attempts hit real infra issues along the way, both fixed live:
   port 8014 collision (see above), and the .env.atlas/.env doc bug from
   #2038 re-confirmed)
-> docker compose up -d atlas-agent
-> downloaded unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf (17.56GB,
   confirmed on disk, matches Unsloth's 16-19GB Q4 estimate)
-> nvidia-smi confirmed GPU1 (index 1) at 18.4GB used
-> docker healthcheck flipped to "healthy"
-> POST /v1/chat/completions "Say hello in exactly five words." ->
   "Hello there, how are you?" (5 words) with reasoning_content populated
   (confirms --reasoning-format deepseek + chat_template_kwargs.
   reasoning_effort=xhigh both actually work on this model/build,
   system_fingerprint "b10398-8e7f22b67" confirms the required image tag)
```

```text
GET http://localhost:8210/routes (orion-llm-gateway, athena)
-> "agent": {"status": "up", "model": "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"}
   (all 7 routes report up)
```

GPU1 was left on `agent` as the live default per the intended design — not
reverted back to `affect`.

## Review findings fixed

- Finding: `services/orion-llamacpp-host`'s `ATLAS_AGENT_HOST_PORT` default
  (8014) was fixed on circe's live `.env` but not in the checked-in
  `.env_example` or the compose file's `${...:-8014}` fallback — a fresh
  deploy from these templates would recreate the exact port collision this
  patch fixes.
  - Fix: both defaults moved to 8015.
  - Evidence: `services/orion-llamacpp-host/.env_example`,
    `docker-compose.atlas-workers.yml`.
- Finding: stale "port 8014" comment in `orion-llm-gateway/.env_example`,
  contradicting the corrected value 3 lines below.
  - Fix: comment updated, also refreshed the stale "Muse Glimmer is live"
    claim.
  - Evidence: `services/orion-llm-gateway/.env_example`.
- Finding: README examples in both services still documented port 8014 (6
  occurrences total), and `orion-llamacpp-host/README.md`'s quickstart still
  told operators to use `.env.atlas` (confirmed live that file doesn't
  exist).
  - Fix: all updated.
  - Evidence: `services/orion-llm-gateway/README.md`,
    `services/orion-llamacpp-host/README.md`.
- Finding: static docker CLI + compose plugin binaries in the Dockerfile had
  no checksum verification, unlike `orion-hub/Dockerfile`'s GPG-verified apt
  install of the same binaries.
  - Fix: added `sha256sum -c` verification for both, checksums computed
    locally via `curl` + `sha256sum` against the exact fetch URLs (not read
    off any webpage) — TOFU for the CLI tarball (Docker publishes no
    checksums file for the static x86_64 build), cross-checked exactly
    against compose's own published `.sha256` sidecar (matched byte-for-byte).
    Rebuilt on circe afterward to confirm the pinned checksums are actually
    correct, not just present.
  - Evidence: `services/orion-gpu-lane-controller/Dockerfile`; rebuild output
    (`Image ... Built`, no `sha256sum -c` failure).
- Finding: pinned docker/compose versions could silently drift from circe's
  host if this image isn't rebuilt after a host docker upgrade — no
  detection mechanism.
  - Fix: **not fixed** — documented as an accepted, known gap in the
    Dockerfile comment rather than left unmentioned. A real drift-detection
    check is a separate, larger patch.
  - Evidence: `services/orion-gpu-lane-controller/Dockerfile` comment above
    the version pins.

## Restart required

Already done live during this session (not a pending instruction):
- `orion-circe-gpu-lane-controller` rebuilt + recreated on circe (checksum fix)
- `orion-atlas-llamacpp-agent` (`orion-circe-atlas-llamacpp-agent`) brought up
  on circe, healthy, serving the new profile
- `orion-llm-gateway` recreated on athena with the corrected route table

No further restart needed for this PR's changes.

## Risks / concerns

- Severity: low
  - Concern: Docker publishes no official checksum for the static x86_64 CLI
    tarball, so that one pin is TOFU against this session's own fetch, not a
    publisher-verified value.
  - Mitigation: the compose binary's checksum WAS verified against its
    official sidecar and matched exactly, giving some confidence in the fetch
    path's integrity; documented the TOFU distinction explicitly in the
    Dockerfile rather than overstating the guarantee.
- Severity: low
  - Concern: version-drift detection gap (see review finding 5) — genuinely
    unfixed.
  - Mitigation: none yet; flagged for a future follow-up if it matters enough
    to build.
- Severity: medium (pre-existing, not introduced by this patch, but newly
  visible)
  - Concern: circe's `orion-atlas-llamacpp-agent` had been silently exited
    for 12+ days before tonight — the `agent` LLM-gateway route was
    effectively dead in production with no alerting that caught it.
  - Mitigation: out of scope for this PR (no health/liveness alerting patch
    included); worth a follow-up if Juniper wants proactive detection of a
    dead LLM lane rather than discovering it manually.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2042
