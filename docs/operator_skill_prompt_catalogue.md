# Operator Skill Prompt Catalogue

## Purpose

- This is the operator-facing prompt catalogue for live Orion skills.
- From **Hub Skill Runner**, matching prompts are dispatched as **explicit `skills.*` verbs** (direct cortex-exec execution), independent of Quick Chat / `chat_quick` or other conversational lanes.
- Autonomous Orion may still reach the same concrete skills via planner / agent-chain / supervisor tool choice; this document does not require changing those paths.
- Results depend on runtime dependencies and service availability.
- The planner may route through semantic verbs when the operator is not using the Skill Runner catalogue shortcuts above.

## System / runtime

1. What time is it right now?

   - Intended concrete skill: `skills.system.time_now.v1`
   - Expected result: local and UTC time info

2. Show NVIDIA GPU status on this node.

   - Intended concrete skill: `skills.gpu.nvidia_smi_snapshot.v1`
   - Expected result: GPU inventory / memory / utilization, or precise runtime dependency failure

3. Show Docker container status on this node.

   - Intended concrete skill: `skills.docker.ps_status.v1`
   - Expected result: Docker container inventory for current node

4. Dry-run cleanup of stopped containers.

   - Intended concrete skill: `skills.runtime.docker_prune_stopped_containers.v1`
   - Expected result: preview mode; no mutation

5. Prune stopped containers.

   - Intended concrete skill: `skills.runtime.docker_prune_stopped_containers.v1`
   - Expected result: execute mode; actual prune if policy/environment allow it

## Biometrics

6. Show the current biometrics snapshot.

   - Intended concrete skill: `skills.biometrics.snapshot.v1`
   - Expected result: current biometrics snapshot

7. Show the 10 most recent biometrics readings.

   - Intended concrete skill: `skills.biometrics.raw_recent.v1`
   - Expected result: recent biometrics rows

8. Show the most recent biometrics readings for athena.

   - Intended concrete skill: `skills.biometrics.raw_recent.v1`
   - Expected result: recent biometrics rows filtered to athena

## Mesh / storage / repo

9. Which nodes are up right now?

   - Intended concrete skill: `skills.mesh.tailscale_mesh_status.v1`
   - Expected result: active mesh nodes

10. Run an active mesh probe.

    - Intended concrete skill: `skills.mesh.tailscale_mesh_status.v1`
    - Expected result: mesh status plus active probe if supported

11. Check disk health across active nodes.

    - Intended concrete skill: `skills.storage.disk_health_snapshot.v1`
    - Expected result: disk-health snapshot or precise unsupported-device reasons

12. Summarize recent PR changes.

    - Intended concrete skill: `skills.repo.github_recent_prs.v1`
    - Expected result: recent merged PR digest

## Round-up skill

13. Run a mesh ops round.

    - Intended concrete skill: `skills.mesh.mesh_ops_round.v1`
    - Expected result: mesh/storage/repo operational summary

14. Run a mesh ops round with PR digest and disk health.

    - Intended concrete skill: `skills.mesh.mesh_ops_round.v1`
    - Expected result: ops round including PR and disk-health coverage

15. Run a mesh ops round including docker housekeeping preview.

    - Intended concrete skill: `skills.mesh.mesh_ops_round.v1`
    - Expected result: ops round including docker housekeeping preview

## Notify

16. Send a notification to operators saying "test alert from Orion".

    - Intended concrete skill: `skills.system.notify_chat_message.v1`
    - Expected result: notify request dispatched or precise policy/runtime failure

## Chat history (SQL discussion window)

These prompts target the **read-only** bounded SQL skill `skills.chat.discussion_window.v1` (persisted `chat_history_log`, `created_at` window, contiguous suffix). They are **not** semantic recall and do **not** use `session_id` as the selection key. Cortex Exec must have `DATABASE_URL` or `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL` configured.

17. Run skills.chat.discussion_window.v1 on chat_history_log with lookback_seconds 3600 and max_turns 30 (optional filters: current user_id and hub source).

    - Intended concrete skill: `skills.chat.discussion_window.v1`
    - Expected result: `window_start_utc` / `window_end_utc`, `turn_count`, `transcript_text`, `selection_strategy`; or empty window / DB unavailable message

18. Run skills.chat.discussion_window.v1 on chat_history_log with lookback_seconds 86400 and max_turns 30 (optional filters: current user_id and hub source).

    - Intended concrete skill: `skills.chat.discussion_window.v1`
    - Expected result: same as (17) with a 24-hour lookback

## Mesh service scripts

19. Bring up all Docker services via mesh-utilities.

    - Intended concrete skill: `skills.mesh.up_all_services.v1`
    - Expected result: `mesh-utilities/common/up_all_services.sh` output or precise policy/runtime failure; requires `SKILLS_ALLOW_MESH_SERVICE_SCRIPTS=true` on cortex-exec and repo root (`ORION_REPO_ROOT` when needed)

20. Refresh service environment files via mesh-utilities.

    - Intended concrete skill: `skills.mesh.refresh_service_envs.v1`
    - Expected result: `mesh-utilities/common/refresh_service_envs.sh` output or precise policy/runtime failure; same policy flag as (19)

## Container bring-up

Not part of the numbered catalogue above and not dispatched through the Skill Runner `<select>` / exact-prompt
match at all. `skills.docker.compose_service_bringup.v1` takes a per-service argument (`skill_args.service`), and
`resolve_skill_runner_catalogue_verb()` only maps an exact prompt string to a no-args verb -- there is no
skill_args pass-through on that dispatch path. Instead, the Hub UI's "Container bring-up" panel (bottom of the
operator controls) POSTs directly to a small dedicated endpoint, `POST /api/debug/container-bringup`
(`services/orion-hub/scripts/api_routes.py`), which dispatches straight to cortex-exec via
`CortexChatRequest(verb=..., metadata={"skill_args": {...}})` -- bypassing the catalogue/chat pipeline entirely,
the same direct-dispatch shape `_execute_workflow_schedule_management` already uses for another metadata-carrying
skill in this file.

- Intended concrete skill: `skills.docker.compose_service_bringup.v1`
- Skill args: `{"service": "<services/ subdirectory name>"}`
- Runs, in order, inside cortex-exec: `docker compose --env-file .env --env-file services/<service>/.env -f services/<service>/docker-compose.yml build` then `... up -d`, then polls container health for up to `SKILLS_DOCKER_COMPOSE_BRINGUP_HEALTH_POLL_SEC` (default 60s, ~3s interval).
- **The service list is discovered live**, not a fixed numbered list like the rest of this document: both the
  skill (server-side, `_discover_compose_services()`) and the Hub dropdown (client-side, populated from the
  existing `GET /api/service-logs/services` endpoint) walk `services/*/docker-compose.yml` on every call. A new
  service directory with a compose file becomes selectable without any catalogue/doc edit.
- Requires `SKILLS_ALLOW_DOCKER_COMPOSE_BRINGUP=true` on cortex-exec (default `false` -- bigger blast radius than
  the two mesh-utilities scripts above: arbitrary per-service build/up, not a fixed allow-listed script).
- Requires the read-only host repo bind-mount (`ORION_HOST_REPO_ROOT` in `services/orion-cortex-exec/docker-compose.yml` /
  `.env_example`) to be live. Without it, `docker compose build` run from inside the container would build from the
  image's stale, build-time-baked repo snapshot instead of the current host code. This mount is **read-only** by
  design -- cortex-exec never gets write access to the repo through this skill.
- Result includes explicit path-accessibility diagnostics (`repo_root`, `repo_root_exists`, `services_dir_exists`,
  `compose_file_exists`) so a missing/stale mount fails loudly instead of a generic error.
- Per-container health is reported with `has_healthcheck` alongside `state`/`health` -- a container with no
  Dockerfile `HEALTHCHECK` reports `running_no_healthcheck` rather than being treated as unhealthy for lacking a
  status that will never arrive. A container that looks settled on one poll is re-confirmed once more after a
  short delay before being trusted, so a crash-loop caught mid-`running` window is not misreported as healthy.
- `orion-cortex-exec` and `orion-hub` are hard-denylisted server-side (`_DOCKER_COMPOSE_BRINGUP_DENYLIST` in
  `verb_adapters.py`) -- rebuilding the service serving this very request, or the Hub in front of it, is not
  allowed through this skill regardless of the policy flag. `POST /api/debug/container-bringup` has no per-caller
  auth (matching every other `/api/debug/*` route in this file today), so the Hub UI also asks for an explicit
  confirmation before firing.
- Build/up stdout+stderr tails are scrubbed for common secret-shaped substrings (GitHub tokens, Slack tokens,
  `Bearer ...` headers, AWS access key IDs) before being returned in the result -- best-effort, not a substitute
  for not leaking secrets into build output in the first place.

## Cognitive workflows (not deterministic skills)

These Hub Skill Runner entries use `data-workflow-id` and stay on the normal chat/workflow path. They are **not** in `SKILL_RUNNER_CATALOGUE_VERBS` (skills-only map).

- Compact the last 24 hours of chat into a memory digest. → workflow `chat_history_compactor_pass`
- Compact the last 6 hours of chat into a memory digest. → workflow `chat_history_compactor_pass`

## Notes

- These prompts are examples, not the only valid phrasings.
- Skill reachability depends on semantic routing and pack exposure.
- Runtime dependencies matter:
  - `nvidia-smi` (and GPU device access) for GPU
  - Docker CLI / engine API for Docker
  - Tailscale binary/socket for mesh
  - Biometrics service for biometrics
  - Notify service for notifications
  - Postgres / `DATABASE_URL` (or exec `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL`) for `skills.chat.discussion_window.v1`
  - `SKILLS_ALLOW_MESH_SERVICE_SCRIPTS=true` (and mounted repo / `ORION_REPO_ROOT`) for mesh service script skills (19–20)
  - `SKILLS_ALLOW_DOCKER_COMPOSE_BRINGUP=true` (and the read-only `ORION_HOST_REPO_ROOT` mount) for the Container bring-up skill above
- Prefer precise failure messages over generic refusal.
