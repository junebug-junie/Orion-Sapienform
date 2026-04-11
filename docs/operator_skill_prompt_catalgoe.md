# Operator Skill Prompt Catalogue

## Purpose

- This is the operator-facing prompt catalogue for live Orion skills.
- These prompts are intended to be run in **agent** mode.
- Results depend on runtime dependencies and service availability.
- The planner may route through semantic verbs, but these prompts are the canonical operator entry points.

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

## Landing pad

9. Show the landing pad metrics snapshot.

   - Intended concrete skill: `skills.landing_pad.metrics_snapshot.v1`
   - Expected result: landing pad metrics snapshot

10. Show the last 10 landing pad events.

    - Intended concrete skill: `skills.landing_pad.last_events.v1`
    - Expected result: recent landing pad events

11. Show the last 10 landing pad events with salience above 0.7.

    - Intended concrete skill: `skills.landing_pad.last_events.v1`
    - Expected result: filtered recent landing pad events

## Mesh / storage / repo

12. Which nodes are up right now?

    - Intended concrete skill: `skills.mesh.tailscale_mesh_status.v1`
    - Expected result: active mesh nodes

13. Run an active mesh probe.

    - Intended concrete skill: `skills.mesh.tailscale_mesh_status.v1`
    - Expected result: mesh status plus active probe if supported

14. Check disk health across active nodes.

    - Intended concrete skill: `skills.storage.disk_health_snapshot.v1`
    - Expected result: disk-health snapshot or precise unsupported-device reasons

15. Summarize recent PR changes.

    - Intended concrete skill: `skills.repo.github_recent_prs.v1`
    - Expected result: recent merged PR digest

## Round-up skill

16. Run a mesh ops round.

    - Intended concrete skill: `skills.mesh.mesh_ops_round.v1`
    - Expected result: mesh/storage/repo operational summary

17. Run a mesh ops round with PR digest and disk health.

    - Intended concrete skill: `skills.mesh.mesh_ops_round.v1`
    - Expected result: ops round including PR and disk-health coverage

18. Run a mesh ops round including docker housekeeping preview.

    - Intended concrete skill: `skills.mesh.mesh_ops_round.v1`
    - Expected result: ops round including docker housekeeping preview

## Notify

19. Send a notification to operators saying "test alert from Orion".

    - Intended concrete skill: `skills.system.notify_chat_message.v1`
    - Expected result: notify request dispatched or precise policy/runtime failure

## Notes

- These prompts are examples, not the only valid phrasings.
- Skill reachability depends on semantic routing and pack exposure.
- Runtime dependencies matter:
  - `nvidia-smi` (and GPU device access) for GPU
  - Docker CLI / engine API for Docker
  - Tailscale binary/socket for mesh
  - Biometrics service for biometrics
  - Notify service for notifications
  - Landing pad RPC/service for landing pad
- Prefer precise failure messages over generic refusal.
