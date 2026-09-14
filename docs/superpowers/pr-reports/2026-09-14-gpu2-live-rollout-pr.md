# GPU2 live rollout and cabinet endpoint correction

The merged GPU2 controller and diffusion drain API are now deployed on Circe.
The first physical round-trip exposed a deployment error: Hub uses host
networking, so durable-runs cannot resolve `orion-athena-hub` through bridge DNS.
Use Athena's Tailscale address for cabinet readings in settings, Compose, the
operator template and the actual production env. This restores real thermal
eligibility without changing any activation or safety gate.

The architecture document now correctly states that GPU2 control uses the
internal service/tailnet boundary without bearer tokens; GPU1 retains its
existing authentication and callers cannot supply arbitrary Docker arguments.

## Deployment and runtime evidence

- Source baseline on Athena and Circe: `e642932d3`, including merges #2215/#2216.
- Local envs synchronized on both hosts, including Circe's six missing GPU2
  keys and Athena's explicit lane policy. All requested enable flags remain
  true; both shadow flags are false; widening remains 1200 seconds.
- Installed missing additive `reverie_visual_attempt` and
  `visual_baseline_checkpoint` schema; confirmed all admission, lease, capacity
  and GPU2 tables exist in the configured production database.
- Real baseline production: chain `385a652b-8d03-477d-90ea-570f08e897d8`,
  831679-byte image, SHA256
  `582019276046ed14be359f92837f1765873cafe5f8d51a2b61a52ceb902f2000`,
  produced `2026-09-14T02:23:26.108336Z`. This cleared overdue baseline debt.
- Agent capability audit through Gateway produced `{"ok":true}` under an
  enforced JSON schema. Live model is
  `/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf`, context 131072, vision false.
- Manual durable intent `gpu2:1:agent-burst` at `02:23:43.279919Z` drained and
  stopped diffusion, loaded burst on GPU2 and reached controller readiness in
  49.37 seconds. Gateway subsequently advertised the exact model/context.
- Unleased burst call returned HTTP 503,
  `agent_burst_requires_durable_capacity_lease`.
- Durable intent `gpu2:2:diffusion` at `02:24:55.762490Z` restored diffusion
  after zero leases/permits and an idle upstream slot. Controller transition
  25.82 seconds, diffusion cold start 14.46 seconds; `/ready` returned true.
- Automatic acceptance run `gpu2-live-20260914T022620` was submitted at
  `02:26:21.287170Z` behind a real streamed Gateway inference on agent. It is
  parked at `resource_wait`, has no lease, and holds no durable graph worker.
  The real 1200-second observation is in progress; no backdating or manual
  assignment was used. Final automatic evidence will be added after restoration.

## Checks and review

- 15 focused elastic API/policy tests and the cabinet deployment-contract
  regression pass. Regression checks the host-network topology and agreement
  of all cabinet defaults on a reachable tailnet address.
- All affected services built and deployed through `safe_docker_build.sh` in
  linked worktrees; primary ignored envs were synchronized.
- Independent code review: no material findings. Added the recommended
  cabinet-address contract regression.
- Sanitized detailed runtime snapshots are retained under
  `/tmp/gpu2-live-evidence` on Athena. They contain no bearer credentials.

World Pulse's reading queue is not integrated with this admission seam.
