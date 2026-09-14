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
- Initial contention probe `gpu2-live-20260914T022620` was submitted at
  `02:26:21.287170Z` behind real streamed Gateway inference on agent. Its wait
  held no durable graph worker. The ordinary request reached Gateway's real
  900-second deadline; the run correctly took the newly free preferred lane at
  `02:41:21.037049Z`. This is not a passing automatic GPU2 acceptance.
- A fresh test uses a temporary HTTP Gateway instance of the same production
  image and shared capacity authority with a 3600-second request budget. It
  does not consume bus requests or alter production activation/request flags.
  Automatic evidence will be added after the full real wait and restoration.

## FCC compatibility correction

The first preferred-lane run exposed a real native Anthropic failure: Claude
SessionStart hooks append a `system` message after the user message, causing
the Qwen template to return HTTP 500 (`System message must be at the beginning`).
Gateway now hoists those blocks into the top-level Anthropic system field,
preserving content, cache metadata and conversational/tool order. A captured
real Claude request reproduced the failure; the normalized request rendered
successfully against the same live model. The automatic run below is the
end-to-end inference evaluation. Gateway has no separate periodic eval harness.

## Review findings fixed

- Finding: malformed hook context could throw after acquiring a permit, before
  entering its cleanup block.
  - Fix: normalize and validate before capacity acquisition; reject unsupported
    system content with HTTP 400.
  - Evidence: both malformed-content regressions assert no capacity acquisition.

## Checks and review

- 21 Gateway Anthropic tests pass, including the captured hook-message shape,
  string/list system content, cache metadata, tool order and malformed content.
- 15 focused elastic API/policy tests and the cabinet deployment-contract
  regression pass. Regression checks the host-network topology and agreement
  of all cabinet defaults on a tailnet address. Reachability was verified by a
  live GET from inside the deployed durable-runs container.
- All affected services built and deployed through `safe_docker_build.sh` in
  linked worktrees; primary ignored envs were synchronized.
- Independent code review: no material findings. Added the recommended
  cabinet-address contract regression.
- Sanitized detailed runtime snapshots are retained under
  `/tmp/gpu2-live-evidence` on Athena. They contain no bearer credentials.

World Pulse's reading queue is not integrated with this admission seam.
