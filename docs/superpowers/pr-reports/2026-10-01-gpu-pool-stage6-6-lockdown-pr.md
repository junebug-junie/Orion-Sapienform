# chore(gpu-pool): stage 6.6 lockdown -- port gate, dead env key tool, circe firewall runbook

## Summary

The GPU pool's promise is "every LLM call waits in one queue". Until now nothing stopped a service
from calling a circe model server directly and skipping that queue, and every key the pool PRs
deleted was still sitting in the live `.env` files on both hosts. This PR closes both, without
deploying anything.

- **CI port gate** (`scripts/check_circe_worker_refs.py`, new step in `orion-static-gates`): fails
  on any circe worker address (100.112.254.99, the LAN addresses, `circe` / `circe.*` + a worker
  port, or a worker container name with any prefix and any port) anywhere in the repo outside
  the pool, the gateway, the actuator and the worker services. Ports are
  read from `config/gpu_pool.yaml` and the worker services' `*_HOST_PORT` keys, worker names from
  the seat compose files, so a new seat is covered with no edit. Every allow entry must match something, and so must every zone.
- **Dead env key tool** (`scripts/report_dead_env_keys.py`): lists keys in a live `.env` that are
  not in `.env_example` and that no code reads (comments and docstrings don't count). `--apply`
  writes `.env.bak.<UTC ts>` first and by default removes only the keys the pool PRs deleted
  (`KNOWN_DEAD`); the code-scan verdicts need `--include-heuristic`. It refuses to apply when the
  code tree is on a different commit than the checkout holding the `.env` files. Secret-named,
  NEVER_SYNC and image/library keys (`LLAMA_ARG_*`, `HF_*`, `CUDA_*`, ...) are never removed unless
  explicitly listed dead.
- **circe firewall** (`scripts/ops/circe_llm_port_gate.sh` + systemd unit, runbook
  `docs/runbooks/2026-10-01-circe-llm-port-firewall.md`): only athena, circe's own containers and
  loopback may open connections to the worker ports and the actuator. Rules sit in mangle
  PREROUTING, before docker's DNAT and out of the docker/tailscale FORWARD ordering race. Commands printed for
  Juniper; nothing applied.
- **CI job rename**: "Gateway — shared capacity and transport lifecycle" is now "Gateway — GPU pool
  dispatch and transport" (the capacity broker it named was deleted in 4.6/5.6).
- **SUPERSEDED headers** on the four stale gpu2/admission docs the stage 6 spec lists; parent spec
  status line updated (stage 6 is **not** done: 6.4, 6.5, 6.7 open).

## Outcome moved

- Acceptance check 5 of the stage 6 spec: the gate exits 0 on this tree and 1 on a planted
  `http://100.112.254.99:8011` in a service `settings.py` (both are tests that run in CI).
- Dead keys are no longer a hand-written list in PR reports: one read-only command per host
  produces them (athena: 54 pool-deleted keys + 123 other dead keys; circe: 16 + 7, below).

## Current architecture

- circe's llama.cpp workers (8011-8013, 8015, 8016), the diffusion host (8014), the bonsai worker
  (8017), the experiment seat (8099) and the lane controller (8090) are docker-published on
  `0.0.0.0`. Inspected read-only on circe: ufw's unit is active but `ENABLED=no`, no nftables /
  firewalld / iptables-persistent, so nothing filters these ports from any tailnet or LAN host.
- The only in-repo direct worker literal outside the pool was thought's diffusion URL (pool-leased
  since 5.4). Nothing enforced that it stayed that way.
- `scripts/sync_local_env_from_example.py` only adds keys; deleted keys stay forever.

## Architecture touched

CI and operator tooling only. No service code, no bus/schema change, no `.env_example` change, no
deploy.

## Files changed

- `scripts/check_circe_worker_refs.py`: the CI port gate.
- `tests/test_check_circe_worker_refs.py`: identity derivation, planted call fails, every address
  shape, comments/tests/zones ignored, stale allow/zone fail, real tree passes, real tree + planted
  `settings.py` fails.
- `scripts/report_dead_env_keys.py`: the dead env key tool.
- `tests/test_report_dead_env_keys.py`: classification (pydantic field names, aliases, env_prefix,
  getenv, prefix scans, compose interpolation, comment/docstring mentions are dead), known-dead
  conflict, report is read-only, apply backs up + removes only dead lines + is idempotent,
  `--known-only`, orphan `.env` never edited, JSON, and every hand-listed key from the 4.6/5.x/6.3
  reports is really unread in this tree.
- `scripts/ops/circe_llm_port_gate.sh`, `scripts/ops/orion-llm-port-gate.service`: the firewall.
- `tests/test_circe_llm_port_gate.py`: DRY_RUN output, firewall covers every port the CI gate
  guards, remove mirrors apply.
- `.github/workflows/orion-static-gates.yml`: gate step + the three test files.
- `.github/workflows/orion-durable-runs-tests.yml`: job display name only (job id `gateway`
  unchanged; main has no required status checks, checked via the branch protection API).
- `docs/runbooks/2026-10-01-circe-llm-port-firewall.md`: install / verify / rollback.
- `docs/architecture/gpu2-elastic-admission.md`, `docs/architecture/gpu2-elastic-evidence.md`,
  `docs/runbooks/gpu2-elastic-admission.md`, `docs/architecture/durable-resource-admission.md`:
  SUPERSEDED header (kept: they hold incident evidence).
- `docs/superpowers/specs/2026-09-24-gpu-pool-design.md`: status line.

## Port gate: what it found

- **Tracked code:** no offender. 6 hits, all allow-listed with reasons:
  - `services/orion-thought/app/settings.py` and `.env_example`: `100.112.254.99:8014`, the
    diffusion host, called only under a validated pool hold (stage 5.4).
  - `services/orion-cortex-exec/app/executor.py` (2 hits, `circe:8011` / `circe:8015`) and
    `services/orion-juniper-affective-state/app/vision_backend.py` (`circe:8011`): docstring prose
    explaining why those callers go through the gateway. No request is made.
- The durable-runs probes, situational context and the `scripts/gpu_pool_*` probe scripts had no
  direct worker literal, so they need no allow
  entry. `docs/` and `bench/` directories are excluded by design.
- **Live `.env` (report only, `--live-env`, athena):** 6 keys hold a direct worker address. 5 are
  dead keys (`DURABLE_RUNS_ELASTIC_CONTROLLER_URL`, `DURABLE_RUNS_ELASTIC_BACKEND`,
  `ORION_VISUAL_ELASTIC_CONTROLLER_URL`, `ORION_VISUAL_CHAIN_GPU2_CAPACITY_BACKEND_KEY`,
  `WM_GPU2_CAPACITY_BACKEND_KEY`); the 6th is thought's allowed `ORION_DIFFUSION_HOST_BASE_URL`.
  circe (grep, read-only): only dead keys (`GPU2_DIFFUSION_URL`, `GPU2_AGENT_URL`,
  `WM_GPU2_CAPACITY_BACKEND_KEY`). No LAN-address (192.168.1.22/.24) use anywhere.

## Dead env keys (read-only `--report`, 2026-10-01)

Not applied. CLAUDE.md and the task forbid editing live `.env` files from an agent session; the
`--apply` commands are below for Juniper.

### athena, `--known-only` (keys the GPU pool PRs deleted on purpose): 54

```text
dead env keys (report, read-only; KNOWN_DEAD only) env-root=/mnt/scripts/Orion-Sapienform
orion-context-exec  (/mnt/scripts/Orion-Sapienform/services/orion-context-exec/.env)
  dead      CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED
orion-cortex-exec  (/mnt/scripts/Orion-Sapienform/services/orion-cortex-exec/.env)
  dead      CORTEX_EXEC_LLM_GATEWAY_URL
orion-diffusion-host  (/mnt/scripts/Orion-Sapienform/services/orion-diffusion-host/.env)
  dead      DIFFUSION_POWER_INTENT_GPU_INDEX
orion-durable-runs  (/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env)
  dead      DURABLE_RUNS_LEASE_SECONDS
  dead      DURABLE_RUNS_CAPACITY_ENABLED
  dead      DURABLE_RUNS_ELASTIC_ENABLED
  dead      DURABLE_RUNS_ELASTIC_SHADOW
  dead      DURABLE_RUNS_ELASTIC_ASSIGNMENTS
  dead      DURABLE_RUNS_ELASTIC_RESTORATION
  dead      DURABLE_RUNS_ELASTIC_CONTROLLER_URL
  dead      DURABLE_RUNS_ELASTIC_BACKEND
  dead      DURABLE_RUNS_ELASTIC_DRAIN_BUDGET_SEC
  dead      DURABLE_RUNS_ELASTIC_TRANSITION_BUDGET_SEC
  dead      DURABLE_RUNS_ELASTIC_COLD_BUDGET_SEC
  dead      DURABLE_RUNS_ELASTIC_IDLE_GRACE_SEC
  dead      DURABLE_RUNS_ELASTIC_MIN_RESIDENCY_SEC
  dead      DURABLE_RUNS_ELASTIC_MAX_BORROW_SEC
  dead      DURABLE_RUNS_ELASTIC_CABINET_URL
  dead      DURABLE_RUNS_ELASTIC_THERMAL_ENABLED
  dead      DURABLE_RUNS_ELASTIC_CONTROLLER_TOKEN
orion-gpu-lane-controller  (/mnt/scripts/Orion-Sapienform/services/orion-gpu-lane-controller/.env)
  dead      GPU_LANE_CONTROLLER_TOKEN
  dead      GPU2_ENABLED
  dead      GPU2_DIFFUSION_URL
  dead      GPU2_AGENT_URL
  dead      GPU2_AUTHORITY_URL
  dead      GPU2_DRAIN_TIMEOUT_SEC
  dead      GPU2_MODEL_READY_TIMEOUT_SEC
orion-gpu-pool  (/mnt/scripts/Orion-Sapienform/services/orion-gpu-pool/.env)
  dead      GPU_POOL_VISUAL_ACTIVITY_URL
orion-hub  (/mnt/scripts/Orion-Sapienform/services/orion-hub/.env)
  dead      HUB_LLM_GATEWAY_URL
  dead      GPU_LANE_MAP_ATHENA_JSON
  dead      GPU_LANE_MAP_CIRCE_JSON
  dead      HUB_CURIOSITY_LEASE_VALIDATION_URL
  dead      HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED
orion-llm-gateway  (/mnt/scripts/Orion-Sapienform/services/orion-llm-gateway/.env)
  dead      LLM_GATEWAY_LEASE_VALIDATION_ENABLED
  dead      LLM_GATEWAY_LEASE_VALIDATION_URL
  dead      LLM_GATEWAY_LEASE_VALIDATION_TIMEOUT_SEC
  dead      LLM_GATEWAY_LEASE_CHECK_INTERVAL_SEC
  dead      LLM_GATEWAY_CAPACITY_ENABLED
  dead      LLM_GATEWAY_CAPACITY_URL
orion-thought  (/mnt/scripts/Orion-Sapienform/services/orion-thought/.env)
  dead      ORION_VISUAL_ELASTIC_STATUS_ENABLED
  dead      ORION_VISUAL_ELASTIC_CONTROLLER_URL
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_ENABLED
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_URL
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_BACKEND_KEY
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_LANE
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_MAX_INFLIGHT
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_BUDGET_SEC
  dead      ORION_VISUAL_CHAIN_GPU2_CAPACITY_POLL_INTERVAL_SEC
orion-world-model  (/mnt/scripts/Orion-Sapienform/services/orion-world-model/.env)
  dead      WM_GPU2_CAPACITY_ENABLED
  dead      WM_GPU2_CAPACITY_URL
  dead      WM_GPU2_CAPACITY_BACKEND_KEY
  dead      WM_GPU2_CAPACITY_LANE
  dead      WM_GPU2_CAPACITY_BUDGET_SEC
  dead      WM_GPU2_CAPACITY_POLL_INTERVAL_SEC
orphan .env files (service directory not in this code tree; never edited -- remove by hand if the service is really gone):
  /mnt/scripts/Orion-Sapienform/services/orion-agent-chain/.env
  /mnt/scripts/Orion-Sapienform/services/orion-landing-pad/.env
  /mnt/scripts/Orion-Sapienform/services/orion-planner-react/.env
  /mnt/scripts/Orion-Sapienform/services/orion-security-watcher/.env
  /mnt/scripts/Orion-Sapienform/services/orion-spark-introspector/.env
  /mnt/scripts/Orion-Sapienform/services/orion-vllm/.env
54 dead key(s) across 101 .env file(s); 85 file(s) clean; 6 orphan file(s).
```

### athena, everything else the full report finds: 123 more (177 total)

Every key below is absent from its service's `.env_example` and appears in no non-comment code in
that service, `orion/` or `config/`. Spot-checked by hand (e.g. `LLM_GATEWAY_ROUTE_TABLE_JSON`,
`LLM_ROUTE_HEALTH_TIMEOUT_SEC`, `DRIVE_DECAY_TAU_SEC`, `RECALL_CARDS_EMBED_TIMEOUT_SEC`,
`FIELD_DIGESTER_LLM_GATEWAY_URL`): each is mentioned only in comments, READMEs or PR reports.
"protected" = secret-named, listed, never auto-removed.

- `orion-context-exec`: CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED
- `orion-cortex-exec`: CHANNEL_AGENT_CHAIN_INTAKE, CHANNEL_PAD_RPC_REQUEST, CHANNEL_PAD_RPC_REPLY_PREFIX, CORTEX_EXEC_LLM_GATEWAY_URL, CORTEX_EXEC_ADMISSION_CUE_TIMEOUT_SEC, SELF_STUDY_NAMED_GRAPH, CORTEX_METACOG_ENRICH_PROMPT_MAX_CHARS, CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET, CHAT_STANCE_DRIVE_STATE_VISIBLE, CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC
- `orion-cortex-orch`: RECALL_TRANSPORT_RENDER_GATE_THRESHOLD
- `orion-curiosity-peer`: CURIOSITY_PEER_REPO_HOST_PATH
- `orion-diffusion-host`: DIFFUSION_POWER_INTENT_GPU_INDEX
- `orion-dream`: CHANNEL_DREAM_BUFFER, CHANNEL_DREAM_COMPLETE, CHANNEL_DREAM_STATUS, CORTEX_GATEWAY_REQUEST_CHANNEL, DREAM_REPLY_PREFIX, DREAM_VERB, CHANNEL_COLLAPSE_SQL_PUBLISH, CHANNEL_COLLAPSE_TAGS_PUBLISH, CHANNEL_TELEMETRY_PUBLISH, CHANNEL_CHAT, DREAM_INTROSPECT_ENABLED, DREAM_SEARCH_CHROMA_URL, DREAM_SEARCH_EMBED_URL, DREAM_SEARCH_COLLECTION, DREAM_SEARCH_MIN_SIMILARITY, DREAM_SEARCH_INDEX_INTERVAL_SEC, DREAM_SEARCH_INDEX_BATCH
- `orion-durable-runs`: DURABLE_RUNS_ADMISSION_SHADOW, DURABLE_RUNS_LEASE_SECONDS, DURABLE_RUNS_WIDENING_ENABLED, DURABLE_RUNS_WIDENING_AFTER_SEC, DURABLE_RUNS_WIDENING_HYSTERESIS_SEC, DURABLE_RUNS_LANE_POLICY_JSON, DURABLE_RUNS_GATEWAY_URL, DURABLE_RUNS_CAPACITY_ENABLED, DURABLE_RUNS_ELASTIC_ENABLED, DURABLE_RUNS_ELASTIC_SHADOW, DURABLE_RUNS_ELASTIC_ASSIGNMENTS, DURABLE_RUNS_ELASTIC_RESTORATION, DURABLE_RUNS_ELASTIC_CONTROLLER_URL, DURABLE_RUNS_ELASTIC_BACKEND, DURABLE_RUNS_ELASTIC_DRAIN_BUDGET_SEC, DURABLE_RUNS_ELASTIC_TRANSITION_BUDGET_SEC, DURABLE_RUNS_ELASTIC_COLD_BUDGET_SEC, DURABLE_RUNS_ELASTIC_IDLE_GRACE_SEC, DURABLE_RUNS_ELASTIC_MIN_RESIDENCY_SEC, DURABLE_RUNS_ELASTIC_MAX_BORROW_SEC, DURABLE_RUNS_ELASTIC_CABINET_URL, DURABLE_RUNS_ELASTIC_THERMAL_ENABLED, DURABLE_RUNS_ELASTIC_CONTROLLER_TOKEN
- `orion-equilibrium-service`: CHANNEL_PAD_SIGNAL, EQUILIBRIUM_METACOG_PAD_PULSE_THRESHOLD
- `orion-field-digester`: FIELD_DIGESTER_LLM_GATEWAY_URL
- `orion-gpu-lane-controller`: GPU_LANE_CONTROLLER_TOKEN, GPU2_ENABLED, GPU2_DIFFUSION_URL, GPU2_AGENT_URL, GPU2_AUTHORITY_URL, GPU2_DRAIN_TIMEOUT_SEC, GPU2_MODEL_READY_TIMEOUT_SEC
- `orion-gpu-pool`: GPU_POOL_VISUAL_ACTIVITY_URL
- `orion-hub`: LANDING_PAD_TIMEOUT_SEC, AGENT_CHAIN_REQUEST_CHANNEL, AGENT_CHAIN_RESULT_PREFIX, HUB_LLM_GATEWAY_URL, HUB_FCC_SERVER_AUTOSTART, HUB_FCC_SERVER_LOG_FILE, HUB_ROOM_CLAUDE_AUTO_RESPOND, HUB_ROOM_CLAUDE_AUTO_MIN_GAP_SEC, GPU_LANE_MAP_ATHENA_JSON, GPU_LANE_MAP_CIRCE_JSON, HUB_CURIOSITY_LEASE_VALIDATION_URL, HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED, HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC, HUB_WORLD_PULSE_READ_DAILY_CAP, HUB_WORLD_PULSE_READ_STAGE2_MIN_COOLDOWN_SEC, HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP
- `orion-llm-gateway`: ORION_LLM_VLLM_URL, ORION_LLM_OLLAMA_URL, ORION_LLM_LLAMA_COLA_URL, ATLAS_METACOG_SERVICE_NAME, LLM_GATEWAY_ROUTE_TABLE_JSON, LLM_ROUTE_CHAT_URL, LLM_ROUTE_METACOG_URL, LLM_ROUTE_LATENTS_URL, LLM_ROUTE_SPECIALIST_URL, LLM_ROUTE_CHAT_SERVED_BY, LLM_ROUTE_METACOG_SERVED_BY, LLM_ROUTE_LATENTS_SERVED_BY, LLM_ROUTE_SPECIALIST_SERVED_BY, LLM_ROUTE_HEALTH_TIMEOUT_SEC, LLM_ROUTE_SPARK_SERVED_BY, LLM_ROUTE_BACKGROUND_SERVED_BY, LLM_ROUTE_AGENT_SERVED_BY, LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK, LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC, LLM_GATEWAY_BACKGROUND_POLL_INTERVAL_SEC, LLM_GATEWAY_BACKGROUND_CONCURRENCY, LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT, LLM_GATEWAY_LEASE_VALIDATION_ENABLED, LLM_GATEWAY_LEASE_VALIDATION_URL, LLM_GATEWAY_LEASE_VALIDATION_TIMEOUT_SEC, LLM_GATEWAY_LEASE_CHECK_INTERVAL_SEC, LLM_GATEWAY_CAPACITY_ENABLED, LLM_GATEWAY_CAPACITY_URL
- `orion-recall`: RECALL_CARDS_EMBED_TIMEOUT_SEC, RECALL_CARDS_EMBED_CONCURRENCY, RECALL_CARDS_CANDIDATE_LIMIT, RECALL_CARDS_MAX_NEW_EMBEDS_PER_CALL, RECALL_VECTOR_COLLECTIONS
- `orion-room-companion`: protected ROOM_COMPANION_CLAUDE_CREDENTIALS_HOST_PATH
- `orion-signal-gateway`: ORION_VECTOR_HOST_URL
- `orion-spark-concept-induction`: BUS_DRIVE_STATE_OUT, BUS_TENSION_EVENT_OUT, BUS_DRIVE_AUDIT_OUT, BUS_GOAL_PROPOSAL_OUT, CONCEPT_CHAT_PG_LOOKUP_ENABLED, CONCEPT_CHAT_PG_LOOKUP_RETRIES, CONCEPT_CHAT_PG_LOOKUP_RETRY_DELAY_SEC, DRIVE_DECAY_TAU_SEC, DRIVE_SATURATION_GAIN, DRIVE_ACTIVATION_ON, DRIVE_ACTIVATION_OFF, GOAL_PROPOSAL_COOLDOWN_MINUTES, GOAL_GENERATION_MODE, GOAL_DRIVE_ORIGIN_SOURCE, ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE, ORION_HOMEOSTATIC_DRIVES_ENABLED, ORION_DRIVE_LEAKY_MATH_ENABLED, DEVIATION_EWMA_ALPHA, DEVIATION_Z_THRESHOLD, DEVIATION_SIGMA_FLOOR, SIGNAL_TENSION_IMPULSE_K, SIGNAL_TENSION_CAP_PER_WINDOW, SIGNAL_TENSION_WINDOW_SEC, HOMEOSTATIC_FAILURE_SEVERITY, ORION_ENDOGENOUS_ORIGINATION_ENABLED, ORIGINATION_WINDOW, ORIGINATION_THRESHOLD, ORIGINATION_COOLDOWN_SEC, ENDOGENOUS_MAG_CAP, ORIGINATION_W_DRIFT, ORIGINATION_W_DWELL, ORIGINATION_W_AGENCY, ORIGINATION_EXOGENOUS_FLOOR, protected CONCEPT_CHAT_PG_DSN
- `orion-substrate-runtime`: DRIVES_AUDIT_CHANNEL
- `orion-thought`: ORION_THOUGHT_MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC, ORION_VISUAL_CHAIN_MESH_CONTEXT_CHAR_LIMIT, ORION_VISUAL_CHAIN_MESH_CONTEXT_MAX_AGE_SEC, ORION_VISUAL_ELASTIC_STATUS_ENABLED, ORION_VISUAL_ELASTIC_CONTROLLER_URL, STANCE_REACT_AGENT_LANE_BUDGET_SEC, ORION_VISUAL_CHAIN_GPU2_CAPACITY_ENABLED, ORION_VISUAL_CHAIN_GPU2_CAPACITY_URL, ORION_VISUAL_CHAIN_GPU2_CAPACITY_BACKEND_KEY, ORION_VISUAL_CHAIN_GPU2_CAPACITY_LANE, ORION_VISUAL_CHAIN_GPU2_CAPACITY_MAX_INFLIGHT, ORION_VISUAL_CHAIN_GPU2_CAPACITY_BUDGET_SEC, ORION_VISUAL_CHAIN_GPU2_CAPACITY_POLL_INTERVAL_SEC
- `orion-vector-host`: VECTOR_HOST_CHAT_MESSAGE_COLLECTION, VECTOR_HOST_CHAT_TURN_COLLECTION
- `orion-vector-writer`: VECTOR_WRITER_CHAT_HISTORY_CHANNEL, VECTOR_WRITER_CHAT_COLLECTION
- `orion-vision-frame-router`: CHANNEL_EDGE_ACTIVITY_IN
- `orion-vision-scribe`: CHANNEL_RDF_ENQUEUE
- `orion-world-model`: WM_GPU2_CAPACITY_ENABLED, WM_GPU2_CAPACITY_URL, WM_GPU2_CAPACITY_BACKEND_KEY, WM_GPU2_CAPACITY_LANE, WM_GPU2_CAPACITY_BUDGET_SEC, WM_GPU2_CAPACITY_POLL_INTERVAL_SEC
- `<root>`: BUS_DRIVE_AUDIT_OUT, GOAL_PROPOSAL_COOLDOWN_MINUTES, GOAL_GENERATION_MODE

Orphan `.env` files (service directory no longer in the repo; reported, never edited):
`orion-agent-chain`, `orion-landing-pad`, `orion-planner-react`, `orion-security-watcher`,
`orion-spark-introspector`, `orion-vllm`.

### circe (ssh read-only, checkout `d3c09c9cb`, 2026-09-30): 16 known + 7 other = 23

```text
dead env keys (report, read-only; all) env-root=/mnt/scripts/Orion-Sapienform
orion-diffusion-host  (/mnt/scripts/Orion-Sapienform/services/orion-diffusion-host/.env)
  dead      DIFFUSION_POWER_INTENT_GPU_INDEX
orion-gpu-lane-controller  (/mnt/scripts/Orion-Sapienform/services/orion-gpu-lane-controller/.env)
  dead      GPU_LANE_CONTROLLER_TOKEN
  dead      GPU2_ENABLED
  dead      GPU2_DIFFUSION_URL
  dead      GPU2_AGENT_URL
  dead      GPU2_AUTHORITY_URL
  dead      GPU2_DRAIN_TIMEOUT_SEC
  dead      GPU2_MODEL_READY_TIMEOUT_SEC
  dead      GPU2_AUTHORITY
  dead      GPU2_POOL_FENCE_STATE_PATH
orion-world-model  (/mnt/scripts/Orion-Sapienform/services/orion-world-model/.env)
  dead      WM_GPU2_CAPACITY_ENABLED
  dead      WM_GPU2_CAPACITY_URL
  dead      WM_GPU2_CAPACITY_BACKEND_KEY
  dead      WM_GPU2_CAPACITY_LANE
  dead      WM_GPU2_CAPACITY_BUDGET_SEC
  dead      WM_GPU2_CAPACITY_POLL_INTERVAL_SEC
<root>  (/mnt/scripts/Orion-Sapienform/.env)
  dead      GRAPHDB_PORT
  dead      CHANNEL_RDF_ENQUEUE
  dead      CHANNEL_RDF_CONFIRM
  dead      CHANNEL_RDF_ERROR
  dead      CHANNEL_WORKER_RDF
  dead      BUS_DRIVE_AUDIT_OUT
  dead      GOAL_PROPOSAL_COOLDOWN_MINUTES
23 dead key(s) across 14 .env file(s); 10 file(s) clean; 0 orphan file(s).
```

### `--apply` commands [GO, Juniper]

Run each from the checkout that holds the `.env` files, after that checkout is pulled to a main
that includes this PR and the deploys that stopped reading these keys are live (a service's
rollback image may still read its own deleted keys). `--apply` touches only `KNOWN_DEAD` keys
unless `--include-heuristic` is given.

```bash
# athena
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
python3 scripts/report_dead_env_keys.py --known-only                  # review
python3 scripts/report_dead_env_keys.py --apply                       # pool keys; writes services/*/.env.bak.<ts>
python3 scripts/report_dead_env_keys.py                               # review the rest
python3 scripts/report_dead_env_keys.py --apply --include-heuristic   # only if the rest looks right

# circe
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && python3 scripts/report_dead_env_keys.py --apply'
```

Removing a key from a `.env` changes nothing in a running container until it is recreated, and
every affected settings model is `extra="ignore"`, so a recreate is also a no-op for these keys.
`LLM_LANE_*` is excluded (deleting the lane keys is its own 6.4 follow-up).

## Circe firewall [GO, Juniper, sudo]

Full runbook with verify and rollback: `docs/runbooks/2026-10-01-circe-llm-port-firewall.md`.
Rules (from `DRY_RUN=1 scripts/ops/circe_llm_port_gate.sh apply`, after a cleanup pass):

```bash
iptables -w -t mangle -N ORION-LLM-GATE
iptables -w -t mangle -A ORION-LLM-GATE -i lo -j RETURN
iptables -w -t mangle -A ORION-LLM-GATE -i docker0 -j RETURN
iptables -w -t mangle -A ORION-LLM-GATE -i br-+ -j RETURN
iptables -w -t mangle -A ORION-LLM-GATE -s 100.92.216.81/32 -j RETURN
iptables -w -t mangle -A ORION-LLM-GATE -j DROP
iptables -w -t mangle -I PREROUTING 1 -p tcp -m multiport --dports 8011:8017,8090,8099 -m addrtype --dst-type LOCAL -j ORION-LLM-GATE
ip6tables -w -t mangle -N ORION-LLM-GATE
ip6tables -w -t mangle -A ORION-LLM-GATE -i lo -j RETURN
ip6tables -w -t mangle -A ORION-LLM-GATE -i docker0 -j RETURN
ip6tables -w -t mangle -A ORION-LLM-GATE -i br-+ -j RETURN
ip6tables -w -t mangle -A ORION-LLM-GATE -s fd7a:115c:a1e0::733:d851/128 -j RETURN
ip6tables -w -t mangle -A ORION-LLM-GATE -j DROP
ip6tables -w -t mangle -I PREROUTING 1 -p tcp -m multiport --dports 8011:8017,8090,8099 -m addrtype --dst-type LOCAL -j ORION-LLM-GATE
```

Install on circe after merge:

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
sudo install -m 0755 scripts/ops/circe_llm_port_gate.sh /usr/local/sbin/orion-llm-port-gate
sudo install -m 0644 scripts/ops/orion-llm-port-gate.service /etc/systemd/system/orion-llm-port-gate.service
sudo systemctl daemon-reload
sudo systemctl enable --now orion-llm-port-gate.service
sudo /usr/local/sbin/orion-llm-port-gate status
```

Verify: `status` shows the `-j ORION-LLM-GATE` jump as the first line of mangle PREROUTING in
both families; from athena `curl -sS -m 5 http://100.112.254.99:8011/health` still answers; from
carbon-x1 the same curl times out; `docker exec orion-circe-gpu-lane-controller` reaching
`100.112.254.99:8011/health` still works. Rollback:

```bash
sudo systemctl disable --now orion-llm-port-gate.service
sudo /usr/local/sbin/orion-llm-port-gate remove
sudo rm -f /etc/systemd/system/orion-llm-port-gate.service /usr/local/sbin/orion-llm-port-gate
sudo systemctl daemon-reload
```

Why mangle PREROUTING: the ports are docker-published, so IPv4 traffic is DNAT'd before INPUT and
ufw (disabled on circe anyway) never sees it. Docker's `DOCKER-USER` hook only works while docker's
FORWARD jump sits above tailscale's `ts-forward`, and a tailscaled restart flips that. PREROUTING
runs before DNAT, matches the host port directly, and covers the docker-proxy (IPv6) path too.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none at runtime.
- Compatibility notes: CI gains two steps in `orion-static-gates`.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template
  changed).
- skipped keys requiring operator action: the dead keys above (`--apply` is an operator step).

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest tests/test_check_circe_worker_refs.py \
  tests/test_report_dead_env_keys.py tests/test_circe_llm_port_gate.py -q -p no:cacheprovider
87 passed in 16.05s
python scripts/check_circe_worker_refs.py            -> PASS (6 allowed hits, 4 allow entries, 12 zones)
python scripts/check_circe_worker_refs.py --live-env -> PASS; 6 live keys, 5 dead + 1 allowed
python scripts/check_chat_route_poachers.py          -> PASS
python scripts/check_scripts_dir_no_stdlib_shadow.py -> clean
python scripts/check_definition_drift.py             -> No definition changes
bash -n scripts/ops/circe_llm_port_gate.sh           -> ok (shellcheck not installed)
```

## Evals run

```text
None: this is CI/operator tooling with no runtime behavior to evaluate. The live-data check is the
read-only --report runs on athena and circe above, and the --live-env scan.
```

## Docker/build/smoke checks

```text
None: no service code or compose change.
circe firewall: DRY_RUN only (tests); applying it needs sudo on circe.
```

## Review findings fixed

Code review ran in a subagent against this diff. Every finding was fixed.

- Finding (must): the firewall lived in `DOCKER-USER`, which tailscale's `ts-forward` (accept-all
  for `tailscale0`) jumps ahead of whenever tailscaled restarts after docker. That would silently
  reopen the gate to every tailnet host.
  - Fix: rules moved to mangle PREROUTING (before DNAT, plain host port, no conntrack, no
    docker/tailscale rules there). The unit applies them once per boot, not tied to docker.
  - Evidence: `test_apply_allows_local_and_athena_then_drops_for_both_families` asserts no
    `DOCKER-USER`/`FORWARD` rule; the runbook's verify step checks the jump is first.
- Finding (should): `remove` deleted jumps by exact port text, so a port-list change left old jumps
  behind and the next `apply` failed with an empty, fail-open chain.
  - Fix: delete every jump by chain name (parse `-S PREROUTING`); `-w` on every call.
  - Evidence: `test_remove_deletes_jumps_by_chain_name_not_by_port_list`.
- Finding (should): bridges were allowed by `172.16.0.0/12`; docker pools can move into 192.168.x.
  - Fix: allow `-i docker0` and `-i br-+` instead. Evidence: same apply test.
- Finding (should): the gate missed `orion-atlas-llamacpp-chat:8080` (compose default name),
  `${PROJECT}-...`, `bonsai-worker`, `dsv41-flash`, `diffusion-host`.
  - Fix: names are read from the seat compose files' service keys and `container_name`, matched
    with any prefix and any port. Evidence: `test_other_address_shapes_are_caught` (12 shapes x 5
    locations), `test_worker_names_come_from_the_seat_compose_files`.
- Finding (should): the gate scanned only four top-level dirs and a narrow suffix list, and not
  circe's LAN addresses.
  - Fix: whole repo minus hidden/excluded dirs; templates, units, ini, txt added; 192.168.1.22/.24
    added. Evidence: the shapes test covers `deploy/`, `Makefile`, templates, `mesh-utilities/`.
- Finding (should): the dead-key tool treated `--` lines as comments everywhere, hiding compose
  `command:` flag lines like `--max-rows ${KEY}`.
  - Fix: `--` is a comment only in `.sql`. Evidence: `test_double_dash_flag_lines_read_keys_outside_sql`.
- Finding (should): keys read only by the image or libraries (`LLAMA_ARG_*`, `HF_*`, `CUDA_*`,
  `TZ`, ...) would look dead.
  - Fix: added to `PROTECTED_PATTERNS`, and `--apply` now removes only `KNOWN_DEAD` unless
    `--include-heuristic`. Evidence: `test_image_and_library_keys_are_protected`,
    `test_apply_defaults_to_known_dead_only`.
- Finding (should): the code tree (often a worktree) and the live `.env` checkout could be on
  different commits. Confirmed live during this PR: before merging main, the report flagged
  `HEARTBEAT_ORGAN_FIRE_WINDOW_SEC`, a key a newer main reads.
  - Fix: `--apply` refuses on a HEAD mismatch unless `--allow-tree-mismatch`.
    Evidence: `test_apply_refuses_when_code_tree_is_on_another_commit`.
- Nits fixed: v6 allows athena's tailnet address; one port list drives both families (tested
  equal); ops zones listed explicitly; atomic `.env` rewrite; multi-line quoted values are never
  half-removed (`test_multiline_value_is_never_half_removed`); file size cap raised to 4 MB;
  docstring notes that `.env.bak.*` holds secrets. Kept as documented: ALLOW keys are per file +
  host + port, not per line.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the firewall cannot tell the gateway from other athena containers (same
  source IP). Mitigation: the CI gate covers in-repo code; documented in the runbook.
- Severity: medium. Concern: `--apply` deletes lines from live `.env` files; a key read only by
  dynamically built names (`os.getenv(f"{x}_URL")`) would look dead. Mitigation: backup first,
  `.env_example` keys are never dead, `FOO_` prefix strings keep their whole family alive,
  `--apply` is KNOWN_DEAD-only by default and refuses a code/env commit mismatch, orphan files
  never edited.
- Severity: low. Concern: UNVERIFIED that nothing on circe inserts an ACCEPT above the gate's jump
  in mangle PREROUTING (reading it needs sudo). Mitigation: the runbook's verify step checks the
  order; re-check after tailscale upgrades.
- Severity: low. Concern: 192.168.1.x LAN clients lose access to the worker ports. Mitigation:
  none found in the repo or live `.env` files; the runbook has a one-line RETURN escape.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2453

🤖 Generated with [Claude Code](https://claude.com/claude-code)
