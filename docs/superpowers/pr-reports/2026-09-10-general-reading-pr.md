# General reading ingress — implementation report

## Arsonist summary

Orion can deliberately queue a source from Unified Chat or an autonomous curiosity turn and ask for its durable status later. Both use the existing World Pulse reading workers, queue, wallets, journal and Concept Atlas landing. The patch adds no service, submission HTTP endpoint, intent router, dashboard, graph writer or second queue.

Worktree: `/mnt/scripts/Orion-Sapienform-general-reading`; branch: `feat/general-reading`; base: `aff23fac0` (current `origin/main` at final verification). Published as [PR #2199](https://github.com/junebug-junie/Orion-Sapienform/pull/2199) from `feat/general-reading`. Seven upstream commits were integrated by fast-forward; the shared checkout and its six unrelated Graphify notes were preserved.

## Summary

- Two narrow MCP tools: `recommend_reading(url, why_now)` and `reading_status(request_id)`.
- Runtime bindings supply requester, invocation context and real parent run/trace IDs. Model arguments cannot supply those fields.
- Postgres commits precede accepted events and receipts; retries and duplicate active URLs do not create another active read.
- Stage 1 and Stage 2 preserve request lineage. Followup reads use the same queue and root-level round-trip cap.
- Journal messages can be replayed from saved artifacts without another model run or wallet debit. Completion requires stored journal entries.
- Source-reading models can use only WebFetch/WebSearch. Candidate materialization remains server-owned.

## Outcome moved

A model-selected recommendation can reach the durable reading queue from either caller, return immediately after acceptance, survive lost Pub/Sub notifications, and produce a source-attributed status/result through the existing worker path. Neither URL presence nor phrases such as “read this” trigger submission by themselves.

## Current architecture and actual bus semantics

Repository inspection found `OrionBusAsync.publish()` calls Redis `PUBLISH`, and `subscribe()` creates a Pub/Sub subscription. Its RPC helper subscribes to a reply channel and waits for a reply; it is not a durable command queue. Other World Pulse paths use Streams, but the Hub/FCC tools and these workers use the Pub/Sub client plus Postgres.

The existing `world_pulse_read_seed` table owns Stage 1/Stage 2 claims, `FOR UPDATE SKIP LOCKED`, pending/claimed/done/failed/skipped transitions, interruption reasons and stale-claim recovery. Hub polls this queue. Wallet A gates/debits heavy reading; Wallet B gates/debits Stage 2. Curiosity has its own run accounting and is not debited by submission.

Both caller paths currently reach the harness governor's FCC motor and per-turn **stdio MCP** configuration. The durable curiosity runner's existing bus round-trip back to Hub is preserved. There was no caller-specific HTTP MCP transport to replace on current HEAD.

## How Unified Chat invokes it

The WebSocket unified-chat wrapper and the existing HTTP chat handler pass an explicit `unified_chat` runtime context to the unified turn. This travels in `HarnessRunRequestV1.reading_binding` to the FCC configuration. The stdio tool wrapper derives `requested_by=juniper` and `invocation_context=unified_chat`. Chat payload fields or tool arguments cannot override it.

The model decides whether to invoke the tool. The description distinguishes immediate WebFetch/search work from deliberate asynchronous reading. It returns a receipt with a request ID; `reading_status` reads the queue and saved result using that ID.

## How curiosity invokes it

Both direct curiosity execution and durable curiosity turn dispatch pass the actual curiosity `run_id` separately from its correlation/trace ID. Their bound tool derives `requested_by=orion`, `invocation_context=curiosity`. The shared stdio adapter calls the same Hub application function as chat. No curiosity wallet or Curiosity Atlas run-count API is called by submission.

Other autonomous turns receive no reading binding by default. Stage 1/2 use a restricted reader profile and cannot recursively call the submission tool; additional heavy reading is requested through the existing typed Stage 2 result and capped server-side reentry.

## How requests become durable

1. The caller-bound MCP wrapper validates its two arguments and creates a stable request ID for the same turn, canonical URL and reason. A later turn gets a new ID and can deliberately reread a completed source.
2. An **ephemeral internal tool RPC** reaches Hub. This RPC's publication means no durable acceptance and is never reported as `submitted` or `queued` by itself.
3. Hub validates public URL/DNS, calls `enqueue_reading` → `enqueue_seeds`, and commits the existing Postgres queue transaction.
4. It then emits `reading.requested.v1` and returns the durable status. A pending accepted request returns `queued`. A transport timeout means acceptance is unknown; retry uses the same request ID.
5. Workers poll Postgres. Lost accepted notifications therefore do not lose queued work.

Transactions serialize equal canonical URLs with advisory locks. A new request for an already-active URL becomes a skipped alias row **in the same table**, retaining its own context and pointing to the active work. Status resolves that work. Exact request delivery is idempotent; completed/failed work does not ban future reads of the URL.

Status distinguishes `queued`, `started`, `stage1_completed`, `stage2_started`, `landing_pending`, `completed`, `failed`, `skipped` and `not_found`. A completed model pass alone does not imply the journal landed. Stage 2 replays missing journal messages from saved handoff/result artifacts using stable entry IDs, then confirms both existing `journal_entries` rows before setting `landing_at` and emitting `landing_completed`.

## Architecture touched

- Hub owns the thin bus listener and the existing Postgres application operation.
- Harness carries server-authored reading bindings and the restricted reader flag.
- Generated MCP configuration pins Python imports to the deployed package, avoiding an older sandbox checkout shadowing the new tool module.
- Reading workers retain existing scheduling, wallets and Concept Atlas materialization. Explicit recommendations do not inherit World Pulse's section-index discovery heuristic.
- Existing adapters still produce attributed candidates. The newly merged upstream Stage 2 `_priors_write_section` helper taught the model to write Cypher directly into `orion_worldview`; this patch removes that helper and its call to enforce the requested knowledge boundary. Candidate content and trace/evidence fields remain available in the handoff. There is no replacement direct graph writer.

## Schema / bus / API changes

Added contracts: `ReadingRequestedV1`, `ReadingToolBindingV1`, `ReadingToolRequestV1`, `ReadingToolResultV1`, `ReadingLifecycleV1`, plus strict tool-argument models.

Added subjects:

| Subject | Contract | Meaning |
| --- | --- | --- |
| `orion:reading:tool:request` | `ReadingToolRequestV1` | Internal ephemeral RPC |
| `orion:reading:tool:result:*` | `ReadingToolResultV1` | Durable receipt/status or error |
| `orion:reading:requested` | `ReadingRequestedV1` | Accepted request, after commit |
| `orion:reading:lifecycle` | `ReadingLifecycleV1` | Started, stage completion/failure, confirmed landing |

Additive migration: `services/orion-sql-db/manual_migration_general_reading_v1.sql`. Adds `request_id`, `request_json`, `root_request_id`, `duplicate_of`, `stage2_result_json`, `landing_at`, and indexes; extends the existing kind check to accept `reading`. Existing rows, claims, timestamps and physical table name are retained. Apply after the two existing World Pulse reading migrations.

`WorldPulseReadSeedV1` and Stage 2 results gain optional request lineage; Stage 2 persists actual round-trip counts for honest journal replay. `HarnessRunRequestV1` gains optional reading binding and a default-off restricted-reader flag. Legacy callers remain valid. No domain-level HTTP API was added.

## Files changed

- `orion/schemas/reading.py`, `world_pulse_read.py`, `harness_finalize.py`, `registry.py`: contracts, lineage and registry.
- `orion/bus/channels.yaml`: four bus subjects.
- `orion/world_pulse_read/{queue,urls,events,tools,mcp_server,journal}.py`: shared durable ingress/status, public URL checks, events, bound tools and replayable journal commands.
- `services/orion-hub/scripts/{reading_listener,main,api_routes,curiosity_investigation,world_pulse_read_pipeline,world_pulse_read_stage2,world_pulse_read_backfill}.py`: runtime wiring and existing worker/backfill integration.
- `orion/hub/turn_orchestrator.py`, `orion/harness/{runner,fcc_motor}.py`, `orion/fcc/mcp_config.py`: trusted turn context and restricted reader tool configuration.
- `orion/substrate/adapters/world_pulse_read.py`: request and evidence lineage on candidate nodes.
- `services/orion-sql-db/manual_migration_general_reading_v1.sql`: additive SQL migration.
- `services/orion-harness-governor/requirements.txt`: explicit MCP dependency.
- Hub, FCC and harness tests; Hub reading eval and minimal test requirements; `.github/workflows/orion-reading-tests.yml`: regression/CI coverage.
- `config/metrics/metric_definitions.lock.json`: deterministic catalog inventory refresh for the four bus subjects; no numeric cognition metric or detector added.
- `graphify-out/{graph.json,manifest.json,GRAPH_REPORT.md}`: guarded code-graph refresh; node count increased from 77,504 to 77,716.
- Hub and harness READMEs, this report and `2026-09-10-general-reading-cli-smoke.json`: operational contract, handoff and sanitized CLI evidence.

## Env/config changes

No operator env keys added, removed or renamed. `.env_example` files are unchanged; local env-template synchronization is not applicable. Existing `HARNESS_FCC_MCP_ENABLED`, reading worker enable flags, wallet settings, Hub Postgres DSN and Tailscale `ORION_BUS_URL` are reused.

`ORION_READING_BINDING` is generated per turn inside the MCP process environment, not operator configuration. It contains caller identity and lineage, never a model argument. The reader-only profile uses an explicit empty MCP configuration even if general MCP is disabled.

Ignored copies of the operator's existing env files were used for image builds and removed afterward. Originals and unrelated Graphify notes were preserved. No skipped operator keys.

## Metric / lifecycle provenance check

The catalog treats every bus subject as an inventory entry, so its definition lock was regenerated. These are request/receipt and state-transition facts, not ranking, trust or independent cognition scores. Producers are the queue's post-commit accepted publisher, worker lifecycle calls and Hub's tool reply. Durable timestamps/results and SQL journal confirmation are the authority. No detector, training default, schema score or model signal consumes these as a new metric. Live event statistics remain **UNVERIFIED** and are not claimed as usable cognitive signal.

## Tests run

Fresh environment: `/tmp/orion-reading-ci`, installed from the dependency set now in `services/orion-hub/tests/requirements-reading.txt`.

```bash
RUN_READING_POSTGRES=1 /tmp/orion-reading-ci/bin/python -m pytest -q \
  services/orion-hub/tests/test_reading_ingress.py \
  services/orion-hub/tests/test_reading_postgres.py \
  services/orion-hub/tests/test_world_pulse_read_*.py \
  services/orion-hub/tests/test_curiosity_investigation.py \
  services/orion-hub/tests/test_curiosity_self_inquiry.py \
  services/orion-hub/tests/test_turn_orchestrator_ws_frames.py \
  orion/fcc/tests/test_reading_mcp.py orion/fcc/tests/test_mcp_config.py \
  orion/harness/tests/test_fcc_motor_mcp.py \
  tests/test_world_pulse_read_*.py tests/test_unified_turn_schemas.py \
  tests/test_unified_turn_bus_catalog.py
# 444 passed, 2 warnings in 10.36s (after integrating current origin/main)

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-durable-runs/tests -q
# 8 passed in 2.57s

/mnt/scripts/Orion-Sapienform/.venv-world-pulse/bin/python -m pytest \
  services/orion-world-pulse/tests/test_curiosity.py \
  services/orion-world-pulse/tests/test_curiosity_followups_schema.py \
  services/orion-world-pulse/tests/test_renderers_curiosity.py -q
# 16 passed in 1.30s
```

The five Postgres integration scenarios start/stop a disposable local cluster; no production DSN is accepted. Migrations, locks, concurrent commits, aliasing, results and journal-row confirmation are real SQL. Models, DNS and event delivery are mocked. MCP protocol tests use the actual SDK with an in-memory transport; FCC tests inspect actual generated subprocess arguments.

Broader regression attempts (before the final seven-commit upstream integration; no affected reading test failed):

```bash
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests -q --maxfail=20
# 20 failed, 1484 passed, 3 skipped in 149.42s; stopped at failure ceiling

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  orion/fcc/tests orion/harness/tests tests/test_unified_turn_schemas.py \
  tests/test_unified_turn_bus_catalog.py -q --maxfail=10
# 4 failed, 409 passed, 1 skipped in 16.56s
```

The Hub failures concern UI expectations, model-label fixtures, the field-channel glossary count, memory-graph routes and consolidation fixtures. They do not establish a green repository-wide suite. Harness failures are an undiscovered room-companion caller, two `mind_coloring` template-fixture errors, and an FCC error-code expectation. The error-code failure was separately reproduced with the original HEAD runner loaded from `git show HEAD:orion/harness/runner.py`; other failing target files are unchanged by this patch, but their failures were not all separately reproduced in a pristine checkout. No unrelated fixes were mixed in.

Exact outputs: `/tmp/general-reading-ci-gate-final.log`, `/tmp/general-reading-durable-runs.log`, `/tmp/general-reading-world-pulse-regression.log`, `/tmp/general-reading-hub-suite.log`, `/tmp/general-reading-harness-suite.log`, `/tmp/general-reading-baseline-error.log`.

Remote CI on implementation commit `604e36692` passed all seven checks: reading; static repo gates; Hub signals cache; signal gateway/adapters/biometrics; SQL writer unit/shape; SQL writer Postgres integration; schedule browser smoke. The [reading workflow](https://github.com/junebug-junie/Orion-Sapienform/actions/runs/34544347176) reproduced **444 passed, 2 warnings in 12.76s** and **6 evals passed in 0.74s**. Final publication metadata and generated graph artifacts are in the followup commit; its check results remain visible on the PR.

## Evals run

```bash
/tmp/orion-reading-ci/bin/python -m pytest services/orion-hub/evals/test_reading_handoff_eval.py -q
# 6 passed in 0.73s (after upstream integration)
```

Offline behavioral fixtures cover all three producers, nonempty source-attributed candidates, lineage/evidence retention, and rejection of empty or direct graph-shaped output. Generated responses are injected; this is not a live model-quality evaluation.

## Docker/build/smoke checks

```bash
scripts/safe_docker_build.sh orion-hub build
# PASS — orion-hub-hub-app built
scripts/safe_docker_build.sh orion-harness-governor build
# PASS — orion-harness-governor-harness-governor built
python3 /tmp/orion-reading-cli-profile-review.py
# PASS — actual Claude Code 2.1.268; local mocked provider
```

The real CLI exposed exactly WebFetch/WebSearch, no plugins/MCP servers, and rejected forced Bash and plugin execution calls. No execution marker was created. Sanitized evidence: [`2026-09-10-general-reading-cli-smoke.json`](2026-09-10-general-reading-cli-smoke.json).

A network-disabled one-shot container smoke loaded the MCP server from the built governor image, enumerated both tools, rejected spoofed provenance and returned a mocked receipt. Output: `IMAGE_MCP_SMOKE_PASS tools=2 spoofed_arguments=rejected domain=mocked network=disabled`. This checks the packaged tool path; it is not a production reading smoke. The builds preceded upstream integration; final Stage 2 source is byte-identical to the implementation that was built.

Static gates: metric lineage, definition drift, inner-state registry, scripts stdlib shadowing, service hostnames, compose mount safety, journal dispatch registry, SystemHealth producers, control-surface store parity and async-route blocking checks pass. `check_definition_drift.py --update` was required only for the four added bus catalog entries; the subsequent `--gate` passed. See `/tmp/general-reading-static-gates.log` and `/tmp/general-reading-definition-gate.log`. Exact static gate commands (all exit 0 with the installed repository environment; the definition gate also passed again after upstream integration):

```bash
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_metric_lineage.py --gate
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_definition_drift.py --gate
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_inner_state_registry.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_scripts_dir_no_stdlib_shadow.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_service_hostname_refs.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_compose_no_relative_mounts.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_compose_no_host_claude_json_mount.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_journal_dispatch_registry.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_system_health_producers.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_control_surface_store_parity.py
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_async_routes_not_blocking.py
git diff --check
```

Graph maintenance: `scripts/safe_graphify_update.sh` passed, **77,504 → 77,716 nodes**, 168,769 edges. Graph/report/manifest are committed together. `graphify prs --worktrees` and `graphify prs --conflicts` ran before publication; GitHub reported the branch mergeable. Graphify reported 71 existing source files that produced no code nodes; no graph-shrink guard fired and no semantic extraction was run.

## Review findings fixed

- Lost Pub/Sub journal writes could remain pending forever.
  - Fix: replay saved artifacts with stable entry IDs; confirm actual SQL journal rows.
  - Evidence: real Postgres replay test, including preserved round-trip count.
- Curiosity run ID was distinct from correlation ID.
  - Fix: pass the actual run ID through direct and durable-turn bindings.
  - Evidence: runtime-binding tests.
- Discovery URL heuristics incorrectly skipped explicit recommendations.
  - Fix: apply section-index filtering only to World Pulse discovery.
  - Evidence: existing discovery behavior preserved; shared explicit ingress coverage.
- Reading stages inherited arbitrary execution/graph capabilities.
  - Fix: WebFetch/WebSearch-only built-ins and an explicit empty strict MCP configuration.
  - Evidence: spawn/config/protocol tests and actual CLI adversarial smoke.
- Post-read failures omitted a lifecycle event.
  - Fix: emit failure after the durable failure write.
  - Evidence: worker regression checks and reviewed path.

Independent subagent final verdict: **approved; no outstanding findings**. A second focused review after upstream integration also approved removal of the graph-write instructions and confirmed preservation of candidate content, lineage, wallets, reentry and journal replay.

## Restart required

Not executed against production. After reviewing/landing the code and arranging the worktree's normal ignored operator configuration:

```bash
psql "$RECALL_PG_DSN" -f services/orion-sql-db/manual_migration_general_reading_v1.sql
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The migration must precede running the new queue code. Existing environments without the original World Pulse tables must apply both prior reading migrations first. Use the configured Tailscale bus URL (`redis://100.92.216.81:6379/0` on this checkout's operator templates).

## Runtime-unverified / known limitations

- **UNVERIFIED:** production migration, deployed chat/curiosity tool selection, real bus receipt/event traffic, real model reading, wallet counters and final Concept Atlas/journal landings. No production restart or test submission was performed.
- Remote GitHub CI status is recorded on the pull request. A dedicated reading workflow is included; local validation results below distinguish real SQL checks from mocked model/bus behavior.
- Broader existing test suites remain red as detailed above. The dedicated affected gate is green.
- Tool transport trusts the existing internal bus/service boundary. Provenance is not model-supplied; this does not introduce authentication for arbitrary privileged host/bus clients.
- Ingress and pre-read checks reject nonpublic addresses and mixed/private DNS answers. They are not an egress sandbox; downstream redirect/rebinding behavior remains governed by WebFetch and has not been exercised against adversarial live sources.
- Status lookup requires a saved request ID. Completion refers to that source's confirmed landing; followup sources are separate linked queue rows.
- Restricting source stages to WebFetch/WebSearch intentionally removes arbitrary execution from those stages. Fetch coverage for paywalled, script-heavy or binary sources needs a real model/source eval.
- Pub/Sub lifecycle notifications remain best-effort. Postgres state and artifacts remain authoritative even if a notification is lost.

## Non-goals

No new service, queue, wallet, scheduler, UI, HTTP submission route, automatic intent/ranking router, graph database, model-authored graph mutation, or automatic promotion of article claims into settled beliefs.

## Recommended next patch

Run one approved chat recommendation and one curiosity recommendation on the deployed rail, follow their receipts through both wallets and journal/Concept Atlas landings, and record source-fetch quality plus redirect behavior. Exact-evidence relation extraction remains a later governed patch.

## PR link / completion status

[PR #2199](https://github.com/junebug-junie/Orion-Sapienform/pull/2199). Remote check results are attached to the PR. **DONE_WITH_CONCERNS** — implemented, reviewed and affected gates pass; broader suite failures and production runtime verification remain as documented.

## Graphify merge repair — 2026-09-11

After PRs #2196–#2198 landed, the generated graph, report and manifest conflicted. The installed Graphify 0.9.15 merge driver rejected this repo's ~108MB graph at a hardcoded 50MiB limit. Its implementation also ignored the common ancestor and discarded top-level metadata when serializing through NetworkX.

The existing LFS wrapper now invokes `scripts/merge_graphify_json.py`, a bounded stdlib three-way union. It preserves both parents' node/edge/hyperedge identities, applies independent field edits against the merge base, normalizes the two hyperedge storage locations, and refuses competing non-derived scalar edits without overwriting its input. The cap is 512MiB per input/output and 100,000 merged nodes. Graphify remains the report/community renderer; no package installation or host-package patch was needed. The wrapper/installer comments and git attributes document the new seam. The shared checkout's configured driver picks up this fix after the PR is merged and that checkout is updated; this repair explicitly exercised the worktree wrapper.

Merged `origin/main` at `15d76a51e`. The graph is the exact identity union of both parents: **77,966 nodes, 169,111 links, 104 hyperedges**, with zero duplicate or dangling IDs. Both hyperedge representations retain all records. A separate audit verified 2,273 independent attribute edits from this branch and 889 from main, all 6,217 manifest paths, and the staged LFS pointer's SHA256 and byte size. Manifest hashes survive only when they match both an existing parent extraction and the merged file's content; stale extraction channels remain empty. Communities/report were regenerated from the union without a source rescan and say so explicitly. Evidence: [`2026-09-11-general-reading-graph-merge.json`](2026-09-11-general-reading-graph-merge.json).

Verification:

```bash
python3 -m unittest discover -s tests/scripts -p test_merge_graphify_json.py -v
# 10 passed in 1.076s; includes >50MiB input and actual Git LFS pointer round trip
sh -n scripts/graphify_lfs_merge_driver.sh
# PASS
# Repeated the dedicated reading gate listed above after merging main:
# 444 passed, 2 warnings in 9.79s
/tmp/orion-reading-ci/bin/python -m pytest services/orion-hub/evals/test_reading_handoff_eval.py -q
# 6 passed in 0.71s
```

The real ~108MB parent graphs also passed through the worktree's LFS wrapper. A CI step now runs the merger regressions in the static-gates workflow. Independent review found and fixed two additional metadata cases: one-sided deletions dropping retained hyperedges, and root-only versus nested-only hyperedge layouts diverging. The implementation and preservation audit were rechecked after those fixes.
