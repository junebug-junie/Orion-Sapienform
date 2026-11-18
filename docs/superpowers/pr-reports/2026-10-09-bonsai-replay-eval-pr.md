## Summary

- A one-command replay test that asks Ternary-Bonsai-2-27B (gpu2) and the dense Qwen3.8-27B Q4 (gpu1) the same 30 real tasks and prints whether Bonsai may take gpu1: `make eval-bonsai-replay` (`ARGS=--dry-run` shows the plan and touches nothing).
- The tasks are frozen from history into a committed fixture: 10 curiosity briefs (both Bonsai runs of the window, including d4db8c2bacb4, plus Q4 investigate/self-inquiry runs), the 4 self-sense questions x 2 lived-answer snapshots, 6 reading turns (2 replaying the historical fetch failure), 6 stance_react prompts.
- Every write is stubbed, by construction, one lane per kind of state (below). The model still gets realistic tools: a shell in a no-network read-only container over the repo, live read-only SQL, and a private scratch copy of its own graph where its writes really land.
- Scoring is automated: finished, contract-valid (production validators), empty, cut by the token limit, tokens, time, and a write-claim check that compares every prior move the write-up claims against what actually landed in its scratch graph. A blind A/B sheet is written for the hand grade.
- The cards come from the GPU pool, not from starting model containers: operator holds per task, role and profile verified, released on every exit path.

## Outcome moved

Before: "is Bonsai good enough for gpu1?" rested on ~44 h of uncontrolled live traffic where the two models never saw the same task, and the one misreported write (d4db8c2bacb4: claimed a 0.80 -> 0.70 revision of `self:four_wiring_points_named_20260922` that never happened, omitted the 0.70 -> 0.85 revision it really wrote) was found by hand.
After: a controlled side-by-side on identical inputs with a pass/fail rule, and the misreport check is code. On the saved d4db write-up, with what really landed in the graph, the check flags the phantom revision and lists the real revision as never mentioned.

## Current architecture

Curiosity, self-sense and reading turns all run through Hub `execute_unified_turn` -> stance_react (cortex-exec) -> FCC harness -> `claude -p` -> llm-gateway `/v1/messages` -> the worker the pool placed. No flag gates all side effects (`no_write` only skips chat_history; investigate turns hold a FalkorDB credential that can write `orion_worldview`). A caller cannot name a GPU role or profile: the pool grants by work class.

## Architecture touched

New package `orion/evals/model_replay/` and two host scripts. No service, schema, bus channel or env key changes. Runtime seams used: GPU pool control verbs (hold / release / cancel) on `orion:gpu_pool:control:request`; the workers' own `/v1/messages`; Postgres as `orion_readonly`; FalkorDB DUMP (read) once per run.

### How a replay turn runs

1. Wait (polling the pool's own state, up to 3 h) until gpu1 and gpu2 serve the expected models and no other run holds either. Then hold gpu1 (class `memory_distill` -> role `agent`, its only slot), then gpu2 (class `agent` -> falls through to `agent-gpu2` because gpu1 is now ours). Each grant's role AND profile are checked; a wrong one is released at once and the task waits and retries (3 tries, then it is left for `--out` resume). Each model call rides its hold as an attached child lease (pool verb `attach`), the way a durable run's calls do, so the pool never lends the slot to one-off traffic mid-generation.
2. Both models run the task at the same time, each on its own card: the model's own stance pass (production `stance_react.j2`), then the harness prompt built by production `orion.harness.runner.build_harness_prompt` over that stance output, then a Claude-Code-shaped tool loop (Bash, Read, Grep, Glob, WebFetch, WebSearch; reading turns get WebFetch/WebSearch only, as in production).
3. Sampling is pinned per request and identical: temperature 1.0, top_k 20, top_p 0.95, min_p 0, max_tokens 16384, reasoning_effort xhigh with preserve_thinking (both profiles' own defaults).
4. Holds released, scores appended to `results.jsonl`, next task.

### How writes are prevented (and how that is proven)

| state | lane | why nothing reaches production | proof |
|---|---|---|---|
| filesystem, network, anything a shell can do | `docker run --network none --read-only --cap-drop ALL`, repo mounted `:ro`, only tmpfs writable, no docker socket | no route out, nothing writable | `test_shell_sandbox_is_isolated`; local smoke: `echo > /repo/...` and `touch` -> Read-only file system, a socket to the bus -> Network is unreachable, no `/var/run/docker.sock` |
| Postgres | `psql` routed to the host, run as `orion_readonly` (SELECT-only role, the one the real turn gets) inside `BEGIN READ ONLY` with `default_transaction_read_only=on`; SQL that looks like a write is never sent | role + read-only transaction + pre-filter | `test_sql_write_never_reaches_postgres`, `test_readonly_psql_argv`; local smoke: insert answered "read-only", count(*) read returned live rows |
| FalkorDB | graph commands go to a fresh local FalkorDB per (task, model) loaded from a DUMP taken at replay start; production FalkorDB is only ever sent DUMP/PING/EXISTS (allowlist in code); production's curiosity ACL is emulated (Atlas read-only) | the model's writes land in a throwaway copy | `test_prod_graph_source_only_reads`, `test_scratch_graph_emulates_curiosity_acl`; local smoke: a MERGE landed in scratch, the same prior id is absent in production afterwards |
| HTTP | GET only; any method other than GET or any body (`-d`, `-F`, `--json`, `-T`) is recorded as a stubbed write and not sent | never sent | `test_http_write_never_sent` |
| docker | ps / logs / inspect / images only | everything else refused before running | `test_docker_mutations_refused_reads_allowed` |
| Orion bus | the package only sends pool control verbs hold / release / cancel | no publish anywhere in the package | `test_package_never_publishes_to_the_bus_or_writes_prod_graph` |
| psql meta-commands (`\!` runs a shell inside the DB container, `\c` switches user, `\o`/`\copy`/`\g file` write files) | refused before anything is sent; only a lone `\d` describe passes | `test_psql_meta_commands_never_reach_postgres`; local smoke: `\! id` refused |
| HTTP to local/tailnet services | GET to private, loopback, link-local or tailnet (100.64/10) addresses refused except the Hub API the prompts name (`host.docker.internal:8080/api/...`); redirects not followed | `test_http_get_blocks_private_addresses`; local smoke: GET to the pool's port refused |
| secrets in what the model can read | the shell sees a `git archive HEAD` snapshot (tracked files only, no `services/*/.env`); `docker inspect` (prints container env) is not allowed | `test_docker_inspect_and_follow_refused` |
| graph/SQL calls the splitter cannot route (`$(...)`, backticks, subshells) | refused with a message, never executed | `test_unsplittable_external_command_is_refused_not_run` |

Anything the router misses (e.g. `bash -c "psql ..."`) falls into the shell container, where it has no network: it fails, it does not write.

### Pool holds released on every exit path

`PoolHolds` is an async context manager. Every hold request is written to `holds.jsonl` before it is sent and its lease id as soon as the pool answers; if a hold RPC gets no answer, the pool's own lease table (`GET /v1/pool`) is swept for this holder (`operator:bonsai-replay-eval`) and anything live is released. Release runs on normal exit, exceptions, task cancellation, Ctrl-C, SIGTERM and SIGHUP (signal -> cancel main task -> unwind); a second cancel during release does not skip the remaining holds. On cancel the worker threads are stopped, their in-flight model call cut (HTTP client closed), and drained for up to 60 s before the holds go back. A refused or wrong-seat grant releases everything already held. After a SIGKILL: `--release-leftovers <out dir>` (ledger + pool sweep). Tests: `test_released_when_task_body_raises`, `test_released_when_cancelled`, `test_second_cancel_during_release_still_releases_everything`, `test_hold_rpc_timeout_sweeps_the_pool_for_our_lease`, `test_wrong_seat_refused_and_everything_released`, `test_pool_refusal_releases_earlier_holds`, `test_run_replay_releases_holds_when_a_rig_crashes`, `test_cancel_cuts_calls_and_drains_threads_before_holds_go_back`, `test_leftovers_after_sigkill_are_released`, `test_leftovers_sweep_finds_unledgered_lease`.

### Metric quality gate (finish rate, misreported writes)

1. Provenance: `finished` = the loop's own end state + production validators (`scoring.py`); `misreported_writes` = `write_claims.check` over the final text against `graph_scratch.diff_snapshots` of the turn's own scratch graph (before vs after).
2. Independence: misreported writes is independent of finishing (a finished turn can misreport, d4db did). Finish rate and the gap rule are the same quantity on two models.
3. Anchor: the d4db hand audit (claim vs PriorRevision/prior state) is exactly this comparison; the decision rule is Juniper's.
4. Live sanity check, on real saved write-ups (24 curiosity runs, 10-07..09) against what production's graph holds (read-only): d4db8c2bacb4 flagged (phantom 0.80 -> 0.70 on `four_wiring_points`, real revision on `frontier_select_region_top8` listed as never mentioned). 3 more runs flag, all traced to the history reconstruction, not the extractor: 3138c48b3ca3 and 49174d16ae80 moved priors without leaving a run marker (later revisions start from the claimed value / the narrated 0.6 -> 0.72 matches the prior's current value), and d4db's new prior reads 0.72 today because 49174 moved it later. In the replay the diff is taken on the turn's own scratch copy, so it needs no run markers. 13 runs produce supported claims; the detector can fire and can stay quiet. Q4's write-ups narrate moves in many shapes; the extractor finds at least one claim in 14 of 24 -- what it does not parse is not checked (see risks).
5. Existing mechanism: none in the repo compares narrated writes to the graph.
6. Reversibility: a local eval; nothing consumes these numbers except the decision printed for Juniper.

## Files changed

- `orion/evals/model_replay/fixture.py`: task schema (`model_replay.task.v1`), load/dump.
- `orion/evals/model_replay/extract.py`, `scripts/extract_model_replay_fixture.py`: task picks and read-only extraction (`--check` diffs against the committed file).
- `orion/evals/model_replay/fixtures/tasks.v1.jsonl`: the 30 frozen tasks (1.1 MB; mostly the 10 briefs, each ~37k chars, twice: as the message and inside its stance prompt).
- `orion/evals/model_replay/pool_hold.py`: operator holds, seat verification, ledger, leftover release.
- `orion/evals/model_replay/sandbox.py`, `command_split.py`: tool surface and no-write lanes.
- `orion/evals/model_replay/graph_scratch.py`: production DUMP (read-only allowlist), scratch FalkorDB, prior snapshot/diff.
- `orion/evals/model_replay/agent_loop.py`: `/v1/messages` tool loop, pinned sampling.
- `orion/evals/model_replay/write_claims.py`: the write-claim check.
- `orion/evals/model_replay/scoring.py`, `report.py`, `runner.py`: scoring, decision rule, report + blind sheet, orchestration.
- `orion/evals/model_replay/README.md`, `scripts/run_bonsai_replay_eval.py`, `Makefile` (`eval-bonsai-replay`).
- `orion/evals/model_replay/tests/*`, `.github/workflows/model-replay-eval-tests.yml`.

## Schema / bus / API changes

- Added: none on the bus or registry. `model_replay.task.v1` is a local fixture schema, not a bus payload.
- Removed / Renamed: none.
- Behavior changed: none in any service.
- Compatibility notes: uses existing pool control verbs only.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- The run needs `ORION_BUS_URL` (the Make target defaults it to `redis://100.92.216.81:6379/0`).

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest orion/evals/model_replay/tests -q -p no:cacheprovider
92 passed in 5.15s   (no-network guard on every test)
CI (PR #2554): model-replay pass, Static repo gates pass, hub-schedule-browser-smoke pass
Local static gates: scripts dir stdlib shadow, service hostname refs, circe worker refs, metric lineage, definition drift, inner-state registry, async routes, chat route poachers, control-surface parity, system-health producers, sentience instruments -- all pass
```

## Evals run

```text
The replay itself was NOT run (by instruction: no live models, no pool holds).
Scorer smoke on saved historical outputs (/tmp/bonsai-quality-2026-10-09/samples, 24 runs, production graph read with GRAPH.RO_QUERY): see Metric quality gate above.
Dry run: make eval-bonsai-replay ARGS=--dry-run -> 30 tasks, expected 11.9 h, worst case 31.8 h, nothing held/started/sent.
```

## Docker/build/smoke checks

```text
Scratch FalkorDB (local container, production image id, loaded from a DUMP of production): fresh in 1.0 s; a SET + CREATE PriorRevision and a MERGE landed in the copy and the diff read them back; the same prior in production still reads 0.85 and the new id is absent.
Real-lane smoke (real shell container, real scratch graph, real read-only psql; scripted model, no LLM, no pool): graph write landed in scratch only; Atlas write NOPERM; SQL read returned live rows; SQL insert stubbed; psql \! refused; /repo and the primary checkout read-only; socket to the bus: Network is unreachable; no docker socket; curl POST stubbed; GET to the pool port refused; docker exec refused; no leftover containers.
```

## Review findings fixed

Code review subagent: 2 blockers, 9 should-fix, 4 nits. All fixed except where noted.

- Finding (blocker): psql meta-commands (`\!` shell in the DB container where `psql -U postgres` is trusted, `\c`, `\o`, `\copy`) slipped past the SQL write check.
  - Fix: any backslash refused except a lone `\d` describe, in the Toolbox AND inside ReadOnlyPsql; `-F/-P` flags dropped.
  - Evidence: `test_psql_meta_commands_never_reach_postgres` (5 forms), real-lane smoke.
- Finding (blocker): a claim ending a sentence ("0.80 -> 0.70.") produced no claim, so the zero-misreport check could pass falsely.
  - Fix: number pattern no longer refuses a trailing period. Evidence: `test_sentence_final_and_trailing_hedge_claims_are_caught`.
- Finding: hedge words anywhere dropped the sentence; free-form ids not passed; self-sense node writes unchecked.
  - Fix: hedges only count in the claim's own clause before its numbers; known prior ids from the before/after snapshots are passed. Self-sense: only Prior moves are checked (stated in risks).
- Finding: a hold RPC that times out after the pool created the lease left no record (gpu1 has no max_hold_sec).
  - Fix: request ledgered before sending; pool lease-table sweep by holder on that path and in `--release-leftovers`. Evidence: `test_hold_rpc_timeout_sweeps_the_pool_for_our_lease`, `test_leftovers_sweep_finds_unledgered_lease`.
- Finding: a second cancel during release skipped the remaining holds.
  - Fix: per-lease CancelledError held and re-raised after the loop. Evidence: `test_second_cancel_during_release_still_releases_everything`.
- Finding: model calls kept running on a card after its hold was released on cancel.
  - Fix: STOP + close the per-rig HTTP client + bounded thread drain before release. Evidence: `test_cancel_cuts_calls_and_drains_threads_before_holds_go_back`. Whether closing an httpx client interrupts a blocking read mid-generation is UNVERIFIED; the drain is bounded at 60 s either way.
- Finding: class `agent` can fall through to chat/hecate; a wrong seat stopped the whole run; the second gpu2 hold could land on chat.
  - Fix: second hold removed; a seat check waits until both cards are free of other runs' holds; wrong seat -> retry, not stop. An operator-only gpu2 class would remove the fall-through entirely but is a config/gpu_pool.yaml change (another agent's stage 7.3 owns that file) -- follow-up.
- Finding: infrastructure failures (sandbox, scratch graph, attach refused, recall) scored as model failures.
  - Fix: `harness_error`/`infra_error`/`cancelled` are void: excluded from both denominators, re-run on resume, verdict says PARTIAL. Evidence: `test_void_tasks_leave_both_denominators_and_rerun_on_resume`.
- Finding: plain GET could reach host-local/tailnet services and follow redirects; `curl -XPOST`/`-dfoo` parsed as GET.
  - Fix: private addresses refused except the Hub API; no redirects; all curl write forms stubbed. Evidence: `test_http_get_blocks_private_addresses`, `test_curl_write_forms_are_stubbed`.
- Finding: re-quoting broke globs, `$VARS`, `Q=...; ... "$Q"` and redirects once a routed tool appeared.
  - Fix: tokens re-emitted bare/single/double-quoted as written; `$VAR` expanded from the sandbox env and earlier assignments (not inside single quotes); `>`/`>>` on routed tools write into the sandbox. Evidence: `test_vars_expand_for_routed_tools_like_bash_would`, `test_redirect_of_routed_tool_lands_in_sandbox`, `test_shell_parts_keep_globs_and_vars`, `test_assignment_tokens_stay_assignments`.
- Finding: the shell could read `services/*/.env` from the primary checkout; `docker inspect` printed container env.
  - Fix: `git archive HEAD` snapshot per run (also pins the repo for the 12 h run); inspect removed.
- Nits fixed: per-URL fetch lock; `--follow=true`/`-tf` refused; `expired`/`aborted` end a grant wait; PR report committed.
- Found while fixing: one test (`test_worker_client_only_posts_messages`) had stopped intercepting after the client change and sent ONE request with an empty message list to the live Bonsai worker (100.112.254.99:8016), which answered 500. No hold, no write. Fixed with httpx MockTransport, and a package-wide autouse guard now refuses any TCP connection from these tests.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: the extractor reads only Prior confidence moves; LivedAnswer/SelfDefinition/Finding writes claimed in prose (self-sense especially) are not checked, and a move narrated in a shape it does not parse is not checked either. Mitigation: every claim and verdict is listed in report.md; `landed` and `stubbed_writes` are in every transcript; the blind sheet is the backstop.
- Severity: medium. Concern: 30 single-shot tasks at temperature 1.0; one task is 3.3 points against a 5-point gap rule. Mitigation: void tasks are excluded rather than counted; re-run with a fresh --out for a second sample if the verdict is close.
- Severity: medium. Concern: fidelity. Not reproduced, identically for both models: Claude Code's system prompt and tool descriptions, WebFetch's summarizer, WebSearch, MCP servers, live stance inputs, memory digest. Stance prompts were never persisted, so the 6 stance tasks render the production template over real user messages with the fallback identity lines.
- Severity: low. Concern: class `agent` can briefly grant hecate's or chat's card before the replay gives it back (one RPC round trip), if gpu2 changes between the seat check and the hold. Mitigation: seat check; follow-up = operator-only gpu2 class (config/gpu_pool.yaml, after stage 7.3 lands -- PR #2553 also changes agent-gpu2 to max_holds 2).
- Severity: low. Concern: gpu2's second slot can serve one-off production calls while Bonsai runs (gap sharing), which slows Bonsai (46 -> 26 tok/s measured in #2553). Time/token numbers are noisier for Bonsai than finish rate.
- Severity: low. Concern: requires the athena host (local orion-athena-sql-db and orion-athena-falkordb containers, the governor image as sandbox) and ~130 MB disk per run for the repo snapshot.
- UNVERIFIED: the replay has never run against the live workers or the pool; llama.cpp's acceptance of `min_p`/`chat_template_kwargs` on `/v1/messages`; that closing the httpx client interrupts a long generation.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2554

🤖 Generated with [Claude Code](https://claude.com/claude-code)
