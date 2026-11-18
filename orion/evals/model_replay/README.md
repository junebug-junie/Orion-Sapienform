# Model replay: Bonsai (gpu2) vs Q4 (gpu1)

Decides whether Ternary-Bonsai-2-27B can replace the dense Qwen3.8-27B Q4 on gpu1, from 30 real
tasks run through both models with identical sampling and every write stubbed.

```bash
make eval-bonsai-replay ARGS=--dry-run     # the plan; touches nothing
make eval-bonsai-replay                    # the run (~12 h expected, 32 h worst case)
```

Decision rule (printed last, and first line of `report.md`): Bonsai eligible iff finish rate >= 90%,
zero misreported writes, and finish rate within 5 points of Q4.

| file | what |
|---|---|
| `fixtures/tasks.v1.jsonl` | 10 curiosity briefs, 4 self-sense questions x 2 snapshots, 6 reading turns (2 with the historical fetch failure), 6 stance_react prompts. Re-extract: `scripts/extract_model_replay_fixture.py` (`--check` diffs). |
| `pool_hold.py` | background operator holds per task (`LeaseHoldTransport`, heartbeated): gpu1 via class `operator_agent`, gpu2 via class `operator_agent_gpu2`, role+profile verified, waits behind Orion's own work, a recall stops the run (partial report, exit 3), model calls go through orion-llm-gateway as children of the hold, released on every exit path; `holds.jsonl` ledger + pool sweep + `--release-leftovers`. |
| `sandbox.py` | the tool surface and the no-write lanes (shell in a `--network none --read-only` container; SQL as `orion_readonly` in a READ ONLY transaction; SQL meta-commands refused; HTTP GET only, no private addresses except the Hub API; docker ps/logs/images only; the shell sees a `git archive` snapshot of the repo, no .env files). |
| `graph_scratch.py` | graph reads/writes go to a fresh local FalkorDB per (task, model) loaded from a start-of-run DUMP; production FalkorDB only ever sees DUMP/PING/EXISTS. |
| `agent_loop.py` | Anthropic `/v1/messages` tool loop through orion-llm-gateway under the task's hold. |
| `write_claims.py` | the write-claim check: write-up vs what landed in the scratch graph (catches d4db8c2bacb4). |
| `scoring.py`, `report.py` | finished / contract / length / tokens / time, decision rule, `report.md`, blind A/B sheet. |

Results go to `~/.orion/model-replay/<UTC timestamp>/` (`--out DIR` to resume).
