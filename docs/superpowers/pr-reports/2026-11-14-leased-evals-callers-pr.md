# Leased evals, part 1 of 3: the last direct callers take a GPU seat and go through the gateway

Design: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md` (PR #3237, all 9
defaults approved 2026-11-12). Phase 1 shipped in PR #3242 and left these callers behind. This PR moves
them. Part 2 (the shadow lane door) and part 3 (refusal + firewall) are stacked on it.

## Summary

- The llama.cpp host's operator probes and its Qwen3.5 fast-lane eval no longer talk to a model server
  directly. Each takes a GPU pool operator hold (background priority, behind all of Orion's work) and
  sends every model call through orion-llm-gateway.
- The slot-bleed probe takes two holds on agent-gpu2, one per slot, so its "side by side" phases really
  run side by side and Orion's work is not on the seat while they run.
- The Qwen3.5 eval's parallel test sends ten calls as ten separate gateway leases and says
  "inconclusive" when the pool never let them overlap.
- Shell scripts use a new helper command, `python -m orion.evals.leased_llm chat|hold`.
- The CI address gate now scans the llama.cpp host's `scripts/` and `evals/` (they sat inside the
  worker's own exempt zone before). The GPU power bench is named as the one by-design direct caller.

## Outcome moved

Before: five operator tools could load a model seat with nobody in the pool's books (the 11-12
incident class). After: each one shows in `gpu_pool_events` as `work_class like 'operator%'`, waits its
turn, and stops when Orion needs the seat. There is no direct-URL fallback left in any of them.

## Current architecture

- `probe_slot_bleed.py` posted to `http://100.112.254.99:8016` straight from athena.
- `probe_structured_output.py` and `verify_atlas_quick_llamacpp_thinking_off.sh` posted to a worker URL
  from env (default `127.0.0.1:8013`).
- `run_qwen35_fast_eval.py` posted to an SSH tunnel (`127.0.0.1:18013`) into the fast worker.
- `verify_qwen3_thinking_off_live.sh` started its own llama-server container on a card with no lease.
- `check_circe_worker_refs.py` skipped all of `services/orion-llamacpp-host/*` as "the worker itself".

## Architecture touched

- `orion/evals/leased_llm.py` (shared helper).
- `services/orion-llm-gateway` HTTP passthrough: one new request header, honoured only on operator routes.
- `services/orion-llamacpp-host` scripts and evals.
- CI gate `scripts/check_circe_worker_refs.py`; CI workflows.

## Files changed

- `orion/evals/leased_llm.py`:
  - `OperatorLane.worker_url`, `worker_read` (only the no-model reads: GET /health /props /slots
    /v1/models /models /metrics, POST /tokenize /detokenize /apply-template), `post_raw`, `http_client()`.
  - `LaneClient`: an httpx-shaped client that sends model calls to the gateway under the hold and reads
    to the granted URL, whatever host the caller names.
  - `gateway_call`: one unheld per-call lease on an operator route, for parallel load.
  - CLI `chat` (one leased call, body on stdin) and `hold` (hold a seat while a command runs; a recall or
    Ctrl+C stops the command before the hold is released).
- `services/orion-llm-gateway/app/passthrough_proxy.py`, `openai_passthrough.py`,
  `anthropic_passthrough.py`: `X-Orion-Operator-Holder` names the holder of a one-off operator call.
- `services/orion-llamacpp-host/scripts/probe_slot_bleed.py`: two holds; busy check skips the probe's own
  leases and their children; recall stops the run INCONCLUSIVE; optional `/metrics` overlap check.
- `services/orion-llamacpp-host/scripts/probe_structured_output.py`: `--lane` replaces `--base-url`.
- `services/orion-llamacpp-host/scripts/verify_atlas_quick_llamacpp_thinking_off.sh`: helper `chat`.
- `services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh`: runs its container under
  helper `hold`, with `--init`, a named container, an EXIT trap and PCI device order.
- `services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py`: rewritten on the helper.
- `services/orion-llamacpp-host/README.md`: run instructions.
- `scripts/check_circe_worker_refs.py`: `ZONE_EXCEPTIONS`, `DIRECT_CALLERS`.
- `.github/workflows/leased-evals-tests.yml` (new): the helper had no CI job before this PR.
- `.github/workflows/orion-gpu-pool-tests.yml`: runs the new llama.cpp host tests.
- Tests: `orion/evals/tests/test_leased_llm.py`, `services/orion-llm-gateway/tests/test_operator_holder_header.py`,
  `services/orion-llamacpp-host/tests/test_probe_slot_bleed.py`, `test_probe_structured_output.py`,
  `test_qwen35_fast_eval_leased.py`, `test_operator_scripts_leased.py`, `tests/test_check_circe_worker_refs.py`.

## Schema / bus / API changes

- Added: gateway request header `X-Orion-Operator-Holder` (`operator:<name>`, only on `operator*` routes).
- Removed: `--url` (slot-bleed probe), `--base-url` (structured-output probe, Qwen3.5 eval),
  `--gateway-url` (Qwen3.5 eval), `ATLAS_LLAMACPP_QUICK_URL`/`ATLAS_QUICK_CHAT_MODEL` (quick check).
- Behavior changed: the probes need `ORION_BUS_URL` and the gateway; they wait in the pool's queue.
- Compatibility: no schema, bus channel or registry change.

## Env/config changes

- Added / removed / renamed keys: none in any `.env_example`.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/evals/tests/test_leased_llm.py + orion/harness/evals            72 passed
tests/test_check_circe_worker_refs.py                                 included above (148 passed total)
services/orion-llm-gateway/tests (full suite)                        553 passed
services/orion-llamacpp-host CI set (LLM_PROFILE_NAME=ci): worker_announce, prism_seat,
  bonsai_chat_template, probe_slot_bleed, strata_apply_defaults, probe_structured_output,
  qwen35_fast_eval_leased, operator_scripts_leased                   155 passed
scripts/check_circe_worker_refs.py                                   PASS
tests/test_no_personal_terms.py                                      passed
git diff --check                                                     clean
```

Gate evidence: on `origin/main`'s copy of `probe_slot_bleed.py`, the widened scan reports two hits
(lines 34 and 382, `100.112.254.99:8016`). The structured-output probe, the quick check and the Qwen3.5
eval used env or tunnel URLs, which no address scan can see; the new lexical test (every model-call path
in those folders must go through `leased_llm`) and the lane door (part 2) cover them.

## Evals run

```text
No live run. Each tool now needs a granted operator hold; running one would put eval load on a live
seat, and part 2 is what measures unkeyed calls. Live path: UNVERIFIED.
```

## Docker/build/smoke checks

```text
No image or compose change.
```

## Review findings fixed

A code-review subagent reviewed the first two commits. No blocker.

- Finding: a recall SIGTERM never reached the Qwen3 test container (bash as PID 1), so it could outlive
  the hold; Ctrl+C released the hold with the command still running.
  - Fix: `--init`, a named container, an EXIT trap that removes it; `_cli_hold` stops the command on any
    exit path before release.
  - Evidence: `test_cli_hold_stops_the_command_before_releasing_on_ctrl_c`, `test_cli_hold_stops_the_command_on_recall`.
- Finding: the hold on an LLM lane does not free the card's VRAM; the lane map was circe's.
  - Fix: documented in the script; refuses off circe unless `HOLD_LANE` is set; `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Finding: a recall during the slot-bleed quiet check read as "/slots unreadable" for up to 300 s, and a
  recall of the second hold went unseen.
  - Fix: the quiet check and every phase boundary check every hold.
  - Evidence: `test_a_recall_seen_by_the_quiet_check_stops_at_once`.
- Finding: an Orion call could slip into the probe's idle second slot mid-phase unseen.
  - Fix: when the worker serves `/metrics`, the probe compares `llamacpp:prompt_tokens_total` with the
    tokens it sent. The summary says which check ran. Without `/metrics` the gap remains and is stated.
  - Evidence: `test_metrics_catch_work_that_slipped_into_an_idle_slot_mid_phase`.
- Finding: the Qwen3.5 overlap count included Orion's own busy slots.
  - Fix: a baseline sampled before the calls is subtracted.
  - Evidence: `test_orions_own_busy_slots_do_not_count_as_the_evals_overlap`.
- Finding: a busy pool on the parallel calls was reported as FAIL.
  - Fix: `gpu_pool_unavailable` or a gone hold is INCONCLUSIVE (exit 3).
- Finding: one-off gateway leases lost the eval's holder.
  - Fix: `X-Orion-Operator-Holder`, operator routes only.
  - Evidence: `test_operator_holder_header.py`.
- Finding: the helper cited a parity test that did not exist yet.
  - Fix: comment reworded; the parity test ships with the door in part 2.
- Nits fixed: query string on a model call, 128+N exit for signal deaths, `DIRECT_CALLERS` relabelled as
  documentation, structured-output summary records the route that ran, workflow jq line dropped and
  triggers widened, a test on the real `LaneClient` wiring (model calls to the gateway, reads to the worker).

## Restart required

```text
The gateway change (holder header) takes effect on its next rebuild:
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
Until then gateway_call's leases still work; they are recorded under the default holder http:openai.
```

## Risks / concerns

- Severity: low. Concern: the slot-bleed probe's mid-phase overlap check needs `--metrics` on the worker;
  agent-gpu2 does not serve `/metrics` today. Mitigation: the summary says `before_after_only` so a PASS
  is read with that limit.
- Severity: low. Concern: the Qwen3 container check shares the card's VRAM with the resident model.
  Mitigation: stated in the script; use a small GGUF or the gpu2 reservation tool.
- Severity: low. Concern: these tools now wait behind Orion's work, sometimes for a long time (chat's card
  waits until Juniper lends it). That is the approved design (answer 4).
- Issues opened/closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3404

🤖 Generated with [Claude Code](https://claude.com/claude-code)
