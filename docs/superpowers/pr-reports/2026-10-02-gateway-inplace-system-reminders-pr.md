## Summary

- Orion's harness turns re-read the whole 13-30k-token prompt on every step (~38 s/step) because the LLM gateway moved every Claude Code `role: system` message (per-step `<total_tokens>` reminder, PreToolUse/SessionStart hook context) into the top-level system block, which grew every step and shifted everything after it.
- The gateway now moves only the system messages that come **before** the conversation starts. Every later one stays in its position as a user turn wrapped in `<system-reminder>…</system-reminder>`, so each step's prompt is the previous prompt plus new text and llama.cpp reuses its cache.
- The wrapper is written inside the first/last text block so Claude Code's two shapes for the same reminder (one-block list with `cache_control` when new, plain string when resent) render byte-identically.
- Fix B: the :8011 chat profile now sets `preserve_thinking: true` (as :8015 effectively already does), so its render is also strictly append-only.
- Root cause and repro scripts: design PR #2471 (`docs/superpowers/specs/2026-10-02-fcc-prompt-prefix-cache-design.md`).

## Outcome moved

Per-step prompt processing on a replay of 4 captured Claude Code steps on :8015, through the new function:
step 2-4 re-read **153 / 325 / 308 tokens in 0.9-1.3 s** (llama-server `f_keep = 1.000`), versus 12.9-13.6k tokens in 16-18 s with the old hoist (design doc). Expected live: ~0.5-4 s/step instead of ~38 s. Live harness turn after deploy: UNVERIFIED until deployed.

## Current architecture

`claude -p` (harness governor, Claude Code 2.1.287) -> `orion-llm-gateway` `/v1/messages` (`anthropic_passthrough.py`) -> llama-server `/v1/messages` on circe (:8011 chat, :8015 agent, b10398, `--parallel 1`). `normalize_anthropic_system_messages` hoisted every `role: system` message into `system`. :8015's template raises on a non-leading system message; :8011's silently drops it.

## Architecture touched

- `orion-llm-gateway` Anthropic passthrough request shaping (one function).
- `config/llm_profiles.yaml` chat-lane profile (llama-server launch flag on circe :8011).
- No bus, schema, or env changes.

## Files changed

- `services/orion-llm-gateway/app/anthropic_passthrough.py`: hoist only the leading run of system messages; convert later ones in place to wrapped user turns; no `system` key added when nothing was hoisted.
- `services/orion-llm-gateway/tests/test_anthropic_passthrough.py`: Claude-Code-shaped multi-step fixture (no real system prompt) asserting stable `system` and append-only `messages`; leading hoist kept; no mid-conversation system role forwarded; non-text edge blocks; route-level forwarding test updated.
- `services/orion-llm-gateway/README.md`: explains why later system messages stay in place.
- `config/llm_profiles.yaml`: `chat_template_kwargs: {preserve_thinking: true}` on `qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition`.
- `services/orion-llamacpp-host/tests/test_profile_forwarding.py`: pins the emitted `--chat-template-kwargs` and that `--reasoning-budget 0` is still emitted.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `/v1/messages` passthrough forwards mid-conversation system messages as `<system-reminder>` user turns in place, instead of appending them to `system`.
- Compatibility notes: leading system messages are still hoisted (the :8015 template requirement). Two adjacent user turns are accepted by llama.cpp and rendered as separate turns on both lanes (verified).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-llm-gateway/tests -q            -> 384 passed
pytest services/orion-llamacpp-host/tests/test_profile_forwarding.py -q
  -> new test passes; test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k fails identically on origin/main (pre-existing, needs LLM_PROFILE_NAME in env)
pytest services/orion-gpu-pool/tests -q              -> 147 passed, 15 skipped
scripts/check_gpu_pool_config.py, check_fcc_context_parity.py, check_circe_worker_refs.py -> ok
```

## Evals run

```text
Render check (/apply-template on each lane, 4 captured Claude Code steps, through the new function):
  :8015                         step1->2,2->3,3->4: 0 chars lost (pure append)
  :8011 (no preserve_thinking)  6 chars lost per step ("<think>" stripped from the previous turn)
  :8011 + preserve_thinking     0 chars lost (pure append)
Live replay on :8015 (max_tokens 1, slot idle first):
  step1 4591 tok (rest from host RAM cache) | step2 153 tok 874 ms | step3 325 tok 1152 ms | step4 308 tok 1192 ms
  llama-server log: "f_sim_best = 0.988 ... f_keep = 1.000" / "prompt eval time = 874.71 ms / 153 tokens" etc.
:8011 accepts the converted shape: 200, prompt eval 55 tokens, no template error.
```

No standing eval harness covers prompt-cache reuse; the acceptance check is live (below).

## Docker/build/smoke checks

```text
Not deployed (per task). Live checks hit llama-server directly with the new function, no gateway restart.
```

## Review findings fixed

Review subagent: no must-fix findings.

- Finding: a reminder between an assistant `tool_use` and its `tool_result` would be emitted as a user turn between them, which breaks tool pairing (the old hoist-all code could not do this).
  - Fix: such reminders are held and emitted right after the next message that does not await a tool result, or at the end of the request. This is still append-only.
  - Evidence: `test_reminder_between_tool_use_and_tool_result_waits_for_the_result`. The captured Claude Code order (reminder after the result) is unaffected; the render check was re-run after the fix and still shows 0 chars lost.
- Finding: `preserve_thinking` applies server-wide on :8011, not only to FCC.
  - Fix: the profile comment and the Risks section now say this.
  - Evidence: `config/llm_profiles.yaml` comment.
- Finding (nit): an empty late system message produced an empty reminder turn.
  - Fix: it is now dropped, the same as on the leading path.
  - Evidence: `test_empty_late_system_message_is_dropped`.
- Finding (nit): a malformed non-dict entry ended the leading run.
  - Fix: only a user or assistant message ends it now.
- Finding (nit): the README overstated "content preserved".
  - Fix: the wording now says the wrapper goes inside the edge text blocks, and the figures are sourced to the design doc.

## Restart required

```bash
# athena (gateway)
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-llm-gateway/.env -f services/orion-llm-gateway/docker-compose.yml up -d --build llm-gateway
# circe (chat lane, Fix B; config is baked into the image)
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-llamacpp-host/.env -f services/orion-llamacpp-host/docker-compose.yml -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml up -d --build --no-deps atlas-chat'
```

Live acceptance after deploy: during one real Orion harness turn, `ssh circe@circe 'docker logs orion-circe-atlas-llamacpp-chat'` should show `f_keep` ~1.0 from step 2 on and `prompt eval time` lines with hundreds to low thousands of tokens, not 20-30k.

## Risks / concerns

- Severity: medium. Concern: reminders now sit as user turns right after the tool result they refer to instead of in the system prompt; model behavior change is UNVERIFIED. Mitigation: watch the first live turns; revert is this one function.
- Severity: low. Concern: without Fix B deployed, :8011 strips the previous step's thinking whenever a reminder user turn appears (cache cost is only ~6 chars, but the model loses its own prior reasoning). Mitigation: deploy Fix B with the gateway.
- Severity: low. Concern: `preserve_thinking` is server-wide on :8011, so any client history that carries reasoning now keeps it in the prompt (more tokens). Most Hub/cortex chat carries none. Mitigation: revert the one profile line and rebuild atlas-chat.
- Severity: low. Concern: other traffic on a `--parallel 1` lane between two of Orion's steps still evicts the slot. Out of scope (GPU-pool hold design).
- FCC no-lease path untouched (out of scope).

## PR link

PR_LINK_PLACEHOLDER
