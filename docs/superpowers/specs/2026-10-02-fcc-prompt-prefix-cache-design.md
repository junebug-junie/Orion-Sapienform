# FCC prompt prefix cache: why every step re-reads the whole prompt

Status: design only, nothing implemented. Investigation date 2026-10-02.

## Arsonist summary

Every step of an Orion harness turn makes the model re-read the whole 20-30k-token
prompt from scratch (~38 s per step on circe). The cause is **our own LLM gateway**, not
Claude Code, not llama.cpp and not the chat template.

Claude Code sends its per-step reminders as extra `role: "system"` messages at the **end**
of the conversation: a `<total_tokens>N tokens left</total_tokens>` line on every step,
plus the context-mode plugin's `PreToolUse:<Tool> hook additional context` after every
tool call. The gateway's `normalize_anthropic_system_messages`
(`services/orion-llm-gateway/app/anthropic_passthrough.py:51`) moves **all** of them into
the top-level `system` field. The Qwen chat template renders that `system` text right
after the tool list, before the first user message. So each step adds text in the middle
of the prompt, at the end of the system block. These Qwen3.6/3.8 models (hybrid/recurrent
layers) can only resume from a saved checkpoint near the end of the last prompt, so a
change that far back means re-reading everything.

Fix (owned by the gateway): stop moving system messages that come after the conversation
starts. Turn each one into a user turn in the same position. Tested by replaying the
captured requests on :8015: step-to-step prompt processing drops from **12.9-13.6k tokens
(16-18 s)** to **153-325 tokens (1.0-1.3 s)**.

## Current architecture

```text
claude -p (governor, Claude Code 2.1.287)
  -- with GPU lease --> orion-llm-gateway /v1/messages (anthropic_passthrough.py)   <- Orion's urgent runs
  -- no lease -------> orion-athena-fcc :8082 -> gateway /v1                         (FCC rejects role=system, see Risks)
gateway: normalize_anthropic_system_messages(body)  -> hoists EVERY role=system message into body.system
  -> llama-server /v1/messages (b10398) -> server_chat_convert_anthropic_to_oai -> jinja template
```

- `orion/harness/fcc_motor.py:782-791`: with a `gpu_lease`, `ANTHROPIC_BASE_URL` is the
  gateway itself. FCC is bypassed. Gateway log for the 2026-10-01 urgent run:
  `gpu_pool_lease_granted route=agent ... holder=http:anthropic role=chat url=http://100.112.254.99:8011`.
- Served templates (both lanes, `GET /props`) render
  `<|im_start|>system\n# Tools ... </tools> ...<IMPORTANT>...` + `\n\n` + **merged system text** +
  `<|im_end|>`, then the conversation. Tools come first, then the system text.
- The :8015 template raises `System message must be at the beginning.` for a system
  message in the middle of the conversation. The :8011 template silently drops it. That
  is why the gateway hoists in the first place.

## Root cause, with evidence

### 1. What Claude Code sends (captured)

A recording proxy sat in the harness-governor container between a real `claude -p` and
llama-server :8015. It was launched with `run_fcc_turn`'s own argv/env builders
(`launch.py`). There were 4 steps with tool calls (Bash, Read, Bash, answer). The message
layout in each request:

```text
req1: user[4 text]  system(SessionStart hook ctx, 13.5k chars)
req2: ...           assistant[thinking,tool_use] user[tool_result] system("<total_tokens>14987040 tokens left</total_tokens>")
req3: ...           assistant[...] user[tool_result] system("PreToolUse:Read hook additional context: <context_guidance>...")
req4: ...           assistant[...] user[tool_result] system("PreToolUse:Bash hook additional context: ...")
```

After the gateway's hoist, `system` had 5 -> 7 -> 9 -> 11 blocks, one more pair per step.

### 2. The exact diff point (rendered with the servers' own `/apply-template`)

The requests were converted with a mirror of llama.cpp b10398's
`server_chat_convert_anthropic_to_oai` (`render.py`) and rendered with each lane's
`/apply-template`. Then the prompt for step N was compared with the prompt for step N+1:

```text
:8015 step1->2 first diff char 47883 of 53264 (90%)
  prev: ...<total_tokens>15000000 tokens left</total_tokens>\n\nToday's date is 2026-10-02.<|im_end|>\n<|im_start|>user\n...
  next: ...<total_tokens>15000000 tokens left</total_tokens>\n\nToday's date is 2026-10-02.\n\n<total_tokens>14987040 tokens left</total_tokens><|im_end|>...
:8015 step2->3 first diff 47934/53755  next adds "\n\nPreToolUse:Read hook additional context: <context_guidance>..."
:8015 step3->4 first diff 48429/54866  next adds "\n\nPreToolUse:Bash hook additional context: ..."
:8011 same three diff points (47674/53055, 47725/53546, 48220/54657)
```

In every step the first changed character is right before the `<|im_end|>` that closes
the system block. Everything after it shifts. In the probe this is 88-90% of the way
into the prompt, because the probe's user prompt is tiny. Orion's real first user message
is 24,488 characters, so the same boundary lands around the ~57% llama-server reported
(`f_keep 0.57-0.64`). That 57% figure is consistent with this cause but was not
re-rendered from Orion's own request, because the request body is not persisted.

### 3. The live effect (llama-server :8015 log, probe session)

```text
f_sim_best = 0.882, f_keep = 0.886 -> prompt eval 16187 ms / 12877 tokens
f_sim_best = 0.863, f_keep = 0.879 -> prompt eval 16445 ms / 13014 tokens
f_sim_best = 0.854, f_keep = 0.868 -> prompt eval 17079 ms / 13323 tokens
                                    -> prompt eval 17678 ms / 13615 tokens
```

This is the same symptom as the urgent run: high similarity, full re-read every step.

### 4. Why "most of it matches" still costs a full re-read

This is the mid-edit test on :8015 (Qwen3.8-27B, `/completion`, `cache_prompt=true`,
7,090 tokens, `n_predict 4`; script `midedit.py`):

```text
cold              cache_n=0     prompt_n=7090 8618 ms
identical         cache_n=7086  prompt_n=4     233 ms
append sentence   cache_n=7086  prompt_n=10    401 ms
edit at 50%       cache_n=0     prompt_n=7086 8621 ms
restore original  cache_n=7086  prompt_n=4     251 ms   (host-RAM prompt cache)
edit at 95%       cache_n=6574  prompt_n=512   920 ms   (checkpoint ~512 tokens before the end)
```

:8015 behaves like :8011. It can resume from the end of the last prompt, or from a
checkpoint about 512 tokens before the end. An edit further back than that means
re-reading from token 0. So the only way to stay fast is for every new prompt to be
the old prompt plus new text on the end.

## Proposed schema / API changes

No bus, schema or env changes. This is a behavior change inside one gateway function.

**Fix A (required, gateway):** `normalize_anthropic_system_messages` should only hoist
**leading** system messages, meaning those before the first user/assistant message. Each
`role: "system"` message after that becomes a user message in the same position:

```json
{"role": "user", "content": [{"type": "text", "text": "<system-reminder>\n...\n</system-reminder>"}]}
```

`<system-reminder>` is the wrapper Claude Code already uses for the same kind of
content. Two user messages in a row are fine for llama.cpp, and both templates render
them as separate `<|im_start|>user` turns.

Verified with timings by replaying the 4 captured requests on :8015 (`replay.py`,
`max_tokens=1`):

```text
current hoist:  step1 12877 tok 16.3s | step2 13014 tok 16.5s | step3 13323 tok 17.2s | step4 13615 tok 17.8s  (cache_read 0 every step)
in-place fix:   step1 12893 tok 17.0s | step2   153 tok  1.0s | step3   325 tok  1.2s | step4   308 tok  1.3s  (cache_read 12893/13046/13371)
```

Rendered diff points with the fix (`/apply-template`):

- :8015 (server already runs `--chat-template-kwargs {"preserve_thinking":true}`):
  each new prompt is the old prompt plus new text, 0 characters lost.
- :8011 (no `preserve_thinking`): the reminder user turn becomes the template's "last
  query" on every step. The template then drops the previous step's `<think>` block, so
  the diff moves to the very end of the old prompt: `assistant\n<think>\n` vs
  `assistant\n<tool_call>`, 6 characters. That falls inside the end-of-prompt checkpoint,
  so only the previous step's generated tokens are re-read.

**Fix B (recommended, chat-lane server flag, operator change):** add
`--chat-template-kwargs '{"preserve_thinking":true}'` to the :8011 llama-server launch,
the same as :8015 already has. With it, the :8011 render is also pure append (verified
with `/apply-template`, 0 characters lost on all 3 steps). The alternative is having the
gateway inject `chat_template_kwargs: {"preserve_thinking": true}` into
`/v1/messages` bodies. llama.cpp passes `chat_template_kwargs` through
(`server-chat.cpp`). It is cheaper to roll back than a server flag, but it is per-request
plumbing. Pick one. The server flag is simpler and matches :8015.

Not recommended: dropping `total_tokens_reminder`. It is only one of the per-step
injections. The PreToolUse hook context breaks the cache the same way.

## Expected per-step time after fix

The probe went from 16-18 s per step to 1.0-1.3 s. For a live Orion turn (20-30k
prompt, about 790 tokens/s prompt processing on :8011), per-step prompt time becomes
`new tokens / 790`. The new tokens are the previous step's output, its tool results and
one reminder: usually 0.3-3k tokens, so about **0.5-4 s instead of about 38 s**. Large
tool outputs still cost their own size, which is correct.

## Files likely to touch

- `services/orion-llm-gateway/app/anthropic_passthrough.py`: `normalize_anthropic_system_messages`.
- `services/orion-llm-gateway/tests/`: a test of the passthrough normalizer.
- `services/orion-llm-gateway/README.md`: one paragraph on why later system messages stay in place.
- circe llama-server launch config for the chat lane (Fix B), which lives outside this
  repo's gateway. The operator applies it.

## Non-goals

- Changing Claude Code, the FCC translator, or llama.cpp.
- Changing the llama.cpp checkpoint or prompt-cache settings (`--ctx-checkpoints`,
  `--cache-ram`). The behavior is understood and the fix makes it irrelevant.
- Fixing cache loss from **other traffic** on the same single-slot lane between two of
  Orion's steps. `--parallel 1` means anyone else's request evicts the slot. The
  host-RAM prompt cache can sometimes restore it ("restore original" above), but that is
  a separate question for the GPU-pool hold design.
- The FCC (no-lease) path. See Risks.

## Acceptance checks

1. **Gate test (deterministic, no network):** build two Claude-Code-shaped request bodies
   for step N and step N+1. Each has the trailing `role: system` reminder the captured
   requests had (fixture copied from `rec/00{1,2}_msg_in.json`'s shape). Assert:
   (a) `forward(N+1)["system"] == forward(N)["system"]`;
   (b) `forward(N)["messages"]` is a prefix of `forward(N+1)["messages"]`, compared as
   JSON values;
   (c) a request that **starts** with a system message still has it hoisted, so the
   :8015 template does not raise.
2. **Render check:** `render.py` on a fresh capture shows each new prompt equals the old
   prompt plus new text on both lanes. Before the fix, the first diff sits right before
   the system block's `<|im_end|>`.
3. **Live check (the one that counts):** during one real Orion harness turn after deploy,
   `docker logs orion-circe-atlas-llamacpp-chat` shows `f_keep` ≈ 1.0 from step 2 on.
   The `prompt eval time = X ms / N tokens` lines should show N ≈ new tokens (hundreds
   to low thousands), not the full 20-30k, and the Anthropic `usage.cache_read_input_tokens`
   should be greater than 0.

## Risks / open questions

- **FCC path (no lease):** FCC's request model has `role: Literal["user", "assistant"]`
  (`api/models/anthropic.py:145` in the FCC image), so Claude Code's trailing system
  messages may be rejected or mangled on that path. UNVERIFIED. It was not exercised here
  because Orion's urgent runs take the lease path. FCC also strips `thinking` blocks
  (`ENABLE_MODEL_THINKING=false`). Under `preserve_thinking` that would re-render an
  empty `<think>` where the model generated real thinking, which loses only the tail
  (that step's own output).
- **:8011 and `--reasoning-budget 0`:** Orion's 10-01 transcript still contains real
  `thinking` text from that run. Whether the budget is honored is UNVERIFIED and does not
  affect this fix.
- **Prompt semantics:** reminders move from the system prompt into user turns. This is
  where Claude Code puts them for real Anthropic models, so behavior should match
  upstream more closely, not less. Watch the first live turns anyway.

## Reproduction scripts (in `2026-10-02-fcc-prompt-prefix-cache/`)

- `proxy.py`: recording Anthropic proxy that applies the gateway's hoist and forwards to llama-server.
- `launch.py`: spawns `claude -p` with `run_fcc_turn`'s argv/env builders (no per-turn MCP config), pointed at the proxy.
- `render.py`: mirror of llama.cpp b10398's Anthropic->OAI conversion plus the server's `/apply-template`.
- `replay.py`: replays captured requests with `hoist` or `inplace` shaping, `max_tokens=1`.
- `midedit.py`: `/completion` partial-reuse test.

Captured request bodies are not committed. They contain the full Claude Code system
prompt and environment context.

## Recommended next patch

`fix/gateway-inplace-system-reminders`: Fix A plus gate test 1, deployed to
`orion-llm-gateway`, verified with live check 3 on one urgent or harness turn. Then ask
the operator to add Fix B's flag to the :8011 launch.
