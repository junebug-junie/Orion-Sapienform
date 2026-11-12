# Qwen3.5 9B fast-lane trial on Circe

## Scope

Only `atlas-fast` / pool role `fast` / port 8013 changes model. It keeps four
4,096-token slots (`--ctx-size 16384 --parallel 4`) on Circe GPU3. `atlas-metacog`
on port 8012 stays on `qwen3-8b-q5km-v100-16gb-atlas-metacog-16k`.

The trial profile is `qwen35-9b-udq5kxl-v100-32gb-circe-fast-4k` in
`config/llm_profiles.yaml`, using Unsloth's
`Qwen3.5-9B-UD-Q5_K_XL.gguf` (6,743,680,224 bytes; SHA-256
`96cf42ddb97f9572410a72b9ed6f2299b1e887ee08da4c2a6c01e897cfa9f673`).
The original Qwen3 8B Q4 profile remains available for rollback.

The server defaults to thinking off but keeps `--reasoning auto` and **no**
launch-wide reasoning budget. A caller can set
`options.chat_template_kwargs.enable_thinking` to `true` or `false`; the gateway
already forwards this option. Direct OpenAI requests can send
`chat_template_kwargs` in their JSON body. Structured JSON calls should use
thinking off. The gateway disables thinking only when a caller supplies a
`structured_output_schema` and selects `structured_output_thinking_policy=disabled_for_artifact`.

## Fast-worker caller census

This is a read-only query of `llm_call_usage` for the seven days before the
trial, filtered to `pool_role='fast' AND recorder='gateway'`. It includes calls
that reached a model; pool refusals and timeouts before model admission are not
in this table. `node`, `holder`, and `verb` are the ledger's own attribution.

| Origin / route | Calls | Non-served | Behavior to inspect |
| --- | ---: | ---: | --- |
| `orion-topic-foundry`, `quick` | 8,051 | 8 | JSON extraction, topic labels, completion truncation in `app/services/training.py` |
| `http:openai`, `quick_background` | 3,740 | 0 | Likely AI Town NPC dialogue (`services/orion-ai-town/scripts/wire_llm_gateway.sh` sets that route); ledger has no node, so this attribution is inferred |
| `cortex-exec`, `quick_turn` | 1,074 | 22 | Pre-turn repair-pressure logprob probe (`app/pre_turn_appraisal.py`); failed/unparseable result must stay unknown |
| `cortex-orch`, `quick_background` | 414 | 0 | Typed memory annotation JSON (`app/memory_extractor.py`); inspect parse/fallback counts |
| `hub`, `quick` | 11 | 0 | Hub quick work, including memory-graph suggestion and retries |
| `cortex-exec`, `journal.compose`, `quick_background` | 7 | 0 | Journal composition output and its JSON parser |
| Other metacog-class grants served by `fast` | 7 | 0 | Pool fallback can put metacog work on fast despite the metacog worker being unchanged |

The ledger's complete history also has two `orion-memory-consolidation` calls
on `quick`; none occurred in the seven-day window. The row above for
metacog-class grants includes `orion-mind`, `vision-council`, and `cortex-exec`.

The last 24 hours before the trial had 571 `http:openai` calls, 413 topic-foundry
calls, 190 pre-turn probes (6 non-served), 88 cortex-orch calls, and 10 Hub calls
on the fast role. The pre-turn probe's largest recorded prompt was 4,082 tokens,
close to the 4,096-token slot boundary; reserve room for generation.

Static paths that may use fast even if absent from the seven-day ledger:

- `orion-cortex-exec/app/executor.py` maps `chat_quick`, `chat_kids_story`,
  `chat_general` stance work, and `introspect_spark` to `quick`; autonomous
  work can be demoted to `quick_background`.
- `orion-embodiment` speech uses `quick_embodied` at interactive priority;
  its optional Hub mode uses `quick`.
- `orion-actions` world-pulse journaling uses `quick_background`.
- `orion-memory-consolidation/app/classify.py` can fall back to `quick`.
- The gateway default route is `quick`, so callers without an explicit route
  are also in scope. The pool's `metacog` class currently lists `fast` as a
  fallback; this trial does not change that routing policy.

To refresh the census on Athena:

```sql
SELECT coalesce(node,''), coalesce(holder,''), coalesce(verb,''),
       coalesce(route,''), path, count(*) AS calls,
       count(*) FILTER (WHERE outcome <> 'served') AS non_served
FROM llm_call_usage
WHERE recorded_at > now() - interval '7 days'
  AND pool_role = 'fast' AND recorder = 'gateway'
GROUP BY 1,2,3,4,5 ORDER BY calls DESC;
```

## Isolated Circe evidence

Using Circe's existing llama.cpp build 10398, a shadow server on GPU3 with
`--reasoning auto`, launch default `enable_thinking=false`, and the exact new
GGUF booted at four 4k slots. It ran alongside the existing fast, metacog, and
Kev processes; GPU3 used 26,421 MiB with 6,074 MiB free during the probe.
The GGUF SHA-256 matched Unsloth's published checksum.

Direct probes passed: non-thinking text had no `reasoning_content`; thinking-on
text had `reasoning_content` and a final answer; strict JSON prompts parsed;
logprobs were present; a required tool call had typed integer arguments; four
simultaneous codeword prompts returned only their own codewords. The first
request spent ~48 seconds in prompt evaluation after cold load; warmed requests
were much faster. A thinking-on call capped at 128 tokens ended with empty
`content` and `finish_reason=length`; at 512 tokens the same type of call
finished with final content. Callers enabling thinking need a sufficient output
budget and room inside the 4k slot.

One JSON quirk surfaced: a weak prompt ("Return JSON with status ok and count 3")
with `response_format: {"type":"json_object"}` sometimes returned a fenced
` ```json ` block even though `finish_reason=stop`. A schema-bearing response
format returned bare JSON in that probe. The actual memory-annotation prompt
begins "Return strict JSON only"; five repeated direct probes with its response
shape parsed as bare JSON. Topic Foundry's caller also says "STRICT JSON only"
and already extracts fenced JSON defensively. The eval checks both strict JSON
and the Topic Foundry labels shape. Watch cortex-orch's
`memory_extractor_annotation_unparseable_text` log after rollout; its parser
uses strict `json.loads` and falls back to regex when the model disobeys.

Re-run the executable check against a shadow or deployed fast worker:

```bash
python3 services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py \
  --base-url http://127.0.0.1:18013
```

## Rollout watch and rollback

The checked-in `.env_example` does not overwrite Circe's existing local `.env`.
Set only `ATLAS_FAST_PROFILE_NAME=qwen35-9b-udq5kxl-v100-32gb-circe-fast-4k`
in Circe's `services/orion-llamacpp-host/.env` before deploying `atlas-fast`;
leave `ATLAS_METACOG_PROFILE_NAME` unchanged. Until that worker is restarted and
`/props` reports the new model, the production fast rail is **UNVERIFIED**.

After deployment, verify `/health`, `/slots` (four 4096-token slots), `/props`
(new model path), and GPU3 VRAM with Kev and metacog still running. Run the eval
against port 8013 and then watch each caller above in `llm_call_usage`, gateway
`upstream_error`/`context_overflow` logs, topic-foundry label artifacts,
pre-turn probe failure reasons, cortex-orch annotation parses, and AI Town NPC
dialogue. A server boot is not proof that those caller rails moved.

If parse, latency, or error rates regress, set `ATLAS_FAST_PROFILE_NAME` back to
`qwen3-8b-q4km-v100-16gb-balanced` and deploy from clean synchronized main.
This trial does not change the metacog worker profile.
