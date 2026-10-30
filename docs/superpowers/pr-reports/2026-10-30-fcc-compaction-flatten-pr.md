## Summary

- When the claude CLI compacts a long turn, orion-fcc now sends the whole history as ONE user message: a quoted `<transcript>`, then the CLI's own summary prompt. It still sends no tools.
- Root cause of the 5th "compaction lost the task": the earlier fix (#2802) took away the tools, but the history still ended in the middle of an agent loop. Qwen took the next step anyway and typed its tool call as plain text (`<tool_call><function=Bash>`). The CLI kept that text as the "summary".
- Tool calls go into the transcript as `[called tool X with {...}]`, followed by their results. Thinking blocks are dropped. Images and documents become `[image omitted]` / `[document omitted]`.
- New opt-in eval `evals/compaction_replay.py`. It replays one real logged compaction request in both shapes against the live lane.

## Outcome moved

The compaction summary keeps the task. Live failure, 2026-10-30 06:10Z, Orion transcript `7e25128b` ("v2 test of the body dig"). The tools were stripped (`fcc_compaction_tools_stripped`) and the model ran for 41 s, but the summary was a tool call. Same request, same `harness` lane, replayed:

| shape | time | `<summary>` | tool call as text |
|---|---|---|---|
| stripped only (old) | 3 s | no | yes |
| flattened (new) | 39 s | yes, opens with Juniper's request | no |

## Current architecture

claude CLI → orion-fcc `prepare_request` → llm-gateway → llama.cpp. Before this patch, `prepare_request` removed `tools`/`tool_choice` from compaction requests and left the history as it was. `guard_local_optimizations` (#2912) keeps FCC's title shortcut away from these requests.

## Architecture touched

`services/orion-fcc` only. No bus, schema or env changes.

## Files changed

- `services/orion-fcc/orion_fcc_messages_transport.py`: `flatten_compaction_text`, `_render_block`. `prepare_request` now flattens every compaction request, including ones sent without tools. The log line is now `fcc_compaction_flattened`.
- `services/orion-fcc/tests/test_messages_transport.py`: tests that the flattened request is one user turn, with no tools, that it keeps the request, tool call and result, that the prompt is the tail, that the guard still detects it, and that thinking and images are not leaked.
- `services/orion-fcc/evals/compaction_replay.py`: the live A/B replay.
- `services/orion-fcc/README.md`: the compaction section rewritten. It also corrects the stale claim that the title detector does not match.

## Schema / bus / API changes

- Behavior changed: compaction requests reach upstream as one user message, not the original multi-turn history. The log key was renamed from `fcc_compaction_tools_stripped` to `fcc_compaction_flattened`.

## Env/config changes

None.

## Tests run

```text
docker run (live FCC image) python -m unittest discover -s tests  -> Ran 33 tests, OK
```

## Evals run

```text
compaction_replay.py --line 288 (live request from 06:10Z)
stripped: 3s  summary_ok=False  ("...Let me check... <tool_call> <function=Bash>")
flattened: 39s summary_ok=True  ("1. Primary Request and Intent: Juniper requested: 'let's run a v2 test of the body dig...'")
```

## Docker/build/smoke checks

Tests ran in the deployed FCC image with the patched module on PYTHONPATH. Live deploy proof after merge: `fcc_compaction_flattened` in `docker logs orion-athena-fcc`, plus the next agent compaction summary containing `Summary:`. Until then: UNVERIFIED.

## Review findings fixed

- Finding: tool results and other blocks in the compaction turn landed under the previous ASSISTANT heading, or outside the quote (reminders, queued messages).
  - Fix: only text blocks holding a CLI prompt mark stay outside; everything else goes into the transcript under `### USER`.
  - Evidence: `test_extra_blocks_in_the_compaction_turn_stay_inside_the_quote_as_user`.
- Finding: a literal `</transcript>` in a tool result (agents read this file) closed the quote early.
  - Fix: escape any closing tag, case- and space-insensitive, to `<\/transcript>`.
  - Evidence: `test_a_quoted_closing_tag_cannot_end_the_transcript_early`.
- Finding: web search/fetch results and tool results that are lists of bare strings rendered blank.
  - Fix: render web results as clipped JSON (4,000 chars), and tool-result items through `_render_block`.
  - Evidence: `test_web_results_and_bare_string_tool_results_are_not_blank`.
- Re-ran live replay after the fixes: stripped 3 s summary_ok=False; flattened 45 s summary_ok=True.

## Restart required

```bash
scripts/safe_docker_build.sh orion-fcc up -d --build
```

## Risks / concerns

- Severity: low. Concern: the flattened transcript loses prompt-cache reuse with the turn's earlier requests, so the summary prefill is slower (39 s vs the cached 3 s). Mitigation: compaction is rare, and a correct summary is worth it.
- Severity: low. Concern: a CLI upgrade that rewords the prompt makes the match stop. Mitigation: no change from before; the governor's `fcc_compaction_degenerate` is still the signal.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
