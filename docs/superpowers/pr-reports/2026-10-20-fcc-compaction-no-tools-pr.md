## Summary

- The FCC proxy now sends the claude CLI's compaction (summarize) request upstream with no tools attached.
- Cause found: the CLI compacts with the turn's tools still attached and asks only in text for no tool calls ("Tool calls will be REJECTED"). Claude models obey. The local Qwen lanes often call a tool, and the CLI keeps the text before that call as the "summary" ("Now let me read the remaining key files..."). The task is then lost.
- Detection: the last turn is a user message holding one of the CLI's fixed texts. These are its "TEXT ONLY, do not call any tools" preamble, or the opening its full and partial summary prompts share. Every other request passes through unchanged.
- The patch script now fails loudly if an upstream anchor is missing, instead of silently skipping it.

## Outcome moved

The failure: 33 of 51 live compactions on 2026-10-17 had no `Summary:` section (`fcc_compaction_degenerate`, PR #2744). Their "summaries" were the text a model writes just before a tool call. With no tools offered, the model can only answer in text, which is what the CLI's prompt asks for.

## Current architecture

claude CLI → orion-fcc (pinned free-claude-code, patched at image build by `install_transport_patch.py`) → llama.cpp Anthropic `/v1/messages`. The proxy forwarded the compaction request as-is, tools included. The anchors from #2749 and #2773 limit the damage for self-inquiry and focus runs only.

## Architecture touched

- `services/orion-fcc/orion_fcc_messages_transport.py`: new `COMPACTION_PROMPT_MARKS`, `is_compaction_request` and `prepare_request`.
- `services/orion-fcc/install_transport_patch.py`: the route now calls `handler.create(prepare_request(request_data))`, and anchors are asserted (`_replace`).

## Files changed

- `services/orion-fcc/orion_fcc_messages_transport.py`: detection and tool strip, with a log line `fcc_compaction_tools_stripped` on the `uvicorn.error` logger (the one that reaches `docker logs`).
- `services/orion-fcc/install_transport_patch.py`: the route rewrite, plus anchor assertions.
- `services/orion-fcc/tests/test_messages_transport.py`: 5 tests against the real pinned `MessagesRequest` and the patched route.
- `services/orion-fcc/README.md`: a section on the compaction strip.

## Schema / bus / API changes

- Added: none.
- Behavior changed: compaction requests reach llama.cpp without `tools` and `tool_choice`. Earlier `tool_use` and `tool_result` blocks in the history are kept.
- Compatibility notes: if a CLI upgrade rewords both marks, the match stops and requests pass through as today. Nothing breaks; the governor's `fcc_compaction_degenerate` count shows it.

## Env/config changes

None.

## Tests run

```text
docker build -f services/orion-fcc/Dockerfile -t orion-fcc:compaction-test .   (patch applied, anchors asserted)
docker run ... orion-fcc:compaction-test -m unittest discover -s /tests
  Ran 18 tests ... OK   (13 existing + 5 new)
```

## Evals run

```text
No eval harness for compaction quality. Live check after deploy:
  docker logs orion-athena-fcc | grep fcc_compaction_tools_stripped
  governor log: fcc_context_compacted not followed by fcc_compaction_degenerate for the same corr.
```

## Docker/build/smoke checks

```text
Both deployed CLIs (governor, hub): Claude Code 2.1.304; both contain both marks (grep of the binary).
CLI 2.1.304 summary wordings: "...of the conversation so far", "...of the RECENT portion of the conversation", "...of this conversation".
Live llama.cpp lanes 8011/8015/8016/8012, POST /v1/messages/count_tokens with tool_use + tool_result history and no tools:
  {"input_tokens":100} / 140 / 140 / 88. All four accepted it (conversion and template, no generation).
/apply-template (OpenAI shape) on the same four: rendered, past tool calls kept, no tools block.
```

## Review findings fixed

- Finding: the live Anthropic path was unverified, because `/apply-template` is OpenAI-shaped.
  - Fix: smoked `/v1/messages/count_tokens` on all four lanes.
  - Evidence: all four returned `input_tokens`, with no template error.
- Finding: partial-compaction requests use a different wording and would still carry tools.
  - Fix: the marks are now the shared opening "Your task is to create a detailed summary of " plus the no-tools preamble.
  - Evidence: the new test covers the "RECENT portion" wording.
- Finding: the new tests sat after `if __name__ == "__main__"`.
  - Fix: the guard was moved to the end of the file.
- Finding: the patch script's `str.replace` would silently skip a missing anchor.
  - Fix: `_replace` raises `FCC patch anchor missing`.
  - Evidence: the image build ran it successfully.
- Finding (nit): removing tools exposes the request to FCC's no-tools intercepts.
  - Fix: documented in the README and the docstring. Those intercepts match title, filepath and safety-classifier shapes, never a CLI transcript.

## Restart required

```bash
scripts/safe_docker_build.sh orion-fcc up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: **UNVERIFIED live**. Curiosity is paused (cabinet_hot), so compactions are rare right now.
  - Mitigation: the log checks above.
- Severity: low.
  - Concern: a model can still refuse in text ("I cannot provide a traditional summary..."). This patch does not cover that.
  - Mitigation: the governor still flags it.
- Severity: low.
  - Concern: matching depends on the CLI's prompt text.
  - Mitigation: if a CLI upgrade rewords both marks, the strip silently stops (no breakage). It is visible through `fcc_compaction_degenerate`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2802

🤖 Generated with [Claude Code](https://claude.com/claude-code)
