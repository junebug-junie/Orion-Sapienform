## Summary

- Adds `SmolagentsCodeEngine` to `orion-context-exec`: a real REPL-based reasoning engine that runs `smolagents CodeAgent` with a local coder model via the `agent` lane (Circe node / llama-cola)
- The agent writes Python code to call 4 read-only tools (`repo_grep`, `repo_read`, `repo_list`, `recall_query`), observes results, and loops — replacing the fake/hallucinating engines for agent mode
- Activate with `CONTEXT_EXEC_RLM_ENGINE=smolcode`; all existing engine paths unchanged

## Changes

**`repo_tools.py`** — adds `repo_list(path)` for directory navigation

**`smolcode_engine.py`** — new engine: `OrionSmolagentsModel(Model)` subclasses `smolagents.models.Model`, implements `generate()` (smolagents 1.26 API); bridges async `OrganRuntime.llm_chat(route="agent")` via `run_coroutine_threadsafe`; lazy-imports `CodeAgent` inside `run()`

**`rlm_engine.py`** — wires `build_engine("smolcode")`

**`requirements.txt`** — adds `smolagents>=1.2.0`

**`.env_example`** — documents `CONTEXT_EXEC_RLM_ENGINE=smolcode` option

## Test plan

- [ ] `make test SERVICE=orion-context-exec ARGS="services/orion-context-exec/tests/test_smolcode_engine.py -v"` — 6 tests pass
- [ ] `test_smolcode_engine_real_agent_loop` — real `CodeAgent` (not mocked) verifies smolagents 1.26 `generate()` API contract end-to-end
- [ ] Manual smoke: set `CONTEXT_EXEC_RLM_ENGINE=smolcode`, confirm ≥2 tool calls before `final_answer`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
