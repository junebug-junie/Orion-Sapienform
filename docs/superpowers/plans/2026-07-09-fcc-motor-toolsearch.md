# FCC motor ToolSearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set `ENABLE_TOOL_SEARCH=true` on every FCC Claude spawn via the shared subprocess env helper so MCP schemas stay deferred until ToolSearch pulls them, without changing MCP attachment or adding stance/intent gates.

**Architecture:** One `setdefault` in `orion/fcc/context_budget.extend_fcc_subprocess_env` covers both consumers (harness motor + hub agent-claude). Tests assert the helper and both `_build_subprocess_env` paths. Docs note the override; GitHub toolset defaults are re-verified, not redesigned.

**Tech Stack:** Python 3.11+, pytest, existing FCC MCP render path (`orion/fcc/mcp_config.py`), Claude Code env contract (`ENABLE_TOOL_SEARCH`).

**Spec:** `docs/superpowers/specs/2026-07-09-fcc-motor-toolsearch-design.md`

**Branch:** `feat/fcc-motor-toolsearch` (or fresh worktree from `main` per AGENTS.md)

---

## File map

| File | Responsibility |
|------|----------------|
| `orion/fcc/context_budget.py` | `extend_fcc_subprocess_env`: `setdefault("ENABLE_TOOL_SEARCH", "true")` |
| `orion/fcc/tests/test_context_budget.py` | Unit: default + override preservation |
| `orion/harness/tests/test_fcc_motor_mcp.py` | Motor spawn env includes ToolSearch |
| `services/orion-hub/tests/test_fcc_claude_bridge_run.py` | Hub bridge spawn env includes ToolSearch |
| `services/orion-hub/.env_example` | Comment near MCP flags |
| `services/orion-harness-governor/.env_example` | Same |
| `services/orion-hub/README.md` | Short ToolSearch note on FCC MCP section |

**Do not modify:** MCP template servers, stance/imperative classifiers, mid-turn MCP attach/respawn, Claude built-in tool allowlists, `GITHUB_TOOLSETS` default logic in `mcp_config.py` (already correct — only re-run existing tests).

**Already covered (no new code):** `services/orion-hub/tests/test_fcc_mcp_config.py::test_render_defaults_github_toolsets_lean_and_read_only` and `orion/harness/tests/test_fcc_motor_mcp.py::test_maybe_render_mcp_config_defaults_github_toolsets_when_env_absent` assert `GITHUB_TOOLSETS=repos,pull_requests` when FCC env omits it. Task 4 re-runs them as the acceptance gate.

---

### Task 1: Failing tests for `extend_fcc_subprocess_env` ToolSearch

**Files:**
- Modify: `orion/fcc/tests/test_context_budget.py`

- [ ] **Step 1: Write the failing tests**

Append to `orion/fcc/tests/test_context_budget.py`:

```python
from orion.fcc.context_budget import extend_fcc_subprocess_env


def test_extend_fcc_subprocess_env_sets_enable_tool_search_when_unset() -> None:
    env: dict[str, str] = {}
    extend_fcc_subprocess_env(env)
    assert env["ENABLE_TOOL_SEARCH"] == "true"


def test_extend_fcc_subprocess_env_preserves_explicit_tool_search_override() -> None:
    env = {"ENABLE_TOOL_SEARCH": "false"}
    extend_fcc_subprocess_env(env)
    assert env["ENABLE_TOOL_SEARCH"] == "false"


def test_extend_fcc_subprocess_env_preserves_auto_tool_search_override() -> None:
    env = {"ENABLE_TOOL_SEARCH": "auto:5"}
    extend_fcc_subprocess_env(env)
    assert env["ENABLE_TOOL_SEARCH"] == "auto:5"
```

Also add `extend_fcc_subprocess_env` to the existing import block at the top of the file (and remove the duplicate inline import above if you prefer a single import site):

```python
from orion.fcc.context_budget import (
    CONTEXT_PRESSURE_NUDGE,
    annotate_harness_step,
    apply_context_overflow_hint,
    build_context_pressure_step,
    context_risk_level,
    extend_fcc_subprocess_env,
    is_context_overflow_text,
    measure_step_payload_chars,
)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest orion/fcc/tests/test_context_budget.py::test_extend_fcc_subprocess_env_sets_enable_tool_search_when_unset \
  orion/fcc/tests/test_context_budget.py::test_extend_fcc_subprocess_env_preserves_explicit_tool_search_override \
  orion/fcc/tests/test_context_budget.py::test_extend_fcc_subprocess_env_preserves_auto_tool_search_override -v
```

Expected: FAIL with `KeyError: 'ENABLE_TOOL_SEARCH'` (or AssertionError if the key is missing / wrong) on the unset case.

- [ ] **Step 3: Commit tests**

```bash
git add orion/fcc/tests/test_context_budget.py
git commit -m "test(fcc): require ENABLE_TOOL_SEARCH in extend_fcc_subprocess_env"
```

---

### Task 2: Implement ToolSearch setdefault

**Files:**
- Modify: `orion/fcc/context_budget.py` (function `extend_fcc_subprocess_env`, after the existing `setdefault` lines ~97–99)

- [ ] **Step 1: Add one line to the helper**

In `extend_fcc_subprocess_env`, after:

```python
    env.setdefault("ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS", str(mcp_tool_result_max_chars()))
    env.setdefault("HARNESS_FCC_CONTEXT_PRESSURE_PCT", str(context_pressure_threshold_pct()))
    env.setdefault("HARNESS_FCC_MAX_CONTEXT_TOKENS", str(max_context_tokens()))
```

add:

```python
    # Claude Code: custom ANTHROPIC_BASE_URL disables ToolSearch unless set explicitly.
    # Keep MCP attached; defer schema load until the model ToolSearches.
    env.setdefault("ENABLE_TOOL_SEARCH", "true")
```

Do not set the key with `env["ENABLE_TOOL_SEARCH"] = "true"` — that would clobber operator debug overrides (`false` / `auto` / `auto:N`).

- [ ] **Step 2: Run helper tests to verify they pass**

```bash
pytest orion/fcc/tests/test_context_budget.py -q
```

Expected: PASS (all tests in the file, including the three new ones).

- [ ] **Step 3: Commit**

```bash
git add orion/fcc/context_budget.py
git commit -m "feat(fcc): enable ToolSearch on FCC Claude subprocess env"
```

---

### Task 3: Consumer spawn-env tests (motor + hub)

**Files:**
- Modify: `orion/harness/tests/test_fcc_motor_mcp.py`
- Modify: `services/orion-hub/tests/test_fcc_claude_bridge_run.py`

Both consumers already call `extend_fcc_subprocess_env` from `_build_subprocess_env` / `_fcc_context_env`. These tests lock the contract at the spawn boundary so a future refactor that bypasses the helper fails loudly.

- [ ] **Step 1: Write failing motor test**

Append to `orion/harness/tests/test_fcc_motor_mcp.py` (near `test_build_subprocess_env_sets_context_ceiling`):

```python
def test_build_subprocess_env_enables_tool_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_TOOL_SEARCH", raising=False)
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["ENABLE_TOOL_SEARCH"] == "true"


def test_build_subprocess_env_preserves_tool_search_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "false")
    env = motor._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["ENABLE_TOOL_SEARCH"] == "false"
```

Note: `_build_subprocess_env` starts from `os.environ.copy()`, so the override test must set the process env via `monkeypatch.setenv`, not only a local dict.

- [ ] **Step 2: Write failing hub bridge test**

Append to `services/orion-hub/tests/test_fcc_claude_bridge_run.py` (near the other `_build_subprocess_env` tests):

```python
@pytest.mark.asyncio
async def test_build_subprocess_env_enables_tool_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_TOOL_SEARCH", raising=False)
    env = bridge._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["ENABLE_TOOL_SEARCH"] == "true"


@pytest.mark.asyncio
async def test_build_subprocess_env_preserves_tool_search_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "auto")
    env = bridge._build_subprocess_env(fcc_server_url="http://127.0.0.1:8082", auth_token="tok")
    assert env["ENABLE_TOOL_SEARCH"] == "auto"
```

- [ ] **Step 3: Run consumer tests**

```bash
pytest orion/harness/tests/test_fcc_motor_mcp.py::test_build_subprocess_env_enables_tool_search \
  orion/harness/tests/test_fcc_motor_mcp.py::test_build_subprocess_env_preserves_tool_search_override \
  services/orion-hub/tests/test_fcc_claude_bridge_run.py::test_build_subprocess_env_enables_tool_search \
  services/orion-hub/tests/test_fcc_claude_bridge_run.py::test_build_subprocess_env_preserves_tool_search_override -v
```

Expected: PASS (Task 2 already landed the helper; these assert the call chain). If either fails with missing key, the consumer is not calling `extend_fcc_subprocess_env` — fix the call site, do not duplicate `ENABLE_TOOL_SEARCH` in motor/hub.

- [ ] **Step 4: Commit**

```bash
git add orion/harness/tests/test_fcc_motor_mcp.py \
  services/orion-hub/tests/test_fcc_claude_bridge_run.py
git commit -m "test(fcc): assert ToolSearch on motor and hub spawn env"
```

---

### Task 4: Re-verify GitHub toolset restriction (no code change)

**Files:**
- Test only (existing): `services/orion-hub/tests/test_fcc_mcp_config.py`
- Test only (existing): `orion/harness/tests/test_fcc_motor_mcp.py`

- [ ] **Step 1: Run the existing toolset gates**

```bash
pytest \
  services/orion-hub/tests/test_fcc_mcp_config.py::test_render_defaults_github_toolsets_lean_and_read_only \
  services/orion-hub/tests/test_fcc_mcp_config.py::test_render_github_toolsets_override_from_env \
  orion/harness/tests/test_fcc_motor_mcp.py::test_maybe_render_mcp_config_defaults_github_toolsets_when_env_absent \
  orion/harness/tests/test_fcc_motor_mcp.py::test_maybe_render_mcp_config_passes_github_toolsets -v
```

Expected: PASS. Defaults remain `repos,pull_requests` when `GITHUB_TOOLSETS` is absent from FCC env; explicit FCC env overrides still win; docker `-e GITHUB_TOOLSETS` passthrough still present.

If any fail, stop — that is a regression outside ToolSearch and must be fixed before docs. Do **not** invent a new Orion tool allowlist.

- [ ] **Step 2: No commit if nothing changed**

If all green and no file edits, skip commit. If you had to fix a real regression, commit that fix separately with a clear message.

---

### Task 5: Env example comments + hub README

**Files:**
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-harness-governor/.env_example`
- Modify: `services/orion-hub/README.md`

Do **not** add a required `ENABLE_TOOL_SEARCH=` key to `.env_example` (happy path is code-set). Comment-only documentation avoids env sync churn. If you choose to add an optional commented line `# ENABLE_TOOL_SEARCH=true`, that is fine; still no required key and no `settings.py` field.

- [ ] **Step 1: Hub `.env_example` comment**

Immediately after the MCP block around:

```text
HUB_AGENT_CLAUDE_MCP_ENABLED=true
```

insert:

```text
# FCC Claude spawns set ENABLE_TOOL_SEARCH=true via orion.fcc.context_budget.extend_fcc_subprocess_env
# (defers MCP schema load on custom ANTHROPIC_BASE_URL). Override only for debug: false | auto | auto:N
```

- [ ] **Step 2: Harness-governor `.env_example` comment**

Immediately after:

```text
HARNESS_FCC_MCP_ENABLED=true
```

insert the same two comment lines:

```text
# FCC Claude spawns set ENABLE_TOOL_SEARCH=true via orion.fcc.context_budget.extend_fcc_subprocess_env
# (defers MCP schema load on custom ANTHROPIC_BASE_URL). Override only for debug: false | auto | auto:N
```

- [ ] **Step 3: Hub README note**

In `services/orion-hub/README.md`, in the section `### fcc-claude MCP (GitHub + Firecrawl + AI Town)`, after the paragraph that starts with `When \`HUB_AGENT_CLAUDE_MCP_ENABLED=true\``, add:

```markdown
**ToolSearch:** FCC Claude subprocesses set `ENABLE_TOOL_SEARCH=true` through `orion.fcc.context_budget.extend_fcc_subprocess_env` (shared with harness-governor). MCP servers stay attached; Claude Code loads tool schemas into context only when ToolSearch pulls them. This counters the custom-`ANTHROPIC_BASE_URL` fallback that otherwise dumps all MCP schemas at spawn. Operators may override `ENABLE_TOOL_SEARCH` in the process environment for debugging (`false` / `auto` / `auto:N`). Optional further GitHub surface tightening: set `GITHUB_TOOLSETS` in `~/.fcc/.env` (code default remains `repos,pull_requests` when unset).
```

- [ ] **Step 4: Env sync only if a real key was added**

If `.env_example` gained only comments (recommended), skip sync.

If you added an actual `ENABLE_TOOL_SEARCH=` key line, run:

```bash
python scripts/sync_local_env_from_example.py
```

and report any skipped keys.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/.env_example \
  services/orion-harness-governor/.env_example \
  services/orion-hub/README.md
git commit -m "docs(fcc): document ToolSearch on FCC MCP spawn path"
```

---

### Task 6: Full gate + live smoke note

**Files:** none (verification only)

- [ ] **Step 1: Run focused unit gates**

```bash
pytest orion/fcc/tests/test_context_budget.py \
  orion/harness/tests/test_fcc_motor_mcp.py \
  services/orion-hub/tests/test_fcc_claude_bridge_run.py \
  services/orion-hub/tests/test_fcc_mcp_config.py \
  services/orion-hub/tests/test_fcc_claude_bridge_mcp.py -q
```

Expected: PASS.

- [ ] **Step 2: Live smoke (when runnable) — else mark UNVERIFIED**

After restarting consumers that spawn FCC Claude:

```bash
# Restart (print for Juniper; do not sudo from the agent):
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build
```

Smoke intent (operator-observed, not a new automated eval in this patch):

1. Ordinary chat turn: context should **not** advertise/load the full ~71 MCP tool schema set at spawn.
2. A turn that needs GitHub/Firecrawl should still reach those tools via ToolSearch then MCP.

If live smoke cannot run in this environment, PR report must say `UNVERIFIED` for the live path and include the restart + observation steps above. Do **not** add a stance/intent MCP gate as a fallback in this PR. If the proxy rejects `tool_reference` or schemas still dump, status is `DONE_WITH_CONCERNS` with a follow-up on FCC/gateway ToolSearch support.

- [ ] **Step 3: Code review subagent + PR report**

Per AGENTS.md: run code review skill in a subagent, fix material findings, push branch, produce the standard Markdown PR description. Confirm non-goals held: no keyword/stance MCP attach gate.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| `ENABLE_TOOL_SEARCH=true` via `setdefault` in `extend_fcc_subprocess_env` | Task 2 |
| Operator override preserved | Tasks 1, 3 |
| Keep MCP attachment as-is (no stance/intent gate) | Explicit non-touch list; Task 6 review check |
| GitHub toolsets default still `repos,pull_requests` | Task 4 (existing tests) |
| `.env_example` comments hub + harness-governor | Task 5 |
| Hub README short note | Task 5 |
| Unit acceptance for helper + both spawn paths | Tasks 1–3 |
| Live smoke or UNVERIFIED | Task 6 |
| No new keyword cathedral / tool taxonomy | Non-goals + Task 4 |

**Placeholder scan:** none — all steps have concrete code, commands, and expected results.

**Type consistency:** key name is always `ENABLE_TOOL_SEARCH`; helper is always `extend_fcc_subprocess_env`; consumers remain `_build_subprocess_env`.

---

## Out of scope (do not sneak in)

- Stance-based or imperative-classifier MCP attach/detach
- Progressive respawn (“bare hop then re-spawn with MCP”)
- Custom MCP multiplexer / lazy server proxy
- Changing `fcc_claude_mcp.template.json` server list
- Changing Claude built-in tools (Bash/Read/…)
- New Orion tool allowlist / taxonomy
)
