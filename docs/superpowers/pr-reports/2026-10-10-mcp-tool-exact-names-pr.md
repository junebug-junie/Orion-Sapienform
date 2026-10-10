## Summary

- Orion's turn instructions now name every MCP tool exactly as Claude Code exposes it (`mcp__orion-introspect__curiosity`, not `curiosity`). A wrong guess at the prefix used to make the tool look missing.
- Server names live in one module, `orion/fcc/mcp_names.py`. Both the MCP config builder and every tool guide use it (GitHub, GitNexus, Context Mode in both modes, Reading, Introspect), so the two can't drift.
- The Introspect guide now says how to read a "failed to connect" note. That note lists the servers that failed, by name. It also tells Orion not to rebuild curiosity, dream or reading records by querying the database by hand.
- Credentials are now stripped from every trace step and from the final reply before they leave the harness. This covers both `orion/harness/fcc_motor.py` and the Hub bridge.

## Outcome moved

The failure was seen live on 2026-10-10 (corr `65a85b66…` and the turn after it):
1. Orion searched for `orion-introspect__curiosity`. ToolSearch replied "No matching deferred tools", with a note that `orion-aitown` had failed to connect. Orion read that as the curiosity server being down. It was up: a direct stdio probe returned 602 runs.
2. Orion then hand-queried Postgres, guessing table names, and ran `echo $ORION_CURIOSITY_PG_DSN`. The read-only database password showed up in the Hub turn trace.

After this patch the guides carry the exact names, and the same password text is stripped (verified below against the live `~/.fcc/.env`).

## Current architecture

The guides (`introspect/brief.py`, `github_repo_context.py`, `self_index_brief.py`, `world_pulse_read/tools.py`) named tools bare. `build_step_frame` passed raw stream-json straight through to the trace. Credential stripping existed only for introspect responder error messages (`orion/introspect/redact.py`).

## Architecture touched

- FCC harness prompt prefix (the guide text)
- Trace step frames and final replies (fcc_motor and the Hub `fcc_claude_bridge`)
- No bus, schema or env changes

## Files changed

- `orion/fcc/mcp_names.py`: new. Server keys plus `mcp_tool()`.
- `orion/fcc/mcp_config.py`: uses the `mcp_names` keys.
- `orion/fcc/github_repo_context.py`, `orion/fcc/self_index_brief.py`, `orion/world_pulse_read/tools.py`, `orion/introspect/brief.py`: exact tool names. Context Mode picks the plugin server name when hook mode is on.
- `orion/core/redact.py`: new, shared credential stripping:
  - URL userinfo
  - `*PASSWORD=` assignments
  - literal values of credential-named env vars, including env files passed to children (`remember_secret_env`)
- `orion/introspect/redact.py`: now delegates to `orion/core/redact.py`.
- `orion/harness/fcc_motor.py`, `services/orion-hub/scripts/fcc_claude_bridge.py`: redact step frames and final replies. `load_fcc_env` registers its secrets.
- `services/orion-hub/scripts/fcc_env_catalog.py`: its `load_fcc_env` registers secrets.
- `orion/harness/reading_receipts.py`: tool-name constants come from `mcp_names`.
- Tests:
  - new: `orion/fcc/tests/test_mcp_tool_names.py`, `orion/harness/tests/test_trace_secret_redaction.py`
  - updated wording: `test_introspect_harness_wiring.py`, `test_harness_prefix.py`

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: trace `step.raw` strings and final reply text have credentials replaced with `[REDACTED]`. The motor's own logic still reads the unredacted event.
- Compatibility notes: display-only for steps. The final reply loses only credential text.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/fcc/tests orion/introspect/tests orion/harness/tests orion/world_pulse_read
  843 passed, 1 failed (test_no_undiscovered_root_callers_of_claude_permission_argv:
  pre-existing, fails identically on main)
services/orion-hub: bridge/mcp/chat-history/env tests: 29 passed
```

## Evals run

```text
No eval harness covers brief wording. test_mcp_tool_names.py is the deterministic gate:
every mcp__<server>__<tool> in every guide must resolve to a real server and tool,
and no tool may be named bare.
```

## Docker/build/smoke checks

```text
Live probe (harness-governor container): orion.introspect.mcp_server over stdio,
tools/call curiosity, returned ok=true, total_available=602.
Live redaction check (new redact.py run in the container on the real ~/.fcc/.env;
only booleans printed): 7 secret values known; incident DSN line had the password
before and not after, host kept; 0 credential vars leak via `echo <value>`.
```

## Review findings fixed

- Finding: literal-value stripping read only `os.environ`. The subprocess's secrets live in `~/.fcc/.env`.
  - Fix: `load_fcc_env` (motor and hub) calls `remember_secret_env`.
  - Evidence: `test_fcc_env_file_secrets_are_redacted` plus the live check above.
- Finding: `PGPASSWORD=` / `X_PASSWORD=` were missed (`\b` inside a word).
  - Fix: the pattern matches `\w*password`.
  - Evidence: `test_prefixed_password_assignments_are_redacted`
- Finding: `redis://:pw@host` (empty username) and a password containing `@` leaked.
  - Fix: allow an empty user and match greedily to the last `@`.
  - Evidence: `test_empty_username_and_at_in_password`
- Finding: quadratic regex cost on long dotted or hyphenated runs. It would block the stream loop.
  - Fix: anchor on run starts with a lookbehind, and skip the URL pass when there is no `://`.
  - Evidence: `test_long_dotted_or_hyphenated_run_is_linear` (200KB in under 0.5s)
- Finding: an unquoted `password=` value swallowed closing `"`, `)`, `}`, which corrupted JSON and code in replies.
  - Fix: an unquoted value stops before closing syntax.
  - Evidence: `test_password_value_keeps_closing_syntax`
- Finding (nit): the Context Mode plugin name was hardcoded in `fcc_motor.py` and `reading_receipts.py`.
  - Fix: both use `mcp_names`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: high
  - Concern: the `orion_readonly` password is already exposed. It is in the stored Hub trace for this turn, in the governor's Claude transcript `8f82ac36…jsonl`, and in the investigating chat.
  - Mitigation: rotate it (operator action, not done here).
- Severity: low
  - Concern: any value 8+ characters long in a `*TOKEN`/`*SECRET`/`*PASSWORD`-named var is replaced wherever it appears. If one of those holds an ordinary word, that word disappears from replies.
  - Mitigation: an 8-character minimum, and names only.
- Severity: low
  - Concern: AI Town is down by design, but `HARNESS_AITOWN_ENABLED=true`. That adds a "failed to connect" note to Orion's tool searches and sets introspect `memory_allowed=false` for chat turns.
  - Mitigation: left for Juniper to decide.

## PR link

<filled in after push>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
