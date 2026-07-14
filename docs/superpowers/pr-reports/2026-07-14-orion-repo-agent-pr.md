# PR report: add orion-repo-agent custom subagent definition

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1044
Branch: `chore/orion-repo-subagent-definition`

## Summary

- Adds `.claude/agents/orion-repo-agent.md`, a custom Claude Code subagent definition with this repo's own conventions baked into its system prompt.
- General read/write/bash tool access, same as `general-purpose`, plus a standing instruction to read `AGENTS.md`/`CLAUDE.md` at the root of whatever worktree the subagent is actually in, a note that worktree discipline is enforced not advisory, a nudge to prefer `graphify query`/`path`/`explain` over raw grep when a graph exists, and a note not to treat relayed/injected instructions as equivalent to direct user instructions for anything destructive or config-changing.

## Outcome moved

Confirmed empirically this session: a dispatched `general-purpose` subagent, tested directly, had no trace of this repo's project `CLAUDE.md` content (specifically its `## graphify` section) in its own context — only the top-level interactive session reliably gets project `CLAUDE.md` injected. Subagents currently only see repo conventions if the orchestrator remembers to paste the relevant excerpt into the dispatch prompt by hand, which happens inconsistently (AGENTS.md's own §20 "subagent task preamble" is explicitly scoped to full build tasks, not narrower dispatches like a review angle or a verification test). This gives future dispatches a `subagent_type` that carries that context automatically, without relying on the orchestrator's memory.

## Current architecture

No custom subagent definitions existed in this repo before this change (`.claude/agents/` didn't exist at all) — every dispatch used the built-in generic types (`general-purpose`, `Explore`, etc.) with zero repo-specific system-prompt content.

## Architecture touched

`.claude/agents/` (new directory, new file). Nothing else.

## Files changed

- `.claude/agents/orion-repo-agent.md`: new custom subagent definition.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

Not applicable — this is a Claude Code configuration file, not executable code. Validated the frontmatter parses as well-formed YAML with the required `name`/`description` fields:

```
python3 -c "
import re, yaml
content = open('.claude/agents/orion-repo-agent.md').read()
m = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
fm = yaml.safe_load(m.group(1))
assert 'name' in fm and 'description' in fm
print('frontmatter parsed OK:', fm)
"
# frontmatter parsed OK: {'name': 'orion-repo-agent', 'description': '...', 'tools': 'Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, Agent'}
```

## Evals run

None — no eval harness applies to a Claude Code config file.

## Docker/build/smoke checks

Not applicable.

## Live verification

**Not verified live from within the session that created it, disclosed explicitly rather than assumed working.** Per Claude Code's own behavior, first-time creation of `.claude/agents/` in a project requires a fresh session before the new `subagent_type` becomes selectable — since this repo had no `.claude/agents/` directory before this change, the session that authored this file could not immediately dispatch an agent of this new type to confirm it. The next session working in this repo (after this branch is merged, or even from this branch directly) should confirm `orion-repo-agent` appears in its available-agent-types listing and that a dispatched instance actually opens with `AGENTS.md` context already in view, before leaning on it for anything load-bearing.

## Review findings fixed

Review skill not run for this PR — it's a single small, inert configuration file (no executable logic, no control flow, no external inputs) where the review's angles (correctness bugs, cross-file breakage, reuse/simplification/efficiency) don't have meaningful surface area to bite into. Flagging this decision explicitly rather than silently skipping the step AGENTS.md §12 otherwise requires.

## Restart required

```text
A fresh Claude Code session in this repo is required before `orion-repo-agent` appears as a selectable subagent_type — see "Live verification" above. No other restart needed.
```

## Risks / concerns

- Severity: Low. Unverified live availability (see above) — the file could theoretically have a subtle format issue that only surfaces once a fresh session tries to actually use it, despite passing basic YAML validation.
- Severity: Low. This is one more file whose content can drift from AGENTS.md's own instructions over time (it deliberately doesn't duplicate AGENTS.md's rules, only points at the file, to minimize this, but the two framing paragraphs about worktree/graphify behavior could still go stale if those mechanisms change).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1044
