## Summary

- Hub's chat now has an **Ask Orion** button right above **Send**. It opens a large modal of things Juniper can ask Orion to do.
- Each card has a short title, a plain description, a preview of the message, and an **Insert** button. Insert puts a plain-English request into the chat box at the cursor. It never sends. Juniper edits and presses Send herself.
- Each inserted message is a normal sentence asking Orion for the thing. A second line names the exact command in backticks, as a hint (Orion routes by topic, and Juniper should not need command words).
- The list lives in one checked-in file, `services/orion-hub/static/catalog/orion-abilities.json`: 9 sections, 52 entries.
- A drift test fails CI when Orion gains a dig command, a `capacity` sub-verb, a body-economy section or a chat MCP tool that the list does not describe (or hide with a reason).

## Outcome moved

Before: to ask Orion to use a dig tool, Juniper had to remember what exists and how to phrase it. Now she picks it from a searchable list, and the request lands in the chat box ready to edit.

## Current architecture

- Main chat composer: `services/orion-hub/templates/index.html` (`#composerRow`, `#chatInput`, `#sendButton`), wired in `static/js/app.js`. Only `index.html` has this composer.
- An older "Operator tools" modal (`#operatorToolsModalRoot`) has a "Skill runner" dropdown of classic-lane prompts (mesh ops, cortex workflows). It replaces the chat box text on Insert and its prompts target the classic lane, not the unified turn. It is left as is.
- What Orion can do from a unified chat turn is told to the motor by the turn rules in `orion/harness/prefix.py`: the dig line (`orion/dig/brief.py`), the introspect and reading MCP briefs, the GitNexus brief, and the AI Town MCP server (attached when `HARNESS_AITOWN_ENABLED`).

## Architecture touched

Hub frontend only (template, one new JS file, one new CSS file, one JSON file), plus tests, an eval and one CI step. No backend route, bus, schema or env change. The modal reads the catalog as a static file.

## Files changed

- `services/orion-hub/templates/index.html`: the button (in a new `#sendStack` above Send), the modal markup (`role="dialog"`, `aria-modal`), and the CSS/JS links with the `?v={{HUB_UI_ASSET_VERSION}}` cache-buster.
- `services/orion-hub/static/js/orion-abilities.js`: loads the catalog, renders sections and cards, search, Insert at the caret, Esc, focus trap and focus return. Pure helpers are exported for tests.
- `services/orion-hub/static/css/orion-abilities.css`: the look (Hub's dark gray/indigo palette; Hub has no light theme), full screen and one column at phone width, reduced motion.
- `services/orion-hub/static/catalog/orion-abilities.json`: the catalog. Each entry says what it `covers` and where the ability is defined (`source`).
- `services/orion-hub/static/js/orion-abilities.test.js`: node:test for insert/compose/filter, and that every message is a sentence, not a command.
- `services/orion-hub/tests/test_orion_abilities_catalog.py`: the drift gate, the entry-shape checks, the template/asset checks, and a check that the catalog is not gitignored.
- `services/orion-hub/evals/orion_abilities_browser.py` + `services/orion-hub/tests/test_orion_abilities_browser_smoke.py`: real-browser smoke.
- `.github/workflows/schedule-browser-smoke.yml`: runs the new smoke and fails if it is skipped.

## How the drift test works

It reads the real sources as text (no imports, no services):

1. Every top-level command in the dig help (`HELP` in `orion/dig/__main__.py`), every command `main()` dispatches, and every command named in the turn-rules line (`DIG_BRIEF_LINE`) must be in some entry's `covers.dig`, or in `hidden.dig` with a reason.
2. Every `capacity` sub-verb `main()` accepts (claim, dispute, write, withdraw, help) must be covered.
3. Every body-economy section (`SECTIONS` in `orion/physiology/report_text.py`, plus any section the dig line names) must be covered.
4. Every MCP tool named in the introspect, reading and GitNexus briefs, and every `@mcp.tool()` in the AI Town MCP server, must be covered.
5. The reverse also holds: an entry that covers a command or tool that no longer exists fails.
6. A hint that names `orion.dig <cmd>` must name a real command. Every `source` path must exist.

A parser self-test checks that the parsers find known commands, so a broken parser cannot make everything pass. Checked by hand: removing the `reminisce` entry's `covers` fails the gate with `dig commands with no catalog entry ...: ['reminisce']`.

## Sections and entries (52)

My body 5 · Jobs & capacities 6 · Self-portrait 2 · Past & history 10 · Energy & cost 12 · Dreams, curiosity & self 7 · Reading & the web 4 · My code 2 · Play & world 4.

Hidden on purpose: dig `help` (every entry already points at its command); AI Town `aitown_resume_world` (refused by design; Juniper uses Hub's Unfreeze button) and `aitown_send_input` (raw god-mode input with no safe plain request).

Left out because a unified chat turn cannot reach them (checked in code): retro/Zelda co-play (timer and co-play routes only), self-inquiry and other cortex workflows (classic lane only), Ask-Claude (removed from the chat path), day letter and office reach-out (timers). The day letter can still be read back, and that entry is in.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none outside the Hub page
- Compatibility notes: the catalog is a static file at `/static/catalog/orion-abilities.json`

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/test_orion_abilities_catalog.py      -> 10 passed
node --test <every services/orion-hub/static/js/*.test.js>            -> 421 pass, 0 fail
pytest services/orion-hub/tests (whole folder, local)                 -> 4641 passed, 151 skipped, 11 failed
  8 of the 11: the catalog tests, because I moved the catalog file mid-run; they pass now (10/10).
  3 of the 11 also fail on clean main here, for local reasons, and are not touched by this PR:
    retro/test_experience.py, retro/test_room_memory.py (no pandas in the local venv),
    test_fcc_model_labels_api.py (permission denied on ~/.fcc/.env).
tests/test_no_personal_terms.py, tests/test_organ_names_partition.py,
tests/test_gitignore_no_bare_catchall.py                              -> 19 passed
Drift gate checked by hand: removing the reminisce entry's `covers` fails with
  "dig commands with no catalog entry ...: ['reminisce']".
```

## Evals run

```text
python services/orion-hub/evals/orion_abilities_browser.py --screenshot ...  -> 37 PASS, 0 FAIL
pytest services/orion-hub/tests/test_orion_abilities_browser_smoke.py       -> 1 passed (also added to schedule-browser-smoke CI)
Covers: button above Send; open; all 9 sections / 52 cards; focus on search; page behind is inert;
section chip moves focus; Shift+Tab/Tab trapped; Esc closes and returns focus; search filter + count;
"nothing matches" text; Insert keeps typed text, adds a blank line, focuses the chat box, caret at end;
Enter in search inserts first match with caret after the fill-in colon; nothing sent (no chat POST,
no websocket frame, no new bubble) plus a control that a real Send IS detected; 390 px full screen,
one column.
Screenshots (local, not committed; the repo keeps no images):
  /tmp/claude-1000/-mnt-scripts-Orion-Sapienform/b4ef51d1-f327-456d-8e60-25b4b418ac3a/scratchpad/oa.png
  /tmp/claude-1000/-mnt-scripts-Orion-Sapienform/b4ef51d1-f327-456d-8e60-25b4b418ac3a/scratchpad/oa-phone.png
  /tmp/claude-1000/-mnt-scripts-Orion-Sapienform/b4ef51d1-f327-456d-8e60-25b4b418ac3a/scratchpad/oa-composer.png
```

## Docker/build/smoke checks

```text
No image build: the change is static files only (template, JS, CSS, JSON). The Dockerfile copies all of
services/orion-hub into the image, no .dockerignore excludes static/catalog/, and Hub serves the whole
static/ folder (HubStaticFiles in scripts/main.py). A test asserts the catalog is not gitignored (the
first location, static/data/, was silently ignored by the root .gitignore `data/` rule).
```

## Review findings fixed

- Finding: the AI Town "seems stuck, kick the engine" card treated the town's normal frozen state (it runs only in sessions) as a fault, and offered stop, which chat cannot undo.
  - Fix: card now says the town runs only in sessions and Hub's Unfreeze opens it; kick only while a session is open. `aitown_stop_world` moved to `hidden` with a reason.
  - Evidence: catalog `town-engine`; `services/orion-ai-town/mcp/orion_aitown_mcp/server.py` resume refusal.
- Finding: the drift gate read only the briefs, so a tool added to an MCP server without a brief line would pass.
  - Fix: it now also reads `ToolSpec(...)` in `orion/introspect/tools.py` and `Tool(name=...)` in `orion/world_pulse_read/mcp_server.py`.
  - Evidence: parser finds 24 tools; gate green.
- Finding: capacity sub-verbs were found only on one-line `elif`s; stale `hidden` names were never checked; two entries (self-model, read-now) had no gate.
  - Fix: sub-verbs also come from the `capacity help` text; `hidden` names must still exist; a `source` of `file:sym1,sym2` now requires each symbol to be in that file.
  - Evidence: `test_orion_abilities_catalog.py` 10 passed.
- Finding (accessibility): message preview hidden from screen readers; background not inert; chips scrolled without moving focus; smooth scroll ignored reduced motion; footer contrast ~4.2:1; live count spoke every keystroke; an untouched chat box reported caret 0.
  - Fix: preview readable; page behind the dialog is `inert` (siblings along the modal's path, never its ancestors: the first try inerted the dialog itself and the eval caught it); chips focus the section heading; reduced motion respected; footer gray-400; count debounced 300 ms; caret trusted only after Juniper has touched the box, otherwise insert at the end.
  - Evidence: eval 37 PASS incl. "page behind the dialog is inert", "dialog itself is not inert", "a section chip moves focus to that section".
- Not changed (nit): a message ending in ":" with no hint keeps one trailing space. It is where Juniper types next.

## Restart required

Hub only. The post-pull rebuild on main rebuilds `orion-hub` because `services/orion-hub/**` changed. Nothing to do by hand.

## Risks / concerns

- Severity: low
- Concern: the catalog text describes what each ability does. A change in what a command does (same name) would not fail the drift gate.
- Mitigation: each entry names its `source`, so it is quick to re-check; the gate catches every add, remove and rename.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
