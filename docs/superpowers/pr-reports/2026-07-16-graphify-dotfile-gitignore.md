# gitignore graphify's ephemeral pipeline dotfiles

## Summary

- `graphify-out/.graphify_python` (the interpreter path graphify's Step 1 writes for later pipeline steps to reuse) was sitting untracked in the shared checkout, flagged by Juniper. Never committed, no git history — just local scratch state nobody cleaned up.
- Same class of file recurred repeatedly this session: `.graphify_detect.json`, `.graphify_chunk_NN.json`, `.graphify_ast.json`, `.graphify_semantic*.json`, `.graphify_cached.json`, `.graphify_uncached.txt`, `.graphify_analysis.json`, `.graphify_labels.json(.sig)`, `.graphify_seed_payload.json` — all ephemeral pipeline state, none of it meant to be tracked (only `graph.json`, `GRAPH_REPORT.md`, `manifest.json` are the real committed outputs).
- Added `graphify-out/.graphify_*` to `.gitignore` so this whole class of file stops showing up as clutter in `git status` for any future session, instead of hand-cleaning one file at a time.

## Outcome moved

`git status` in a fresh or long-running checkout no longer accumulates stray `.graphify_*` scratch files as untracked noise. One gitignore pattern closes the whole class, rather than each individual scratch filename needing its own fix as it's noticed.

## Files changed

- `.gitignore`: added `graphify-out/.graphify_*`.

## Tests run

```text
$ touch graphify-out/.graphify_python graphify-out/.graphify_chunk_00.json
$ git check-ignore -v graphify-out/.graphify_python graphify-out/.graphify_chunk_00.json
.gitignore:154:graphify-out/.graphify_*   graphify-out/.graphify_python
.gitignore:154:graphify-out/.graphify_*   graphify-out/.graphify_chunk_00.json

$ git check-ignore -v graphify-out/graph.json graphify-out/GRAPH_REPORT.md graphify-out/manifest.json
(no output, exit 1 -- correctly NOT ignored)
```

Confirms the pattern catches the ephemeral dotfiles without touching the three real tracked outputs.

## Restart required

```text
No restart required.
```

## Risks / concerns

None -- pure `.gitignore` addition, no tracked file affected.

## PR link

<link>
