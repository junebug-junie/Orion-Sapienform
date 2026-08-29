# PR: label every node in the Concept Atlas, gate on zoom instead of node count

Branch: `fix/concept-atlas-labels-zoom`

## Summary

- Reported from a screenshot: at **136 nodes the atlas rendered as a field of
  unlabeled dots** with five purple exceptions.
- The declutter gate was all-or-nothing -- `nodeCount >= 60 && hasGodNodes` hid every
  non-god-node label -- so it hid **131 of 136**. That is not decluttering, it is
  blanking, and it is the same failure fixed at 24 nodes on 2026-08-28. That patch
  moved the cliff instead of removing it; this one removes it.
- Replaced with cytoscape's own `min-zoomed-font-size`. Every node always carries its
  label; the renderer skips one only while its on-screen text would be too small to
  read. There is no graph size at which a node becomes a permanently anonymous dot.
- God nodes get a larger font and therefore stay legible further out -- the useful
  half of what the old gate reached for, without discarding the other 131 labels.

## How my own acceptance check missed it

Acceptance check A9 in the concept-induction spec reads "no node in the network
payload renders an empty label at default settings". I verified the **payload**,
which carries all 136 labels correctly, and never checked what the page draws. The
check was worded around the previous bug (labels missing server-side) and did not
survive the bug moving into the renderer.

## Outcome moved

| | before | after |
|---|---|---|
| labels drawn at 136 nodes, default view | 5 | every label whose text is large enough to read; all of them when zoomed in |
| behaviour as the graph grows | hard cliff at 60 nodes | none -- threshold is zoom, not count |
| "Show all labels" | the only way to see 131 of the labels | an override for forcing labels at any zoom |
| label legibility over the edge mesh | plain text on top of edges | background plate |

## Files changed

- `services/orion-hub/static/js/concept-atlas.js`: `shouldDeclutterLabels` and
  `LABEL_DECLUTTER_MIN_NODES` deleted outright (kill means kill -- no dormant knob
  left behind); `labelMinZoomedFontSize` / `edgeLabelMinZoomedFontSize` added; node
  and edge stylesheets wired to them; label background plate.
- `services/orion-hub/templates/concept_atlas.html`: page copy no longer promises
  god-node-only labels.
- `services/orion-hub/static/js/concept-atlas.test.js`: rewritten.

## Design notes

- **Edge labels get a higher threshold than node labels** (11px vs 9px). 461 edges vs
  136 nodes in the live graph, and `supports` / `co_occurs_with` carry far less
  information per pixel than a concept name.
- **"Show all labels" sets the threshold to 0, not to a small number.** A small
  non-zero value still culls when zoomed far out, which is exactly what someone
  checking that box is complaining about. Only 0 disables culling in cytoscape.
- **God-node legibility is bought by font size, not a separate rule.**
  `min-zoomed-font-size` compares against the rendered font size, so a god node
  survives further out because its font is bigger. A test asserts
  `GOD_NODE_FONT_PX > NODE_FONT_PX`, because if those ever invert the threshold
  silently does the opposite of what is intended.

## Schema / bus / API changes

None. Client-side rendering only; the `/api/substrate/concepts/network` payload is
untouched.

## Env/config changes

None.

## Tests run

```text
$ node --test services/orion-hub/static/js/
# pass 61  # fail 0

$ pytest services/orion-hub/tests -q -k "concept or atlas or asset or ui"
8 failed, 392 passed
```

The 8 failures are **pre-existing**. Verified by running the same five test files
against the primary checkout on `main`, which fails 9 in those files -- a strict
superset. None of them read `concept-atlas.js` or `concept_atlas.html`; they read
`app.js`, `memory.js`, and the LLM route selector.

### Mutation testing

Each mutation asserted present in the file before running.

| mutation | result |
|---|---|
| god-node threshold equals the ordinary one | 2 tests fail |
| show-all lowers the threshold to 2 instead of disabling it | 1 fails |
| edge labels use the node threshold | 1 fails |
| node stylesheet hardcodes the value instead of calling the helper | 1 fails |
| `GOD_NODE_FONT_PX` no longer larger than `NODE_FONT_PX` | 2 fail |
| label background plate removed | 1 fails |

Two tests also assert that `shouldDeclutterLabels` and `LABEL_DECLUTTER_MIN_NODES`
never reappear in the source, so a count-based gate cannot be reintroduced quietly.

## Docker/build/smoke checks

No rebuild needed. `services/orion-hub/docker-compose.yml` mounts `./static` and
`./templates` from the checkout, and `build_hub_ui_asset_version()` is called inside
the `/concept-atlas` route handler rather than at import, so the `?v=` token
recomputes per request from the files' mtimes.

**Because those mounts are relative to the primary checkout, this cannot be verified
from a worktree without repointing the live Hub's mounts at a directory that is about
to be deleted.** It goes live when the branch merges and the primary checkout
updates; a hard refresh is enough on the browser side.

## Restart required

```text
No restart required. Merge, then hard-refresh /concept-atlas (Ctrl-Shift-R).
```

## Risks / concerns

- Severity: low. Concern: 9px is a judgement call, not a measurement -- it is the
  point below which the label text stops being readable, which depends on the
  viewer's display. Mitigation: it is a named constant with no cliff behaviour on
  either side of it, and "Show all labels" overrides it entirely.
- Severity: low. Concern: not verified in a browser at the live graph size, for the
  mount reason above. The rendering rule is cytoscape's own documented behaviour
  rather than custom code, and every wiring point is pinned by a test.

## Status

DONE_WITH_CONCERNS -- correct by test and by cytoscape's documented semantics, not
yet seen rendered at 136 nodes.
