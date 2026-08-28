# Put the Curiosity Atlas in a tab

## Summary

Follow-up to #1915. That PR merged while its last commit was still in flight, so
main has the Curiosity Atlas page and its route but **nothing in the Hub UI
points at it** — asking "what tab is it in" was the only way to find out.

- Adds the **Curiosity** tab, beside Concept Atlas: tab button, iframe panel,
  standalone link.
- The page stops polling when its panel is hidden.

## Outcome moved

The surface is reachable. Before this it existed only as a URL you had to know.

## Architecture touched

Nine touchpoints in `static/js/app.js`, not one — element refs, the missing-panel
guard, the `is*` flag, the panel toggle, `styleTabButton`, the hash branch, the
known-hash fallback list, the click handler, and the refresh handler. Miss the
hash branch and a deep link to `#curiosity-atlas` silently lands on `#hub`; miss
`styleTabButton` and the button never looks selected. All nine are asserted in a
test now rather than trusted to a careful read.

## Files changed

- `services/orion-hub/templates/index.html`: tab button + iframe panel.
- `services/orion-hub/static/js/app.js`: the nine wires.
- `services/orion-hub/templates/curiosity_atlas.html`: `window.OrionCuriosityAtlas`
  = { refresh, activate, deactivate }.
- `tests/test_curiosity_atlas_template.py`: four tab tests; fixture now reads the
  clock.

## Why the lifecycle contract

An iframe keeps running behind a hidden tab, so the page's 60s poll would have
been a FalkorDB read every minute for a panel nobody is looking at. It takes the
same `activate`/`deactivate` contract Concept Atlas already established rather
than inventing a second convention, and also stops on `visibilitychange` — the
Hub panel being hidden and the browser tab being backgrounded are independent,
and either one alone leaves the poll running.

## Schema / bus / API changes

None. No new route: the tab embeds the existing `/curiosity`.

## Env/config changes

None.

## Tests run

```text
$ .venv/bin/python -m pytest services/orion-hub/tests/test_curiosity_investigation.py \ tests/test_curiosity_worldview.py tests/test_curiosity_atlas.py \ tests/test_curiosity_atlas_template.py -q
192 passed, 18 warnings in 4.83s

$ node --check services/orion-hub/static/js/app.js
(clean)
```

Also fixes a test that would have broken every midnight: the fixture hardcoded
`2026-08-27` as "today" and started failing when the date rolled over
mid-session, in the two tests whose whole subject is whether a run happened
today.

## Review findings fixed

Carried from #1915; this commit was reviewed there before the merge landed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

`index.html` and `app.js` are baked into the image, so a restart alone is not
enough.

## Risks / concerns

- Severity: LOW. The page has still not been seen in a browser — no browser on
  the build host. Layout, wrapping and paint are unverified by anything but
  reading, in the tab as well as standalone.

## PR link

This PR.

