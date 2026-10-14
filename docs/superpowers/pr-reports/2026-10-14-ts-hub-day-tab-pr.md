## Summary
- New Hub tab **Temporal Self** (`#temporal-self`, nav + launcher): Orion's day as a timeline of arcs grouped by kind, the active arc, open threads, day phase, and regulation now (arousal + since, rest drive, immune findings). ◀/▶ and a date picker load closed days. Read-only.
- `scripts/temporal_self_routes.py`: thin read-only proxies to orion-durable-runs (`/api/temporal-self/frame|day/{id}|arcs|threads|cursors|regulation|immune`), 5 s timeout, 502 with a named reason when upstream is down, 404/503 passthrough, strict input validation. Arcs are slimmed to rendered fields unless `full=true` (420 KB → 82 KB/day).
- Timeline uses `/arcs` (the frame's list is capped at 64; live day had 228).

## Env/config changes
None (reuses `HUB_CURIOSITY_DURABLE_RUNS_URL`; Hub runs on the host network, live value `http://127.0.0.1:8124`).

## Tests run
```text
tests/test_temporal_self_routes.py tests/test_hub_panel_reachability.py: 33 passed
node --test hub *.test.js: 263 passed (9 new)
full hub suite: 3591 passed, 36 failed — failures pre-exist on main (checked 4 files from main)
```
Live: route handlers against real durable-runs — today 225 arcs over 7 kinds; closed day 2026-10-12 200; unclosed day 404; upstream down → 502. Page JS run in Node against live data rendered today and 2026-10-12.

## Review findings fixed
Malformed day id could 500 → strict match + 502; stale reload race (load tickets); background polling while hidden; reachability exceptions split; "not run yet" for immune; real close time; picker reset.

## Restart required
```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns
- UNVERIFIED in a real browser (no headless browser on host); layout/mobile unchecked.
- No endpoint lists closed days; picker shows "not closed" for unchronicled dates.
- Regulation panel is "now" even when viewing a past day (labelled).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2657
