# Curiosity sittings strip: 14-day pager + line filter

## Intent

The Curiosity atlas sittings strip showed only the latest 14 days with `line=all` hardcoded. Operators need to step back through earlier fortnights and filter by the three curiosity lines.

## Lines (exactly three)

| id | Label |
|----|--------|
| `investigate` | World question |
| `self_inquiry` | Self question |
| `self_sense_eval` | Self-sense check |

`self_study.reflect` stays excluded. No fourth line chip.

## UI

- Line chips: All · World question · Self question · Self-sense check
- **← Older** / **Newer →** beside the sittings header (Newer disabled on the live window)
- Strip note: sitting count · date range · active line
- Hash: `#run=…&line=…&until=…` so refresh keeps the window

Budget tiles, priors, and peer briefs stay on “now”; only the sittings strip pages.

## API

`GET /curiosity/api/runs?days=14&line=<line|&all>&until=<iso|epoch-ms>`

- `until` omitted → now (live window)
- Window is `[until - days, until)` clamped so `since` is not older than `now - 90d`
- Response adds: `until`, `since`, `has_older`, `has_newer`, `until_is_now`
- `runs_seen_today` schedule compare only when `until_is_now`

## Non-goals

Infinite scroll, per-day chip pagination, reflect as a filter, changing Run Story internals.
