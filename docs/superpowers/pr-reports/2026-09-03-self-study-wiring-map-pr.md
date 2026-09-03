# Self-study wiring map

Branch: `docs/self-study-wiring-spec`
Status: **DONE** (docs only)

## Summary

- Wrote a current-state map of Orion self-study: inspect / induce / reflect /
  retrieve, the separate analysis verb, the commit-triggered enrichment
  service, journal vs retired RDF writeback, and who actually reads any of it.
- Named the homonym: self-model vs telemetry analysis vs Hub curiosity all
  stamping `source_kind='self_study'`.
- Recorded that there is no dedicated reducer, that GraphDB writeback is
  retired, and that the chat-stance SPARQL producer is likely a no-op
  (UNVERIFIED live).

## Outcome moved

Someone asking "where is self-study wired" can read one spec instead of
reconstructing it from `self_study.py`, enrichment README (stale on cache
consumption), channels.yaml comments, and thought's privacy allowlist.

## Current architecture

See `docs/superpowers/specs/2026-09-03-self-study-wiring-map.md`.

## Architecture touched

Docs only. No service, contract, config, or runtime seam.

## Files changed

- `docs/superpowers/specs/2026-09-03-self-study-wiring-map.md`: the map
- `docs/superpowers/pr-reports/2026-09-03-self-study-wiring-map-pr.md`: this report

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: n/a

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a
- skipped keys requiring operator action: none

## Tests run

```text
docs-only; no pytest.
```

## Evals run

```text
No eval harness for a wiring map. Existing analysis eval remains
services/orion-cortex-exec/evals/run_self_study_analysis_eval.py (untouched).
```

## Docker/build/smoke checks

```text
No runtime change. No compose.
```

## Review findings fixed

- Finding: prior turn claimed the spec+PR were done; they were not.
  - Fix: this changeset is the actual writeup.
  - Evidence: files on `docs/self-study-wiring-spec`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: GraphDB emptiness and analysis journal rate since deploy are
  UNVERIFIED. The spec says so; a reader could still treat the map as live
  proof of those two claims.
- Mitigation: labeled UNVERIFIED; recommended next patch is a live query, not
  more docs.
)
