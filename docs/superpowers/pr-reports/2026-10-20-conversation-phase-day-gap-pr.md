## Summary

- A gap of 12-48 hours since Juniper's last message fell through every conversation-phase bucket and was labeled `unknown` with `continuity=continue_directly`, the same framing as "we just talked".
- `long_gap` now covers up to 48 h. Crossing a local day boundary still lifts it to `next_day`. 48 h and over is `stale_thread` (was strictly over 48 h).
- The pinned `test_KNOWN_GAP_twelve_to_forty_eight_hours_falls_through_to_unknown` test is turned into the regression guard, plus a test built from the live Lagoon timestamps.

## Outcome moved

Live 2026-10-20 04:52Z: Juniper's previous message ("Big day later today at Lagoon") was 45 h earlier (Sun 10-18 01:40 MDT). The situation block told the motor `Conversation phase: unknown; continuity=continue_directly`, and Orion answered "hope Lagoon was magical today" on Monday night. 11 live turns with gaps of 15.0-46.2 h carry the `unknown` stamp in `chat_history_log.spark_meta.conversation_phase`. After this patch those gaps render `next_day`/`long_gap` with `continuity=reorient`, and the `temporal_resume` affordance fires.

## Current architecture

`classify_conversation_phase` (`orion/situational/context.py`) buckets the time since Juniper's last turn. Buckets were `<2m`, `<20m`, `<3h`, `<12h`, `>48h`; there was no branch for 12-48 h, and the `crossed_day` override skips `unknown`.

## Architecture touched

`orion/situational/context.py` only. Consumers read the same labels as before; no schema change.

## Files changed

- `orion/situational/context.py`: extend `long_gap` to 48 h; `else` branch is `stale_thread`.
- `orion/situational/tests/test_conversation_phase_stamp.py`: live-timestamp regression and band tests.
- `services/orion-cortex-exec/tests/test_situation_conversation_phase.py`: KNOWN_GAP test inverted to guard the fix.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: 12-48 h gaps stamp `long_gap`/`next_day` instead of `unknown`; `next_day` only when exactly one local midnight was crossed; exactly 48 h stamps `stale_thread`.
- Compatibility notes: memory-consolidation Rule 3 already split these turns through its time-gap fallback for `unknown`; now it splits on `v2:phase_*`. The legacy window rule hides the phase by default (`MEMORY_LEGACY_BOUNDARY_USE_PHASE` off).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/situational/tests/test_conversation_phase_stamp.py           19 passed
services/orion-cortex-exec/tests/test_situation_conversation_phase.py  15 passed
services/orion-memory-consolidation/tests/test_episode_boundary_rule3.py + test_boundary.py  42 passed
```

## Evals run

```text
No eval change: the live check is the next chat turn's spark_meta.conversation_phase after a 12-48 h gap.
```

## Docker/build/smoke checks

```text
None run; pure-function threshold change, no config or dependency change.
```

## Review findings fixed

- Finding (low): with the dead zone closed, a 24-48 h gap crossing two local midnights (Sat 22:00 -> Mon 07:00) was labeled `next_day`, the same wrong-day class as the incident.
  - Fix: the `next_day` override now needs exactly one local date crossed; otherwise it stays `long_gap`. `crossed_day` keeps its meaning.
  - Evidence: `test_gap_over_two_local_midnights_is_long_gap_not_next_day`.
- Finding (nit): test name `test_no_gap_band_falls_through_to_unknown` read backwards.
  - Fix: renamed `test_every_12_to_48h_gap_reorients`.
- Finding (nit): memory-consolidation replay evals re-run `classify_conversation_phase`, so phase counts will shift against the committed `evals/results/2026-10-06-*` JSON.
  - Fix: noted here; the shift is expected, not drift.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a separate miss on the follow-up turn (05:00Z): with a correct `short_pause` clock, the 35B chat model read "It was eventful" as Monday, not Lagoon. This patch does not address reference resolution.
- Mitigation: reported to Juniper; not patched with rules.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
