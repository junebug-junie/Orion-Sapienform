## Summary

- New stdlib script `scripts/gpu_power_bench.py`: runs one card's llama-server at several power limits and records reading speed, writing speed, joules per token and temperature.
- First step agreed with the body-economy agent before any power-cap actuator is built: if joules per token barely moves with the cap, drop the idea.
- Run on the card's host as root: `sudo python3 scripts/gpu_power_bench.py --gpu 2 --port 8016`.
- No pool hold. A request goes out only when every slot is idle and the card is near idle power; samples are thrown away when another slot worked, the server counted extra tokens, or the request waited.
- The starting limit is restored on every exit except SIGKILL, with signals ignored during the restore.

## Outcome moved

Gives the power-cap question a measured answer instead of a guess (body-economy handoff, 2026-10-29).

## Current architecture

Nothing in the repo sets GPU power limits. The power-intent loop (orion-biometrics) only measures draw.

## Architecture touched

None at runtime. One operator script and its tests.

## Files changed

- `scripts/gpu_power_bench.py`: the bench.
- `tests/test_gpu_power_bench.py`: energy math, dirty-sample rules, restore on error/SIGTERM/Ctrl+C, restore retry under signals, lock, refusal when the card is not at its default.
- `docs/superpowers/pr-reports/2026-10-29-gpu-power-bench-pr.md`: this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none.

## Env/config changes

- None. `.env_example` not touched.

## Tests run

```text
pytest tests/test_gpu_power_bench.py -q  -> 22 passed
```

## Evals run

```text
No eval harness: this script IS the measurement. Live smoke below.
```

## Docker/build/smoke checks

```text
circe gpu2 (Bonsai 2 27B, :8016), no root, limit untouched at 250 W:
limits (250, 250, 100, 250); busy slots []; /metrics disabled (501) -> extra-token check off
quiet True; dirty False; waited 0.837 s
read 662.9 tok/s, write 49.0 tok/s, 194.2 W while writing, 3.96 J/token, 51 C
```

## Review findings fixed

- Finding: a second signal during the restore could skip it and leave the card capped.
  - Fix: `restore_limit` ignores SIGINT/TERM/HUP/QUIT, catches BaseException, retries 3x; nvidia-smi runs in its own session.
  - Evidence: `test_restore_retries_after_an_interrupt_and_ignores_signals_meanwhile`.
- Finding: nvidia-smi calls had no timeout.
  - Fix: 30 s timeout on every call.
  - Evidence: `smi()`.
- Finding: a production request on our own slot was invisible; windows started at send time.
  - Fix: all slots must be idle before sending; windows anchored to reply arrival; samples dropped when the round trip exceeds the server clocks by > 2 s.
  - Evidence: `test_busy_slots_counts_our_own_slot_unless_ignored`, `test_all_dirty_attempts_end_the_cap_and_still_restore`, `test_sample_windows_are_anchored_to_the_reply`.
- Finding: a dead sampler thread made each cap burn ~27 min capped.
  - Fix: sampler catches every error; quiet wait aborts when readings are stale.
  - Evidence: `test_dead_sampler_aborts_the_quiet_wait`.
- Finding: sample timestamps lagged; quiet test needed a sample count.
  - Fix: midpoint timestamps; quiet test uses time coverage.
- Finding: another writer could make us "restore" its cap.
  - Fix: flock per card; refuse when the limit is not the card's default.
  - Evidence: `test_card_not_at_default_is_refused_before_touching_it`, `test_second_bench_on_the_same_card_is_refused`.
- Finding: missing `is_processing` silently disabled detection; repeated /slots errors dropped every sample.
  - Fix: refuse at start; abort after 3 failed polls.
  - Evidence: `test_slots_without_is_processing_abort_instead_of_reading_clean`.
- Finding: short requests could slip between polls.
  - Fix: final poll after the reply; /metrics token-count delta when metrics are on (off on :8016 today).
- Finding: summary switched baseline silently; results lost on a late error.
  - Fix: baseline is the first requested cap or None; outputs written after every sample and in `finally`.
  - Evidence: `test_summary_compares_to_the_first_cap_and_never_switches_baseline`.
- Found during the fix (not in review): an Event named `_stop` shadows `Thread._stop`, so `join()` would crash the first request.
  - Fix: renamed `_halt`. Evidence: `test_thread_join_works`.
- Found by the live smoke: a clean idle request spends 0.84 s outside server clocks, so a 0.5 s wait limit dropped everything.
  - Fix: limit 2.0 s; each sample records its wait.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: world-model and diffusion on gpu2 are not visible to the server's slots. Mitigation: idle-power gate before each request; their draw shows as noise.
- Severity: low. Concern: while capped, Orion's real gpu2 work runs slower for ~10 min. Mitigation: restored at the end; each cap waits for the card to be quiet.
- Severity: low. Concern: the power-intent loop will settle gpu2's watts lower during the bench. Mitigation: one ~10 min window.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2945

🤖 Generated with [Claude Code](https://claude.com/claude-code)
