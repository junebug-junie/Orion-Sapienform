## Summary

- Benchmarked Strata (MoE expert-tiering engine) serving Qwen3.8-Flash-Next IQ3_S on hecate's single V100-SXM2-32GB.
- Reused the 2026-09-30 circe bench kit, so the scores line up with Bonsai, 27B Q4 and 35B Q5.
- Added a streaming timing path to the kit for servers without llama.cpp `timings`, and fixed the September kit's nested reasoning setting.
- Results and caveats: `docs/bench/2026-10-30-strata-hecate/README.md`.

## Outcome moved

One V100 serves a model that llama.cpp needed four V100s for in September (UD-IQ4_XS on circe).
Hard reasoning 20/20, reasoning at depth 12/12, recall 22/22 to 75K, 94 tok/s average output. Prompt reading is about 960-1,280 tok/s on new tokens.
CPU socket pinning: about 2%, within noise.

## Current architecture

hecate runs one vLLM worker (`qwen3-4b-instruct`, `agent-deep`). Nothing in Orion uses Strata. This PR is bench docs and scripts only.

## Architecture touched

None at runtime. New folder `docs/bench/2026-10-30-strata-hecate/`.

## Files changed

- `README.md`: results, setup, caveats.
- `run_hecate.sh`: pinned docker run plus the bench sequence.
- `bonsai_bench.py`, `bonsai_depth.py`, `reason_depth.py`, `reason_hard.py`, `agent_prompt_max.txt`: copied from the 09-30 kit. Changes: `BENCH_DIR`, stream mode, API key, the dict-effort fix.
- `out-pin-IQ3_S/`, `out-nopin-IQ3_S/`: raw JSONL.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: n/a

## Env/config changes

- None. `.env_example` not touched. No sync needed.

## Tests run

```text
bash -n run_hecate.sh -> ok
Fake SSE server checks: cache split (800 cached / 200 new) ok; tool_calls and 'reasoning' captured; a stream with no usage raises; the non-stream request shape is unchanged.
```

## Evals run

```text
The live bench on hecate is itself the eval: reason_depth 12/12, reason_hard 20/20, depth recall 22/22 (JSONL committed).
```

## Docker/build/smoke checks

```text
ghcr.io/niko1221/strata:0.2.7-cuda12, pinned --cpuset-cpus 20-39,60-79 --cpuset-mems 1 (verified via docker inspect).
hecate vLLM restored afterwards: healthy, and a chat request returned "ready".
```

## Review findings fixed

- Finding: the stored prefill speed included cached tokens (2,077 -> 12,197 tok/s fake climb).
  - Fix: read `usage.prompt_tokens_details.cached_tokens`; prompt_n = new tokens. The README notes the old JSONL.
  - Evidence: fake server, 800/200 split recorded correctly.
- Finding: the API key file sat in a tracked folder, world-readable.
  - Fix: `~/.config/strata-bench/api_key`, umask 077. The old file on hecate was deleted.
- Finding: the wait loop had no timeout exit, and failed steps went unnoticed.
  - Fix: a `ready` flag, `pipefail`, and `|| exit 1` per step.
- Finding: README ranges were slightly off; the reasoning setting differed from September.
  - Fix: corrected to 960-1,280 and 83-106; the difference is noted.
- Finding: a top-level `reasoning_effort` was sent to llama.cpp runs too.
  - Fix: sent in stream mode only, and only when the caller set it.
- Finding: the stream path missed tool calls and `reasoning` deltas, and logged 0 speeds when usage was missing.
  - Fix: all three handled.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: single-request only; concurrency and gateway tool calls are untested. Mitigation: listed as the next test in the README.
- Severity: low. Concern: hecate's card is SXM2 and circe's September card is PCIe. Mitigation: noted in the README.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3015

🤖 Generated with [Claude Code](https://claude.com/claude-code)
