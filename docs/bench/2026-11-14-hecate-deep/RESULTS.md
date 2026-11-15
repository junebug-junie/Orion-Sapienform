# Hecate deep-thinking measurements

Commissioning date: 2026-11-14. All inference uses Orion Gateway and the existing
operator admission lease. DFlash2 selected and promoted 2026-11-15; Strata comparison open.

## Dense baseline: verified

Pinned Qwen3.8-27B QUASAR NVFP4, 1Cat-vLLM 1.5.3, TP2, FP16 compute,
E4M3 KV, FLASH_ATTN_V100, one sequence, prefix cache, chunked prefill 4096.
No speculative draft or CPU offload. See the lock file and runbook for hashes.

| Actual prompt tokens | Fresh TTFT median (s) | Replay TTFT median (s) | Fresh prefill median (tokens/s) | Fresh decode median (tokens/s) | Fresh sampled GPU energy median (kJ) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.11 | 0.71 | 2257.03 | 48.23 | 12.62 |
| 32768 | 16.40 | 1.08 | 2096.62 | 46.67 | 18.07 |
| 65536 | 35.95 | 1.46 | 1889.01 | 45.15 | 32.47 |
| 132000 | 86.11 | 2.53 | 1571.65 | 42.31 | 61.03 |
| 196608 | 149.62 | 3.84 | 1342.28 | 39.81 | 99.52 |
| 245760 | 206.92 | 4.66 | 1210.86 | 38.13 | 130.25 |

Each depth has three seeded problems and an exact replay of each: **36/36 passed**.
The scored task combines six distributed records, updates, a decoy, unit conversion,
date arithmetic and original-value recall. Provider input counts match exactly.
Every request allows 16384 output tokens; ordinary quality requests finish naturally.
Fresh means a new prompt prefix on an already loaded engine, not model cold start.

The separate capacity test completed **245760 input + 16384 generated = 262144**
user tokens, with a length stop and a correct subsequent 8192-token recovery.
It took 643.48 seconds, TTFT 207.11 seconds, sampled GPU board energy 325.19 kJ,
and decode 37.55 tokens/s. This forced-output test establishes capacity, not quality.
Peak sampled memory was 28719/28713 MiB; both GPUs reached 100% utilization.
The engine recorded zero preemptions. The model stayed loaded throughout the grid,
endurance run, Gateway maintenance restart and operational tests.

## Operational acceptance: verified

All five checks passed (`1cat-operations.jsonl`):

- A second operator hold queued behind a 132000-token request and ran after release.
- Prompt plus output overflow returned 400 without truncation.
- Multiple completions returned 400.
- Streaming cancellation returned to idle by the first sample, 0.11 seconds, followed by a correct answer.
- A five-second client read timeout interrupted observed active generation; the engine was idle after 1.36 seconds and the next answer was correct.

The Gateway cancellation patch was deployed with the authorized brief maintenance
restart. Circe model workers and Hecate biometrics were not restarted. These tests
verify admission/refusal and cancellation recovery; they do not deliberately exhaust
CUDA memory or establish simultaneous full-context concurrency. Initial concurrency
remains one. Prefix cache retention after cancellation is expected and is distinct
from an active request retaining its lease.

## Task quality: dense baseline

All **12/12 runs passed**: four unique exact-answer problems, repeated three times
at temperature zero. These repetitions establish consistency, not twelve independent
reasoning problems. Median completed-task results:

| Task | Elapsed seconds | Generated tokens | Sampled GPU energy (kJ) |
| --- | ---: | ---: | ---: |
| Lease-generation code analysis | 51.09 | 2450 | 21.19 |
| Bounded resource planning | 102.71 | 4967 | 43.59 |
| Memory provenance | 86.43 | 4179 | 36.42 |
| Transaction retry analysis | 19.92 | 943 | 7.95 |

The tasks, exact gold answers, reasoning text, final answers and provider usage are
preserved in the harness and `1cat-quality.jsonl`. Resource summaries are in
`1cat-quality-summary.jsonl`. This is a narrow suite; do not infer general reasoning
superiority from a tie or a perfect score here.

## Measurement limits

Energy is trapezoidal integration of sampled GPU board power, not wall-plug energy.
CPU utilization covers the whole host, including unrelated services. Peaks are
sampled NVML observations, not allocator peaks. Host memory-bandwidth counters are
permission-restricted; NVLink byte counters report unavailable. Connectivity/P2P
support does not establish measured NVLink bandwidth. Raw results and deterministic
resource summaries are stored alongside this report. Compressed raw baseline
samples are `1cat-baseline-host-samples.jsonl.gz` and
`1cat-baseline-engine-samples.jsonl.gz`. Reproduce resource summaries with:

```bash
gzip -cd docs/bench/2026-11-14-hecate-deep/1cat-baseline-host-samples.jsonl.gz > /tmp/hecate-host-raw.jsonl
python services/orion-vllm-host/evals/summarize_deep.py --host /tmp/hecate-host-raw.jsonl \
  --results docs/bench/2026-11-14-hecate-deep/1cat-hard-grid.jsonl \
  --out /tmp/hecate-hard-recomputed.jsonl
```

`process_memory_snapshot.py` captures per-process RSS, CPU/memory affinity and
mapped resident pages by NUMA node from `/proc` inside the worker's PID namespace:

```bash
docker exec -i orion-hecate-vllm-host python - < docs/bench/2026-11-14-hecate-deep/process_memory_snapshot.py
```

Baseline TP workers each used about 3.38 GiB RSS, mostly on NUMA node 1, with zero
swap. This includes shared mappings; do not sum process RSS as unique physical
memory. The raw snapshot is `1cat-process-memory.json`.

## DFlash2 context grid: verified (recovered 2026-11-15)

Same target model and lane; DFlash2 draft, seven speculative tokens, draft TP2,
FP16 draft KV. Same seeded hard suite, Gateway, operator lease and 16384-token
output allowance. Raw results: `dflash-hard-grid.jsonl` (copied from Hecate's
`/tmp/hecate-deep-bench/`; the original Codex session ended before committing it).

| Actual prompt tokens | Fresh TTFT median (s) | Replay TTFT median (s) | Fresh decode median (tokens/s) | Baseline fresh decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 3.87 | 0.67 | 190.45 | 48.23 |
| 32768 | 15.55 | 1.05 | 130.88 | 46.67 |
| 65536 | 34.26 | 1.65 | 131.65 | 45.15 |
| 132000 | 83.02 | 2.87 | 133.98 | 42.31 |
| 196608 | 144.90 | 4.17 | 120.44 | 39.81 |
| 245760 | 200.72 | 4.94 | 104.38 | 38.13 |

**36/36 passed.** The capacity test completed 245760 input + 16384 generated tokens
in 322.11 s (baseline 643.48 s), TTFT 200.67 s, decode 134.94 tokens/s, followed by a
correct 8192-token recovery. Prefill is unchanged; the gain is decode only. Decode
speed depends on draft acceptance, so it varies with the text being generated.

### DFlash2 task quality: 12/12 passed

Same `deep_quality.py` suite, three repeats, run 2026-11-15 in a throwaway container
from the worker image through the leased Gateway. Raw: `dflash-quality.jsonl`.

| Task | Baseline pass | DFlash2 pass | Median seconds (base / DFlash2) | Median output tokens | Median decode tokens/s |
| --- | :---: | :---: | ---: | ---: | ---: |
| lease-generation | 3/3 | 3/3 | 51.1 / 28.0 | 2450 / 2977 | 48.8 / 108.3 |
| bounded-plan | 3/3 | 3/3 | 102.7 / 70.3 | 4967 / 7686 | 48.7 / 110.1 |
| memory-provenance | 3/3 | 3/3 | 86.4 / 20.5 | 4179 / 2512 | 48.7 / 126.3 |
| retry-transaction | 3/3 | 3/3 | 19.9 / 7.2 | 943 / 972 | 48.9 / 157.4 |

Decode is 2.2-3.2x faster on these tasks. Reasoning length differs between engines
(+55% on bounded-plan, -40% on memory-provenance) with identical correct answers,
so per-task time savings partly reflect different sampled reasoning paths, not only
speed. Four problems cannot show a quality difference in either direction.

Not yet measured for DFlash2: completed-task energy and the source-text distractor
run (`deep_context.py --filler-file`, added in this patch).

The Strata challenger is still unmeasured. No model
ranking follows from capacity alone. Strata reserves eight internal tokens, so its
maximum comparison uses 245752 prompt + 16384 output = 262136 user tokens. The
paired seed helper preserves identical reasoning facts across this boundary.
