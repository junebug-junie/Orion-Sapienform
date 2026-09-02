# Streamed sentence-chunk TTS — Orion's voice starts in ~3.3s instead of ~17s

## Summary

- Orion said nothing until the **entire** reply had been synthesized, so voice latency was the whole LLM turn *plus* the whole clip, stacked end to end.
- Measured the real rail before changing anything: XTTS synthesis is ~0.2s fixed + ~0.33s per second of audio (RTF 0.33, flat from 54 to 814 chars). Audio renders ~3x faster than it plays.
- So the reply is now split on sentence boundaries and each chunk is queued to the browser the moment it is synthesized. Time-to-first-sound stops scaling with reply length.
- Chunk targets **double** (80 → 160 → 280 chars) rather than jumping straight to the cap — a flat "small first, big rest" pair starts fast and then runs the voice dry.
- **No frontend change was needed**: `handleTtsFields` already pushes every `audio_response` onto `audioQueue`, and `playAudio`'s `onended` drains it in order.
- Ruled out hardware: benchmarked the same synthesis on a V100-32GB (RTF 0.337) vs the P100 it runs on (RTF 0.326). Identical.

## Outcome moved

Time to first spoken sound, 814-char reply, measured live over the real bus:

| | time to first sound |
|---|---|
| before (single shot) | **17.06 s** |
| after (streamed) | **3.33 s** |

**5.1x**, zero starvation gaps. Total synthesis work is unchanged (slightly higher, by ~0.2s per extra chunk); only the moment you first hear something moves.

This does **not** address the larger half of the wait. Turn latency over the last 7 days (`harness_turn_trace`, n=127) is **p50 36.6 s, max 156.6 s**. TTS was roughly 15–30% of the "minutes" complaint; the LLM turn is the rest and is untouched here.

## Current architecture (before)

`run_unified_turn()` returns a frame list. `extract_unified_turn_final_text()` scans it for the `type == "final"` frame and pulls `llm_response` — i.e. TTS could not begin until the turn was completely finished. `dispatch_tts_reply()` then fire-and-forgets `run_tts_remote()`, which made exactly **one** `speak()` RPC for the whole reply and put exactly **one** message on `tts_q`.

## Architecture touched

Only the hub, and only inside the shared TTS path. `run_tts_remote` is the single function both the classic and orion-unified lanes reach through `dispatch_tts_reply`, so both lanes get this from one place. No bus contract change, no schema change, no TTS-service change, no frontend change.

## Files changed

- `services/orion-hub/scripts/websocket_handler.py`: new `chunk_text_for_speech()`; `run_tts_remote()` loops chunks and queues each as it completes; `chunk_index`/`chunk_total` threaded into `synthesize_tts_reply` and its `voice.tts.*` traces.
- `services/orion-hub/app/settings.py`: three new keys.
- `services/orion-hub/.env_example`: same keys, with the measured rationale.
- `services/orion-hub/static/js/app.js`: comment only — documents that clearing `audioQueue` no longer stops speech (see Risks).
- `services/orion-hub/tests/test_tts_streaming_chunks.py`: new, 13 tests.
- `services/orion-hub/tests/test_orion_unified_turn_tts.py`: fixed a test that was **already failing on main**.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the WS lane now emits **N** `audio_response` messages per turn where it previously emitted exactly one. The browser already handled this (sequential `audioQueue`).
- The HTTP-fallback lane (`api_routes.py:3267`) calls `synthesize_tts_reply` directly and is **unaffected** — it still synthesizes the whole reply in one call, correctly, since a single JSON response body can only carry one clip.
- Compatibility: `chunk_index`/`chunk_total` default to `0`/`1`, so every existing caller is unchanged.

## Env/config changes

- Added keys: `HUB_TTS_STREAM_ENABLED` (true), `HUB_TTS_STREAM_FIRST_CHUNK_CHARS` (80), `HUB_TTS_STREAM_CHUNK_CHARS` (280).
- Removed / renamed: none.
- `.env_example` updated: yes.
- Local `.env` synced: **yes, by hand.** `scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary* checkout, so keys added in a worktree are invisible to it — it reported no change for these three. Written directly into `services/orion-hub/.env` and verified present.
- Skipped keys requiring operator action: none.

## Tests run

```text
# new + directly related suites (in the hub image, worktree mounted)
pytest tests/test_tts_streaming_chunks.py tests/test_orion_unified_turn_tts.py -q
  -> 37 passed

pytest tests/test_orion_unified_turn_tts.py tests/test_handle_tts_fields_frontend.py \
       tests/test_voice_tts_hang_guards.py tests/test_tts_client_errors.py \
       tests/test_handle_chat_request_http_fallback_tts.py tests/test_tts_streaming_chunks.py -q
  -> 53 passed + the new suite

# full hub suite, branch vs main, same image and command
branch: 62 failed, 1985 passed, 9 skipped
main:   67 failed, 1968 passed, 9 skipped
```

The pre-existing failure count is high on **both** sides. Diffing the FAILED sets rather than the counts:

- **New on branch:** `test_substrate_mutation_manual_route_routing.py::test_routing_dry_run_produces_trial_and_decision_without_side_effects`. Run in isolation it **passes on the branch and fails on main** — the reverse of the suite result. It is order/state-dependent and pre-existing; it touches substrate mutation routing and nothing in this diff.
- **No longer failing on branch:** 6, of which one (`test_dispatch_fires_when_all_conditions_are_met`) is fixed deliberately here. The other 5 are cache-bust/mtime assertions that flip because editing files changes mtimes — cosmetic, not real fixes.

## Evals run

```text
None. services/orion-hub has no evals/ directory.
```
The eval gap is real and pre-existing for this service. The live measurement below is the substitute evidence.

## Docker / build / smoke checks

Measured against the **live** whisper-tts on circe over the real bus (warm model), not a mock:

```text
single-shot baseline, by length:
   54 chars -> rpc  2.18s  synth  1.68s  audio  4.59s   (RTF 0.365)
  237 chars -> rpc  6.37s  synth  6.32s  audio 15.16s   (RTF 0.417)
  814 chars -> rpc 16.16s  synth 16.04s  audio 49.13s   (RTF 0.326)

streamed, 814-char reply -> 4 chunks [179, 179, 334, 119]:
  chunk 1/4  synth 3.33s  ready_at  3.33s  audio 10.0s
  chunk 2/4  synth 3.64s  ready_at  6.97s  audio 11.0s
  chunk 3/4  synth 6.31s  ready_at 13.29s  audio 19.4s
  chunk 4/4  synth 2.40s  ready_at 15.69s  audio  7.0s
  first sound 17.06s -> 3.33s (5.1x); starvation gaps: NONE

GPU comparison (same text, same image, device_ids pinned):
  P100-16GB (current): RTF 0.326
  V100-32GB:           RTF 0.337
```

## Review findings fixed

The mandated `/code-review` subagent **did not complete** — it terminated on an API session rate limit (429, resets 08:50 UTC). I reviewed the diff myself against the same checklist. Findings I found and fixed:

- **Finding:** `test_chunks_are_sentence_aligned_and_lossless` asserted `" ".join(chunks) == text`, claiming byte-losslessness. The inherited `split_sentences` collapses newlines to spaces and strips, so this is false for any text with newlines or double spaces — the test passed only because its fixture had neither.
  - **Fix:** assertion now compares against `split_sentences`' own output (the true guarantee: no sentence lost or reordered), renamed accordingly, plus a new `test_whitespace_is_normalized_not_preserved` pinning the real behavior.
  - **Evidence:** `chunk_text_for_speech("First line here.\nSecond line here.", ...)` returns `["First line here.", "Second line here."]` — roundtrip_equal=False.

- **Finding:** with streaming, the interrupt handler's `audioQueue = []` no longer stops Orion talking — the server loop keeps pushing chunks, which get queued and played. Orion would resume after being interrupted.
  - **Fix:** documented at the handler rather than built around. `#interruptButton` ships `class="hidden"` in `index.html` and **nothing ever un-hides it**, so the path is unreachable dead UI today; building suppression machinery for it would be speculative.
  - **Evidence:** `grep -n interruptButton` finds exactly one template hit (the hidden button) and no code that removes `hidden`.

- **Finding (checked, clean):** `%`-placeholder/arg counts in the three `voice.tts.*` log calls after threading `chunk_index`/`chunk_total`. Verified by actually emitting the records with a handler calling `getMessage()`, which raises on mismatch — both format cleanly (`voice.tts.start ... chunk=3/5`). My first automated check reported a mismatch; that check was itself buggy.

- **Finding (checked, clean):** every other caller of the two functions. `api_routes.py`'s HTTP fallback calls `synthesize_tts_reply` directly with the new defaults and is unaffected.

## Restart required

```bash
# Hub only; whisper-tts and the bus are untouched.
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

Not deployed by me — this restarts Juniper's live chat UI, so it is her call when to take it.

## Risks / concerns

- **Severity: medium.** Interrupt/barge-in is incompatible with streaming (above). Latent only because the button is dead UI. Mitigation: documented in place; must be handled by whoever revives it.
- **Severity: low.** A single very long sentence lands in one chunk whole (this never splits mid-sentence) and, after a short first chunk, can briefly outrun playback and insert a pause. No audio is lost or reordered, and it is still far shorter than the 17 s of silence the old path opened with. Not engineered around deliberately.
- **Severity: low.** The measured win depends on where sentence boundaries fall. A uniform-sentence model predicted 1.95 s; real text gave 3.33 s because this reply's first two sentences total 179 chars. Left at 80/280 rather than tuned lower — further gains fit one sample's sentence lengths, while the starvation risk is general.
- **Severity: informational.** This does not touch the dominant cost. p50 turn latency is 36.6 s before TTS is even asked to speak.

## PR link

<filled on push>
