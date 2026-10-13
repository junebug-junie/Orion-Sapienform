## Summary

- A WebFetch result that is the tool's own HTTP error notice ("The server returned HTTP 404 Not Found.") no longer counts as reading the source. The seed ends `failed` with `no_read_evidence:http_error` (terminal), and no atlas nodes, journal, or claims are created.
- Stage 2's `need_stage1_urls` re-entry follows only URLs that appear in text Orion actually fetched (the parent's retained Stage 1 text, or the Stage 2 turn's own fetch results), or URLs a Stage 2 fetch reached with a real page. Every refusal is logged (`world_pulse_read_stage2_reentry_refused reason=url_not_in_fetched_content`).
- A parent read whose only retained text is an error notice no longer gets a Stage 2 pass: no turn, no Wallet B debit, no claims, no hop chain (`stage2_status=skipped`, `no_read_evidence:http_error`).
- Claims never draw on a notice: `load_retained_texts` drops notices by default.
- Chat receipt grounding: a fetch that hit a 404 is not counted as `source_read`.
- Prompts: Stage 1 asks WebFetch for the URLs the page links to, so real links land in retained text. Stage 2 is told that guessed URLs are dropped.

## Outcome moved

Agent-GPU turns stop being spent reading pages that do not exist and recording them as `done`. Live 2026-10-13: one CoreWeave read asked Stage 1 to read three sibling URLs it had guessed. Each one returned the 215-char 404 notice, was marked `done`, and two of them started their own Stage 2 with further guessed hops.

## Current architecture

- Stage 1 (`services/orion-hub/scripts/world_pulse_read_pipeline.py`) counts a read as evidence when there is a same-site, non-listing fetch of at least `MIN_SOURCE_CONTENT_CHARS=200` (`orion/world_pulse_read/read_evidence.py`). Fetch text rides harness -> Hub on reading turns (`reading_only` -> `retain_text`) and is retained by sha256 in `reading_document_snapshot`.
- Stage 2 (`world_pulse_read_stage2.py`) took `need_stage1_urls` straight from model JSON and enqueued each one, up to `max_round_trips`. Nothing checked where the URL came from.
- The existing thin-fetch guard was the 200-char floor. The notice is 215–223 chars, so it passed.

## Architecture touched

- New pure module `orion/world_pulse_read/fetch_provenance.py`: the status-line rule plus URL extraction and allowlisting.
- Hub Stage 1 evidence rule, Hub Stage 2 tick (dead-parent skip, hop gate), the retained-text loader, and the harness receipt tracker (`orion/harness/reading_receipts.py`, runs in the harness governor).
- No bus, schema, channel, or env change. `SourceFetchEvidenceV1` is unchanged.

## Files changed

- `orion/world_pulse_read/fetch_provenance.py`: new. `fetch_http_error_status`, `urls_in_text`, `hop_url_key`, `hop_allowlist`, `hop_refusal_reason`.
- `orion/world_pulse_read/read_evidence.py`: excludes notice fetches; adds the `NO_READ_EVIDENCE_HTTP_ERROR` label; floor comment updated with live findings.
- `orion/world_pulse_read/fetch_text.py`: `load_retained_texts(..., drop_http_errors=True)`.
- `orion/harness/reading_receipts.py`: a notice is reported but is not `source_read`.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: `_parent_read`, `_hop_allowlist`, the refusal loop, `GenerateOutcome.source_fetches`, a prompt sentence.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: Stage 1 prompt asks for linked URLs.
- Tests: `orion/world_pulse_read/tests/test_fetch_provenance.py` (new), `test_read_evidence.py`, `orion/harness/tests/test_reading_receipts.py`, `services/orion-hub/tests/test_world_pulse_read_pipeline.py`, `test_world_pulse_read_stage2.py` (existing hop tests now supply link provenance), `reading_queue_fakes.py` (serves snapshot loads).
- Evals: `services/orion-hub/evals/run_reading_dead_url_replay.py` (live, read-only) and `test_reading_dead_url_eval.py` (offline, frozen live cases, added to CI).
- `.github/workflows/orion-reading-tests.yml`: runs the new eval.

## Schema / bus / API changes

- Added: none (new `last_error` / `stage2_error` label value `no_read_evidence:http_error`, terminal, prefix-compatible with existing `no_read_evidence*` handling).
- Removed / Renamed: none.
- Behavior changed: Stage 1 evidence rule; Stage 2 hop admission; Stage 2 skip on dead parent.
- Compatibility notes: Hub needs no new harness field. The harness change only affects chat `source_read`. Either side can deploy first.

## Metric quality gate (status-line rule)

1. **Provenance.** The tool_result text the reader received: `ReadingReceiptTracker._observe_tool_result` (`orion/harness/reading_receipts.py`) -> `SourceFetchEvidenceV1.content_text` -> Hub `source_read_evidence`.
2. **Independence.** This replaces nothing and adds no score. It is a pass/fail filter next to the size floor, and it uses the tool's status channel, not the text size.
3. **Theory anchor.** WebFetch does not digest the body of a 4xx/5xx response. It returns a fixed notice whose first line carries the HTTP status code. That is the same kind of tool-status signal as the "Redirect detected" notice the tracker already rejects. Anchored to the start of the text and parsed as a number, so it is not a list of error words.
4. **Live data.** 853 harness fetches; 74 with retained text (since 2026-10-11). The notice appears with codes 400/403/404/429, always as the first line, at 215–223 chars. Real digests in the same window run from 97 chars up. Real CoreWeave digests were 139–208 chars, so **no size cut separates them**, which is why the size floor stays a backstop and is not raised. 218-char rows before text retention match the 401 notice; the Falcon-ASR read's own model output confirms it was a 401.
5. **Existing mechanism.** `_usable_fetch_result` already rejects WebFetch's redirect notice; this extends the same idea. `MIN_SOURCE_CONTENT_CHARS` was checked and does not catch it.
6. **Reversibility.** One pure function, plus label strings. No schema or stored-shape change.

Known residual (not fixed, by design): WebFetch's inner model sometimes summarises an error page in prose ("The provided text is an error page, not an article", 119–161 chars). Catching that would need a phrase list, which is a keyword cathedral. Most of these already fall under the 200-char floor.

## Live replay (read-only, 14 days, `run_reading_dead_url_replay.py`)

- Stage 1: 30 `done` reads. **3 would become `failed`**: all three guessed CoreWeave siblings (`/infrastructure`, `/architecture`, `/gpu-cpu-compute`, 215-char 404 notice). One more is likely dead but cannot be judged because it predates text retention: Falcon-ASR-1.6B, 3x 218 chars (401). Borderline reads that stay `done`: the CoreWeave `/products/gpu-compute` 202-char digest (real content, verified in `reading_document_snapshot`) and a YouTube read (202/211 chars). **No real read is lost.**
- Stage 2: 16 parents asked for 30 hops. 4 allowed; 26 refused.
  - 3 refused with retained text available: exactly the guessed CoreWeave URLs (2 of them were 404 reads).
  - 4 refused as hops from a dead parent: the CoreWeave 404 parents.
  - 19 refused because the parent predates text retention (`no_retained_text`), so there is nothing to find a link in. 7 of those hops historically led to real pages (arXiv 2609.10712 and 2609.02849, an arXiv PDF, the llama.cpp `ggml-quants.c` source, the NASA flight-director FAQ, an HF collection, an HF dataset). These are the borderline losses; see Risks.
  - `nasa.gov/maxpower` (a real 1786-char read) is allowed because the parent digest wrote it without a scheme ("details at nasa.gov/maxpower"). The extractor handles that form.
- Duplicates (goal 4): 5 seeds repeat a URL in their evidence. **Not a pipeline bug.** The harness trace shows separate WebFetch calls the model made on the same URL with different prompts (e.g. arXiv 207/306/502 chars, ar5iv 396/277/393/1573), and the Falcon read retried a 401 three times. No change made.

## Env/config changes

- Added / Removed / Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest -q services/orion-hub/tests/test_world_pulse_read_*.py services/orion-hub/tests/test_reading_*.py orion/world_pulse_read/tests orion/harness/tests/test_reading_receipts.py services/orion-hub/evals/test_reading_*.py
reading CI group (orion-reading-tests.yml list): 716 passed, 64 skipped (Postgres-gated)
reading evals group: 26 passed
mutation check: removing the _stage2_pass -> source_fetches wiring fails test_stage2_turn_fetches_reach_the_hop_gate_through_the_real_pass
```

## Evals run

```text
services/orion-hub/evals/test_reading_dead_url_eval.py   8 passed
services/orion-hub/evals/run_reading_dead_url_replay.py --days 14   (numbers above)
```

## Docker/build/smoke checks

```text
Not rebuilt from this worktree (prod deploys from the primary checkout on main).
Live effect UNVERIFIED until Hub (and the harness governor, for chat source_read) is redeployed.
Post-deploy check: the next 404 read logs world_pulse_read_no_read_evidence ... reason=no_read_evidence:http_error,
and any guessed hop logs world_pulse_read_stage2_reentry_refused.
```

## Review findings fixed

Reviewer subagent: no blockers, no deploy-order hazard, assertions.py untouched.

- Finding: a parent could be marked dead when a page's retained text failed to load but a notice did.
  - Fix: `fetch_provenance.parent_read_state` says dead only when every evidence row is retained, every distinct text loaded, and all are notices.
  - Evidence: `test_parent_is_dead_only_when_every_retained_text_is_a_notice`.
- Finding: the `_stage2_pass` -> `GenerateOutcome.source_fetches` wiring was untested (hop tests injected fetches directly).
  - Fix: a test drives the real `_stage2_pass` with a stubbed `_generate`.
  - Evidence: `test_stage2_turn_fetches_reach_the_hop_gate_through_the_real_pass`; mutation check fails it.
- Finding: `http_error` fired when *any* same-site fetch was a notice, which polluted `thin_fetch` calibration.
  - Fix: `all(...)` over the candidates.
  - Evidence: `test_a_404_sibling_does_not_relabel_a_thin_real_page`.
- Finding: relative links (`](/x)`, `href="/x"`) were invisible.
  - Fix: `hop_allowlist` takes `(page_url, text)` pairs and resolves root-relative targets.
  - Evidence: `test_relative_links_resolve_against_the_fetched_page`.
- Finding: refused hops left no trace beyond a log line.
  - Fix: `hops_refused_total` counter by reason, plus a per-seed summary log. The stored result still keeps the model's list as asked: persisting refusals would need a schema change on a forbid-extra model, which is out of scope here.
  - Evidence: asserted in the real-pass test.
- Finding: default ports, percent-encoding, and `&amp;` were not normalized.
  - Fix: `hop_url_key` drops :80/:443 and unquotes the path; text is HTML-unescaped.
  - Evidence: `test_default_port_percent_encoding_and_html_entities_normalize`.
- Finding: the replay and eval re-implemented the dead/reached logic.
  - Fix: `parent_read_state` and `reached_pages` live in `fetch_provenance.py`, and Stage 2, the replay, and the eval all call them.
- Not changed (nits): bare `host.tld/path` can match `llama.cpp/server` (documented; this only widens the allowlist by text that was literally fetched). Comma-joined URLs are not split (a comma is a legal URL character). Two `load_retained_texts` round trips per Stage 2 (minor). `_stage2_fetches` instance state (single tick loop, reset per claim).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: WebFetch digests summarise. A real link the page contained may not appear in the digest, so a good hop gets refused. The replay cannot measure this for reads before 2026-10-11, and 7 historical good hops fall in that bucket.
  - Mitigation: the Stage 1 prompt now asks WebFetch for the page's linked URLs, and Stage 2's own fetch results count. Watch `world_pulse_read_stage2_reentry_refused` against `reading_claim_text_missing` for a week. Refusals are logged with the URL so a too-strict gate is visible.
- Severity: low
  - Concern: `no_read_evidence:http_error` is terminal, including for a 429 (rate limit) on the seed's own page.
  - Mitigation: none in this patch. No seed-page 429 has been seen live; the only 429s were on search-engine URLs, which are off-site and never evidence.

- Severity: low
  - Concern: reads stored `done` on a notice before this patch (the 3 CoreWeave siblings, plus the likely-401 Falcon-ASR read) stay `done`. `introspect.py` still counts them as reads, and re-entry dedup keeps treating those URLs as already read.
  - Mitigation: Stage 2 and hops are protected by the dead-parent skip. Flipping them to `failed` is a production write, so it is left as a follow-up backfill under the section 14 protocol. The replay lists the exact seed_ids.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2633

🤖 Generated with [Claude Code](https://claude.com/claude-code)
