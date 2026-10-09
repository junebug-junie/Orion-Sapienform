# Reading internal documents by path (2026-09-28)

## Arsonist summary

The Reading pipeline only accepted public web URLs, so Orion could not read
their own specs. This patch adds one more source kind, `file:///abs/path?sha256=<hex>`.
It goes through the same seed queue, the same durable Stage 1 turn, the same
evidence gate, and the same Concept Atlas landing as a web read. No new service,
channel, or reader is added.

## Current architecture (before)

- `ReadingRequestedV1.url` was an `HttpUrl`. `validate_source_url` refused
  anything that was not public http(s).
- The Stage 1 prompt told the reader to fetch the URL. Read evidence was the
  reader's own `WebFetch` tool calls, matched against the seed URL.
- Dedup (`already_read` / `already_queued`) was keyed by the seed `url` column.

## Design

- **Capture once, at acceptance.** `accept_source` (`orion/world_pulse_read/queue.py`)
  reads the file through `DocumentPolicy`. It stores the bytes in the
  content-addressed `reading_document_snapshot` table and rewrites the source
  to the pinned ref. The durable binding is immutable, so the prompt has to be
  built from a snapshot that never changes, not from whatever the file says later.
- **Versioned dedup for free.** The sha lives in the seed URL. An unchanged
  file dedups as `already_read`, and an edited file is a new read. A status
  lookup by the bare path matches any version (`split_part(url, '?', 1)`).
- **Evidence is Hub's, not the model's.** Stage 1 puts the text inside a fenced
  block in the prompt. After the turn, Hub checks that the text is in the
  *bound* prompt and records `SourceFetchEvidenceV1(tool_name="orion_document_snapshot",
  content_sha256=...)`. `source_read_evidence` accepts only that tool name for
  document seeds, with a sha that matches the URL, so no model tool call can
  forge document evidence.
- **Size.** The reader is launched as `claude -p <prompt>`. A single argument
  is capped at 131072 bytes, and the agent lane has a 32k-token context. The
  default cap is 49152 bytes; larger files are refused, never truncated. The
  eval measured 279 of 282 repo specs (98.9%) accepted, with a largest prompt
  of 50 KB.

## Privacy boundary

- Only files under `HUB_READING_DOCUMENT_ROOTS` are readable. The default is
  the repo checkout plus Orion's sandbox copy, both already mounted into Hub.
  Roots are resolved with realpath, so a symlink that escapes the roots is
  refused.
- Always denied: `.git`, `.ssh`, `.gnupg`, `.aws`, `.docker` path components;
  `.env*` and `id_*` names; key/cert suffixes.
- Only the allowlisted extensions are read, and only as UTF-8 text with no NUL
  bytes. Non-regular files (FIFOs, devices) are refused before `open`.
- The path can be swapped between the checks and the open (for example, for a
  symlink, or for a named pipe that would block forever). The file is opened
  without following a final symlink and without blocking, then judged by what
  was actually opened: it must be a regular file, and its real path must be the
  path that was checked. Otherwise the result is `document_changed_during_read`.
- A request that already carries `?sha256=` is never re-read. It still passes
  the same path checks and the on/off switch, and it is accepted only if Hub
  captured those bytes from that exact path (the snapshot's `first_source`, or
  an existing seed row with that ref). A known hash cannot vouch for a
  different file.
- Refusals return a short policy code only, never file text or a stack trace.
- Captured text is stored in Hub's Postgres and sent to the same reader lane as
  web reads. Nothing leaves the host beyond what a web read already sends.

## Trace

- The seed row URL `file:///...?sha256=...` in `world_pulse_read_seed`.
- The snapshot row in `reading_document_snapshot`.
- `read_evidence[].tool_name == "orion_document_snapshot"` on the handoff, with
  a Reading tab badge that says "Hub document snapshot".
- Refusal codes on the receipt / `last_error`: `document_*`,
  `no_read_evidence:document_not_in_prompt`.

## Dangerous failure modes

- Secrets read by path. Guarded by the roots, the denylist, and the extension
  allowlist, and covered by tests in `orion/world_pulse_read/tests/test_documents.py`.
- Evidence faked without the text. Guarded by the bound-prompt check and the
  tool-name-only gate, and covered by `test_world_pulse_read_pipeline.py`.

## Rollback

Set `HUB_READING_DOCUMENT_ROOTS=` (empty) and recreate Hub. Every document
request is then refused with `document_reading_disabled`. Web reads are
untouched, and the snapshot table can stay in place because it is inert.

## Non-goals

- Stage 2 re-entry for documents (it stays web-only).
- Reading from worktrees, binary formats, or PDFs.
- Uploads (the next PR reuses `capture_text` and the snapshot table).

## Acceptance checks

- `pytest orion/world_pulse_read/tests/test_documents.py services/orion-hub/tests -q`
- `pytest services/orion-hub/evals/test_reading_document_eval.py -s`
- `RUN_READING_POSTGRES=1 pytest services/orion-hub/tests/test_reading_postgres.py -q`
- Live: paste a spec path into the Reading tab, watch Stage 1 finish with the
  snapshot badge, and see a second submit return `already_read`.
