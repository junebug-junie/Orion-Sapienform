# Hub chat: image input + readable rendering

Status: **proposal mode, approved by Juniper 2026-08-14** ("write this up in a spec
and hit it"). Implementation lands on `feat/hub-chat-vision-markdown`.

Date: 2026-08-14. Every runtime claim below carries live evidence from the
running Athena/Circe/Atlas nodes on that date, or is marked `UNVERIFIED`.

---

## Thesis

Juniper can talk to Orion but cannot *show* Orion anything. Every image in the
conversation today has to be transcribed into words by the human first. That is
a bandwidth ceiling on the relationship, not just a missing UI affordance.

Separately: Orion's replies render as `textContent` in a `whitespace-pre-wrap`
paragraph. Code comes back as an unformatted wall, tables as pipe soup. The
chat window is the highest-traffic surface in the whole system and it is the
least legible one.

Both are fixed here. They are independent — the rendering work does not depend
on the vision work — but they share one file (`appendMessage`) and one review.

---

## Live evidence: what can actually see

Probed `/props` on all four workers in `LLM_GATEWAY_ROUTE_TABLE_JSON`
(`services/orion-llm-gateway/.env:51`), 2026-08-14:

| route | port | model | `modalities.vision` | n_ctx |
|---|---|---|---|---|
| `chat` | :8011 | `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | **false** | 131072 |
| `agent` | :8014 | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | **true** (+video) | 32768 |
| `metacog` | :8012 | `Qwen_Qwen3-8B-Q5_K_M.gguf` | false | 4096 |
| `quick` | :8013 | `Qwen_Qwen3-8B-Q4_K_M.gguf` | false | 4096 |

So the premise "chat compute and agent compute run models that can do vision"
is **half true**. Agent sees. Chat is blind *at runtime*.

But the chat lane's blindness is a launch-flag omission, not a model
limitation. Two facts:

1. The chat worker's own chat template opens with
   `{%- set image_count = namespace(value=0) %}` and a matching `video_count` —
   that is the Qwen3-VL-family template. The weights understand images.
2. `unsloth/Qwen3.6-35B-A3B-GGUF` — the exact repo pinned at
   `config/llm_profiles.yaml:337` — ships `mmproj-BF16.gguf`, `mmproj-F16.gguf`,
   and `mmproj-F32.gguf` (HF model API, 31 files total, checked 2026-08-14).

`llama-server` reports `vision: false` when started without `--mmproj`. The
chat worker was started without it. That is the entire gap.

### The multimodal payload shape is proven, not assumed

Sent a hand-built 64×64 solid-red PNG to the live agent lane as an OpenAI
content-parts array:

```
POST http://100.112.254.99:8014/v1/chat/completions
content: [ {"type":"text","text":"What single color fills this image? Answer with one word."},
           {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}} ]
```

Reply `content`: `"red"`. Reply `reasoning_content` included
`"The image is a red square."` — real perception, not a lucky guess from the
text prompt. `usage.prompt_tokens` 79 with the image vs 64 for a text-only
control, so the projector cost ~15 tokens for a 64×64 image.

This pins three things the implementation depends on:
- the exact wire format llama.cpp mtmd accepts,
- that `data:` URIs work (no dependency on worker-side URL fetching),
- that Muse-Glimmer returns `reasoning_content`, which the gateway already
  handles (`llm_backend.py` reads it at several sites).

---

## Current architecture: what would break

Four chokepoints, all of which would silently mangle an image today.

**Three schemas type message content as a bare `str`:**

| file | line | field |
|---|---|---|
| `orion/core/bus/bus_schemas.py` | 167 | `LLMMessage.content: str` (`extra="forbid"`, `frozen=True`) |
| `services/orion-llm-gateway/app/models.py` | 8 | `ChatMessage.content: str` (`extra="forbid"`) |
| `orion/schemas/cortex/contracts.py` | 172 | `CortexChatRequest.prompt: str` |

**One serializer flattens whatever survives:**

`services/orion-llm-gateway/app/llm_backend.py:463-474` —

```python
def _serialize_messages(messages) -> List[Dict[str, str]]:
    ...
    serialized.append({"role": ..., "content": str(dumped["content"])})
```

and it builds the *actual outbound* `"messages"` at lines 700, 810, and 945.
Even with all three schemas widened, a content-parts list would reach
llama.cpp as a Python `repr` string.

**One dead flag:** `supports_vision` is declared at
`services/orion-llm-gateway/app/profiles.py:41` and referenced **nowhere else
in the repo**. Per `AGENTS.md` §0A that is a keyword cathedral — a label with no
runtime behavior. This patch wires it to real routing or it goes.

**Hub UI:**
- `services/orion-hub/static/js/app.js:6876` `appendMessage()` sets
  `body.textContent` on a `whitespace-pre-wrap` `<p>`. Plain text by
  construction — which is also why it has never had an XSS surface.
- No markdown or sanitizer library anywhere in the hub (nothing in
  `templates/index.html`, nothing in `package.json`).
- No upload endpoint exists (no `UploadFile`/multipart anywhere in
  `services/orion-hub/scripts/`). The only static mount is read-only
  `/static` (`main.py:853`).
- Normal chat turns are **not** token-streamed. Only the FCC harness streams
  step frames (`websocket_handler.py:621`). Replies land as one payload, so
  markdown is a one-shot render per message — no incremental-parse problem.

---

## Design: attachments ride beside content, never inside it

The way to break chat turns is to widen `content` from `str` to
`str | list[part]` across the saga. Spark ingest (`llm_backend.py:513`),
recall, `chat_history.py`, the memory-graph bridge, every trace consumer and
every reducer assumes a string. Widening that type is a cathedral and it would
regress paths nobody in this patch is looking at.

So content stays `str`. Attachments travel as a **typed sibling field**:

```python
class AttachmentRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha256: str          # content address; also the storage key
    mime: str            # image/png | image/jpeg | image/webp | image/gif
    bytes: int           # stored size, post client-side downscale
    width: int | None
    height: int | None
    kind: Literal["image"] = "image"
    source_url: str      # gateway-reachable URL to fetch the bytes
```

Refs on the wire. **Never base64 on the bus.** Per-event blobs into Postgres
already killed this host once (TOAST OOM crash loop, 2026-07-23) — a 2MB image
becomes ~2.7MB of base64 replicated through Redis streams, `chat_history`, and
every trace store that mirrors the turn.

### Wire path

```
Hub UI  --multipart-->  POST /api/chat/attachments        -> AttachmentRefV1
Hub UI  --chat payload {attachments:[ref]}-->  hub ws / POST /api/chat
Hub     --> CortexChatRequest.attachments      --> cortex-gateway
gateway --> CortexClientContext.attachments    --> cortex-orch --> cortex-exec
exec    --> ChatRequestPayload.attachments     --> llm-gateway
llm-gateway: resolve route -> if route is vision-capable,
             GET source_url, build data: URI parts, send multimodal
             else -> explicit refusal or explicit lane switch (never silent)
```

Every added field is additive with a default of `[]`. `extra="forbid"` rejects
*unknown* fields; these become known. **With `attachments == []` — which is
every turn Orion has ever taken — the outbound gateway payload is byte-identical
to today.** That property is the whole safety argument, and it gets a
golden-file test rather than a promise.

### Why the gateway fetches bytes

The gateway is the only service on the final leg, and it is where the
vision-capability decision is made. It already makes outbound HTTP to workers;
one more capped, host-allowlisted fetch to the hub is a smaller cost than
carrying blobs through five hops. Contract: `source_url` must be
gateway-reachable, size-capped, and host-allowlisted via env.

`_serialize_messages` is **not modified**. A sibling
`_serialize_messages_multimodal` is used only when attachments are present and
the route can see. The text path keeps its exact current behavior.

---

## Routing decision: make chat see (option A)

Three options were on the table. Recorded here because the rejected ones are
the tempting ones.

**A — load mmproj on the chat lane. CHOSEN.** Add `mmproj_filename` to
`qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition` in `config/llm_profiles.yaml`.
`services/orion-llamacpp-host` already implements the whole path
(`_ensure_mmproj_file`, `app/main.py:133`); Muse-Glimmer uses it today
(`config/llm_profiles.yaml:1058`). No new infrastructure. Orion's own voice
answers, on its own 131k-context lane, with no continuity break.

Cost, and it is real: ~1-2GB VRAM for the projector on top of ~24GB Q5_K_M
weights plus a 131k KV cache across 2×V100 32GB, and vision tokens consume
context. **This cannot be verified without restarting the chat worker**, which
is Juniper's call and Juniper's `sudo`. The config change ships here; the
bring-up is a separate, reversible step with the exact commands in the PR
report. Until that restart happens, the chat lane still reports
`vision: false` and the runtime gate below will correctly refuse images on it.

**B — route image-bearing turns to the agent lane.** Ships today, zero risk to
the chat worker, because the agent lane already sees. But a *different model
answers*. Kept as the interim fallback, and under §0A it cannot be silent: the
message gets a visible "seen by: agent lane (Muse-Glimmer-30B)" badge and the
lane lands in the trace.

**C — small VL model captions the image, caption injected as text. REJECTED.**
This is the empty-shell-cognition failure in §0A: Orion would say "I see" when
a different model saw and Orion read a transcript. If it is ever built, the UI
must say so out loud.

### The capability gate is runtime-derived, not config-declared

`supports_vision` in the profile registry is a *claim*. `/props`'s
`modalities.vision` is the *fact*, and they disagree right now — which is
exactly why the dead flag is dangerous. The gate reads the live worker's
`/props`, cached with a TTL, and falls back to refusing rather than to
assuming. Config being set is not proof (§0A, "runtime truth beats config
truth").

---

## Design: readable chat

Two new modules rather than more mass in the 574KB `app.js`. This matches the
existing `static/js/*.js` + `*.test.js` pattern and makes both testable.

**`chat-attachments.js`** — composer chip row above the input. File button,
drag-and-drop, and **paste-from-clipboard**, which is the one people actually
use. Client-side downscale to a max edge before upload (keeps stored bytes and
vision tokens bounded). Chips are removable pre-send. On send, the thumbnail
renders in the message; click opens a lightbox.

**`chat-markdown.js`** — markdown for `role=assistant` **only**. User turns stay
`textContent`. Code fences get a language class and a per-block copy button.

Sanitization is not optional here. Switching from `textContent` to `innerHTML`
introduces the hub's first XSS surface, and the content being rendered is model
output — partially influenceable through recall and any web content that
reaches the context. `marked` + `DOMPurify`, **vendored into
`static/js/vendor/`, not CDN.** The hub does pull Tailwind and Cytoscape from
CDN today (`index.html:7`, `:3526`), but a compromised CDN serving a broken
stylesheet and one serving script into the surface that renders Orion's replies
are different severity classes.

Emoji need no work beyond the font stack — they already survive `textContent`
and they survive markdown. `:shortcode:` expansion is ornamental; not built.

**Copy to clipboard** — whole transcript and per-message, both copying the
**source markdown**, not `innerText` of the rendered DOM. Copying a rendered
table as space-mangled text is the failure mode to avoid.

---

## Files likely to touch

Contract:
- `orion/schemas/cortex/contracts.py` — `AttachmentRefV1`, `CortexChatRequest.attachments`, `CortexClientContext.attachments`
- `orion/core/bus/bus_schemas.py` — `ChatRequestPayload.attachments`

Producer:
- `services/orion-hub/scripts/chat_attachments.py` — new store + endpoints
- `services/orion-hub/scripts/main.py` — route registration
- `services/orion-hub/scripts/cortex_request_builder.py` — populate `attachments`
- `services/orion-hub/scripts/websocket_handler.py` — accept `attachments` on the ws payload
- `services/orion-cortex-gateway/app/main.py` — pass through to context
- `services/orion-cortex-exec/app/executor.py` — context → `ChatRequestPayload`

Consumer:
- `services/orion-llm-gateway/app/llm_backend.py` — `_serialize_messages_multimodal`, vision gate, byte fetch
- `services/orion-llm-gateway/app/profiles.py` — wire or delete `supports_vision`

Runtime:
- `config/llm_profiles.yaml` — chat-lane `mmproj_filename`

UI:
- `services/orion-hub/static/js/chat-attachments.js` (+ `.test.js`)
- `services/orion-hub/static/js/chat-markdown.js` (+ `.test.js`)
- `services/orion-hub/static/js/vendor/` — `marked`, `DOMPurify`
- `services/orion-hub/static/js/app.js` — `appendMessage` render seam, composer wiring
- `services/orion-hub/templates/index.html` — composer markup, vendor script tags
- `services/orion-hub/static/css/style.css` — markdown + attachment styles

Env:
- `services/orion-hub/.env_example` + local `.env` — attachment dir, size cap, allowed mimes
- `services/orion-llm-gateway/.env_example` + local `.env` — fetch host allowlist, byte cap, `/props` cache TTL

---

## Non-goals

Video. Audio. Attachments in recall, memory-graph, or `chat_history` rehydrate.
Feeding chat-borne images into the perception substrate. Server-side thumbnail
generation (no Pillow dependency — the client downscales). Token-level
streaming. `:shortcode:` emoji.

On the perception substrate specifically: `orion:vision:*` and
`node:substrate.vision` already carry a live cam0 object-count signal
(`orion/substrate/prediction_error.py:770`). A chat-borne image is a
*different* channel — Juniper-authored, intentional, shared attention rather
than passive ambient sensing. Whether it should move any substrate signal is a
real question and a separate proposal. Not answered here, and deliberately not
wired, because wiring it without an answer is how a metric gets baked in before
it has a theory anchor (§0A metric quality gate, step 3).

---

## Acceptance checks

1. `attachments == []` produces a byte-identical gateway outbound payload to
   today — golden-file test over `_serialize_messages`.
2. Upload → serve round-trip against a live hub: `sha256` is stable, content
   address matches bytes, oversize and disallowed-mime uploads are refused.
3. Multimodal assembly emits exactly the shape proven above against :8014.
4. An image sent while the resolved route reports `vision: false` produces an
   explicit refusal or an explicit badged lane switch — **never** a silent
   text-only answer.
5. `supports_vision` gates real routing behavior, or it is deleted.
6. Markdown render test: an assistant turn containing a code fence, a table,
   and a `<script>` tag — the script does not execute.
7. User turns still render via `textContent`.
8. Copy-to-clipboard yields source markdown, not rendered `innerText`.
9. After the chat-worker restart, `/props` on :8011 reports `vision: true`.
   Until then this is `UNVERIFIED` and check 4 is what holds the line.

---

## Rollback

- UI: both modules are additive; `appendMessage` keeps its `textContent` path
  behind a flag, so rendering reverts without touching the vision work.
- Vision: remove `mmproj_filename` from the profile and restart the worker.
  The runtime `/props` gate then refuses images on its own — no code change
  needed to make the system safe again.
- Attachments: the store is content-addressed on disk under a single
  configurable directory. Deleting it orphans refs; it does not corrupt turns.
