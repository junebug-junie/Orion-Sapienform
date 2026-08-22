# orion-percept-store

Short-lived, content-addressed storage for camera frames.

A percept exists so a model can look at it **once**. The interpretation is the
durable artifact; the picture is not. This service exists to make that true
structurally instead of by intention.

## Why this is not Hub

Hub already serves a content-addressed store at `/api/chat/attachments/{sha256}`,
and `orion-llm-gateway` already knows how to fetch from it. Reusing it would
have been fewer moving parts, and it is the wrong call:

| | Hub chat attachments | percept store |
| :--- | :--- | :--- |
| content | files a person chose to upload | camera frames of a private home |
| retention | indefinite | **1 hour**, swept in-process |
| served by | Hub, which holds the docker socket | its own process, no socket |

Three different answers, so: two stores. The separation is enforced by
`AttachmentRefV1.kind` (`"image"` vs `"percept"`), which the gateway reads to
pick a base URL. An unset `LLM_GATEWAY_PERCEPT_BASE_URL` **refuses** rather
than falling back to the chat base — silent reuse is the failure that would
matter.

## Contract

```
POST /percepts            raw image bytes  -> {"sha256", "mime", "bytes"}
GET  /percepts/{sha256}   -> the bytes, or 404 if absent/expired
GET  /stats               count, bytes, oldest age, retention  (never hashes)
GET  /healthz /readyz
```

`X-Orion-Percept-Token` gates both if `PERCEPT_STORE_TOKEN` is set. Empty
disables the check, which is acceptable only on a closed tailnet.

## The properties that matter

- **Non-enumerable.** The only caller-supplied component of a read is a
  64-char hex digest. No directory to traverse, no name to guess: you can only
  ask for a hash you already hold. `/stats` reports size, never hashes.
- **Verified on read.** Bytes are re-hashed and refused on mismatch. A
  content-addressed store returning the wrong content is worse than one
  returning nothing.
- **Sniffed, not trusted.** The declared content-type is ignored; the mime is
  read from magic bytes, because that value ends up in a model prompt.
- **Expiring by default.** The sweep runs at boot and on a timer, so "we will
  add retention later" cannot happen. Age is mtime and a re-store refreshes it,
  so "old" means *not seen recently*, not *first seen long ago* — a frame still
  being requested is not swept out from under an in-flight interpretation.

## Operating

```bash
cd <worktree> && scripts/safe_docker_build.sh orion-percept-store up -d --build
curl localhost:8021/stats
```

`PERCEPT_STORE_DIR` must never point at `HUB_CHAT_ATTACHMENT_DIR`.

Raise `PERCEPT_RETENTION_SECONDS` only with a reason you would be happy to read
back in six months. The default is short on purpose.

## Not done yet

Capture agents still publish a local `image_path`; nothing writes to this store
automatically. The next patch adds `sha256` to `VisionFramePointerPayload` so a
node with no shared filesystem — carbon, or any future camera — can feed the
pipeline. See the seeing-Juniper design doc.
