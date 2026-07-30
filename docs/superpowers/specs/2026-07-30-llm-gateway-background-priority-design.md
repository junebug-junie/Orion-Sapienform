# LLM gateway background-priority lane

Status: implemented (2026-07-30, Juniper: "make ai town generation queue behind
every other process... if this is a good pattern, it can be a pilot infra for
other worker lanes").

## Arsonist summary

AI Town's NPC dialogue was just moved back onto the `quick` route
(`atlas-worker-fast-1`, GPU1) after the circe drift fix earlier today. `quick`
is also the default route for `orion-mind` (semantic + stance), `orion-embodiment`
(hub-mode speech), and `orion-hub` (memory-graph-suggest). Juniper asked: can AI
Town share that one model/GPU without ever making those other, snappier
consumers wait behind it -- no new GPU is available (confirmed live: both V100s
already host one model each, `atlas-worker-2` on GPU0 for `metacog`,
`atlas-worker-fast-1` on GPU1 for `quick`), so a dedicated second small model
was ruled out.

Confirmed live: `atlas-worker-fast-1`'s llama.cpp server runs `--parallel 4`
(continuous batching across 4 slots) and exposes a real-time `/slots` endpoint
reporting each slot's `is_processing` state. The gateway itself
(`services/orion-llm-gateway/app/openai_passthrough.py`) does pure synchronous
proxying today -- no admission control, no queueing, no per-caller priority of
any kind (grepped the whole app package: zero matches for
queue/priority/concurrency/semaphore). Every `quick` consumer hits the
identical route key, so there is no way to distinguish "AI Town" traffic from
"mind" traffic at request-dispatch time without a new signal.

This adds that signal as a generic route-table feature, not an AI-Town special
case: an optional `priority: "background"` field on any `LLM_GATEWAY_ROUTE_TABLE_JSON`
entry. A background-tagged route polls the upstream's own `/slots` before
forwarding and only proceeds once enough slots are free to leave headroom for
foreground traffic; every existing route (no `priority` field) is completely
unaffected -- this is additive, not a behavior change for current consumers.

## Current architecture

- `RouteTarget` (`services/orion-llm-gateway/app/llm_backend.py`): frozen
  dataclass `{url, backend, served_by, model}`, populated generically from
  whatever keys are present in each `LLM_GATEWAY_ROUTE_TABLE_JSON` entry.
- `handle_chat_completions_post` (`openai_passthrough.py`): resolves the
  route, forwards the caller's body verbatim (`forward_body = dict(body)`) via
  `httpx.AsyncClient.post`, returns the response. No admission control.
- `atlas-worker-fast-1` (`quick`): `llama-server --parallel 4`, real `/slots`
  endpoint live (verified: 4 slots, `is_processing: true|false` per slot,
  `total_slots: 4` in `/props`).
- Real current `quick` consumers, all sharing one route key with no way to
  tell them apart: `orion-mind` (`MIND_SEMANTIC_MODEL_ROUTE`,
  `MIND_STANCE_MODEL_ROUTE`), `orion-embodiment`
  (`EMBODIMENT_SPEECH_HUB_LLM_ROUTE`), `orion-hub`
  (`MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE`), the gateway's own
  `LLM_ROUTE_DEFAULT`, and now AI Town (`LLM_MODEL` via Convex, fixed to
  `quick` earlier today in `fix/aitown-no-circe`).

## Missing questions

None outstanding -- direction (new route key, not a header patch to AI Town's
vendored code; fail-open on timeout; reserve 2 of 4 slots for foreground)
confirmed with Juniper via the design-mode exchange preceding this spec.

## Proposed schema / API changes

`RouteTarget` gains two new optional fields, parsed the same way as the
existing ones (`value.get(...)`, absent -> `None`, fully backward compatible):

```python
@dataclass(frozen=True)
class RouteTarget:
    url: str
    backend: Optional[str] = None
    served_by: Optional[str] = None
    model: Optional[str] = None
    priority: Optional[str] = None            # None/"foreground" (default) | "background"
    reserved_free_slots: Optional[int] = None # only meaningful when priority == "background"
```

New route-table entry (same upstream URL as `quick`, AI Town-facing):

```json
"quick_background": {
  "url": "http://100.121.214.30:8013",
  "served_by": "atlas-worker-fast-1",
  "backend": "llamacpp",
  "priority": "background",
  "reserved_free_slots": 2
}
```

New module `services/orion-llm-gateway/app/priority_admission.py`:

```python
async def wait_for_slack(target: RouteTarget, *, poll_interval_sec: float, max_wait_sec: float) -> bool:
    """Poll {target.url}/slots until free slots >= target.reserved_free_slots.
    Returns True if slack was found, False if it timed out (caller forwards
    anyway either way -- fail-open, never silently drops a turn)."""
```

`openai_passthrough.py`'s `handle_chat_completions_post`: after resolving the
route, if `target.priority == "background"`, call `wait_for_slack(...)` before
the existing `_proxy_upstream_json` forward. Log a warning on timeout (still
forwards -- fail-open, matching `orion-embodiment`'s existing fail-open speech
path rather than dropping an NPC's turn silently forever).

New env knobs (`services/orion-llm-gateway/.env_example`):
- `LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC` (default `30`)
- `LLM_GATEWAY_BACKGROUND_POLL_INTERVAL_SEC` (default `0.5`)
- `LLM_GATEWAY_BACKGROUND_CONCURRENCY` (default `1`) -- an `asyncio.Semaphore`
  scoped per background route, capping how many background-tagged requests
  this gateway process will have in flight to one upstream at once, even if
  slots are nominally free. Keeps AI Town (or any future background lane)
  from claiming multiple slots simultaneously just because several NPCs
  happened to want to speak in the same tick.

AI Town side: repoint the live Convex `LLM_MODEL` from `quick` to
`quick_background` (same live admin-API mechanism used for the circe fix
earlier today -- no AI Town code changes). `check_llm_route_not_circe.py`
needs no change: it still resolves `quick_background` to
`served_by: "atlas-worker-fast-1"`, still non-circe, still passes.

## Files likely to touch

- `services/orion-llm-gateway/app/llm_backend.py`: `RouteTarget` fields +
  parsing.
- `services/orion-llm-gateway/app/priority_admission.py` (new): slot-polling
  + semaphore gate.
- `services/orion-llm-gateway/app/openai_passthrough.py`: call the gate for
  `priority == "background"` routes.
- `services/orion-llm-gateway/app/settings.py`: new env knobs.
- `services/orion-llm-gateway/.env_example`, `docker-compose.yml`: new keys
  (env/compose parity).
- `services/orion-llm-gateway/README.md`: document the pattern generically
  (not AI-Town-specific) as the reusable "background priority lane" seam.
- `services/orion-llm-gateway/tests/`: new tests for `wait_for_slack` (mocked
  `/slots` responses: idle -> immediate; busy-then-frees -> waits then
  proceeds; permanently busy -> times out, still returns to let the caller
  fail open) and for the passthrough wiring (background route calls the
  gate, foreground routes don't).
- Live (not repo-tracked): add `quick_background` to the deployed
  `LLM_GATEWAY_ROUTE_TABLE_JSON`; repoint AI Town's Convex `LLM_MODEL`.
- `services/orion-ai-town/README.md`: note the new default route.

## Non-goals

- Not a full multi-tier fair scheduler -- just foreground/background. More
  tiers can reuse the same `priority` field later if another lane needs it.
- Not touching llama.cpp's own scheduling or config.
- Not patching AI Town's vendored `chatCompletion()` -- the new-route-key
  design needs zero AI Town code changes.
- Not a durable/persistent queue -- a gateway restart mid-wait just means
  that one in-flight HTTP call fails/retries per each caller's own existing
  retry logic (AI Town's embodiment worker already fails open on speech
  errors).
- Not detecting GPU compute saturation beyond slot count (e.g. an
  unusually long prompt occupying a "free" slot for a long time) -- a known,
  named limitation, not solved here.

## Acceptance checks

- Existing `quick` consumers (mind, hub, embodiment hub-speech): zero code
  change, zero observed behavior change (their route entries have no
  `priority` field, so `wait_for_slack` is never called for them).
- AI Town on `quick_background`: dispatches immediately when the upstream is
  idle; waits (bounded by `LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC`) when slots
  are scarce; always eventually forwards (fail-open), never silently drops a
  turn.
- `check_llm_route_not_circe.py` still passes unchanged after the AI Town
  repoint.
- New tests cover: idle dispatch, busy-then-frees dispatch, timeout fail-open,
  and confirm foreground routes never invoke the gate at all.

## Recommended next patch

Implement as scoped above in a single PR on
`feat/llm-gateway-background-priority`
(`/mnt/scripts/Orion-Sapienform-llm-gateway-background-priority`).
