# Vision scene belief — habituation at the window layer

**Date:** 2026-07-04  
**Status:** Draft — pending operator review  
**Scope:** `orion-vision-window` (producer), `orion-vision-council` (consumer gate only)  
**Goal:** ≥98% council metacog skip on stable office scenes by eliminating label flicker; Orion stays silent (contract A) after scene registration.

---

## 1. Problem

Live `orion-athena-vision-council` logs (post PR #810) show ~74% `stable_scene` skip but **100% of remaining metacog calls are `reason=labels_changed`**, often flickering between `labels=-` and `labels=door,package,screen`.

Root cause: `orion-vision-window` builds `evidence.hard_labels` from a single 5s artifact batch with score threshold 0.25. No temporal memory. One weak detection frame → empty label set → council treats it as a scene transition.

`refresh_ttl` (120s forced LLM) also conflicts with the desired contract: **after ~2 minutes of stability, Orion emits nothing until a real salient change.**

---

## 2. Operator contract (confirmed)

| Situation | Orion behavior |
|-----------|----------------|
| First time registering a scene (labels become stable) | Run metacog once; publish grounded events |
| Scene stable ≥2 minutes, nothing salient changed | **Silence** — no LLM, no heartbeat replay, no bus publish |
| Person enters/exits (on **belief**, not flicker) | Metacog on transition |
| Label added/removed persistently (debounced) | Metacog on transition |
| RPC on-demand | Always interpret (unchanged) |

This is **contract A**: silence is correct for stable scenes. Suggestion #3 (cheap `host_fallback` heartbeat) is **out of scope**.

---

## 3. Approach: belief in vision-window (Approach 2)

Habituation lives in the **perception projection layer** (`orion-vision-window`), not council. Window emits both **observed** and **believed** evidence tiers. Council gates metacog only on **belief** transitions.

Rationale:
- Flicker is a projection/aggregation problem; fix at source.
- Belief is inspectable via window HTTP (`/current`) and recovery snapshots.
- Council stays a thin consumer; no duplicate belief state.

**Non-goals:**
- SigLIP / visual embedding familiarity gate (defer unless belief debounce insufficient)
- Durable belief across process restart (ephemeral like `live_state`)
- Composite image sampling or pixel-hash deltas
- Keyword triggers or regex on narratives

---

## 4. Architecture

```text
artifacts (5s batch)
    → summarize_items() → observed hard_labels
    → SceneBeliefTracker.observe(stream, observed)
    → believed_hard_labels
    → VisionWindowPayload.summary.evidence
    → bus orion:vision:windows
    → council evidence_transition (belief snapshot only)
    → metacog OR stable_scene noop
```

### 4.1 Choke points

| Layer | File | Function |
|-------|------|----------|
| Belief producer | `services/orion-vision-window/app/scene_belief.py` | `SceneBeliefTracker` |
| Wire producer | `services/orion-vision-window/app/main.py` | `_flush_and_publish()` after `summarize_items` |
| Belief consumer | `services/orion-vision-council/app/evidence_transition.py` | `snapshot_from_window()` reads `believed_hard_labels` |
| Gate consumer | `services/orion-vision-council/app/main.py` | `_generate_interpretation()` (unchanged wiring) |

---

## 5. Scene belief algorithm (deterministic)

Per `stream_id`, maintain:
- `observation_ring`: deque of last **N** observed label sets (frozenset[str])
- `believed_labels`: frozenset[str] — current habituated inventory
- `person_present`: derived as `"person" in believed_labels`
- `registered_at`: timestamp when belief first became non-empty (optional, for metrics)
- `last_belief_change_at`: timestamp of last symmetric diff on belief (optional)

**Defaults (env-tunable):**
- `WINDOW_BELIEF_VOTE_N=3` — remember last 3 window observations (~15s at 5s flush)
- `WINDOW_BELIEF_ENTER_VOTES=3` — label must appear in all N observations to **enter** belief (production-tuned 2026-07-04; was 2)
- `WINDOW_BELIEF_EXIT_VOTES=0` — label must appear in 0 of last N raw observations to **exit** belief (production-tuned; was 1)

**On each flush:**
1. Compute `observed_labels` from current batch (existing `_build_evidence` logic).
2. Push `observed_labels` onto ring (maxlen N).
3. For each label ever seen in ring ∪ believed:
   - `count =` appearances in ring
   - If label ∉ believed and `count >= ENTER_VOTES` → add to believed
   - If label ∈ believed and `count <= EXIT_VOTES` → remove from believed
4. Emit both in `summary.evidence`:

```json
{
  "hard_labels": ["door", "package", "screen"],
  "believed_hard_labels": ["door", "package", "screen"],
  "belief": {
    "schema": "scene_belief.v1",
    "vote_n": 3,
    "enter_votes": 3,
    "exit_votes": 0,
    "observation_count": 3
  },
  "soft_labels": [],
  "host_person_hits": 0,
  "edge_person_hits": 0,
  "caption_count": 0
}
```

**Empty observations:** An empty `hard_labels` window is a valid ring entry. Empty-empty-empty does not clear belief (exit requires ≤1 of 3). This is the core flicker fix.

**Empty carry-forward (enter only):** For **enter** vote counting, empty ring slots inherit the last non-empty observed label set so a single weak frame does not block promotion. **Exit** uses raw ring counts only (empty slots contribute no labels), so `WINDOW_BELIEF_EXIT_VOTES` remains meaningful.

**Belief transition detection (for logging only in window):** symmetric diff on `believed_labels` before/after update. Log when non-empty:
`[WINDOW] belief_transition stream=cam0 added=[] removed=[] believed=door,package,screen`

---

## 6. Council consumer changes

### 6.1 Snapshot source

`snapshot_from_window()` prefers `evidence.believed_hard_labels` when present and non-empty schema; falls back to `hard_labels` for backward compatibility during rollout.

```python
def _labels_for_gate(window) -> frozenset[str]:
    evidence = (window.summary or {}).get("evidence") or {}
    believed = evidence.get("believed_hard_labels")
    if isinstance(believed, list) and believed:
        return frozenset(...)
    return frozenset(...)  # hard_labels fallback
```

### 6.2 Transition reasons (unchanged semantics, belief input)

| Reason | Condition |
|--------|-----------|
| `first_window` | No prior belief recorded for stream in council tracker |
| `person_entered` | person appears in belief, was absent |
| `person_exited` | person absent in belief, was present |
| `salient_labels_changed` | symmetric diff on belief label sets (rename from `labels_changed` in logs) |
| `stable_scene` | belief unchanged |
| `interpret_in_flight` | concurrent coalesce (existing) |

### 6.3 Disable refresh TTL

Set `COUNCIL_TRANSITION_REFRESH_SEC` default to **0** (disabled). Contract A: no periodic metacog on stable scenes. Operators may re-enable explicitly if needed.

---

## 7. Data flow and restart behavior

| Event | Window belief | Council tracker | Metacog |
|-------|---------------|-----------------|---------|
| Process restart (window) | Cleared | May still hold old state until council restart | Mismatch until both restart; acceptable |
| Process restart (council) | Unchanged | Cleared | `first_window` if belief non-empty → one interpret |
| Empty flicker window | Belief unchanged | No transition | Skip |
| Persistent new label (2/3 windows) | Belief updates | `salient_labels_changed` | Interpret once |
| 2+ min stable | Belief unchanged | `stable_scene` | Skip |

Window belief is **ephemeral** (in `WindowService`, same lifetime as `_buffers`). Matches §4.1 of `2026-05-02-orion-vision-window-projection-design.md` (live_state, not canonical memory).

---

## 8. Schema / bus contract

**Additive only** — `summary.evidence` is already `Dict[str, Any]`. No `VisionWindowPayload` pydantic shape change required.

New documented fields:
- `evidence.believed_hard_labels: list[str]`
- `evidence.belief: { schema, vote_n, enter_votes, exit_votes, observation_count }`

Update:
- `services/orion-vision-window/README.md` — evidence tiers table
- `docs/vision_services.md` — belief tier
- `services/orion-vision-council/README.md` — gate reads believed tier

No new bus channel. No registry schema ID required (projection metadata inside summary dict).

---

## 9. Configuration

### vision-window (`settings.py` + `.env_example` + `docker-compose.yml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `WINDOW_BELIEF_ENABLED` | `true` | Emit believed tier |
| `WINDOW_BELIEF_VOTE_N` | `3` | Observation ring length |
| `WINDOW_BELIEF_ENTER_VOTES` | `2` | Votes to enter belief |
| `WINDOW_BELIEF_EXIT_VOTES` | `1` | Max votes to stay in belief |

### council

| Key | Default | Change |
|-----|---------|--------|
| `COUNCIL_TRANSITION_REFRESH_SEC` | `0` | Was `120`; disabled per contract A |

---

## 10. Testing

### vision-window (gate tests)

- `test_belief_ignores_single_empty_observation` — `door,screen` → `[]` → `door,screen` belief stable
- `test_belief_requires_enter_votes` — new `package` in 1/3 windows → not in belief; 2/3 → in belief
- `test_belief_requires_exit_votes` — label absent 2/3 windows → removed from belief
- `test_flush_payload_includes_believed_hard_labels` — integration via `build_window_payload` path

### council (gate tests)

- `test_snapshot_prefers_believed_hard_labels`
- `test_stable_scene_when_observed_flickers_but_belief_stable` — observed `-`, believed `door,screen` → skip
- `test_refresh_ttl_disabled_by_default` — stable belief + time elapsed → still skip

### Acceptance (runtime)

After deploy, 30-minute cam0 sample:
- `evidence_transition interpret` count ≤2% of window count
- Zero interpret lines with `observed` flicker pattern `labels=-` (only belief transitions)
- `refresh_ttl` interpret count = 0

---

## 11. Metrics / debug

Window log line on belief change (INFO). Council keeps existing `evidence_transition skip|interpret` lines.

Optional HTTP: `summary.evidence.belief` visible on `GET /current?stream_id=cam0` — no new endpoint.

---

## 12. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Slow label appearance (1 window flash) delayed 10–15s | Low | Acceptable; matches human habituation |
| Fast enter/exit during belief update lag | Low | Council `interpret_in_flight` + next window catches exit |
| Council/window restart skew | Low | Single interpret on reconnect; document restart order |
| Belief too sticky after object removed | Medium | Tune `EXIT_VOTES`; default 1-of-3 is responsive (~15s) |

---

## 13. Implementation order

1. `scene_belief.py` + window settings/env/compose/README
2. Wire into `_flush_and_publish` / `projection.py`
3. Window tests
4. Council `snapshot_from_window` belief preference + refresh default 0
5. Council tests + README
6. Deploy window then council; verify log acceptance criteria

---

## 14. Non-goals (explicit)

- Moving belief to Redis recovery store
- LLM on stable scenes (including refresh TTL)
- `host_fallback` heartbeat publish (contract A)
- Edge pipeline changes
- Embedding-based familiarity (Phase 2 only if acceptance fails)
