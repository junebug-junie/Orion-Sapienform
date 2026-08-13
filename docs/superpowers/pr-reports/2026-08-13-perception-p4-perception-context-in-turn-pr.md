# PR report — perception P4: the room in the turn

Implements **P4** of the perception frontier design (PR #1590), after P0
(#1602) and P3 (#1616, #1620). This is the step where perception first reaches
Orion's cognition: before it, the vision stack had **zero** cognition consumers.

## Summary

- New `PerceptionContextV1` on `SituationBriefV1`, carrying the most recent
  camera percept — a natural-language scene summary and its age, nothing else.
- New `perception_reader.py` in cortex-exec: bounded, fail-open, read-only,
  following `metacog_trend_reader.py`'s conventions exactly.
- A hard staleness gate. Past the age threshold the percept is **withheld
  entirely** and renders as "haven't seen anything recently".
- Default **off**. This puts camera-derived content about a private home into
  the prompt, so it is opt-in.

## Outcome moved

The prompt fragment Orion receives, rendered live end-to-end against the real
database with the flag on:

```text
Situation:
- Local context: pre dawn Thursday, America/Denver.
- Conversation phase: unknown; continuity=continue_directly.
- Presence: requestor=Juniper, audience_mode=solo.
- Weather: unavailable or low-confidence; do not infer.
- Lab: unavailable/stub; do not infer.
- Room (seen 7 min ago): Multiple chairs, a door, a desk, a screen, and a table are visible in the scene.
- Situation context is grounding, not a requirement to mention.
```

That last-but-one line is the deliverable. `SituationBriefV1` previously gave
Orion the weather in Denver and nothing about the room it was sitting in.

## Proposal-mode scoping

Per AGENTS.md §0A, recorded here rather than in a separate doc because Juniper
directed proposal-then-implement in one pass.

- **What capability changes.** Orion's turn context gains a summary of what its
  camera most recently saw. It does not gain the ability to look on demand
  (that is `look()`, P1), nor to act on what it sees.
- **What data is touched.** Read-only `SELECT narrative, created_at FROM
  vision_events ORDER BY created_at DESC LIMIT 1`. No writes. `entities` and
  every other column are deliberately not read.
- **Privacy boundary.** The camera watches a private home office. The exposed
  field list *is* the contract: `scene_summary`, `observed_at`,
  `observation_age_seconds`, `stream_id`, `source`, `privacy_mode`. Absent by
  design, and not addable without a separate sign-off: raw frames or frame
  paths, bounding boxes, per-object detections, embeddings, and anything
  identity-bearing. `extra="forbid"` on the model means a future caller cannot
  smuggle one in silently; a test asserts the ban list.
  `privacy_mode="session_only"` is the default and the only value that should
  hold without an explicit change.
- **What trace proves it worked.** The rendered fragment above, plus
  `brief.source_summary["perception"]` and
  `diagnostics.provider_status["perception"]` on every brief, which name the
  provider state (`live`/`stale`/`disabled`/`unavailable`/`error`).
- **What failure mode would be dangerous.** A stale percept presented as a
  current observation — a confabulation with a real referent, which is worse
  than silence because it is checkable and wrong. Mitigated by withholding the
  summary entirely past the age gate, not merely flagging it.
- **How to disable or roll back.** `ORION_SITUATION_PERCEPTION_ENABLED=false`,
  which is the default. No restart of anything but cortex-exec, no data
  migration, nothing persisted.

## Design decisions worth naming

**The stale branch drops the summary from the payload, not just from the
prompt.** A stale summary carried "for debugging" is one refactor away from
reaching a prompt. `observed_at`/`observation_age_seconds` still travel so a
debug surface can say how old, but the text does not.

**"Haven't seen anything recently" is not "the room is empty."** Not seeing and
seeing nothing are different claims and only one of them is true when the
camera is stale or off. A test asserts the fragment never renders the second.

**Age threshold: 900s.** Live `vision_events` arrive roughly every 5 minutes on
a static scene (measured 2026-08-13, post-#1602), so this tolerates a few
missed windows without letting an hour-old percept read as current.

## Files changed

- `orion/schemas/situation.py`: `PerceptionContextV1`; additive `perception`
  field on `SituationBriefV1`.
- `orion/schemas/registry.py`: registered — the sibling nested contexts
  (`LabContextV1`, `EnvironmentContextV1`, `PlaceContextV1`) all are, so this
  is the existing contract rather than a new one.
- `services/orion-cortex-exec/app/perception_reader.py`: new reader.
- `services/orion-cortex-exec/app/situation.py`: `_build_perception_context`,
  the prompt line, settings plumbing.
- `services/orion-cortex-exec/app/settings.py` + `.env_example`: three keys.
- `services/orion-cortex-exec/tests/test_situation_perception_context.py`: new,
  14 tests.

## Schema / bus / API changes

- Added: `PerceptionContextV1`; `SituationBriefV1.perception`;
  `source_summary["perception"]`.
- Additive and defaulted — an unpatched producer or a disabled flag yields
  `available=False`, i.e. "haven't seen anything recently", never a missing
  field or a crash.
- No bus channels touched. No writes anywhere.

## Env/config changes

- Added: `ORION_SITUATION_PERCEPTION_ENABLED` (false),
  `ORION_SITUATION_PERCEPTION_MAX_AGE_SECONDS` (900),
  `ORION_SITUATION_PERCEPTION_STREAM_ID` (cam0).
- `.env_example` updated; local `.env` synced by hand.
- **No docker-compose change needed**: cortex-exec uses `env_file: .env` and
  enumerates only 1 of its 24 `ORION_SITUATION_*` keys explicitly. Checked
  rather than assumed — the opposite is true of `orion-substrate-runtime`,
  where this exact difference cost a deploy during P3.

## Tests run

```text
$ pytest test_situation_perception_context.py test_situation_provider.py \
         test_situation_settings_env.py -q
24 passed
```

Pre-existing failures, confirmed identical on unmodified `main` and unrelated
to this branch:

- `test_situation_prompt_integration.py` — 2 failures, `jinja2 'metadata' is
  undefined`.
- `services/orion-cortex-exec/tests` full run — 13 collection errors,
  `Verb already registered: legacy.plan`.
- repo-root `tests/` — 32 collection errors.

## Evals run

```text
None. services/orion-cortex-exec has no evals/ directory for this surface.
```

The live end-to-end render above is the acceptance evidence. A real eval here
would score whether Orion *uses* the percept appropriately when it is relevant
and stays quiet when it is not — worth building, and not something this patch
can assert about itself.

## Live checks

```text
$ perception_reader.fetch_latest_percept()   (against live conjourney)
{'scene_summary': 'Multiple chairs, a door, a desk, a screen, and a table are
 visible in the scene.', 'observed_at': 2026-08-13T08:07:43Z}
age_sec: 413

$ build_situation_for_ctx(...) with the flag enabled
perception slot: available=True, source='live', observation_age_seconds=424,
                 privacy_mode='session_only'
prompt line:     "Room (seen 7 min ago): Multiple chairs, a door, a desk, a
                  screen, and a table are visible in the scene."
```

## Acceptance checks vs the design doc

P4's stated checks:

- *"a real chat turn whose prompt fragment carries a current percept, with the
  trace ID"* — **partially met.** The fragment is rendered live from the real
  database above, but through a direct `build_situation_for_ctx` call rather
  than an end-to-end chat turn, because the flag ships off. Flipping it on a
  live turn is the operator's call, not this patch's.
- *"a stale case rendering as 'haven't seen anything recently'"* — **met**,
  asserted by test and visible in the disabled/stale branches.

## Restart required

Only if enabling:

```bash
# set ORION_SITUATION_PERCEPTION_ENABLED=true in services/orion-cortex-exec/.env
bash scripts/safe_docker_build.sh orion-cortex-exec up -d
```

Otherwise no restart — the default is off and the code path is inert.

## Risks / concerns

- **Severity: medium (privacy).** This is the first surface that puts
  camera-derived content about a private home into a prompt. The exposed-field
  list is deliberately minimal and test-enforced, but the narratives are
  LLM-generated by the vision council and could in principle describe a person
  ("one person detected" appears in historical rows). That is presence, not
  identity, and identity/face/re-ID remain disabled upstream — but it is the
  reason this ships default-off rather than default-on.
- **Severity: low.** Object counts in the narrative are approximate (a known
  limit from #1602 — the detector reported 3 chairs for 2). Orion may therefore
  state a count that is off by one. The prompt frames the whole block as
  grounding rather than fact, but a consumer treating the summary as an
  inventory would be wrong.
- **Severity: low.** One extra database query per situation-brief build, bounded
  by a 1500ms `statement_timeout` and cached behind the existing brief TTL
  (default 300s), so it is not per-turn in practice.

## What this does not deliver

`look()` — perception Orion *invokes* mid-turn (P1). This is a passive slot: it
reports the most recent percept, and Orion cannot choose to look.
