# One channel name, several unrelated meanings

First patch of the design in PR #2054 (`docs/superpowers/specs/2026-09-03-recall-signal-rendering-design.md`). Nothing else in that spec can be verified until this is true, so it ships alone.

## Summary

- The field channel glossary was keyed by channel name alone.
- That is right for the 48 raw digester channels -- `cpu_pressure` means the same thing on every node -- and wrong for the `node:substrate.*` domain nodes, which are not physical hosts and each carry an unrelated reading under the same name.
- `node:substrate.bus_synaptic.prediction_error` is the share of the message bus running irregular. `node:substrate.vision.prediction_error` is how overdue the camera is. One entry described both.
- Adds node-qualified entries and `resolve_channel(channel, node=...)`, which prefers them and falls back to the bare channel.
- Each node entry also names where its history lives, and its channel in the transport lattice policy when it has one.

## Outcome moved

Live on 2026-09-02, `bus_synaptic.prediction_error` sat at 0.021 while `vision.prediction_error` sat pinned at 1.0. Rendered through the glossary they produced the same sentence -- "how much a recent prediction missed reality" -- for a mildly jittery bus and a blind camera.

They now render differently, which is the acceptance check the rest of the spec depends on.

## Architecture touched

`node_channels:` is a **separate list** in the yaml, not more rows in `channels:`. Hub's Field Channel Glossary panel iterates `entries` keyed on `e.channel` (`field_channel_glossary_routes.py:278`), so same-named rows added there would render as duplicates. Existing consumers are untouched; a test pins that.

`resolve_channel()` returns `None` for a channel nobody has described, never a placeholder -- a caller must not be able to render a confident sentence about an undescribed channel.

## Files changed

- `config/field/field_channel_glossary.v1.yaml`: `node_channels:` with two entries; `version: 1` -> `2`
- `orion/field/channel_glossary.py`: `FieldChannelNodeEntry`, loaded into `node_entries`, and `resolve_channel()`
- `orion/field/tests/test_channel_glossary_node_key.py`: new, 8 tests

## Schema / bus / API changes

- Added: top-level `node_channels:` list; entries carry `node`, `channel`, `category`, `meaning`, optional `trend_source` and `policy_channel`.
- Added: `load_glossary()` returns a `node_entries` key alongside `entries` and `categories`.
- Added: `resolve_channel(channel, *, node=None, path=None)`.
- Behavior changed: none. Every existing consumer reads `entries` and sees an identical list.
- Compatibility: additive. Callers passing no `node` get exactly today's resolution.

## Env/config changes

None.

## Tests run

```text
pytest orion/field/tests/test_channel_glossary_node_key.py -q  -> 8 passed
pytest orion/field orion/metrics -q                            -> 8 passed
pytest services/orion-hub/tests/test_field_channel_glossary_routes.py \
       services/orion-hub/tests/test_field_channel_glossary_hub_tab.py -q
                                                               -> 24 passed, 1 pre-existing failure
```

The failure is `test_channels_endpoint_returns_38_raw_channels_plus_1_derived`, asserting 39 channels against a file that has held 48 for some time. Confirmed pre-existing by reverting both changed files and re-running: identical failure. Not fixed here -- correcting that count is a separate call about whether the test or the file is wrong.

## Deliberate non-goals

- `node:substrate.perception` gets no entry. It ships shadow-only and default-off (`SUBSTRATE_PERCEPTION_PREDICTION_ERROR_TICK_ENABLED=false`), and an entry for a channel nothing reads is the keyword cathedral CLAUDE.md 0A bans. It belongs in the patch that turns it on.
- Nothing consumes `resolve_channel()` yet. The resolver in `conversation_front.py` is the next patch in the spec.

## Restart required

```text
No restart required. Config and a library function; no running service reads
node_channels yet.
```

## Risks / concerns

- **Severity: low.** `load_glossary()` is `lru_cache`d on the path, so a yaml edit needs a process restart to take effect. Pre-existing behavior, unchanged, but now it gates a second list.
- **Severity: low.** Nothing enforces that a `node:` in `node_channels` matches a node that actually exists in the field topology. The test asserts the `node:substrate.` prefix only. A typo'd node id would silently never resolve and fall back to the bare entry -- quiet, not wrong, but quiet.

## PR link

<pending>
