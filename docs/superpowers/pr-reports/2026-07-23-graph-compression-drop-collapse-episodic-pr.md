# PR report: retire `orion:collapse` from `EpisodicFederator`

## Summary

- Fifth in today's series of Fuseki-dependency retirements in `orion-graph-compression`. Same pattern as the prior `orion:chat:social` removal: `EpisodicFederator` stays live, one more dead graph drops out of its list (8 -> 7 graphs).
- Live SPARQL `COUNT` against the running Fuseki container confirmed `orion:collapse` has exactly **0** triples, ever -- not low, zero.
- Traced the real reason, correcting an initial incomplete theory (see Review findings below): `orion-rdf-writer`'s `collapse.mirror.entry` write handler has its own observer/dense gate, but that gate is unreachable -- `orion-rdf-writer` only subscribes to `orion:collapse:intake`, which carries a different `kind` (`collapse.mirror.intake`, from `orion-cortex-exec`). The only real producer of `kind="collapse.mirror.entry"` (`orion-collapse-mirror`) publishes it to a different channel, `orion:collapse:triage`, whose registered consumers (`channels.yaml`) don't include `orion-rdf-writer` at all. This dispatch branch has never received a matching envelope, period -- a real channel/kind mismatch bug in `orion-rdf-writer`, flagged for its own follow-up, not fixed here.

## Outcome moved

`EpisodicFederator`'s SPARQL dependency narrows to 7 graphs: chat, enrichment, cognition, metacog (all dead writers / frozen history at this point), and the 3 autonomy graphs. Collapse tag/entity clustering signal (a separate concern from this raw-entry graph) has a real, if currently dark-by-default, Falkor path already built (`RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` in `orion-meta-tags`).

## Current architecture

Before this patch: `EpisodicFederator` queried 8 named Fuseki graphs via UNION, including `orion:collapse`, which has provably never held any content.

## Architecture touched

- `services/orion-graph-compression/app/federators/episodic.py`, `app/worker.py`, `app/stale_listener.py`, `.env_example`, `README.md`, `tests/test_federator_episodic.py`

## Files changed

- `services/orion-graph-compression/app/federators/episodic.py`: removed `http://conjourney.net/graph/orion/collapse` from `EPISODIC_GRAPHS`, with a comment explaining the verified channel/kind mismatch root cause (corrected mid-review, see below).
- `services/orion-graph-compression/app/worker.py`, `.env_example`: comment updates at the Falkor-union call site, now reflecting both retirements (social + collapse).
- `services/orion-graph-compression/app/stale_listener.py`: added a comment on the now-fully-inert `orion:collapse` -> `episodic` mapping (left in place, harmless, matching the review's suggestion for consistency with how `orion:chat:social`'s equivalent mapping was documented).
- `services/orion-graph-compression/README.md`: Compression Scopes table (8 -> 7 graphs) and a new FalkorDB federators paragraph explaining the collapse removal with the corrected root-cause chain and an accurate "dark by default" disclosure for the collapse-triage Falkor writer (review finding -- see below).
- `services/orion-graph-compression/tests/test_federator_episodic.py`: added `test_episodic_federator_no_longer_queries_collapse`.

Deliberately not touched (confirmed by review against live repo state):
- `orion-rdf-writer`'s `collapse.mirror.entry` handler and its observer/dense gate -- untouched; this PR only stops reading the (structurally guaranteed empty) result.
- The channel/kind mismatch itself (`orion-rdf-writer` not subscribed to `orion:collapse:triage`) -- a real bug, out of scope for a series about retiring Fuseki reads, not reviving Fuseki writes. Flagged for its own follow-up if anyone ever wants this write path alive.
- `orion-meta-tags`' collapse-triage Falkor writer -- pre-existing, unrelated to this session, untouched.

## Schema / bus / API changes

None.

## Env/config changes

None (comment-only `.env_example` edit).

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-graph-compression venv/bin/python3 -m pytest services/orion-graph-compression/tests -q
-> 47 passed (46 + 1 new regression test)

git diff --check -> clean
```

## Evals run

No eval harness exists for this service; the existing dynamic-count UNION test plus the new membership regression test cover the changed behavior.

## Docker/build/smoke checks

No container rebuild/restart performed. Live evidence (Fuseki triple count, channel/kind tracing against `channels.yaml` and the actual producer/subscriber code) gathered directly against the running system in this same session.

## Review findings fixed

- Finding (HIGH, causal-narrative accuracy): the diff's first-pass comments and README paragraph attributed the empty `orion:collapse` graph solely to `_build_raw_collapse_graph`'s Juniper/dense observer gate. Review traced deeper and found that gate is unreachable: `orion-rdf-writer` isn't even subscribed to the channel (`orion:collapse:triage`) that ever carries a `collapse.mirror.entry`-kind envelope -- it only listens on `orion:collapse:intake`, which carries a different kind entirely. The conclusion (safe to remove) survives, and is if anything more solidly justified -- but the causal claim was wrong and needed correcting before being committed as established fact for future readers.
  - Fix: rewrote both the code comment and the README paragraph to state the verified channel/kind mismatch as the root cause, with the observer/dense gate noted as real-but-unreachable rather than the primary explanation. Independently re-verified the channel/kind mismatch myself against `channels.yaml` and the actual producer code before committing the correction.
  - Evidence: commit, this same PR.
- Finding (LOW): the README's Falkor-coverage claim for collapse tag/entity content didn't disclose that `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` ships dark by default in `orion-meta-tags`, unlike the adjacent `chat:social` paragraph's already-careful "live-verified thin" disclosure.
  - Fix: added the same disclosure standard to the collapse paragraph.
- Finding (informational, hygiene): `stale_listener.py`'s now-fully-inert `orion:collapse` mapping had no comment explaining why it's safe to leave (unlike `self_study`'s fully-removed mappings, which got one).
  - Fix: added a comment.
- Informational (no fix needed, verified): the `EPISODIC_GRAPHS` removal itself, the UNION-count construction, and the new regression test were all confirmed correct and non-vacuous by hand-tracing `_build_sparql()` against the new 7-item list.

## Restart required

No restart required to merge. If `orion-athena-graph-compression` is redeployed, the `episodic` scope simply queries one fewer graph.

## Risks / concerns

- Severity: Low.
- Concern: none blocking. The real channel/kind mismatch bug in `orion-rdf-writer` (flagged, not fixed) means nobody should assume fixing the observer/dense gate alone would ever make this write path work -- worth a follow-up ticket if collapse-mirror raw content in Fuseki is ever wanted again (unlikely, given the whole point of this migration).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1302
