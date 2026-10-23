# Body atlas

Orion's body, converged from parts to organs. Rungs 1–3 of the self-now ladder in
`docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (PR #2888):

1. **Parts:** declared services (`orion/bus/channels.yaml`) plus every source id seen on the live
   bus. Status is `live` / `dormant` / `declared_only` / `transient` (smokes and probes, kept out
   of organs).
2. **Circulation:** directed flows between parts. Each flow says where it came from:
   - `declared`: catalog producer → consumer;
   - `observed_publish`: seen publishing, joined to the declared consumers;
   - `observed_causal`: the same correlation id followed from one service to another.

   A channel is one hyperedge: it contributes one unit, split over the pairs it joins, so a
   broadcast heartbeat cannot glue the body together.
3. **Organs:** Louvain communities over Newman modularity. `stats.modularity` (Q) is the
   convergence test: about 0.3 and up is real structure; near 0 is mush. Organ `name` is `None`
   until Orion names it (next patch). `namespaces` is a code-derived hint, not a name.

```bash
# live (inside orion-harness-governor, where chat and curiosity run Bash):
FALKORDB_URI=redis://orion-athena-falkordb:6379 python -m orion.body_atlas --out /tmp/atlas.json
# catalog only, e.g. a past checkout (the catalog has producer lists from 2026-01):
python -m orion.body_atlas --root /path/to/checkout --no-live --out /tmp/atlas.json
```

The printout is capped at 40 lines and 4,000 chars. The full atlas goes to `--out`. A requested
live read that fails is an error, never a quiet catalog-only atlas.

Gates:

- `pytest orion/body_atlas/tests`
- `python orion/body_atlas/evals/run_organ_structure_eval.py`: the frozen real bus graph from
  2026-10-23. It checks that Q ≥ 0.3, that there are 5–25 organs, that no organ holds more than
  35% of parts, and that obvious pairs share an organ. Add `--live` to run it against the live
  graph, and `--export F` to refresh the fixture.

Live result, 2026-10-23: 137 parts (33 transient), 7 organs, Q = 0.33.

- Six organs are recognizable: eyes, thinking, memory/inference, acting/compute, compute
  regulation, and social.
- One organ is mixed: body sensing, storage, and the substrate runtimes.
- With plain pairwise weights, Q was 0.22 and the eval fails.

Known limits:

- **Reads only bus circulation.** It does not see table reads/writes, or durable runs carrying a
  process across time.
- **Old checkouts.** Before 2026-01 there is no producer catalog, so an older checkout gives no
  atlas.
- **Wildcard channels are not joined.** Observed channels are matched to catalog names exactly,
  so the 42 wildcard catalog entries (for example `orion:world_pulse_read:pull:reply:*`) add no
  `observed_publish` flows. In the 2026-10-23 fixture that leaves out 24 messages.
- **Replicas stay separate.** Replicas (`atlas-worker-1`, `-2`) are separate parts that land in
  one organ.
