## Summary

- Town NPCs now see the chat history with speaker roles. Their own lines are `assistant` turns and the partner's lines are `user` turns, as plain text. Before, every line was a `user` turn with a `Name to Name:` prefix.
- A new partner-echo check catches a reply that copies the partner's last line. It uses the same retry-then-leave path as the self-repeat check.
- `orion-npc-no-self-repeat.patch` is rebased with the same behavior. On `main`, a fresh apply of the patch chain stopped at that patch.
- New CI job `aitown-patch-chain` applies the whole chain to the pinned upstream. It also runs the echo check on the real transcript.
- This is PR 1 of the living-world design (#2666, §4b first bullet).

## Outcome moved

- Failure mode: two NPCs copied each other's last line word for word for five turns (live 2026-09-17, Sofia↔Mara, `c:578946`). The shipped check flags exactly those 5 turns and none of the other 4.
- Failure mode: a fresh build of the town stopped at `orion-npc-no-self-repeat.patch`, and no test failed. Now the full chain applies, and CI checks it.

## Current architecture

AI Town is the a16z upstream at a pinned commit. Patch files in `services/orion-ai-town/patches/` customize it, applied in order by `scripts/apply_upstream_patches.sh`. `convex/agent/conversation.ts` builds each turn's LLM messages. The self-repeat gate (#2230) compared a reply only with the speaker's own last 4 lines.

## Architecture touched

AI Town patches only (`convex/agent/conversation.ts`, plus `convex/aiTown/agentOperations.ts` through the rebased patch). No bus, schema, env or service changes.

## Files changed

- `services/orion-ai-town/patches/orion-live-turn-roles.patch` (new): speaker roles, `replyCue`, `isPartnerEcho`, and echo added to the retry/leave gate.
- `services/orion-ai-town/patches/orion-npc-no-self-repeat.patch`: rebased onto the continuity/answer-first chain, with the same behavior.
- `services/orion-ai-town/scripts/apply_upstream_patches.sh`: registers the new patch last.
- `services/orion-ai-town/tests/test_live_turn_roles_patch.py` (new): checks the patch text, runs the echo check under Node ≥ 22.6, and applies the full chain to the pinned upstream.
- `services/orion-ai-town/tests/fixtures/aitown_echo_c578946.json` (new): the real transcript, from `social_room_turns`.
- `services/orion-ai-town/tests/test_mechanical_leave_patch.py`: the patch-order check no longer requires mechanical-leave to be last.
- `services/orion-ai-town/tests/test_npc_no_self_repeat_patch.py`: the hunk counter counts an empty line as blank context, as `git apply` does.
- `.github/workflows/aitown-patch-chain.yml` (new): runs the Node and chain tests.
- `services/orion-ai-town/README.md`: section for the new patch.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the LLM message roles in town continue and leave turns. Echo replies now retry, then leave. A goodbye that copies the partner's goodbye also leaves, which is the same outcome.
- Compatibility notes: `townLastOtherPrompt` and `ownPriorLines` now read roles. The continuity ingest prompt field is unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Metric quality gate (partner-echo threshold)

1. Provenance: `isPartnerEcho` in the patch. It is the word-set overlap (Jaccard) of the normalized reply and the partner's last line.
2. Independence: this is a different comparison from the self-repeat check (own lines, exact match). The two only share `normalizeTownLine`.
3. Theory anchor: copying a partner's line is the failure seen in `c:578946`. Word-set overlap measures copying directly.
4. Live data: 9,391 real town turns from `social_room_turns` (platform `aitown`). At ≥ 0.7, 192 turns are flagged (2.0%), and 136 of them are exact copies. Samples from 0.7–0.8 are pronoun-swapped copies. Samples from 0.5–0.7 share a topic or template. Lines under 4 words are exempt; the only short echo in the data was "You in?".
5. Existing mechanism: extends the self-repeat gate and reuses its retry and leave sentinel.
6. Reversibility: one patch file.

## Tests run

```text
pytest services/orion-ai-town/tests (AITOWN_TEST_NODE=node 24, AITOWN_UPSTREAM_GIT=local clone) -> 141 passed, 1 skipped (pre-existing: upstream/world.ts not cloned)
mutation: PARTNER_ECHO_MIN_OVERLAP 0.7 -> 1.5 -> test_partner_echo_flags_real_transcript fails
tsc --noEmit -p convex on the fully patched upstream -> 0 errors (a planted type error fails it)
fresh clone at pinned ref + apply_upstream_patches.sh -> every patch "Applied"
python3 scripts/check_personal_terms.py -> passed
git diff --check -> passed
```

## Evals run

None. The live echo rate check is acceptance check A3 in #2666 (100 scenes + 20 live NPC↔Orion turns). It needs the town running and is not done.

## Docker/build/smoke checks

Not run. The town runs on circe, and this host cannot SSH there. Live dialogue behavior is UNVERIFIED.

## Review findings fixed

- Finding (medium): a plain answer that restates a yes/no question scored as an echo ("Is the shed locked at night?" → "Yes, the shed is locked at night." = 0.86). The NPC could then leave after a good answer. The fixture also never tested the 0.7 line.
  - Fix: after a question, only an exact copy counts. In live data, all 7 echoes of a question were exact copies. Added boundary rows: two real pairs at 0.72 and 0.64, plus two question cases.
  - Evidence: mutations to 0.99, to 0.6, and with the exemption removed each fail `test_partner_echo_flags_real_transcript`.
- Finding (medium): the retry instruction became the last `user` turn, so the model could read it as the partner speaking.
  - Fix: the instruction is appended to the system turn instead. History and cue are unchanged.
  - Evidence: `tsc --noEmit -p convex` passes. A test asserts `...llmMessages.slice(1)`.
- Finding (low): the last fixture row had the wrong partner label.
  - Fix: relabeled it Sofia Bell.
  - Evidence: fixture.
- Finding (low): CI did not trigger on the README, which holds `AITOWN_UPSTREAM_REF`, and a missing tool would skip and still pass.
  - Fix: added the README to the paths. `AITOWN_REQUIRE_TOOLS=1` turns those skips into failures.
  - Evidence: strict run, 5 passed.
- Not fixed (low): `stopWords`/`trimContentPrefx` still match the old `Name to Name:` form. They are now no-ops, harmless, and left in place.

## Restart required

On circe, after the PR merges and `main` is pulled. This resets the tracked upstream files to the pin and removes the one file a patch creates (untracked `.env.local` stays). Tested on a patched scratch copy. It then re-applies the chain, and pushes the Convex functions. No world wipe is needed:

```bash
cd services/orion-ai-town && git -C upstream checkout 7b242334bfbfef02f7718bded120d431e8f307df -- . && rm -f upstream/src/lib/hubEmbed.ts && bash scripts/apply_upstream_patches.sh && (cd upstream && npx convex dev --once)
```

## Risks / concerns

- Severity: medium. Concern: if you re-run `apply_upstream_patches.sh` on an already-patched upstream, it stops at `orion-conversation-proximity.patch`. This is already true on `main`. Mitigation: deploy from a fresh clone. A follow-up should make re-runs work.
- Severity: low. Concern: Orion's own replies come from the embodiment worker, so this check does not catch Orion echoing an NPC. Mitigation: follow-up in embodiment.
- Severity: low. Concern: the chat model may answer differently now that it sees real roles. Mitigation: A3 live check.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
