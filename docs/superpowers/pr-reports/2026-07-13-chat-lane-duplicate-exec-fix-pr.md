# PR: stop legacy cortex-exec container double-consuming orion:verb:request

Branch: `fix/chat-lane-duplicate-exec` → `main`

## Summary

Direct follow-up to `feat/exec-lane-routing-default-on` (merged earlier today), which fixed duplicate execution for `spark`/`background`-lane verbs. That fix couldn't touch chat-lane traffic by design — `use_direct_exec` explicitly requires `lane != "chat"`. This PR fixes the chat-lane half of the same underlying bug: both `legacy` and `chat` cortex-exec containers were independently subscribed to and fully executing every message on the shared `orion:verb:request` broadcast channel.

## Outcome moved

Chat-lane verbs (`chat_general`, `chat_quick`) now execute exactly once instead of twice. Real duplicate LLM inference cost eliminated on the remaining lane this session's investigation hadn't yet fixed.

## Current architecture

`orion/core/bus/bus_service_chassis.py::Hunter` is documented "Fire-and-forget consumer... Subscribes to patterns" — Redis pub/sub broadcast semantics, not a competing-consumer queue: every subscriber receives and processes every message independently. `services/orion-cortex-exec/app/main.py` (pre-fix, line 922) registered a `Hunter` on `orion:verb:request` for any container whose `_lane in {"chat", "legacy", ""}` — both the `legacy` (base) and `chat` containers matched.

## Investigation (verified, not assumed)

- Confirmed live before the fix, via `request_id` tracing across containers: a real chat_quick request appeared in both `legacy` and `chat` logs with identical `trigger=legacy.plan`, both fully executed.
- **Corrected a wrong assumption in the original task brief**: the brief claimed unset `EXEC_LANE` "already defaults to acting like chat." Verified this is false — `services/orion-cortex-exec/app/settings.py:28` (`exec_lane: str = Field("legacy", alias="EXEC_LANE")`) and `docker-compose.yml`'s `EXEC_LANE: ${EXEC_LANE:-legacy}` for the `legacy` container both default to `"legacy"`. `main.py`'s own `str(settings.exec_lane or "chat")` fallback only fires for a literal empty string, not an unset env var. Unset `EXEC_LANE` therefore resolves to `_lane == "legacy"` and (post-fix) correctly disables the broadcast listener — tested explicitly as its own case, not conflated with the empty-string case.
- Confirmed `legacy`'s *other* job — serving direct RPC traffic via the bare `orion:cortex:exec:request` channel through the separate `Rabbit`-backed `svc`/`handle()` registration (e.g. `orion-thought`'s `stance_react` calls) — is real, distinct, and completely untouched by this fix. The diff has exactly one hunk.

## Architecture touched

`orion-cortex-exec` only. No changes to `orion-cortex-orch` (that fix already landed separately), no env/config/schema/bus contract changes.

## Files changed

- `services/orion-cortex-exec/app/main.py` — one-line fix: `if _lane in {"chat", "legacy", ""}:` → `if _lane in {"chat", ""}:`.
- `services/orion-cortex-exec/tests/test_verb_listener_lane_wiring.py` (new) — 6 parametrized cases (`chat`, explicit `""`, `legacy`, unset, `spark`, `background`), using a guarded fresh-reimport pattern mirroring this directory's existing `_exec_import_guard.py` convention, extended with a targeted `app.main`/`app.settings`-only module purge (not a full `app.*` tree purge) to avoid re-triggering `app.verb_adapters`' `@verb("legacy.plan")` registration into the process-global verb registry across repeated imports in one test file.
- `services/orion-cortex-exec/README.md` — new "Exec lane containers" section; this whole four-container split and the shared broadcast channel had zero prior documentation.

## Schema / bus / API changes

None. No new channel, no schema change. `orion:verb:request`'s consumer set changes (one fewer subscriber), not its contract.

## Env/config changes

None.

## Tests run

```
cd /mnt/scripts/Orion-Sapienform-fix-chat-lane-duplicate-exec
/tmp/orion-test-venv/bin/python -m pytest services/orion-cortex-exec/tests/test_verb_listener_lane_wiring.py -v
→ 6 passed
```
Regression-proof confirmed directly: re-ran the same 6 tests against unfixed `main.py` (`git stash` on just that file) — the exact two `legacy` cases fail (`AssertionError` on `verb_listener is None`), the other 4 pass unchanged. Confirms the test guards precisely the bug this PR fixes, not a tautology.

Whole-directory `services/orion-cortex-exec/tests` sweep has a pre-existing, unrelated collection-order fragility (several files purge the entire `app.*` module tree at collection time, re-triggering `app.verb_adapters`' verb registration and raising `ValueError: Verb already registered: legacy.plan`) — confirmed identical failure count on `git stash` baseline (60 failed / 488 passed / 12 collection errors) vs. this branch. Not introduced or worsened by this change; the new test file passes cleanly in isolation (6/6) and was deliberately written to *not* trigger that fragility (targeted two-module purge instead of a full tree purge).

## Evals run

Not applicable — routing/wiring fix, no cognition/model component.

## Docker/build/smoke checks

Not deployed as part of this PR (code-level fix; `orion-cortex-orch`'s companion fix was applied live separately). Restart section below has the exact deploy + verification commands.

## Review findings fixed

Two parallel review passes run before this report (correctness/cross-file tracing; cleanup/simplification):
- Correctness pass: confirmed no other `exec_lane` reader is affected (`llm_lane.py`'s LLM-routing metadata reads `exec_lane` too, but is orthogonal to `verb_listener` wiring — unaffected). Confirmed test isn't a tautology (fails pre-fix). Confirmed bare-channel `Rabbit`/`handle()` path structurally untouched.
- Cleanup pass: flagged the four-separate-test-functions draft as collapsible — fixed, now one parametrized test with 6 cases and preserved per-case commentary. Flagged the `_exec_import_guard` boilerplate as now copy-pasted a third time — left as-is, matches this directory's established convention and deviating risks reintroducing the exact collection-order fragility documented above.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build cortex-exec
```
(Only the `legacy`/base `cortex-exec` service needs rebuilding — `chat`/`spark`/`background` are unaffected by this specific line, though a full-stack rebuild is harmless if that's the standard deploy flow.)

**Verification** (mirrors the tracing method used to find the bug):
```bash
# After a real chat turn:
docker logs orion-athena-cortex-exec --since 5m | grep -oP "request_id=\S+" | sort -u > /tmp/legacy_ids.txt
docker logs orion-athena-cortex-exec-chat --since 5m | grep -oP "request_id=\S+" | sort -u > /tmp/chat_ids.txt
comm -12 /tmp/legacy_ids.txt /tmp/chat_ids.txt | wc -l   # should be 0 now for chat-lane verbs

# legacy should show zero verb_runtime_intake activity going forward (only its own
# direct-RPC "Incoming Exec Request" lines from orion-thought etc. should remain):
docker logs orion-athena-cortex-exec --since 5m | grep -c verb_runtime_intake   # should be 0
```

## Risks / concerns

- Severity: none identified. `legacy`'s remaining job (bare-channel direct RPC) is structurally separate code (`Rabbit`/`handle()`, not `Hunter`/`handle_verb_request`) and untouched by this diff.
- Severity: informational. The pre-existing whole-suite collection fragility in `services/orion-cortex-exec/tests` (documented above) is a real, separate issue worth its own fix at some point — not introduced by this PR, not fixed by it either; flagged for visibility.

## PR link

Push and open via: `git push -u origin fix/chat-lane-duplicate-exec`, then open the compare URL GitHub prints (no `gh` auth in this environment).
