## Summary

- Reading's market binding (attention patch 3c, PR #3200) now ships **on**: `ORION_PHYSIOLOGY_BID_READING=on` in `.env_example`, `settings.py` and the compose fallback.
- Juniper asked for it live on 2026-11-12 ("bring it all live"). It went live at 01:36Z, before this PR, from the clean primary main checkout. This PR makes the checked-in default match the live value.

## Outcome moved

- Reading stage 1 obeys its market bid. A read with a price or sated deferral is held. It bids again later, and after 2,900 s it asks the pool anyway. It is never dropped.
- The default now matches the live value, so a rebuild or a fresh `.env` keeps reading bound.

## Current architecture

- PR #3200 built the binding and shipped it as `shadow`. In shadow, reading makes only a value bid (patch 3b), and that bid is logged but not obeyed.

## Architecture touched

- orion-durable-runs config only.

## Files changed

- `services/orion-durable-runs/.env_example`: the default is now `on`, and the comment states when it went live.
- `services/orion-durable-runs/app/settings.py`: the field default is now `"on"`.
- `services/orion-durable-runs/docker-compose.yml`: the fallback is now `:-on`.
- `services/orion-durable-runs/tests/test_reading_binding.py`: the default assertion now expects `on`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: reading binds by default.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (the value changed from `shadow` to `on`).
- Local `.env` synced: yes. It was already `on` (set by hand for the 01:36Z deploy), and `sync_local_env_from_example.py` ran.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-durable-runs/tests        863 passed, 132 skipped
scripts/check_env_template_parity.py            PASS
scripts/check_env_key_single_source.py          OK
git diff --check                                clean
```

## Evals run

```text
No new eval. Replay C (services/orion-durable-runs/evals/reading_binding_replay.py) passed for #3200: 773 reads, 0 held.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-durable-runs up -d --build   (primary main, 2026-11-12 01:36Z)
docker exec orion-athena-durable-runs printenv ORION_PHYSIOLOGY_BID_READING  -> on
Container up, serving /physiology requests (200), GPU pool RPCs replying.
First binding decisions expected after world-pulse seeds start (~13:00Z): UNVERIFIED until then.
```

## Review findings fixed

- See the PR conversation. The review ran in a subagent on `git diff origin/main...HEAD`.

## Restart required

```text
No restart required. Live since 2026-11-12 01:36Z.
```

## Risks / concerns

- Severity: low.
  - Concern: the 2,900 s hold end is still Juniper's open decision (#3200 report).
  - Mitigation: replay C held 0 reads. The daily live-week checks in the #3200 report watch held time.

## PR link

(added after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
