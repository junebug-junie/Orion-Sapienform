#!/usr/bin/env python3
"""Deterministic qualifying-change check + one-shot bus publish for
self-study semantic enrichment, invoked from `scripts/git_hooks/post-commit`.

Design notes (see CLAUDE.md sec 4 -- deterministic vs latent split):

- Reuses `orion.structural_mass.git_delta.git_churn_delta` (real, tested,
  replay-verified) for the numeric commit/file/line delta rather than
  reinventing it.
- Path-qualification (does this commit touch anything self-study cares
  about?) is *not* in git_delta.py -- that module only computes aggregate
  counts, not the changed path list -- so this script owns a small,
  independent `git diff --name-only` call and a static pattern list mirrored
  from `services/orion-cortex-exec/app/self_study.py::_SCAN_ROOTS` (services/,
  orion/ -- specifically the subset self_study.py actually scans: service
  dirs, orion/bus/channels.yaml, orion/schemas/, orion/cognition/verbs/).
- State (last-enriched SHA) is a plain JSON file under
  `SELF_STUDY_ENRICHMENT_STATE_PATH` (default `.orion/self_study_enrichment_state.json`
  at repo root) -- deliberately not a bus-consumed cursor: this script runs
  from a git hook (host/shell context), not a long-lived service, so it owns
  its own tiny durable pointer the way `orion-cocreation-signals`' pr_lifecycle
  producer owns its own cold-start lookback.
- Publish is a single one-shot `redis.Redis(...).publish(...)` call -- no
  async client, no bus enforcement/velocity tracking dependency, matching
  the "small python3 -c one-shot publish, mirroring how agent_board.py's
  heartbeat is invoked from the same hook family" instruction. No existing
  lightweight one-shot-bus-publish-from-a-shell-context helper was found in
  this repo (searched scripts/ and orion/core/bus/) -- this is deliberately
  new and deliberately minimal rather than pulling in OrionBusAsync's full
  async/event-loop machinery for a fire-and-forget host-side publish.
- Never raises, never blocks the commit: every failure path is caught and
  logged to stderr, and the script always exits 0. The hook fragment that
  calls this also wraps it in `( ... ) || true` per CLAUDE.md's git-hook
  conventions (graphify's own post-commit hook `exit 0`s for linked
  worktrees, so a plain `exit` here must not kill later hook fragments).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Self-study-relevant path patterns, mirrored from self_study.py's real scan
# surface (services/, orion/bus/channels.yaml, orion/schemas/,
# orion/cognition/verbs/) -- see that module's _SCAN_ROOTS and the
# _service_items/_channel_items/_schema_items/_yaml_verb_items functions.
_QUALIFYING_PREFIXES: tuple[str, ...] = (
    "services/",
    "orion/bus/channels.yaml",
    "orion/schemas/",
    "orion/cognition/verbs/",
)


def is_qualifying_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in _QUALIFYING_PREFIXES)


def qualifying_paths(paths: list[str]) -> tuple[str, ...]:
    return tuple(p for p in paths if is_qualifying_path(p))


def _run_git(args: list[str], repo_path: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout


def changed_paths(prev_sha: str, head_sha: str, repo_path: Path) -> list[str]:
    if prev_sha == head_sha:
        return []
    out = _run_git(["diff", "--name-only", f"{prev_sha}..{head_sha}"], repo_path)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _state_path(repo_root: Path) -> Path:
    override = os.environ.get("SELF_STUDY_ENRICHMENT_STATE_PATH")
    if override:
        return Path(override)
    return repo_root / ".orion" / "self_study_enrichment_state.json"


def read_last_sha(repo_root: Path) -> str | None:
    path = _state_path(repo_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sha = data.get("last_enriched_sha")
        return sha if isinstance(sha, str) and sha else None
    except Exception:
        return None


def write_last_sha(repo_root: Path, sha: str) -> None:
    path = _state_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_enriched_sha": sha}), encoding="utf-8")


def _rate_limit_ok(repo_root: Path) -> bool:
    """Cheap daily-ceiling check shared with the consumer service's own
    enforcement -- this is a *belt* check to avoid publishing events nobody
    will ever process once the service-side ceiling is already known to be
    hit today; the *suspenders* (authoritative) enforcement lives in the
    service itself (app/rate_limit.py), since only it knows how many runs it
    actually executed. This hook-side check only prevents publishing more
    requests than the configured daily max onto the channel; it is not
    itself authoritative and fails open (unlimited) if the counter file is
    unreadable."""
    max_per_day = int(os.environ.get("SELF_STUDY_ENRICHMENT_MAX_PER_DAY", "8") or "8")
    if max_per_day <= 0:
        return False
    counter_path = repo_root / ".orion" / "self_study_enrichment_publish_count.json"
    today = datetime.now(timezone.utc).date().isoformat()
    count_today = 0
    if counter_path.exists():
        try:
            data = json.loads(counter_path.read_text(encoding="utf-8"))
            if data.get("date") == today:
                count_today = int(data.get("count", 0))
        except Exception:
            pass
    if count_today >= max_per_day:
        return False
    try:
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        counter_path.write_text(json.dumps({"date": today, "count": count_today + 1}), encoding="utf-8")
    except Exception:
        pass
    return True


def _read_bus_url_from_env_file(repo_root: Path) -> str:
    """Fallback ORION_BUS_URL lookup for when this hook runs from a shell
    context that never exported it -- confirmed live 2026-08-12: a normal
    git-commit shell (interactive or a coding agent's Bash tool) has NO
    ORION_BUS_URL in its process environment on this host, because the
    canonical value lives only in `.env` at repo root, which is a
    docker-compose `--env-file` input, not something shells source. Without
    this fallback the hook always hit "ORION_BUS_URL unset, skipping
    publish" -- reached correctly, not a crash, but a silent no-op on every
    real commit regardless of whether the qualifying-path/rate-limit/import
    logic upstream was working (it was, once fixed).

    Deterministic KEY=VALUE parse of `.env` -- no new dependency (CLAUDE.md
    sec 10: vanilla libraries first for a trivial parse like this). Returns
    "" (not raising) on a missing file, unreadable file, or missing key, so
    callers keep their existing "unset -> skip" behavior unchanged when
    `.env` genuinely has no value either."""
    env_path = repo_root / ".env"
    if not env_path.exists():
        return ""
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "ORION_BUS_URL":
            return value.strip().strip('"').strip("'")
    return ""


def publish_enrichment_request(payload: dict, *, bus_url: str, channel: str) -> None:
    import redis  # local import: only needed on the publish path

    client = redis.Redis.from_url(bus_url, socket_timeout=3.0, socket_connect_timeout=3.0)
    try:
        client.publish(channel, json.dumps(payload).encode("utf-8"))
    finally:
        try:
            client.close()
        except Exception:
            pass


def main() -> int:
    repo_root_str = _run_git(["rev-parse", "--show-toplevel"], Path.cwd()).strip()
    repo_root = Path(repo_root_str)

    head_sha = _run_git(["rev-parse", "HEAD"], repo_root).strip()
    prev_sha = read_last_sha(repo_root)
    if prev_sha is None:
        # Cold start: diff against the immediate parent only, so the very
        # first hook fire after install doesn't try to diff the whole repo
        # history as "one giant qualifying commit".
        try:
            prev_sha = _run_git(["rev-parse", f"{head_sha}~1"], repo_root).strip()
        except RuntimeError:
            prev_sha = head_sha  # first commit in the repo; no parent

    paths = changed_paths(prev_sha, head_sha, repo_root)
    qualifying = qualifying_paths(paths)
    if not qualifying:
        return 0

    # This script is invoked by the git hook as `python3 <this file's path>`
    # (see scripts/git_hooks/post-commit), not `python3 -c ...` or `-m` --
    # under that invocation form Python puts the SCRIPT's own directory
    # (scripts/) on sys.path[0], not the repo root and not the caller's cwd
    # (even though the hook does `cd "$REPO_ROOT"` first). `orion` lives at
    # the repo root, so the import below silently raised ModuleNotFoundError
    # on every real invocation, caught by __main__'s outer `except Exception`
    # and printed as a "non-fatal error" -- functionally invisible, since the
    # hook already always exits 0. Confirmed live 2026-08-12: zero enrichment
    # requests were ever published despite the hook fragment being installed
    # and the qualifying-path/rate-limit logic above all working correctly.
    # No existing test caught this because none exercised main() as a real
    # subprocess the way the git hook does (see test_main_real_subprocess_
    # invocation_imports_correctly in the test file). repo_root (computed
    # above via `git rev-parse --show-toplevel`) is the correct fix target
    # regardless of where this script physically lives -- worktree, primary
    # checkout, or a future relocation.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Compute the delta BEFORE consuming a rate-limit slot: git_churn_delta
    # can raise (corrupted/shallow clone, prev_sha no longer reachable after
    # a rebase/force-push, etc.), and _rate_limit_ok() has an observable
    # side effect (it persists the incremented daily counter to disk). If
    # the ceiling check ran first and this raised, main()'s outer
    # try/except would swallow it and exit 0 having already spent a slot on
    # nothing -- repeated failures would silently exhaust the day's publish
    # ceiling with zero actual events ever published. Doing the
    # (fallible, no-side-effect) delta computation first means a slot is
    # only ever consumed once we know there is a real payload to publish.
    from orion.structural_mass.git_delta import git_churn_delta

    delta = git_churn_delta(prev_sha, head_sha, repo_path=repo_root)

    if not _rate_limit_ok(repo_root):
        print("self_study_enrichment_hook: daily publish ceiling reached, skipping", file=sys.stderr)
        return 0

    payload = {
        "schema": "SelfStudyEnrichmentRequestV1",
        "kind": "self_study.enrichment.request.v1",
        "repo_root": str(repo_root),
        "prev_sha": prev_sha,
        "head_sha": head_sha,
        "commit_count": delta.commit_count,
        "files_changed": delta.files_changed,
        "lines_changed": delta.lines_changed,
        "touched_paths": list(qualifying),
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }

    bus_url = os.environ.get("ORION_BUS_URL", "") or _read_bus_url_from_env_file(repo_root)
    channel = os.environ.get(
        "CHANNEL_SELF_STUDY_ENRICHMENT_REQUESTED", "orion:self_study:enrichment:requested"
    )
    if not bus_url:
        print("self_study_enrichment_hook: ORION_BUS_URL unset, skipping publish", file=sys.stderr)
        return 0

    publish_enrichment_request(payload, bus_url=bus_url, channel=channel)
    write_last_sha(repo_root, head_sha)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # never block the commit
        print(f"self_study_enrichment_hook: non-fatal error: {exc}", file=sys.stderr)
        sys.exit(0)
