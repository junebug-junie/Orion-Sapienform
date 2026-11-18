"""PR field notes, Patch 2: the poll that makes a note happen after every merge.

Spec: docs/superpowers/specs/2026-11-04-pr-field-notes-design.md ("Trigger: durable-runs polls
GitHub's public API"). Every PR_FIELD_NOTES_POLL_SEC (900 s):

1. One unauthenticated GitHub search call (more pages only past 100 results) for PRs merged since
   ``max(now - lookback, not_before)``. GitHub is the source of truth for "merged"; the 72 h lookback
   repairs a tick missed during a restart. No token: the repo is public and durable-runs has no gh,
   git or clone (and must not get Juniper's write-scoped token).
2. A PR that already has a ledger record (a note or a code record) is done.
3. A PR with no run yet: one ``/pulls/<n>/files`` call classifies it. A dependency-only or tiny
   docs-only PR gets its code record written here, at once, with no model turn (Juniper's decision 2;
   two more calls fetch its merge sha and commit authors). Any other PR gets one ``pr.field_note`` run.
4. A PR whose runs all FAILED gets one more run after RETRY_AFTER_SEC (MAX_RUNS_PER_PR in all), inside
   the lookback: a note past its 24 h window is overdue, not dropped (spec "Body economy").

Bounded: at most MAX_IN_FLIGHT note runs at once (the spec's "one at a time"), at most
MAX_DETAIL_CALLS GitHub detail calls per tick (60/h is the unauthenticated core limit; 4 ticks/h x 10
stays under it), and a rate-limit answer ends the tick (the lookback covers it). Idempotent:
deterministic run ids (``pr-field-note-<n>``, ``-a<k>``) and the ledger's one-live-record rule.

Not a reader of notes: it learns only which PR numbers have a record (app/pr_field_note_ledger.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from app.pr_field_note_ledger import is_code_record_pr
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.pr_field_note_run import (
    PR_FIELD_NOTE_RUN_PREFIX, PR_FIELD_NOTE_WORKFLOW, PrFieldNoteBriefV1, base_run_id, pr_number_of_run,
)
from orion.schemas.resource_admission import ResourceRequirementV1

logger = logging.getLogger("orion-durable-runs.pr_field_note_poll")

GITHUB_API = "https://api.github.com"
HTTP_TIMEOUT_SEC = 20.0
SEARCH_PER_PAGE = 100
SEARCH_MAX_PAGES = 3          # 300 merges in a lookback window; the busiest 72 h on record is ~120
MAX_DETAIL_CALLS = 10         # per tick: /files, /pulls/<n>, /commits
MAX_IN_FLIGHT = 1             # spec: note runs go one at a time
MAX_RUNS_PER_PR = 2
RETRY_AFTER_SEC = 3600.0
# Under DURABLE_RUNS_MAX_AGE_HOURS (24 h). A run that cannot get the agent lane in this long fails, and
# the next run of the PR (RETRY_AFTER_SEC later) carries it on: overdue, never dropped inside the lookback.
DEADLINE_HOURS = 20.0
LANE = "agent"

RUNS_SQL = """
SELECT run_id, terminal, created_at, updated_at FROM durable_admission_runs
WHERE run_id LIKE %s AND created_at >= %s
"""
LAST_RUN_SQL = "SELECT max(created_at) AS last FROM durable_admission_runs WHERE run_id LIKE %s"
# No time filter: a run submitted before this poll's window (a manual since= poll) still counts.
IN_FLIGHT_SQL = "SELECT count(*) AS n FROM durable_admission_runs WHERE run_id LIKE %s AND terminal IS NULL"


class GitHubUnavailable(RuntimeError):
    """GitHub did not answer usefully (down, rate-limited, bad JSON). On the search or a rate limit the
    tick ends (try next tick); on one PR's detail call only that PR waits for the next tick."""

    def __init__(self, message: str, *, tick_wide: bool = True):
        super().__init__(message)
        self.tick_wide = tick_wide   # False: this one request's answer (a 404/422, bad JSON), not GitHub


class GitHubPublic:
    """The few public, unauthenticated GitHub REST calls the poll needs. Blocking (urllib): call
    through ``asyncio.to_thread``."""

    def __init__(self, repo: str, *, opener: Callable[..., Any] | None = None, timeout: float = HTTP_TIMEOUT_SEC):
        self.repo = repo
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout
        self.calls = 0

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(f"{GITHUB_API}{path}", headers={
            "Accept": "application/vnd.github+json", "User-Agent": "orion-durable-runs-pr-field-notes",
            "X-GitHub-Api-Version": "2022-11-28"})
        self.calls += 1
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode() or "null")
        except urllib.error.HTTPError as exc:
            remaining = exc.headers.get("X-RateLimit-Remaining") if exc.headers else None
            limited = exc.code in (403, 429) and remaining in ("0", None)
            kind = "rate_limited" if limited else f"http_{exc.code}"
            raise GitHubUnavailable(f"{kind}: GET {path}", tick_wide=limited or exc.code >= 500) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise GitHubUnavailable(f"{type(exc).__name__}: GET {path}: {exc}"[:300]) from exc
        except ValueError as exc:
            raise GitHubUnavailable(f"bad JSON: GET {path}: {exc}"[:300], tick_wide=False) from exc

    def merged_since(self, since: datetime) -> list[dict[str, Any]]:
        """``[{number, title, body, merged_at}]`` for PRs merged at or after ``since``."""
        q = f"repo:{self.repo} is:pr is:merged merged:>={since.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        out: list[dict[str, Any]] = []
        for page in range(1, SEARCH_MAX_PAGES + 1):
            data = self._get(f"/search/issues?q={urllib.parse.quote(q)}&per_page={SEARCH_PER_PAGE}&page={page}")
            if not isinstance(data, dict):
                raise GitHubUnavailable("search answered no object")
            items = data.get("items") or []
            for it in items:
                merged = (it.get("pull_request") or {}).get("merged_at")
                if merged:
                    out.append({"number": int(it["number"]), "title": it.get("title") or "",
                                "body": it.get("body") or "", "merged_at": merged})
            if len(items) < SEARCH_PER_PAGE or page * SEARCH_PER_PAGE >= int(data.get("total_count") or 0):
                break
        return out

    def files(self, number: int) -> list[dict[str, Any]]:
        data = self._get(f"/repos/{self.repo}/pulls/{int(number)}/files?per_page=100")
        if not isinstance(data, list):
            raise GitHubUnavailable(f"files of #{number}: no list", tick_wide=False)
        return data

    def pull(self, number: int) -> dict[str, Any]:
        data = self._get(f"/repos/{self.repo}/pulls/{int(number)}")
        if not isinstance(data, dict):
            raise GitHubUnavailable(f"pull #{number}: no object", tick_wide=False)
        return data

    def commits(self, number: int) -> list[dict[str, Any]]:
        data = self._get(f"/repos/{self.repo}/pulls/{int(number)}/commits?per_page=100")
        if not isinstance(data, list):
            raise GitHubUnavailable(f"commits of #{number}: no list", tick_wide=False)
        return data


FILES_PAGE = 100


def gh_shape(item: dict[str, Any], files: list[dict[str, Any]], pull: dict[str, Any] | None = None,
             commits: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The REST answers in the shape ``gh pr view --json`` gives the dig CLI (orion/body_atlas/
    field_notes.flags_for reads it), so the poller and ``orion.dig field-note`` classify a PR alike.
    A file list of exactly 100 may be cut (GitHub's page): changedFiles says so, and such a PR is never
    tiny (tiny docs is at most 40 changed lines)."""
    fl = [{"path": f.get("filename") or "", "additions": int(f.get("additions") or 0),
           "deletions": int(f.get("deletions") or 0)} for f in files]
    # A full page may be cut: never treated as a code-record PR (see is_code_record_pr's caller).
    pull = pull or {}
    out = {
        "number": int(item["number"]), "title": pull.get("title") or item.get("title") or "",
        "body": pull.get("body") if pull.get("body") is not None else item.get("body") or "",
        "mergedAt": pull.get("merged_at") or item.get("merged_at"),
        "files": fl,
        "additions": int(pull["additions"]) if "additions" in pull else sum(f["additions"] for f in fl),
        "deletions": int(pull["deletions"]) if "deletions" in pull else sum(f["deletions"] for f in fl),
        "changedFiles": int(pull["changed_files"]) if "changed_files" in pull
        else (len(fl) + 1 if len(fl) >= FILES_PAGE else len(fl)),
        "headRefName": (pull.get("head") or {}).get("ref") or "",
        "author": {"login": (pull.get("user") or item.get("user") or {}).get("login") or ""},
        "mergeCommit": {"oid": pull.get("merge_commit_sha") or ""},
        "commits": [{"authors": [{"email": ((c.get("commit") or {}).get("author") or {}).get("email") or ""}]}
                    for c in (commits or [])],
    }
    return out


def next_run_id(pr_number: int, attempts: list[dict[str, Any]], *, now: datetime,
                max_runs: int = MAX_RUNS_PER_PR, retry_after_sec: float = RETRY_AFTER_SEC) -> str | None:
    """The run to submit for a PR with no ledger record, or None. Same rule as the episode-distill
    reconciler: none yet -> the base id; one in flight or completed, or max runs spent -> nothing; all
    failed and the newest older than ``retry_after_sec`` -> ``<base>-a<k>``."""
    base = base_run_id(pr_number)
    if not attempts:
        return base
    if any(a.get("terminal") is None for a in attempts):
        return None
    if any(a.get("terminal") == "completed" for a in attempts):
        return None
    if len(attempts) >= max_runs:
        return None
    latest = max(a["updated_at"] for a in attempts)
    if now - latest < timedelta(seconds=retry_after_sec):
        return None
    taken = {a["run_id"] for a in attempts}
    k = len(attempts) + 1
    while f"{base}-a{k}" in taken:
        k += 1
    return f"{base}-a{k}"


def run_request(item: dict[str, Any], run_id: str, *, now: datetime, timeout_sec: float) -> DurableRunRequestV1:
    merged = datetime.fromisoformat(str(item["merged_at"]).replace("Z", "+00:00"))
    return DurableRunRequestV1(
        run_id=run_id,
        workflow=PR_FIELD_NOTE_WORKFLOW,
        # Deterministic: a re-submit of the same run id is the same request (the store's dedup).
        correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:pr_field_note:{run_id}")),
        requested_at=now,
        brief=PrFieldNoteBriefV1(field_note="pr.field_note", pr_number=int(item["number"]), merged_at=merged,
                                 title=str(item.get("title") or "")[:300], timeout_sec=float(timeout_sec)),
        admission=ResourceRequirementV1(resource=f"llm.route.{LANE}", preferred_lane=LANE, priority="background",
                                        deadline_at=now + timedelta(hours=DEADLINE_HOURS)),
    )


@dataclass
class PollResult:
    at: str
    since: str
    merged_seen: int = 0
    recorded: int = 0
    submitted: list[str] = field(default_factory=list)
    code_records: list[int] = field(default_factory=list)
    waiting_turn: int = 0          # classified, a run is due, but MAX_IN_FLIGHT is in use
    skipped_budget: int = 0        # not classified this tick (detail-call budget spent)
    github_calls: int = 0
    gap_warning: str | None = None
    error: str | None = None
    pr_errors: dict[int, str] = field(default_factory=dict)   # one PR's trouble never blocks the rest

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class FieldNotePoller:
    def __init__(self, *, settings: Any, pool: Any, submit: Callable[[DurableRunRequestV1], Awaitable[Any]],
                 ledger: Any, github: GitHubPublic | None = None, now: Callable[[], datetime] | None = None):
        self.settings = settings
        self.pool = pool
        self.submit = submit
        self.ledger = ledger
        self.github = github or GitHubPublic(settings.field_note_github_repo)
        self.now = now or (lambda: datetime.now(timezone.utc))
        # PR -> (is a code-record PR, its /files answer): one classify call per PR, kept only while
        # the PR is in the window.
        self.classified: dict[int, tuple[bool, list[dict[str, Any]]]] = {}
        self.last: PollResult | None = None
        self._lock = asyncio.Lock()

    def floor(self, now: datetime) -> datetime:
        lookback = now - timedelta(hours=float(self.settings.field_note_lookback_hours))
        return max(lookback, self.settings.field_note_not_before)   # settings make it UTC-aware

    async def _runs(self, since: datetime) -> dict[int, list[dict[str, Any]]]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(RUNS_SQL, (PR_FIELD_NOTE_RUN_PREFIX + "%", since - timedelta(hours=1)))).fetchall()
        out: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            n = pr_number_of_run(row["run_id"])
            if n is not None:
                out.setdefault(n, []).append(dict(row))
        return out

    async def _gap_warning(self, now: datetime, recorded: dict[int, Any]) -> str | None:
        """Spec: an outage longer than the lookback loses merges; say so (a manual since= poll fixes it).

        Only a REAL gap: when nothing has happened for longer than the lookback (no run, no record), one
        more search looks at the stretch the lookback no longer covers, and a warning names the merges in
        it that have neither a record nor a run. A quiet stretch with no merges warns about nothing."""
        lookback = timedelta(hours=float(self.settings.field_note_lookback_hours))
        edge = now - lookback
        not_before = self.settings.field_note_not_before
        if not_before >= edge:
            return None   # the lookback still reaches back to not_before: nothing can fall out of it
        async with self.pool.connection() as conn:
            row = await (await conn.execute(LAST_RUN_SQL, (PR_FIELD_NOTE_RUN_PREFIX + "%",))).fetchone()
        last_run = row["last"] if row else None
        last_record = await asyncio.to_thread(self.ledger.last_written_at)
        stamps = [t for t in (last_run, datetime.fromisoformat(last_record) if last_record else None) if t]
        last = max(stamps) if stamps else not_before
        if now - last <= lookback:
            return None
        start = max(last, not_before)
        older = [m for m in await asyncio.to_thread(self.github.merged_since, start)
                 if datetime.fromisoformat(m["merged_at"].replace("Z", "+00:00")) < edge]
        runs = await self._runs(start)
        missed = sorted(m["number"] for m in older if m["number"] not in recorded and m["number"] not in runs)
        if not missed:
            return None
        return (f"{len(missed)} merged PR(s) fell out of the {lookback} lookback with no record or run: "
                + " ".join(f"#{n}" for n in missed[:20])
                + f"; POST /pr-field-notes/poll?since={start.isoformat()}")

    async def poll_once(self, *, since: datetime | None = None) -> PollResult:
        async with self._lock:
            result = await self._poll(since=since)
            self.last = result
            return result

    async def _poll(self, *, since: datetime | None) -> PollResult:
        now = self.now()
        floor = since or self.floor(now)
        result = PollResult(at=now.isoformat(), since=floor.isoformat())
        calls0 = self.github.calls
        try:
            merged = await asyncio.to_thread(self.github.merged_since, floor)
            merged = sorted((m for m in merged
                             if datetime.fromisoformat(m["merged_at"].replace("Z", "+00:00")) >= floor),
                            key=lambda m: m["merged_at"])
            window = {m["number"] for m in merged}
            self.classified = {n: v for n, v in self.classified.items() if n in window}   # bounded
            result.merged_seen = len(merged)
            recorded = await asyncio.to_thread(self.ledger.live_records)
            if since is None:
                result.gap_warning = await self._gap_warning(now, recorded)
                if result.gap_warning:
                    logger.warning("field_note_poll_gap_possible %s", result.gap_warning)
            runs = await self._runs(floor)
            async with self.pool.connection() as conn:
                row = await (await conn.execute(IN_FLIGHT_SQL, (PR_FIELD_NOTE_RUN_PREFIX + "%",))).fetchone()
            state = {"in_flight": int(row["n"]) if row else 0, "budget": MAX_DETAIL_CALLS}
            for item in merged:
                n = item["number"]
                if n in recorded:
                    result.recorded += 1
                    continue
                try:
                    await self._one(item, runs.get(n, []), now=now, state=state, result=result)
                except GitHubUnavailable as exc:
                    if exc.tick_wide:
                        raise
                    result.pr_errors[n] = f"github: {exc}"[:300]
                    logger.warning("field_note_poll_pr_failed pr=%s err=%s", n, exc)
                except OSError:
                    raise   # the ledger itself: every PR would fail the same way
                except Exception as exc:  # noqa: BLE001 -- one bad PR (a refused code record, bad data) waits alone
                    self.classified.pop(n, None)
                    result.pr_errors[n] = f"{type(exc).__name__}: {exc}"[:300]
                    logger.warning("field_note_poll_pr_failed pr=%s err=%s", n, exc, exc_info=True)
        except GitHubUnavailable as exc:
            result.error = f"github: {exc}"
            logger.warning("field_note_poll_github_unavailable %s", exc)
        except OSError as exc:
            result.error = f"ledger: {exc}"
            logger.warning("field_note_poll_ledger_unreadable %s", exc)
        result.github_calls = self.github.calls - calls0
        logger.info("field_note_poll_tick %s", json.dumps({k: v for k, v in result.as_dict().items() if v}))
        return result

    async def _one(self, item: dict[str, Any], attempts: list[dict[str, Any]], *, now: datetime,
                   state: dict[str, int], result: PollResult) -> None:
        """One merged PR with no record: classify once, then its code record or its run."""
        n = item["number"]
        run_id = next_run_id(n, attempts, now=now)
        if run_id is None:
            return
        if not attempts:
            if n not in self.classified:
                if state["budget"] < 1:
                    result.skipped_budget += 1
                    return
                state["budget"] -= 1
                files = await asyncio.to_thread(self.github.files, n)
                # A full first page may hide more files: such a PR always gets a note, never a code record.
                code = len(files) < FILES_PAGE and is_code_record_pr(gh_shape(item, files))
                self.classified[n] = (code, files)
            code_record, files = self.classified[n]
            if code_record:
                if state["budget"] < 2:
                    result.skipped_budget += 1
                    return
                state["budget"] -= 2
                pull = await asyncio.to_thread(self.github.pull, n)
                commits = await asyncio.to_thread(self.github.commits, n)
                note_id = await asyncio.to_thread(self.ledger.write_code_record, pr=gh_shape(item, files, pull, commits))
                self.classified.pop(n, None)
                if note_id:
                    result.code_records.append(n)
                    logger.info("field_note_poll_code_record pr=%s note_id=%s", n, note_id)
                return
        if state["in_flight"] >= MAX_IN_FLIGHT:
            result.waiting_turn += 1
            return
        request = run_request(item, run_id, now=now, timeout_sec=self.settings.field_note_turn_timeout_sec)
        try:
            await self.submit(request)
        except ValueError as exc:   # duplicate run id (a concurrent poll): nothing to do
            logger.info("field_note_poll_duplicate run=%s err=%s", run_id, exc)
            return
        state["in_flight"] += 1
        result.submitted.append(run_id)
        logger.info("field_note_poll_submitted run=%s pr=%s merged_at=%s", run_id, n, item["merged_at"])

    async def run_forever(self, stop: asyncio.Event) -> None:
        interval = max(60.0, float(self.settings.field_note_poll_sec))
        while not stop.is_set():
            try:
                await self.poll_once()
            except Exception:  # noqa: BLE001 -- e.g. DB away: log, try next tick (the lookback covers it)
                logger.exception("field_note_poll_tick_failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def health(self) -> dict[str, Any] | None:
        return self.last.as_dict() if self.last else None
