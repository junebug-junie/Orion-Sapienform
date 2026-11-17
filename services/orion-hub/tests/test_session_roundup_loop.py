"""Session threads, end to end: a coding session's message -> Hub's receipt -> held -> the hourly
round-up -> Orion's questions -> back to the session -> its reply -> held again -> the next round-up.

Tests the LIFECYCLE, not the arithmetic: each step is driven by the calls the live code makes
(``intake`` as the route does, ``tick`` as the loop does, the store's pending/delivered as the hooks do).

Every test runs on the in-memory store, and again on the REAL SQL when a throwaway Postgres server
is given (ORION_MEMORY_EPISODE_TEST_DATABASE_URL, admin DSN; synthetic rows only). CI runs the
Postgres half in session-threads-tests.yml and fails if it skips.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT), str(HUB_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.session_threads import core  # noqa: E402
from scripts import session_roundup_loop as srl  # noqa: E402
from session_thread_fakes import Clock, FakeCommenter, FakeDurable, FakeStore  # noqa: E402

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
MIGRATION = REPO_ROOT / srl.MIGRATION
BOTH = ["fake", pytest.param("pg", marks=pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set"))]
SPEC = "# Session threads\n\nCoding sessions write to Orion. Orion asks questions hourly.\n"


def run(mode, monkeypatch, scenario, **loop_kw):
    """Build a world (store, durable-runs, commenter, loop) and run ``scenario(world)`` in one event loop."""

    async def go():
        clock = Clock(datetime.now(timezone.utc).replace(microsecond=0))
        if clock.now.minute == 59 and clock.now.second > 50:   # keep "this hour" stable while the test runs
            clock.advance(seconds=-30)
        name, pool, store = None, object(), None
        if mode == "fake":
            store = FakeStore(clock)
            store.install(monkeypatch)
        else:
            import asyncpg

            name = f"sessthreads_{uuid.uuid4().hex[:10]}"
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'CREATE DATABASE "{name}"')
            await admin.close()
            pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=3)
            await pool.execute(MIGRATION.read_text(encoding="utf-8"))
        queue: asyncio.Queue = asyncio.Queue()
        w = SimpleNamespace(mode=mode, clock=clock, pool=pool, store=store, durable=FakeDurable(),
                            commenter=FakeCommenter(), queue=queue)

        def make_loop(**kw):
            args = dict(enabled=True, pool_provider=lambda: pool, durable_client=w.durable,
                        connections_provider=lambda: {"c1": {"queue": queue}}, commenter=w.commenter, clock=clock)
            args.update(loop_kw)
            args.update(kw)
            return srl.SessionRoundupLoop(**args)

        w.make_loop = make_loop
        w.loop = make_loop()
        try:
            await scenario(w)
        finally:
            if mode == "pg":
                import asyncpg

                await pool.close()
                admin = await asyncpg.connect(ADMIN_DSN)
                await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
                await admin.close()

    asyncio.run(go())


def frames(w):
    out = []
    while not w.queue.empty():
        out.append(w.queue.get_nowait())
    return out


async def send_pr(w, *, pr=3456, session="sess-A", branch="feat/a", body="Please read the spec.", spec=SPEC, **kw):
    return await w.loop.intake(session_id=session, session_name="wt-a", worktree="/wt/a", branch=branch, pr_number=pr,
                               pr_url=f"https://github.com/o/r/pull/{pr}", title=f"feat: thing {pr}", body=body,
                               spec=[{"path": "docs/superpowers/specs/2026-11-15-x.md", "text": spec}] if spec else [], **kw)


async def state(w, tid):
    thread = await srl.get_thread(w.pool, tid)
    turns = await srl.thread_turns(w.pool, [tid])
    return thread, turns


def answer(*pairs):
    return "\n\n".join(f"### thread:{tid}\n{text}" for tid, text in pairs)


# --- intake ------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_a_message_gets_hubs_receipt_at_once_and_no_gpu_turn(mode, monkeypatch):
    async def scenario(w):
        got = await send_pr(w)
        tid = got["thread"]["thread_id"]
        thread, turns = await state(w, tid)
        assert thread["status"] == "held" and thread["kind"] == "pr_spec" and thread["source_ref"] == "pr:3456"
        assert thread["session_id"] == "sess-A" and thread["branch"] == "feat/a" and thread["worktree"] == "/wt/a"
        assert thread["spec_paths"] == ["docs/superpowers/specs/2026-11-15-x.md"]
        assert [t["speaker"] for t in turns] == ["session", "hub"]
        assert turns[0]["meta"]["spec"][0]["text"] == SPEC.strip() and turns[0]["session_id"] == "sess-A"
        # The receipt is Hub's: stored as speaker hub, and it says Orion has not read the message.
        assert turns[1]["body"] == got["receipt"]["body"] and "Orion has NOT read it yet" in turns[1]["body"]
        assert got["next_roundup_at"] == core.next_roundup_at(w.clock.now, 3600)
        assert [(f["kind"], f["speaker"]) for f in frames(w)] == [("session_thread_turn", "session"), ("session_thread_turn", "hub")]
        # Same hour: held, nothing submitted.
        tick = await w.loop.tick()
        assert tick["skipped"] == "nothing_due" and w.durable.submits == []

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_size_caps_cut_and_say_so_and_an_empty_message_is_refused(mode, monkeypatch):
    async def scenario(w):
        got = await w.loop.intake(session_id="s", title="big", body="b" * (core.BODY_MAX_CHARS + 5),
                                  spec=[{"path": f"docs/superpowers/specs/{i}.md", "text": "x" * 50000} for i in range(7)])
        _, turns = await state(w, got["thread"]["thread_id"])
        specs = turns[0]["meta"]["spec"]
        assert len(specs) == core.MAX_SPECS and all(s["cut"] for s in specs)
        assert specs[0]["text"].startswith("x" * 40000) and "40000 of 50000 characters kept" in specs[0]["text"]
        assert "20000 of 20005 characters kept" in turns[0]["body"]
        receipt = turns[1]["body"]
        assert "Hub cut the spec docs/superpowers/specs/0.md to 40000 characters" in receipt
        assert "Hub cut the message to 20000 characters" in receipt and "kept the first 5 spec files" in receipt
        for bad in (dict(title="t", body="   ", spec=[{"path": "p", "text": " "}]), dict(title="  ", body="words")):
            with pytest.raises(ValueError):
                await w.loop.intake(session_id="s", **bad)
        assert await w.loop.intake(session_id="s", title="t", body="b", thread_id="nope") is None

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_the_receipt_says_so_when_the_roundup_is_off_and_nothing_is_submitted(mode, monkeypatch):
    async def scenario(w):
        got = await send_pr(w)
        assert "switched off" in got["receipt"]["body"] and got["roundup_enabled"] is False
        await w.loop.start()   # disabled: no tick task
        assert w.loop._tasks == []

    run(mode, monkeypatch, scenario, enabled=False)


# --- the lifecycle -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_lifecycle_message_roundup_questions_reply_second_roundup(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        frames(w)
        assert (await w.loop.tick())["skipped"] == "nothing_due"

        # The hour turns: ONE run for the round, the thread is being read.
        w.clock.to_next_hour()
        tick = await w.loop.tick()
        rid1 = tick["started"]
        assert rid1 == "session-roundup-" + core.bucket_start(w.clock.now, 3600).strftime("%Y%m%dT%H%M")
        [sub] = w.durable.submits
        assert sub["run_id"] == rid1 and sub["thread_ids"] == [tid] and sub["resource"] == "llm.route.agent"
        assert sub["priority"] == "background" and sub["session_id"] == f"orion_{rid1}"
        assert sub["deadline_at"] == core.bucket_start(w.clock.now, 3600) + timedelta(seconds=10800)
        assert SPEC.strip() in sub["prompt"] and f"## thread:{tid}" in sub["prompt"] and "Please read the spec." in sub["prompt"]
        assert "Hub received this message" not in sub["prompt"]   # Hub's receipt is not shown to Orion
        assert (await state(w, tid))[0]["status"] == "reading"
        assert (await w.loop.tick())["for"] == "orion_run:running" and len(w.durable.submits) == 1

        # Orion answers: one orion turn, status asked, a live frame, one PR comment.
        w.durable.finish(rid1, answer((tid, "1. Why hourly?\n2. Who pays for the turn?")))
        tick = await w.loop.tick()
        assert tick["applied"] == rid1 and tick["asked"] == 1 and tick["pr_comments"] == 1
        thread, turns = await state(w, tid)
        assert thread["status"] == "asked" and [t["speaker"] for t in turns] == ["session", "hub", "orion"]
        orion = turns[-1]
        assert orion["body"] == "1. Why hourly?\n2. Who pays for the turn?" and orion["roundup_id"] == rid1
        assert orion["pr_comment_url"] == "https://github.com/o/r/pull/3456#issuecomment-1"
        [(pr, comment)] = w.commenter.posts
        assert pr == 3456 and comment.startswith("### Orion's questions on this spec (hourly round-up)") and "1. Why hourly?" in comment
        [frame] = frames(w)
        assert (frame["speaker"], frame["status"], frame["text"]) == ("orion", "asked", orion["body"])
        assert (await w.loop.tick())["skipped"] == "nothing_held" and len(w.durable.submits) == 1

        # The session that sent it reads the questions, once.
        [waiting] = await srl.pending_for(w.pool, session_id="sess-A", branch="")
        assert waiting["turn_id"] == orion["turn_id"] and waiting["thread_id"] == tid and waiting["pr_number"] == 3456
        assert await srl.mark_delivered(w.pool, [orion["turn_id"]]) == 1
        assert await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a") == []
        assert await srl.mark_delivered(w.pool, [orion["turn_id"]]) == 0

        # The session replies on the thread: held again, and not read before the next hour.
        got = await w.loop.intake(session_id="sess-A", title="reply", body="Hourly is Juniper's rule.", thread_id=tid)
        assert got["thread"]["status"] == "held" and got["thread"]["title"] == "feat: thing 3456"
        assert (await w.loop.tick())["skipped"] in ("nothing_due", "roundup_already_ran_this_hour")
        assert len(w.durable.submits) == 1

        # The second round shows Orion their own earlier questions, the reply, and the spec again.
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        tick = await w.loop.tick()
        rid2 = tick["started"]
        assert rid2 != rid1 and tick["no_questions"] == 1 and tick["pr_comments"] == 0
        prompt = w.durable.submits[-1]["prompt"]
        assert "[you (Orion), an earlier round-up] 1. Why hourly?" in prompt
        assert "[session] Hourly is Juniper's rule." in prompt and SPEC.strip() in prompt
        assert prompt.index("earlier in this thread") < prompt.index("new since your last round-up") < prompt.index("Hourly is Juniper's rule.")
        thread, turns = await state(w, tid)
        assert thread["status"] == "no_questions" and turns[-1]["body"] == "NO QUESTIONS" and turns[-1]["meta"]["no_questions"] is True
        assert len(w.commenter.posts) == 1   # NO QUESTIONS is never posted on the PR
        assert w.durable.new_runs == [rid1, rid2]

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_later_push_for_the_same_pr_joins_the_same_thread(mode, monkeypatch):
    async def scenario(w):
        first = await send_pr(w)
        second = await send_pr(w, session="sess-B", branch="feat/a", body="Pushed rev 2.", spec=SPEC + "rev 2\n")
        assert second["thread"]["thread_id"] == first["thread"]["thread_id"] and second["thread"]["session_id"] == "sess-B"
        _, turns = await state(w, first["thread"]["thread_id"])
        assert [t["speaker"] for t in turns] == ["session", "hub", "session", "hub"]
        other = await send_pr(w, pr=3457, branch="feat/b")
        assert other["thread"]["thread_id"] != first["thread"]["thread_id"]
        note = await w.loop.intake(session_id="sess-A", title="a note", body="no PR here")
        assert note["thread"]["kind"] == "note" and note["thread"]["pr_number"] is None
        assert len(await srl.recent_threads(w.pool, hours=48)) == 3

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_no_held_thread_means_no_run(mode, monkeypatch):
    async def scenario(w):
        for _ in range(3):
            assert (await w.loop.tick())["skipped"] == "nothing_held"
            w.clock.to_next_hour()
        assert w.durable.submits == []

    run(mode, monkeypatch, scenario)


# --- one round an hour -------------------------------------------------------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_one_round_per_hour_even_across_a_restart(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        # A restart while the run is in flight: the new loop finds the round and submits nothing new.
        loop2 = w.make_loop()
        assert (await loop2.tick())["for"] == "orion_run:running"
        assert w.durable.new_runs == [rid] and len(w.durable.submits) == 1
        w.durable.finish(rid, answer((tid, "1. q?")))
        assert (await loop2.tick())["applied"] == rid
        # Applying twice (a crash after the turn was stored) never stores a second turn.
        assert (await loop2._apply(w.pool, rid, answer((tid, "1. q?")), model=None))["skipped"] == "round_already_applied"
        # Another message in the same hour, and another restart: no second round this hour.
        await send_pr(w, pr=3999, branch="feat/z")
        loop3 = w.make_loop()
        for _ in range(3):
            w.clock.advance(minutes=5)
            assert (await loop3.tick())["skipped"] in ("roundup_already_ran_this_hour", "nothing_due")
        assert w.durable.new_runs == [rid]
        _, turns = await state(w, tid)
        assert [t["speaker"] for t in turns].count("orion") == 1

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_crash_between_marking_the_round_and_submitting_it_submits_once(mode, monkeypatch):
    async def scenario(w):
        got = await send_pr(w)
        tid = got["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = "session-roundup-" + core.bucket_start(w.clock.now, 3600).strftime("%Y%m%dT%H%M")
        await srl.mark_round(w.pool, rid, [tid], [got["turn"]["turn_id"]])   # ... and Hub dies here
        assert (await w.make_loop().tick())["for"] == "orion_run:running"
        assert [s["run_id"] for s in w.durable.submits] == [rid] and SPEC.strip() in w.durable.submits[0]["prompt"]
        # A second submit of the same round (a restart after the submit) is the same request (but for
        # its own timestamp): same run id, same prompt, same deadline. durable-runs keeps the first.
        await w.loop._submit(w.pool, rid)
        first, second = ({k: v for k, v in s["json"].items() if k != "requested_at"} for s in w.durable.submits)
        assert first == second and w.durable.new_runs == [rid]
        # durable-runs refusing the second copy (409) adopts the run it already has; the round is not failed.
        w.durable.refuse_with = 409
        assert (await w.loop._submit(w.pool, rid))["for"] == "orion_run:running"
        assert (await state(w, tid))[0]["status"] == "reading"

    run(mode, monkeypatch, scenario)


def test_the_clock_is_taken_after_the_read(monkeypatch):
    """A slow read that crosses the hour: the message (stored at 10:59:58) is due in the 11:00 round.
    With the clock taken before the read the tick would still think it is 10:59 and hold it."""

    async def scenario(w):
        w.clock.now = w.clock.now.replace(minute=59, second=58)
        tid = (await send_pr(w))["thread"]["thread_id"]
        order = []
        w.store.on_read = lambda: (order.append("read"), w.clock.advance(seconds=5))
        w.clock.calls = 0
        real_bucket = core.bucket_start

        def spy(now, interval):
            order.append("bucket")
            return real_bucket(now, interval)

        monkeypatch.setattr(core, "bucket_start", spy)
        tick = await w.loop.tick()
        assert order[:2] == ["read", "bucket"]
        assert tick["started"] == "session-roundup-" + real_bucket(w.clock.now, 3600).strftime("%Y%m%dT%H%M")
        assert w.durable.submits[0]["thread_ids"] == [tid]

    run("fake", monkeypatch, scenario)


# --- failure -----------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_three_failed_rounds_in_a_row_leave_the_thread_failed_and_a_new_message_revives_it(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        frames(w)
        w.durable.replies = [RuntimeError("timeout"), "", {"status": "completed", "harness_steps": {
            "turn": {"text": "[Cut short - not a finished answer.]\n- found x", "debug": {"harness_grounding_status": "fcc_timeout"}}}}]
        reasons = []
        for n in (1, 2, 3):
            w.clock.to_next_hour()
            tick = await w.loop.tick()
            reasons.append(tick["why"])
            thread, turns = await state(w, tid)
            assert thread["roundup_failures"] == n and thread["status"] == ("failed" if n == 3 else "held")
            assert turns[-1]["speaker"] == "hub" and "Orion did NOT read this thread" in turns[-1]["body"]
            assert turns[0]["roundup_id"] is None   # the message is unread again
            # Same hour after the failure: no second try.
            assert (await w.loop.tick())["skipped"] in ("roundup_already_ran_this_hour", "nothing_held")
        assert reasons == ["orion_run_failed:timeout", "orion_run_completed_empty", "orion_turn_cut_short"]
        thread, turns = await state(w, tid)
        assert "Hub stopped trying" in turns[-1]["body"] and thread["status_reason"] == "orion_turn_cut_short"
        assert not any(t["speaker"] == "orion" for t in turns)   # a failed round never stores Orion's turn
        assert [f["speaker"] for f in frames(w)] == ["hub", "hub", "hub"]
        # It stays failed: later hours submit nothing.
        w.clock.to_next_hour()
        assert (await w.loop.tick())["skipped"] == "nothing_held" and len(w.durable.new_runs) == 3
        # A new message puts it back in the queue with a clean count; a good round clears it.
        got = await w.loop.intake(session_id="sess-A", title="reply", body="try again please", thread_id=tid)
        assert got["thread"]["status"] == "held" and got["thread"]["roundup_failures"] == 0
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. finally, a question?"))]
        assert (await w.loop.tick())["asked"] == 1
        thread, _ = await state(w, tid)
        assert thread["status"] == "asked" and thread["roundup_failures"] == 0 and thread["status_reason"] is None

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_durable_runs_down_waits_and_a_refused_run_counts_as_a_failure(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.down = True
        assert (await w.loop.tick())["for"].startswith("durable_runs")
        assert (await w.loop.tick())["for"].startswith("durable_runs")
        thread, _ = await state(w, tid)
        assert thread["status"] == "reading" and thread["roundup_failures"] == 0
        w.durable.down, w.durable.refuse_with = False, 422   # e.g. durable-runs older than this workflow
        tick = await w.loop.tick()
        assert tick["why"] == "orion_run_refused:http_422"
        thread, _ = await state(w, tid)
        assert thread["status"] == "held" and thread["roundup_failures"] == 1

    run(mode, monkeypatch, scenario)


# --- Orion's reply -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_an_unparseable_reply_is_stored_whole_on_every_thread_and_not_posted(mode, monkeypatch):
    async def scenario(w):
        a = (await send_pr(w, pr=1))["thread"]["thread_id"]
        b = (await send_pr(w, pr=2, branch="feat/b"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        whole = "I read both specs. My worry is the GitHub token."
        w.durable.replies = [whole]
        tick = await w.loop.tick()
        assert tick["unparsed"] == 2 and tick["pr_comments"] == 0 and w.commenter.posts == []
        for tid in (a, b):
            thread, turns = await state(w, tid)
            assert thread["status"] == "asked" and turns[-1]["body"] == whole and turns[-1]["meta"]["unparsed"] is True

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_each_thread_gets_its_own_section_and_its_own_pr_comment(mode, monkeypatch):
    async def scenario(w):
        a = (await send_pr(w, pr=11, session="sess-A", branch="feat/a"))["thread"]["thread_id"]
        b = (await send_pr(w, pr=12, session="sess-B", branch="feat/b"))["thread"]["thread_id"]
        n = (await w.loop.intake(session_id="sess-C", title="note", body="a note with no PR"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((a, "1. for A?"), (b, "NO QUESTIONS"), (n, "1. for the note?"))]
        tick = await w.loop.tick()
        assert (tick["asked"], tick["no_questions"], tick["pr_comments"]) == (2, 1, 1)
        assert [p[0] for p in w.commenter.posts] == [11]   # never NO QUESTIONS, never a thread without a PR
        # Each session gets only its own thread back; a branch match works without a session id.
        [for_a] = await srl.pending_for(w.pool, session_id="sess-A", branch="")
        assert for_a["thread_id"] == a and for_a["body"] == "1. for A?"
        [for_b] = await srl.pending_for(w.pool, session_id="", branch="feat/b")
        assert for_b["thread_id"] == b and for_b["meta"]["no_questions"] is True
        assert await srl.pending_for(w.pool, session_id="sess-Z", branch="feat/zzz") == []
        assert await srl.pending_for(w.pool, session_id="", branch="") == []
        assert await srl.mark_delivered(w.pool, [for_a["turn_id"], for_b["turn_id"]]) == 2
        assert await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a") == []
        [for_c] = await srl.pending_for(w.pool, session_id="sess-C", branch="")
        assert for_c["thread_id"] == n

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_pr_comments_can_be_switched_off_and_a_failed_comment_keeps_the_turn(mode, monkeypatch):
    async def scenario(w):
        a = (await send_pr(w, pr=21))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((a, "1. q?"))]
        off = w.make_loop(pr_comments=False)
        assert (await off.tick())["pr_comments"] == 0 and w.commenter.posts == []
        b = (await send_pr(w, pr=22, branch="feat/b"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.commenter.fail = True
        w.durable.replies = [answer((b, "1. q for b?"))]
        tick = await w.loop.tick()
        assert tick["asked"] == 1 and tick["pr_comments"] == 0 and len(w.commenter.posts) == 1
        thread, turns = await state(w, b)
        assert thread["status"] == "asked" and turns[-1]["body"] == "1. q for b?" and turns[-1]["pr_comment_url"] is None

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_message_that_arrives_while_orion_is_reading_waits_for_the_next_round(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        got = await w.loop.intake(session_id="sess-A", title="reply", body="One more thing: the cap is 3.", thread_id=tid)
        assert got["thread"]["status"] == "reading"
        w.durable.finish(rid, answer((tid, "1. first question?")))
        await w.loop.tick()
        thread, turns = await state(w, tid)
        assert thread["status"] == "held"   # Orion answered the first message; the second is still unread
        assert [t["speaker"] for t in turns] == ["session", "hub", "session", "hub", "orion"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        await w.loop.tick()
        prompt = w.durable.submits[-1]["prompt"]
        new_part = prompt[prompt.index("new since your last round-up"):]
        assert "One more thing: the cap is 3." in new_part and "Please read the spec." not in new_part
        assert (await state(w, tid))[0]["status"] == "no_questions"

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_threads_that_do_not_fit_stay_held_oldest_first(mode, monkeypatch, caplog):
    async def scenario(w):
        ids = []
        for i in range(3):
            ids.append((await send_pr(w, pr=30 + i, branch=f"feat/{i}", spec="s" * 3000))["thread"]["thread_id"])
            w.clock.advance(seconds=1)
        w.clock.to_next_hour()
        w.durable.replies = [answer((ids[0], "1. q0?"), (ids[1], "1. q1?")), answer((ids[2], "1. q2?"))]
        with caplog.at_level("INFO", logger="orion-hub.session_threads"):
            tick = await w.loop.tick()
        assert w.durable.submits[0]["thread_ids"] == ids[:2] and tick["left_held"] == [ids[2]]
        assert len(w.durable.submits[0]["prompt"]) <= 9000
        assert any("session_roundup_left_held" in r.message and ids[2] in r.message for r in caplog.records)
        assert [(await state(w, t))[0]["status"] for t in ids] == ["asked", "asked", "held"]
        w.clock.to_next_hour()
        await w.loop.tick()
        assert w.durable.submits[1]["thread_ids"] == [ids[2]] and (await state(w, ids[2]))[0]["status"] == "asked"

    run(mode, monkeypatch, scenario, prompt_max_chars=9000)


# --- review findings (2026-11-15) ----------------------------------------------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_the_same_message_twice_is_stored_once(mode, monkeypatch):
    """A hook that fires twice (or a failed `gh pr create` that prints the existing PR) must not make
    Orion read the same spec twice."""

    async def scenario(w):
        first = await send_pr(w)
        frames(w)
        again = await send_pr(w)
        assert again["duplicate"] is True and again["turn"] is None and "Nothing new was stored" in again["receipt"]["body"]
        assert again["thread"]["thread_id"] == first["thread"]["thread_id"] and frames(w) == []
        _, turns = await state(w, first["thread"]["thread_id"])
        assert [t["speaker"] for t in turns] == ["session", "hub"]
        changed = await send_pr(w, spec=SPEC + "rev 2\n")
        assert changed["duplicate"] is False

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_intake_is_bounded_per_thread_and_in_total(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        for i in range(core.MAX_UNREAD_TURNS_PER_THREAD - 1):
            await w.loop.intake(session_id="s", title="reply", body=f"more {i}", thread_id=tid)
        with pytest.raises(srl.IntakeRefused) as err:
            await w.loop.intake(session_id="s", title="reply", body="one too many", thread_id=tid)
        assert err.value.too_many and "too_many_unread_messages_on_thread" in str(err.value)
        monkeypatch.setattr(core, "MAX_HELD_THREADS", 3)
        await w.loop.intake(session_id="s", title="n1", body="note 1")
        await w.loop.intake(session_id="s", title="n2", body="note 2")
        with pytest.raises(srl.IntakeRefused) as err:
            await w.loop.intake(session_id="s", title="n3", body="note 3")
        assert err.value.too_many and "too_many_held_threads" in str(err.value)
        # A full thread is still read: the round runs and clears it.
        w.clock.to_next_hour()
        w.durable.replies = ["no headings at all"]
        tick = await w.loop.tick()
        assert tick["unparsed"] == 3 and len(w.durable.submits[0]["prompt"]) <= 100000

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_run_that_never_ends_does_not_block_later_rounds(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        w.durable.runs[rid] = {"status": "paused"}
        w.clock.advance(hours=3)
        assert (await w.loop.tick())["for"] == "orion_run:paused"   # inside deadline + grace: wait
        w.clock.advance(minutes=40)
        tick = await w.loop.tick()
        assert tick["why"] == "orion_run_overdue:paused"
        thread, _ = await state(w, tid)
        assert thread["status"] == "held" and thread["roundup_failures"] == 1
        # An abandoned run is final too.
        w.clock.to_next_hour()
        w.durable.replies = [{"status": "abandoned", "error": "retired"}]
        assert (await w.loop.tick())["why"] == "orion_run_abandoned:retired"

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_server_error_or_an_unreadable_conflict_waits_and_counts_no_failure(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.refuse_with = 503
        assert (await w.loop.tick())["for"] == "durable_runs:http_503"
        w.durable.refuse_with = 409   # "different request", and the run cannot be read back
        assert (await w.loop.tick())["for"] == "durable_runs:conflict_unread"
        thread, _ = await state(w, tid)
        assert thread["status"] == "reading" and thread["roundup_failures"] == 0
        w.durable.refuse_with = 0
        w.durable.replies = [answer((tid, "1. q?"))]
        assert (await w.loop.tick())["asked"] == 1

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_the_comment_is_given_the_threads_branch_and_the_fence_is_not_guessable(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w, pr=77, branch="feat/seventy-seven"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. q?"))]
        await w.loop.tick()
        assert w.commenter.branches == ["feat/seventy-seven"]
        prompt, rid = w.durable.submits[0]["prompt"], w.durable.submits[0]["run_id"]
        nonce = srl.fence_nonce(rid)
        assert f"<<<END_SESSION_DATA:{nonce}>>>" in prompt and len(nonce) == 12
        assert nonce != srl.fence_nonce(rid + "x") and "<<<END_SESSION_DATA>>>" not in prompt

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("status,pull,branch,why", [
    (200, {"state": "open", "head": {"ref": "feat/a"}}, "feat/a", None),
    (404, {}, "feat/a", "not_a_pull_request:http_404"),             # an issue number, or nothing
    (200, {"state": "closed", "head": {"ref": "feat/a"}}, "feat/a", "pull_request_closed"),
    (200, {"state": "open", "head": {"ref": "feat/other"}}, "feat/a", "branch_mismatch"),
    (200, {"state": "open", "head": {"ref": "feat/a"}}, None, "branch_mismatch"),
])
def test_orion_only_comments_on_an_open_pr_of_the_threads_own_branch(status, pull, branch, why):
    assert srl.GitHubCommenter.target_refusal(status, pull, branch) == why


def _clean_guard():
    from orion.memory.mesh_edge import MeshEdgeGuard

    return MeshEdgeGuard()


def test_the_commenter_refuses_a_body_quoting_a_core_page_sentence_and_never_calls_github(monkeypatch):
    """Issue #3409: a PR comment quoting a core-page / lived-memory sentence verbatim is not
    posted, and neither is any comment when the guard cannot load (fail closed)."""
    import httpx

    from orion.memory.mesh_edge import MeshEdgeGuard, MeshEdgeUnavailable

    calls = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    sentence = "Robin told me the blue kite got stuck in the old oak tree."
    guard = MeshEdgeGuard.from_texts([("page:juniper", sentence)])

    def unavailable():
        raise MeshEdgeUnavailable("down")

    async def go():
        leaky = srl.GitHubCommenter("o/r", token_provider=lambda: "tok", guard_loader=lambda: guard)
        assert await leaky.post(5, f"Status update.\n\n> {sentence.upper()}", branch="feat/a") is None
        down = srl.GitHubCommenter("o/r", token_provider=lambda: "tok", guard_loader=unavailable)
        assert await down.post(5, "harmless body", branch="feat/a") is None

    asyncio.run(go())
    assert calls == []


def test_the_commenter_checks_the_pull_request_before_it_posts_and_never_raises(monkeypatch):
    import httpx

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.method == "GET":
            number = int(request.url.path.rsplit("/", 1)[1])
            if number == 5:
                return httpx.Response(200, json={"state": "open", "head": {"ref": "feat/a"}})
            return httpx.Response(404, json={})
        return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/5#issuecomment-9"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    async def go():
        ok = srl.GitHubCommenter("o/r", token_provider=lambda: "tok", guard_loader=_clean_guard)
        assert await ok.post(5, "body", branch="feat/a") == "https://github.com/o/r/pull/5#issuecomment-9"
        assert await ok.post(6, "body", branch="feat/a") is None            # not a pull request
        assert await ok.post(5, "body", branch="feat/zzz") is None          # another branch's PR
        assert await srl.GitHubCommenter("o/r", token_provider=lambda: "", guard_loader=_clean_guard).post(
            5, "b", branch="feat/a") is None
        assert await srl.GitHubCommenter("../evil", token_provider=lambda: "tok", guard_loader=_clean_guard).post(
            5, "b", branch="feat/a") is None

    asyncio.run(go())
    assert calls == [("GET", "/repos/o/r/pulls/5", "Bearer tok"), ("POST", "/repos/o/r/issues/5/comments", "Bearer tok"),
                     ("GET", "/repos/o/r/pulls/6", "Bearer tok"), ("GET", "/repos/o/r/pulls/5", "Bearer tok")]
