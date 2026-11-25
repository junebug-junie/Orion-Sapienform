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


def one_round(monkeypatch):
    """Answer-phase mechanics tests run in a world of one question round: the phase rules are the same at
    any count, and the 3-round count has its own lifecycle test (Juniper 2026-11-25: up to 3 per PR)."""
    monkeypatch.setattr(core, "MAX_QUESTION_ROUNDS_PER_THREAD", 1)


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
        assert "Hub cut the spec docs/superpowers/specs/0.md to at most 40000 characters" in receipt
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
    one_round(monkeypatch)
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
        assert pr == 3456 and comment.startswith("### Orion's review questions (session-thread round-up)") and "1. Why hourly?" in comment
        [frame] = frames(w)
        assert (frame["speaker"], frame["status"], frame["text"]) == ("orion", "asked", orion["body"])
        assert (await w.loop.tick())["skipped"] == "nothing_held" and len(w.durable.submits) == 1

        # The session that sent it reads the questions, once.
        [waiting] = await srl.pending_for(w.pool, session_id="sess-A", branch="")
        assert waiting["turn_id"] == orion["turn_id"] and waiting["thread_id"] == tid and waiting["pr_number"] == 3456
        assert await srl.mark_delivered(w.pool, [orion["turn_id"]], session_id="sess-A") == 1
        assert await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a") == []
        assert await srl.mark_delivered(w.pool, [orion["turn_id"]], session_id="sess-A") == 0

        # The session replies on the thread: held again, and not read before the next hour. Orion used
        # the thread's one round of questions, so Hub's receipt says the reply is read as answers.
        got = await w.loop.intake(session_id="sess-A", title="reply", body="Hourly is Juniper's rule.", thread_id=tid)
        assert got["thread"]["status"] == "held" and got["thread"]["title"] == "feat: thing 3456"
        assert got["answer_phase"] is True and "all 1 rounds of questions on this thread" in got["receipt"]["body"]
        assert (await w.loop.tick())["skipped"] in ("nothing_due", "roundup_already_ran_this_hour")
        assert len(w.durable.submits) == 1

        # The second round shows Orion their own earlier questions and the reply (the answers), marked
        # ANSWER PHASE, without the spec again. Orion asks anyway: that text is a closing note.
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "Thanks.\n1. But who pays for the retry?"))]
        tick = await w.loop.tick()
        rid2 = tick["started"]
        assert rid2 != rid1 and tick["closed"] == 1 and tick["asked"] == 0 and tick["pr_comments"] == 0
        assert tick["dropped_question_lines"] == 1
        prompt = w.durable.submits[-1]["prompt"]
        assert "[you (Orion), an earlier round-up] 1. Why hourly?" in prompt
        assert "[session] Hourly is Juniper's rule." in prompt and SPEC.strip() not in prompt
        assert core.ANSWER_PHASE_NOTE in prompt
        assert prompt.index(f"## thread:{tid}") < prompt.index(core.ANSWER_PHASE_NOTE) < prompt.index("earlier in this thread")
        assert prompt.index("earlier in this thread") < prompt.index("new since your last round-up") < prompt.index("Hourly is Juniper's rule.")
        thread, turns = await state(w, tid)
        assert thread["status"] == core.STATUS_CLOSED
        closing = turns[-1]
        assert closing["body"] == "Thanks.\n1. But who pays for the retry?" and closing["meta"]["closing_note"] is True
        assert closing["meta"]["dropped_question_lines"] == 1 and not closing["meta"].get("no_questions")
        assert closing["pr_comment_url"] is None and len(w.commenter.posts) == 1   # a closing note is never posted
        [waiting] = await srl.pending_for(w.pool, session_id="sess-A", branch="")
        assert waiting["turn_id"] == closing["turn_id"] and waiting["meta"]["closing_note"] is True
        assert w.durable.new_runs == [rid1, rid2]

        # Another reply: still read (Orion should see it), still answer phase, still no questions out.
        await w.loop.intake(session_id="sess-A", title="reply", body="The retry is on durable runs.", thread_id=tid)
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        tick = await w.loop.tick()
        assert tick["closed"] == 1 and tick["no_questions"] == 0 and core.ANSWER_PHASE_NOTE in w.durable.submits[-1]["prompt"]
        thread, turns = await state(w, tid)
        assert thread["status"] == core.STATUS_CLOSED and turns[-1]["meta"]["closing_note"] is True
        assert len(w.commenter.posts) == 1

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
def test_an_unparseable_reply_is_kept_sent_nowhere_and_read_again_without_using_the_round(mode, monkeypatch):
    one_round(monkeypatch)
    # Session thread b3053f6df376 (Orion, 2026-11-18): a parse accident is not Orion's silence. The reply
    # is posted nowhere, handed to no session, and the threads are read again next hour as a FIRST
    # question round. It is a failure toward the cap: 3 in a row end the thread. Thread 455a41af5b44 Q4:
    # in a round of two threads the whole reply is NOT stored on either (it holds both threads' text).
    async def scenario(w):
        a = (await send_pr(w, pr=1))["thread"]["thread_id"]
        b = (await send_pr(w, pr=2, branch="feat/b", session="sess-B"))["thread"]["thread_id"]
        frames(w)
        w.clock.to_next_hour()
        whole = "I read both specs. My worry is the GitHub token."
        w.durable.replies = [whole]
        tick = await w.loop.tick()
        tick_id = tick["applied"]
        assert tick["unparsed"] == 2 and tick["asked"] == 0 and tick["pr_comments"] == 0 and w.commenter.posts == []
        for tid in (a, b):
            thread, turns = await state(w, tid)
            assert thread["status"] == "held" and thread["roundup_failures"] == 1
            assert thread["status_reason"] == srl.UNPARSED_REASON
            assert not [t for t in turns if t["speaker"] == "orion"] and whole not in str(turns)
            assert turns[-1]["speaker"] == "hub" and turns[-1]["meta"]["reply_unparsed"] is True
            assert turns[-1]["meta"]["reply_not_stored"] is True
            assert "does not use one of this thread's rounds of questions" in turns[-1]["body"]
            assert "did not store it on this thread" in turns[-1]["body"] and tick_id in turns[-1]["body"]
            assert "Failure 1 of 3" in turns[-1]["body"]
            assert turns[0]["roundup_id"] is None            # the message is unread again
            assert core.in_answer_phase(turns) is False
        assert [f["speaker"] for f in frames(w)] == ["hub", "hub"]
        # Handed to no session, by session or by branch.
        assert await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a") == []
        assert await srl.pending_for(w.pool, session_id="sess-B", branch="feat/b") == []
        # A reply sent now is NOT read as answers: the round is unused.
        got = await w.loop.intake(session_id="sess-A", title="more", body="One more detail.", thread_id=a)
        assert got["answer_phase"] is False and "Orion's questions will appear" in got["receipt"]["body"]
        # Next hour: both threads read again as a first round, the spec shown again; a good reply asks.
        w.clock.to_next_hour()
        w.durable.replies = [answer((a, "1. Which token?"), (b, "1. Why hourly?"))]
        tick = await w.loop.tick()
        prompt = w.durable.submits[-1]["prompt"]
        assert core.ANSWER_PHASE_NOTE not in prompt and SPEC.strip() in prompt
        assert "Please read the spec." in prompt and whole not in prompt   # the re-queued reply is not shown again
        assert tick["asked"] == 2 and tick["unparsed"] == 0 and tick["pr_comments"] == 2
        for tid in (a, b):
            thread, turns = await state(w, tid)
            assert thread["status"] == "asked" and thread["roundup_failures"] == 0 and thread["status_reason"] is None
        [for_a] = await srl.pending_for(w.pool, session_id="sess-A", branch="")
        assert for_a["body"] == "1. Which token?"   # only the parsed questions reach the session
        # Answer phase, a later round: the uncut reply (other threads' sections) is never shown again.
        await w.loop.intake(session_id="sess-A", title="answers", body="It is the Hub token.", thread_id=a)
        w.clock.to_next_hour()
        w.durable.replies = [answer((a, "NO QUESTIONS"))]
        await w.loop.tick()
        prompt = w.durable.submits[-1]["prompt"]
        assert core.ANSWER_PHASE_NOTE in prompt and "1. Which token?" in prompt and whole not in prompt

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_requeue_is_once_per_round_and_a_message_sent_while_reading_is_read_next_round(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        # A reply arrives while Orion reads: intake keeps `reading` and zeroes the count.
        await w.loop.intake(session_id="sess-A", title="more", body="Also see the cap.", thread_id=tid)
        w.durable.finish(rid, "no headings at all")
        assert (await w.loop.tick())["unparsed"] == 1
        thread, turns = await state(w, tid)
        assert thread["status"] == "held" and thread["roundup_failures"] == 1
        assert all(t["roundup_id"] is None for t in turns if t["speaker"] == "session")
        # A second apply of the same round (a crash midway) stores nothing new and counts nothing.
        again = await srl.requeue_unparsed(w.pool, thread_id=tid, roundup_id=rid, body="x", meta={},
                                           reason=srl.UNPARSED_REASON, text_for=lambda n: "line")
        assert again[3] is False and again[2] is None and again[0]["roundup_failures"] == 1
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. ok?"))]
        assert (await w.loop.tick())["asked"] == 1
        prompt = w.durable.submits[-1]["prompt"]
        assert "Please read the spec." in prompt and "Also see the cap." in prompt

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_thread_unparsed_three_times_in_a_row_ends_failed_and_a_new_message_revives_it(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.durable.replies = ["no headings 1", RuntimeError("timeout"), "no headings 3"]
        for n in (1, 2, 3):
            w.clock.to_next_hour()
            await w.loop.tick()
            thread, turns = await state(w, tid)
            assert thread["roundup_failures"] == n and thread["status"] == ("failed" if n == 3 else "held")
        assert "Hub stopped trying" in turns[-1]["body"] and turns[-1]["meta"]["reply_unparsed"] is True
        assert thread["status_reason"] == srl.UNPARSED_REASON and core.in_answer_phase(turns) is False
        w.clock.to_next_hour()
        assert (await w.loop.tick())["skipped"] == "nothing_held" and len(w.durable.new_runs) == 3
        got = await w.loop.intake(session_id="sess-A", title="reply", body="try again", thread_id=tid)
        assert got["thread"]["status"] == "held" and got["thread"]["roundup_failures"] == 0
        assert got["answer_phase"] is False
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. now?"))]
        assert (await w.loop.tick())["asked"] == 1

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
        assert await srl.mark_delivered(w.pool, [for_a["turn_id"], for_b["turn_id"]], session_id="sess-A") == 2
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
    one_round(monkeypatch)
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        got = await w.loop.intake(session_id="sess-A", title="reply", body="One more thing: the cap is 3.", thread_id=tid)
        assert got["thread"]["status"] == "reading"
        # Orion has not answered yet: the receipt says this may be read as answers, not "questions will come".
        assert got["answer_phase"] is False and "reading this thread's earlier message now" in got["receipt"]["body"]
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
        # Orion's round-1 turn sits AFTER this message (it arrived while Orion was reading), and still
        # counts: the thread is in its answer phase.
        assert core.ANSWER_PHASE_NOTE in prompt
        assert "[you (Orion), an earlier round-up] 1. first question?" in prompt   # Orion sees its own questions
        assert (await state(w, tid))[0]["status"] == core.STATUS_CLOSED

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_threads_that_do_not_fit_stay_held_oldest_first(mode, monkeypatch, caplog):
    one_round(monkeypatch)
    async def scenario(w):
        ids = []
        for i in range(3):
            ids.append((await send_pr(w, pr=30 + i, branch=f"feat/{i}", spec="s" * 2400))["thread"]["thread_id"])
            w.clock.advance(seconds=1)
        w.clock.to_next_hour()
        w.durable.replies = [answer((ids[0], "1. q0?"), (ids[1], "1. q1?")), answer((ids[2], "1. q2?"))]
        with caplog.at_level("INFO", logger="orion-hub.session_threads"):
            tick = await w.loop.tick()
        assert w.durable.submits[0]["thread_ids"] == ids[:2] and tick["left_held"] == [ids[2]]
        assert len(w.durable.submits[0]["prompt"]) <= 9300
        assert any("session_roundup_left_held" in r.message and ids[2] in r.message for r in caplog.records)
        assert [(await state(w, t))[0]["status"] for t in ids] == ["asked", "asked", "held"]
        w.clock.to_next_hour()
        await w.loop.tick()
        assert w.durable.submits[1]["thread_ids"] == [ids[2]] and (await state(w, ids[2]))[0]["status"] == "asked"

    run(mode, monkeypatch, scenario, prompt_max_chars=9300)


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
        assert tick["why"] == "orion_run_overdue:paused" and tick["strike"] is True
        thread, _ = await state(w, tid)
        # Paused is not waiting for the pool: a strike, as before (a hung run must not retry for ever).
        assert thread["status"] == "held" and thread["roundup_failures"] == 1
        # An abandoned run is final too.
        w.clock.to_next_hour()
        w.durable.replies = [{"status": "abandoned", "error": "retired"}]
        assert (await w.loop.tick())["why"] == "orion_run_abandoned:retired"
        thread, _ = await state(w, tid)
        assert thread["roundup_failures"] == 2   # abandoned is the run's own end: a strike
        # A run still waiting for the pool well past its deadline is the body's: not a strike.
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        w.durable.runs[rid] = {"status": "waiting_resource"}
        w.clock.advance(hours=3, minutes=40)
        tick = await w.loop.tick()
        assert tick["why"] == "orion_run_overdue:waiting_resource" and tick["strike"] is False
        thread, _ = await state(w, tid)
        assert thread["status"] == "held" and thread["roundup_failures"] == 2

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_hot_week_never_ends_a_thread_and_retries_quietly(mode, monkeypatch):
    # Session thread b3053f6df376 Q3: the body's refusals (the GPU pool kept the run waiting past its
    # deadline) are not strikes. Five in a row leave the thread held with a zero count, one Hub line
    # for the streak, and the next good round asks as usual.
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        frames(w)
        errors = ["workflow_deadline", "gpu_pool_unavailable:deadline", "RuntimeError: gpu_pool_unavailable:backlog_max_age",
                  "gpu_pool_unavailable:deadline", "workflow_deadline"]
        w.durable.replies = [RuntimeError(e) for e in errors]
        for e in errors:
            w.clock.to_next_hour()
            tick = await w.loop.tick()
            assert tick["why"] == f"orion_run_failed:{e}" and tick["strike"] is False
            thread, turns = await state(w, tid)
            assert thread["status"] == "held" and thread["roundup_failures"] == 0 and turns[0]["roundup_id"] is None
        hub_lines = [t for t in turns if t["speaker"] == "hub" and t["meta"].get("body_refusal")]
        assert len(hub_lines) == 1 and "not counted as a failure" in hub_lines[0]["body"]
        assert [f["speaker"] for f in frames(w)] == ["hub"]
        # A strike after the streak still counts and writes its line; the next body refusal writes again.
        w.durable.replies = [RuntimeError("timeout"), RuntimeError("workflow_deadline")]
        w.clock.to_next_hour()
        assert (await w.loop.tick())["strike"] is True
        w.clock.to_next_hour()
        await w.loop.tick()
        thread, turns = await state(w, tid)
        assert thread["roundup_failures"] == 1 and thread["status"] == "held"
        assert len([t for t in turns if t["speaker"] == "hub" and t["meta"].get("body_refusal")]) == 2
        # A request the pool can never place at the floor budget is the round's own fault: it counts.
        w.loop.budget_chars = srl.MIN_PROMPT_CHARS
        w.durable.replies = [RuntimeError("gpu_pool_unavailable:min_ctx_exceeds_class")]
        w.clock.to_next_hour()
        assert (await w.loop.tick())["strike"] is True
        thread, _ = await state(w, tid)
        assert thread["roundup_failures"] == 2
        # A deadline hit after Orion's turn was attempted is the round's failure: a strike.
        w.durable.replies = [{"status": "failed", "error": "workflow_deadline", "harness_turn_meta": {"step_key": "turn"}}]
        w.clock.to_next_hour()
        tick = await w.loop.tick()
        assert tick["why"] == "orion_run_failed:workflow_deadline_after_turn" and tick["strike"] is True
        thread, _ = await state(w, tid)
        assert thread["roundup_failures"] == 3 and thread["status"] == "failed"
        await w.loop.intake(session_id="sess-A", title="again", body="please retry", thread_id=tid)
        w.durable.replies = [answer((tid, "1. finally?"))]
        w.clock.to_next_hour()
        assert (await w.loop.tick())["asked"] == 1

    run(mode, monkeypatch, scenario)


async def age_messages(w, tid, days):
    """Make the thread's session messages ``days`` old (the body-refusal backstop reads their age)."""
    if w.mode == "fake":
        for u in w.store.turns:
            if u["thread_id"] == tid and u["speaker"] == "session":
                u["created_at"] = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        await w.pool.execute("UPDATE orion_session_turn SET created_at = now() - make_interval(days => $2) "
                             "WHERE thread_id = $1 AND speaker = 'session'", tid, days)


@pytest.mark.parametrize("mode", BOTH)
def test_a_body_refusal_on_a_very_old_message_counts_so_no_thread_retries_for_ever(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]   # pending: the run fails only after the messages are old
        await age_messages(w, tid, core.BODY_REFUSAL_MAX_AGE_SEC // 86400 + 1)
        w.durable.runs[rid] = {"status": "failed", "error": "workflow_deadline"}
        await w.loop.tick()
        thread, turns = await state(w, tid)
        assert thread["roundup_failures"] == 1 and not turns[-1]["meta"].get("body_refusal")
        assert turns[-1]["speaker"] == "hub" and "Failure 1 of 3" in turns[-1]["body"]

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


# --- rounds of questions per thread ---------------------------------------------------------


def test_three_question_rounds_per_thread_is_a_fixed_constant():
    assert core.MAX_QUESTION_ROUNDS_PER_THREAD == 3


@pytest.mark.parametrize("mode", BOTH)
def test_orion_writes_on_a_pull_request_at_most_three_times(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        for k in (1, 2, 3):
            w.clock.to_next_hour()
            w.durable.replies = [answer((tid, f"1. Question of round {k}?"))]
            tick = await w.loop.tick()
            assert tick["asked"] == 1 and tick["pr_comments"] == 1, k
            prompt = w.durable.submits[-1]["prompt"]
            if k == 1:
                assert "round(s) of questions left" not in prompt and SPEC.strip() in prompt
            else:
                # Rounds 2 and 3: Hub says how many are left, and the spec is still shown.
                assert f"you have {3 - (k - 1)} round(s) of questions left" in prompt and SPEC.strip() in prompt
                assert core.ANSWER_PHASE_NOTE not in prompt
            got = await w.loop.intake(session_id="sess-A", title="reply", body=f"Answers to round {k}.", thread_id=tid)
            assert got["answer_phase"] is (k == 3)
        # The fourth round is the answer phase: a closing note, never a fourth PR comment.
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. One more?"))]
        tick = await w.loop.tick()
        assert tick["closed"] == 1 and tick["pr_comments"] == 0 and core.ANSWER_PHASE_NOTE in w.durable.submits[-1]["prompt"]
        assert len(w.commenter.posts) == 3
        # The draft gate finds the thread by its pull request (real SQL in the pg lane).
        [found] = await srl.threads_for_pr(w.pool, 3456)
        assert found["thread_id"] == tid and found["turns"] >= 8
        assert await srl.threads_for_pr(w.pool, 4242) == []

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_code_diff_is_kept_cut_with_the_gh_command_and_shown_as_a_diff(mode, monkeypatch):
    async def scenario(w):
        diff = "diff --git a/x.py b/x.py\n" + "+x = 1\n" * 10000
        got = await w.loop.intake(session_id="s", title="feat: code", body="PR opened", pr_number=77,
                                  pr_url="https://github.com/o/r/pull/77", branch="feat/c",
                                  spec=[{"path": "1 file(s)", "text": diff, "kind": "diff"}])
        tid = got["thread"]["thread_id"]
        _, turns = await state(w, tid)
        [att] = turns[0]["meta"]["spec"]
        assert att["kind"] == "diff" and att["cut"] and att["text"].endswith("Full diff: `gh pr diff 77`.]")
        assert len(att["text"]) < core.DIFF_MAX_CHARS + 200
        assert "Hub cut the pull request diff" in got["receipt"]["body"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        await w.loop.tick()
        prompt = w.durable.submits[-1]["prompt"]
        assert "--- pull request code diff (1 file(s)) ---" in prompt and core.CODE_REVIEW_TEMPLATE in prompt

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_new_thread_still_gets_its_round_beside_an_answer_phase_thread(mode, monkeypatch, caplog):
    one_round(monkeypatch)
    async def scenario(w):
        old = (await send_pr(w, pr=1))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((old, "1. Why?"))]
        await w.loop.tick()
        await w.loop.intake(session_id="sess-A", title="reply", body="Because.", thread_id=old)
        new = (await send_pr(w, pr=2, session="sess-B", branch="feat/b"))["thread"]["thread_id"]
        assert new != old
        w.clock.to_next_hour()
        w.durable.replies = [answer((old, "1. And then what?"), (new, "1. What does it cost?"))]
        with caplog.at_level("INFO", logger="orion-hub.session_threads"):
            tick = await w.loop.tick()
        assert tick["asked"] == 1 and tick["closed"] == 1 and tick["pr_comments"] == 1
        assert "session_thread_questions_dropped thread=" + old in caplog.text and "lines=1" in caplog.text
        prompt = w.durable.submits[-1]["prompt"]
        old_part = prompt[prompt.index(f"## thread:{old}"):prompt.index(f"## thread:{new}")]
        new_part = prompt[prompt.index(f"## thread:{new}"):]
        assert core.ANSWER_PHASE_NOTE in old_part and core.ANSWER_PHASE_NOTE not in new_part
        assert SPEC.strip() in new_part and SPEC.strip() not in old_part
        assert (await state(w, old))[0]["status"] == core.STATUS_CLOSED
        assert (await state(w, new))[0]["status"] == core.STATUS_ASKED
        assert [pr for pr, _ in w.commenter.posts] == [1, 2]   # round 1 of the old thread, round 1 of the new
        [b] = await srl.pending_for(w.pool, session_id="sess-B", branch="")
        assert b["body"] == "1. What does it cost?" and not b["meta"].get("closing_note")

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_neither_no_questions_nor_an_unparsed_reply_uses_the_round(mode, monkeypatch):
    one_round(monkeypatch)
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        await w.loop.tick()
        # NO QUESTIONS used no round: a new message on the thread may still be asked about.
        got = await w.loop.intake(session_id="sess-A", title="reply", body="I changed the cap.", thread_id=tid)
        assert got["answer_phase"] is False
        w.clock.to_next_hour()
        w.durable.replies = ["no headings at all, 1. why the cap?"]
        tick = await w.loop.tick()
        assert core.ANSWER_PHASE_NOTE not in w.durable.submits[-1]["prompt"] and tick["unparsed"] == 1
        # The unparsed whole reply went nowhere: the round is NOT used, the thread is read again.
        _, turns = await state(w, tid)
        assert core.in_answer_phase(turns) is False
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. why the cap?"))]
        tick = await w.loop.tick()
        assert core.ANSWER_PHASE_NOTE not in w.durable.submits[-1]["prompt"] and tick["asked"] == 1
        # Now the round is used: an answer-phase reply that is unparsed is a closing note, as before.
        await w.loop.intake(session_id="sess-A", title="reply", body="Answers.", thread_id=tid)
        w.clock.to_next_hour()
        w.durable.replies = ["again no headings. 1. one more?"]
        tick = await w.loop.tick()
        assert tick["closed"] == 1 and tick["unparsed"] == 0 and tick["pr_comments"] == 0
        assert tick["dropped_question_lines"] == 0   # the whole reply is not this thread's own lines
        thread, turns = await state(w, tid)
        assert thread["status"] == core.STATUS_CLOSED
        assert turns[-1]["meta"]["closing_note"] is True and turns[-1]["meta"]["unparsed"] is True

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_failed_round_does_not_use_the_threads_one_question_round(mode, monkeypatch):
    one_round(monkeypatch)
    # Orion's round-up times out (and then comes back empty): no Orion turn is stored, so the next
    # good round is still the thread's FIRST question round, not an answer phase.
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.durable.replies = [RuntimeError("timeout"), ""]
        for why in ("orion_run_failed:timeout", "orion_run_completed_empty"):
            w.clock.to_next_hour()
            assert (await w.loop.tick())["why"] == why
            thread, turns = await state(w, tid)
            assert thread["status"] == "held" and not any(t["speaker"] == "orion" for t in turns)
            assert core.in_answer_phase(turns) is False
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. Why hourly?"))]
        tick = await w.loop.tick()
        assert tick["asked"] == 1 and tick["closed"] == 0 and tick["pr_comments"] == 1
        assert core.ANSWER_PHASE_NOTE not in w.durable.submits[-1]["prompt"]
        assert SPEC.strip() in w.durable.submits[-1]["prompt"]
        thread, turns = await state(w, tid)
        assert thread["status"] == "asked" and not turns[-1]["meta"].get("closing_note")
        # Only now is the round used: a reply is read as answers.
        got = await w.loop.intake(session_id="sess-A", title="reply", body="Juniper's rule.", thread_id=tid)
        assert got["answer_phase"] is True

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_orions_questions_reach_the_session_after_a_later_push_amends_the_thread(mode, monkeypatch):
    one_round(monkeypatch)
    # Delivery is keyed to the thread (its sender sessions and its branch), not to the `gh pr create`
    # event: a second push for the same PR, even from another session, still gets the open questions.
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. Why hourly?"))]
        assert (await w.loop.tick())["asked"] == 1
        _, turns = await state(w, tid)
        question = turns[-1]
        second = await send_pr(w, session="sess-B", body="Pushed rev 2.", spec=SPEC + "rev 2\n")
        assert second["thread"]["thread_id"] == tid and second["answer_phase"] is True
        for who in ({"session_id": "sess-B", "branch": ""}, {"session_id": "sess-A", "branch": ""},
                    {"session_id": "", "branch": "feat/a"}):
            [row] = await srl.pending_for(w.pool, **who)
            assert row["turn_id"] == question["turn_id"] and row["body"] == "1. Why hourly?"
        # Handed back once to each session; a session that sent a message on the thread since is a reply.
        assert await srl.mark_delivered(w.pool, [question["turn_id"]], session_id="sess-A") == 1
        assert await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a") == []
        # The amended spec is read at the next round, as answers (the round is used).
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        tick = await w.loop.tick()
        assert tick["closed"] == 1 and "rev 2" in w.durable.submits[-1]["prompt"]

    run(mode, monkeypatch, scenario)


# --- session thread 455a41af5b44: Orion's round-1 findings (2026-11-18) ---------------------------


@pytest.mark.parametrize("mode", BOTH)
def test_the_prompt_tells_orion_when_a_threads_pull_request_is_merged_or_closed(mode, monkeypatch):
    # Q2: Hub comments only on an OPEN pull request. A thread whose PR merged before the round says so
    # above its fence, so Orion knows its questions reach only the session hook and Hub chat.
    async def scenario(w):
        w.commenter.states = {1: "merged", 2: "open", 3: "closed"}
        ids = []
        for n in (1, 2, 3, 4):
            ids.append((await send_pr(w, pr=n, branch=f"feat/{n}"))["thread"]["thread_id"])
            w.clock.advance(seconds=1)
        w.clock.to_next_hour()
        w.durable.replies = [answer(*[(t, f"1. q{i}?") for i, t in enumerate(ids)])]
        tick = await w.loop.tick()
        prompt = w.durable.submits[-1]["prompt"]
        part = {t: prompt[prompt.index(f"## thread:{t}"):] for t in ids}
        assert core.pr_not_open_note(1, "merged") in part[ids[0]].split("<<<SESSION_DATA")[0]
        assert core.pr_not_open_note(3, "closed") in part[ids[2]].split("<<<SESSION_DATA")[0]
        assert "[Hub] Pull request #2" not in prompt and "[Hub] Pull request #4" not in prompt   # open / unknown
        assert sorted(w.commenter.state_reads) == [1, 2, 3, 4] and tick["asked"] == 4, tick

    run(mode, monkeypatch, scenario)


def test_no_pr_state_lookup_when_pr_comments_are_off(monkeypatch):
    async def scenario(w):
        w.loop = w.make_loop(pr_comments=False)
        w.commenter.states = {1: "merged"}
        await send_pr(w, pr=1)
        w.clock.to_next_hour()
        await w.loop.tick()
        assert w.commenter.state_reads == [] and "[Hub] Pull request" not in w.durable.submits[-1]["prompt"]

    run("fake", monkeypatch, scenario)


@pytest.mark.parametrize("status,pull,state", [
    (200, {"state": "open"}, "open"),
    (200, {"state": "closed", "merged": True}, "merged"),
    (200, {"state": "closed", "merged_at": "2026-11-18T00:00:00Z"}, "merged"),
    (200, {"state": "closed", "merged": False, "merged_at": None}, "closed"),
    (404, {}, None),
    (200, {"state": "weird"}, None),
])
def test_pr_state_reads_githubs_pull_request(status, pull, state):
    assert srl.GitHubCommenter.state_of(status, pull) == state


def test_pr_state_never_raises_and_needs_a_token(monkeypatch):
    import httpx

    def handler(request):
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    async def go():
        assert await srl.GitHubCommenter("o/r", token_provider=lambda: "tok").pr_state(5) is None
        assert await srl.GitHubCommenter("o/r", token_provider=lambda: "").pr_state(5) is None

    asyncio.run(go())


@pytest.mark.parametrize("mode", BOTH)
def test_a_fresh_session_message_restarts_the_backstop_clock(mode, monkeypatch):
    # Q3: the 14-day backstop is read from the NEWEST unread message of the round. A thread whose first
    # message is old but whose session wrote again today is not struck by the body's heat; a new message
    # also zeroes the strike count.
    async def scenario(w):
        tid = (await send_pr(w))["thread"]["thread_id"]
        await age_messages(w, tid, core.BODY_REFUSAL_MAX_AGE_SEC // 86400 + 1)
        await w.loop.intake(session_id="sess-A", title="more", body="Still here, still working on it.", thread_id=tid)
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        w.durable.runs[rid] = {"status": "failed", "error": "workflow_deadline"}
        tick = await w.loop.tick()
        thread, turns = await state(w, tid)
        assert tick["strike"] is False and thread["roundup_failures"] == 0 and thread["status"] == "held"
        assert turns[-1]["meta"].get("body_refusal") is True
        # Once every unread message is old, the body's refusals count again.
        await age_messages(w, tid, core.BODY_REFUSAL_MAX_AGE_SEC // 86400 + 1)
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        w.durable.runs[rid] = {"status": "failed", "error": "workflow_deadline"}
        await w.loop.tick()
        thread, _ = await state(w, tid)
        assert thread["roundup_failures"] == 1
        # A new message zeroes the streak.
        got = await w.loop.intake(session_id="sess-A", title="again", body="New detail.", thread_id=tid)
        assert got["thread"]["roundup_failures"] == 0

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_mixed_parse_stores_no_other_threads_text_on_the_unparsed_thread(mode, monkeypatch):
    # Q4: two threads cut cleanly, the third has no heading. Only the third is requeued, and Orion's whole
    # reply (with the other threads' questions and the session text quoted in them) is NOT stored on it:
    # not in its turns, not in a later prompt, not handed to a session.
    async def scenario(w):
        a = (await send_pr(w, pr=1, branch="feat/a", spec="SPEC-A secret words"))["thread"]["thread_id"]
        b = (await send_pr(w, pr=2, branch="feat/b", session="sess-B", spec="SPEC-B"))["thread"]["thread_id"]
        c = (await send_pr(w, pr=3, branch="feat/c", session="sess-C", spec="SPEC-C"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        reply = answer((a, "1. About 'SPEC-A secret words': why?"), (b, "1. Why B?")) + "\n\nAnd for C: why C?"
        w.durable.replies = [reply]
        # The trailing text sits under b's heading, so make c's section really missing: c gets no heading.
        tick = await w.loop.tick()
        assert tick["asked"] == 2 and tick["unparsed"] == 1
        thread, turns = await state(w, c)
        assert thread["status"] == "held" and thread["roundup_failures"] == 1
        assert not [t for t in turns if t["speaker"] == "orion"]
        assert "SPEC-A secret words" not in str(turns) and "Why B" not in str(turns)
        assert turns[-1]["meta"]["reply_not_stored"] is True
        assert await srl.pending_for(w.pool, session_id="sess-C", branch="feat/c") == []
        # Next hour c is read again, and nothing of a's text is in c's section.
        w.clock.to_next_hour()
        w.durable.replies = [answer((c, "1. Why C?"))]
        tick = await w.loop.tick()
        prompt = w.durable.submits[-1]["prompt"]
        assert w.durable.submits[-1]["thread_ids"] == [c] and "SPEC-A secret words" not in prompt
        assert tick["asked"] == 1

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_an_unparsed_answer_phase_thread_in_a_shared_round_is_requeued_not_closed(mode, monkeypatch):
    one_round(monkeypatch)
    # Q4: before, an answer-phase thread with no section stored the WHOLE reply as its closing note, and
    # a later round showed that note (other threads' text) to Orion as this thread's history.
    async def scenario(w):
        a = (await send_pr(w, pr=1, branch="feat/a"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((a, "1. Why?"))]
        await w.loop.tick()
        await w.loop.intake(session_id="sess-A", title="reply", body="Because.", thread_id=a)
        b = (await send_pr(w, pr=2, branch="feat/b", session="sess-B", spec="SPEC-B private"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((b, "1. About SPEC-B private?"))]   # no section for a
        tick = await w.loop.tick()
        assert tick["asked"] == 1 and tick["unparsed"] == 1 and tick["closed"] == 0
        thread, turns = await state(w, a)
        assert thread["status"] == "held" and "SPEC-B private" not in str(turns)
        w.clock.to_next_hour()
        w.durable.replies = [answer((a, "NO QUESTIONS"))]
        tick = await w.loop.tick()
        assert tick["closed"] == 1 and "SPEC-B private" not in w.durable.submits[-1]["prompt"]

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_questions_stay_pending_for_a_later_session_until_one_replies(mode, monkeypatch):
    # Q5: a session that pulled Orion's questions and exited without replying must not lose them for
    # every later session on the branch. Each session is shown them once; a reply ends it.
    async def scenario(w):
        tid = (await send_pr(w, branch="feat/a"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. Why hourly?"))]
        await w.loop.tick()
        [row] = await srl.pending_for(w.pool, session_id="sess-throwaway", branch="feat/a")
        assert row["delivered_at"] is None
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-throwaway") == 1
        # The same session is not shown them again (its next prompt, or its waiting hook).
        assert await srl.pending_for(w.pool, session_id="sess-throwaway", branch="feat/a") == []
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-throwaway") == 0
        # It exits without replying. A later session on the branch still gets them, once.
        [again] = await srl.pending_for(w.pool, session_id="sess-later", branch="feat/a")
        assert again["turn_id"] == row["turn_id"] and again["delivered_at"] is not None
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-later") == 1
        assert await srl.pending_for(w.pool, session_id="sess-later", branch="feat/a") == []
        # A caller with no session id (an older client) claims nothing already handed back.
        assert await srl.pending_for(w.pool, session_id="", branch="feat/a") == []
        assert await srl.mark_delivered(w.pool, [row["turn_id"]]) == 0
        # A session replies: no later session is handed the questions again.
        await w.loop.intake(session_id="sess-later", title="reply", body="Juniper's rule.", thread_id=tid)
        assert await srl.pending_for(w.pool, session_id="sess-third", branch="feat/a") == []
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-third") == 0

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_no_questions_is_handed_back_once_in_total(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w, branch="feat/a"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        await w.loop.tick()
        [row] = await srl.pending_for(w.pool, session_id="s1", branch="feat/a")
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="s1") == 1
        assert await srl.pending_for(w.pool, session_id="s2", branch="feat/a") == []

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_a_prompt_too_big_for_the_lane_is_hubs_sizing_not_a_strike(mode, monkeypatch):
    # Q6: the pool refusing the prompt as bigger than the lane (min_ctx_exceeds_class) is Hub's sizing.
    # No strike; Hub halves its prompt budget and the next round is built to it. At the floor it counts.
    async def scenario(w):
        tid = (await send_pr(w, spec="s" * 30000))["thread"]["thread_id"]
        w.durable.replies = [RuntimeError("gpu_pool_unavailable:min_ctx_exceeds_class:65536")]
        w.clock.to_next_hour()
        tick = await w.loop.tick()
        assert tick["strike"] is False and w.loop.budget_chars == 20000
        thread, turns = await state(w, tid)
        assert thread["roundup_failures"] == 0 and thread["status"] == "held"
        assert "Hub's prompt was too big" in turns[-1]["body"] and "20000 characters" in turns[-1]["body"]
        w.durable.replies = [answer((tid, "1. ok?"))]
        w.clock.to_next_hour()
        assert (await w.loop.tick())["asked"] == 1
        assert len(w.durable.submits[-1]["prompt"]) <= 20000
        assert w.loop.status()["budget_chars"] == 20000

    run(mode, monkeypatch, scenario, prompt_max_chars=40000)


def test_the_prompt_never_exceeds_the_budget_however_many_threads_wait():
    # Q6: a total ceiling is enforced before submit: 50 threads at their caps still make one prompt
    # within the budget, the oldest first, the rest held.
    def thread(i):
        return ({"thread_id": f"t{i:011d}", "title": "x" * 300, "session_id": "s", "branch": "b", "pr_number": i + 1,
                 "pr_url": "u", "pr_state": "merged"},
                [{"seq": j, "speaker": "session", "body": "m" * core.BODY_MAX_CHARS,
                  "meta": {"spec": [{"path": "p", "text": "s" * core.SPEC_MAX_CHARS}]}}
                 for j in range(core.MAX_UNREAD_TURNS_PER_THREAD)], [])

    for budget in (srl.MIN_PROMPT_CHARS, 20000, 100000):
        cands = [thread(i) for i in range(core.MAX_HELD_THREADS)]
        sel = core.select_for_prompt(cands, budget_chars=budget, nonce="n")
        assert sel.sections and len(core.roundup_prompt(sel.sections, nonce="n")) <= budget


@pytest.mark.parametrize("mode", BOTH)
def test_questions_stop_going_to_new_sessions_once_orion_closed_the_thread(mode, monkeypatch):
    one_round(monkeypatch)
    # Review finding: a reply that arrives WHILE Orion reads sits before the question turn. The next round
    # reads it as answers and writes a closing note: the questions must not go out again after that.
    async def scenario(w):
        tid = (await send_pr(w, branch="feat/a"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        rid = (await w.loop.tick())["started"]
        await w.loop.intake(session_id="sess-A", title="more", body="Answers ahead of time.", thread_id=tid)
        w.durable.finish(rid, answer((tid, "1. Why hourly?")))
        assert (await w.loop.tick())["asked"] == 1
        [row] = await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a")
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-A") == 1
        assert [r["turn_id"] for r in await srl.pending_for(w.pool, session_id="sess-B", branch="feat/a")] == [row["turn_id"]]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "NO QUESTIONS"))]
        assert (await w.loop.tick())["closed"] == 1
        rows = await srl.pending_for(w.pool, session_id="sess-B", branch="feat/a")
        assert row["turn_id"] not in [r["turn_id"] for r in rows]
        assert await srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-B") == 0

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_questions_delivered_before_per_session_delivery_are_not_handed_out_again(mode, monkeypatch):
    # Review finding: a turn claimed under the old rule (no `delivered_to`) stays delivered after deploy,
    # and a question turn older than REDELIVERY_MAX_AGE_SEC is not handed out again either.
    async def scenario(w):
        tid = (await send_pr(w, branch="feat/a"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. Why hourly?"))]
        await w.loop.tick()
        [row] = await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a")
        assert await srl.mark_delivered(w.pool, [row["turn_id"]]) == 1   # an old client: no session id
        assert await srl.pending_for(w.pool, session_id="sess-B", branch="feat/a") == []
        # Per-session claim, then the turn ages past the limit: no later session gets it.
        tid2 = (await send_pr(w, pr=2, branch="feat/b"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid2, "1. Why B?"))]
        await w.loop.tick()
        [row2] = await srl.pending_for(w.pool, session_id="sess-A", branch="feat/b")
        assert await srl.mark_delivered(w.pool, [row2["turn_id"]], session_id="sess-A") == 1
        assert len(await srl.pending_for(w.pool, session_id="sess-C", branch="feat/b")) == 1
        old = datetime.now(timezone.utc) - timedelta(seconds=srl.REDELIVERY_MAX_AGE_SEC + 3600)
        if w.mode == "fake":
            next(u for u in w.store.turns if u["turn_id"] == row2["turn_id"])["created_at"] = old
        else:
            await w.pool.execute("UPDATE orion_session_turn SET created_at = $2 WHERE turn_id = $1", row2["turn_id"], old)
        assert await srl.pending_for(w.pool, session_id="sess-C", branch="feat/b") == []

    run(mode, monkeypatch, scenario)


@pytest.mark.parametrize("mode", BOTH)
def test_two_hooks_of_one_session_claim_a_turn_once(mode, monkeypatch):
    async def scenario(w):
        tid = (await send_pr(w, branch="feat/a"))["thread"]["thread_id"]
        w.clock.to_next_hour()
        w.durable.replies = [answer((tid, "1. Why hourly?"))]
        await w.loop.tick()
        [row] = await srl.pending_for(w.pool, session_id="sess-A", branch="feat/a")
        got = await asyncio.gather(*(srl.mark_delivered(w.pool, [row["turn_id"]], session_id="sess-A") for _ in range(4)))
        assert sum(got) == 1

    run(mode, monkeypatch, scenario)


def test_the_budget_cut_follows_the_lanes_window_and_comes_back_a_day_later(monkeypatch):
    async def scenario(w):
        assert core.shrunk_budget(100000, "orion_run_failed:gpu_pool_unavailable:min_ctx_exceeds_class:16384",
                                  floor=srl.MIN_PROMPT_CHARS) == 32768
        assert core.shrunk_budget(100000, "orion_run_failed:min_ctx_exceeds_class", floor=srl.MIN_PROMPT_CHARS) == 50000
        assert core.shrunk_budget(10000, "x:min_ctx_exceeds_class:1000", floor=srl.MIN_PROMPT_CHARS) == srl.MIN_PROMPT_CHARS
        await send_pr(w)
        w.durable.replies = [RuntimeError("gpu_pool_unavailable:min_ctx_exceeds_class:16384")]
        w.clock.to_next_hour()
        assert (await w.loop.tick())["strike"] is False and w.loop.budget_chars == 32768
        w.clock.advance(seconds=srl.BUDGET_RESTORE_SEC + 1)
        await w.loop.tick()
        assert w.loop.budget_chars == 100000

    run("fake", monkeypatch, scenario)
