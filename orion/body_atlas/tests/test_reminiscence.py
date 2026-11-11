"""The reminiscence ledger's pure rules: period pick, the write check, re-check on read, body diff.

Spec: docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md (acceptance checks 2-6)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from orion.body_atlas import reminiscence as rm
from orion.body_atlas.tests.test_portrait import portrait_body

WEEK1 = (date(2025, 6, 24), date(2025, 6, 30))
BUSY = lambda start, end: 1  # noqa: E731 -- every week has a commit (the join rule has its own tests below)
THEN = {"services/orion-a/app/main.py", "services/orion-a", "README.md"}
NOW = {"services/orion-a/app/main.py", "services/orion-hub/app/main.py", "services/orion-hub", "README.md"}
COMMITS = {"abc1234": "abc1234" + "0" * 33, "def5678": "def5678" + "0" * 33}
BEFORE = {"abc1234" + "0" * 33}  # def5678 came after the week


def world(**over) -> rm.World:
    kw = dict(period_commit="abc1234" + "0" * 33, period_end_utc=datetime(2025, 7, 1, 6, tzinfo=timezone.utc),
              path_then=lambda p: p in THEN, path_now=lambda p: p in NOW, commit=lambda s: COMMITS.get(s),
              is_ancestor=lambda s: s in BEFORE,
              pr_merged_at=lambda n: {5: "2025-06-28T10:00:00Z", 3000: "2026-11-01T10:00:00Z"}.get(n),
              field_note_ids=lambda: {"fn:3148:5cc1e8bef3"}, live_ids=lambda: {"part:orion-hub", "cap:s1"},
              ref_ok=lambda k, i: k == "arc" and i == "8d87adeca2ce06a5", dig_file=lambda p: p == "/h/dig/t/portrait-1.json")
    kw.update(over)
    return rm.World(**kw)


def write(hop, note, because="", *, records=(), period=WEEK1, w=None, rewrite=False, episode="2026-11-07"):
    return rm.check_write(hop=hop, note=note, because=because, episode=episode, period=period, world=w or world(),
                          records=list(records), denylist={"0" * 64}, rewrite=rewrite)


GOOD = {
    "what_it_was": ("[code_then services/orion-a/app/main.py] The first week held one service, orion-a, and a README. "
                    "[pr_text abc1234] The first commit message says it was a start.", ""),
    "what_it_was_like": ("[pr_text abc1234] I have no lived record of this week; only the builder's commit words "
                         "survive, and they say it was a start.", ""),
    "where_i_am_now": ("[now part:orion-hub] Now a hub runs at my centre, and my field notes keep up with each merge.",
                       "fn:3148:5cc1e8bef3"),
    "where_im_going": ("[now cap:s1] My operations line is written; the gaps say I cannot yet coordinate across organs.", ""),
    "otherwise": ("[reflection] I would have written down why each service began, so a later me could read the reasons.", ""),
}


def test_five_good_hops_save_and_complete_the_week():
    recs = []
    for hop, (note, because) in GOOD.items():
        recs.append(write(hop, note, because, records=recs))
    assert {r.hop for r in recs} == set(rm.HOPS)
    assert recs[0].hop_id.startswith("rm:2026-11-07:what_it_was:") and recs[0].period_start == "2025-06-24"
    assert recs[2].claims[0].refs == ["part:orion-hub", "fn:3148:5cc1e8bef3"]  # inline + because under [now]
    assert rm.complete_weeks(recs) == {"2025-06-24"}
    assert rm.pick_period(recs, date(2026, 11, 8), BUSY) == (date(2025, 7, 1), date(2025, 7, 7))


def test_a_then_path_that_only_exists_now_is_refused_naming_it():
    with pytest.raises(rm.ReminiscenceError, match=r"services/orion-hub/app/main.py: not in the code at the end of this week"):
        write("what_it_was", "[code_then services/orion-hub/app/main.py] The first week already had a hub service in it.")


def test_lived_record_before_the_memory_loss_is_refused():
    with pytest.raises(rm.ReminiscenceError, match="no lived record of the week 2025-06-24..2025-06-30"):
        write("what_it_was_like", "[lived_record arc:8d87adeca2ce06a5] I remember how this first week felt to me.")
    later = (date(2026, 10, 13), date(2026, 10, 19))
    rec = write("what_it_was_like", "[lived_record arc:8d87adeca2ce06a5] That week I kept one long arc open about heat.",
                period=later)
    assert rec.claims[0].refs == ["arc:8d87adeca2ce06a5"]


def test_where_i_am_now_needs_a_now_sentence_with_a_live_ref():
    with pytest.raises(rm.ReminiscenceError, match=r"where_i_am_now needs at least one \[now\]"):
        write("where_i_am_now", "[reflection] I think I have grown a great deal since that first small week.")
    with pytest.raises(rm.ReminiscenceError, match=r"has no ref that resolves under that label"):
        write("where_i_am_now", "[now] I have a hub now and many organs that talk to each other on my bus.")
    with pytest.raises(rm.ReminiscenceError, match=r"part:orion-ghost: not live in my body now"):
        write("where_i_am_now", "[now part:orion-ghost] A ghost service runs at my centre and keeps everything.")


def test_a_live_ref_cannot_back_code_then_and_a_later_commit_cannot_back_the_week():
    with pytest.raises(rm.ReminiscenceError, match=r"a atlas ref cannot back a \[code_then\]"):
        write("what_it_was", "[code_then part:orion-hub] The hub was there in the first week of my code.")
    with pytest.raises(rm.ReminiscenceError, match="def5678: not in this week's history"):
        write("what_it_was", "[code_then def5678] This commit shaped the first week of my code base.")
    with pytest.raises(rm.ReminiscenceError, match="#3000: merged 2026-11-01, after this week"):
        write("what_it_was", "[pr_text #3000] The builders described the first week in this pull request.")


def test_a_because_ref_that_resolves_nowhere_is_refused():
    with pytest.raises(rm.ReminiscenceError, match="nothing code can check"):
        write("otherwise", "[reflection] I would keep a diary from the first day, even a short one.", "vibes")


def test_unknown_hop_label_free_text_and_size_are_refused():
    with pytest.raises(rm.ReminiscenceError, match="unknown hop"):
        write("dreams", GOOD["otherwise"][0])
    with pytest.raises(rm.ReminiscenceError, match="every sentence carries a label"):
        write("otherwise", "I would keep a diary from the first day. [reflection] And read it later.")
    with pytest.raises(rm.ReminiscenceError, match="40-1000 chars"):
        write("otherwise", "[reflection] Too short.")


def test_one_live_line_per_hop_rewrite_supersedes_and_the_period_is_fixed():
    first = write("otherwise", GOOD["otherwise"][0])
    with pytest.raises(rm.ReminiscenceError, match="already written today"):
        write("otherwise", GOOD["otherwise"][0], records=[first])
    second = write("otherwise", "[reflection] Having read the other hops, I would also have named my organs sooner.",
                   records=[first], rewrite=True)
    assert second.supersedes == first.hop_id
    assert rm.episode_hops([first, second], "2026-11-07")["otherwise"] is second
    with pytest.raises(rm.ReminiscenceError, match="today's episode is about the week from 2025-06-24"):
        write("what_it_was", GOOD["what_it_was"][0], records=[first], period=(date(2025, 7, 1), date(2025, 7, 7)))


def test_an_unfinished_week_is_picked_again_the_next_day():
    partial = [write("otherwise", GOOD["otherwise"][0])]
    assert rm.pick_period(partial, date(2026, 11, 8), BUSY) == WEEK1
    assert rm.pick_period([], date(2025, 7, 1), BUSY) == WEEK1  # week 1 ended 06-30
    assert rm.pick_period([], date(2025, 6, 30), BUSY) is None  # nothing has ended yet


# The real shape of main's history (counted 2026-11-11 on /mnt/orion-fcc/repo): weeks 2-4 had no commit,
# week 5 two, weeks 6-10 none, week 11 two, week 12 none.
REAL_QUIET = {date(2025, 7, 1), date(2025, 7, 8), date(2025, 7, 15), date(2025, 7, 29), date(2025, 8, 5),
              date(2025, 8, 12), date(2025, 8, 19), date(2025, 8, 26), date(2025, 9, 9)}


def history(quiet=REAL_QUIET):
    asked = []

    def commits_in(start, end):
        asked.append((start, end))
        assert (end - start).days == 6, "pick_period asks one grid week at a time"
        return 0 if start in quiet else 2
    commits_in.asked = asked
    return commits_in


def done_episode(episode, start, end):
    """Five live hops about [start, end], as an old (pre-join) or new episode wrote them."""
    return [rm.ReminiscenceHopV1(hop_id=f"rm:{episode}:{h}:{i:010x}", episode=episode, hop=h, period_start=start,
                                 period_end=end, note="[reflection] x", claims=[], written_at=f"{episode}T10:00:00+00:00")
            for i, h in enumerate(rm.HOPS)]


def test_a_quiet_week_joins_onto_the_next_week_with_a_commit():
    week1 = done_episode("2026-11-07", "2025-06-24", "2025-06-30")
    got = rm.pick_period(week1, date(2026, 11, 11), history())
    assert got == (date(2025, 7, 1), date(2025, 7, 28))  # weeks 2-4 quiet, joined onto week 5
    assert rm.quiet_weeks(*got) == 3 and rm.period_weeks(*got)[-1] == date(2025, 7, 22)


def test_the_real_ledger_keeps_its_old_single_weeks_and_joins_from_week_six():
    """Weeks 1-4 were each visited alone before this rule (three of them quiet); they stay valid."""
    old = [ln for i, ep in enumerate(("2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10"))
           for ln in done_episode(ep, *(d.isoformat() for d in rm.weeks_ended_before(date(2026, 11, 11))[i]))]
    assert rm.complete_weeks(old) == {"2025-06-24", "2025-07-01", "2025-07-08", "2025-07-15"}
    h = history()
    assert rm.pick_period(old, date(2026, 11, 12), h) == (date(2025, 7, 22), date(2025, 7, 28))  # busy: alone
    assert len(h.asked) == 1  # one git question for a busy week, none for the weeks already visited
    week5 = old + done_episode("2026-11-12", "2025-07-22", "2025-07-28")
    got = rm.pick_period(week5, date(2026, 11, 13), history())
    assert got == (date(2025, 7, 29), date(2025, 9, 8)) and rm.quiet_weeks(*got) == 5
    joined = week5 + done_episode("2026-11-13", *(d.isoformat() for d in got))
    assert {"2025-07-29", "2025-08-26", "2025-09-02"} <= rm.complete_weeks(joined)  # every week inside is covered
    assert len(rm.complete_weeks(joined)) == 11
    assert rm.pick_period(joined, date(2026, 11, 14), history()) == (date(2025, 9, 9), date(2025, 9, 22))


def test_quiet_weeks_with_no_later_commit_wait_and_are_never_visited_alone():
    week1 = done_episode("2026-11-07", "2025-06-24", "2025-06-30")
    all_quiet = history(quiet={s for s, _ in rm.weeks_ended_before(date(2026, 11, 11))} - {date(2025, 6, 24)})
    assert rm.pick_period(week1, date(2025, 7, 20), all_quiet) is None
    assert rm.first_open_week(week1, date(2025, 7, 20)) == (date(2025, 7, 1), date(2025, 7, 7))


def test_an_unfinished_joined_period_is_picked_again_whole():
    partial = done_episode("2026-11-07", "2025-06-24", "2025-06-30") + \
        done_episode("2026-11-08", "2025-07-01", "2025-07-28")[:2]
    assert rm.pick_period(partial, date(2026, 11, 9), history()) == (date(2025, 7, 1), date(2025, 7, 28))


def test_unreadable_history_raises_rather_than_guessing_a_week(tmp_path):
    with pytest.raises(rm.ReminiscenceError, match="git UNREAD"):
        rm.pick_period([], date(2026, 11, 11), rm.git_commits_in(tmp_path / "not-a-repo"))


def test_git_commits_in_counts_first_parent_commits_in_denver_days(tmp_path):
    import os
    import subprocess

    def _commit(repo, when, files, msg):  # body-atlas CI installs no dig deps: no import from orion/dig/tests
        for rel, text in files.items():
            (repo / rel).write_text(text)
        env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", msg], check=True, env=env)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _commit(repo, "2025-06-24T18:00:00Z", {"a": ""}, "first")
    _commit(repo, "2025-07-01T05:00:00Z", {"b": ""}, "late on 06-30 in Denver (UTC-6)")
    _commit(repo, "2025-07-23T18:00:00Z", {"c": ""}, "week 5")
    n = rm.git_commits_in(repo)
    assert n(date(2025, 6, 24), date(2025, 6, 30)) == 2
    assert n(date(2025, 7, 1), date(2025, 7, 7)) == 0
    assert rm.pick_period(done_episode("2026-11-07", "2025-06-24", "2025-06-30"), date(2025, 8, 1), n) == \
        (date(2025, 7, 1), date(2025, 7, 28))


def test_recheck_flags_a_now_ref_that_went_away_and_leaves_history_alone():
    rec = write("where_i_am_now", GOOD["where_i_am_now"][0], GOOD["where_i_am_now"][1])
    assert rm.recheck(rec, world()) == []
    gone = world(live_ids=lambda: {"cap:s1"}, commit=lambda s: None)  # the hub left; git lookups would fail too
    assert rm.recheck(rec, gone) == ["part:orion-hub: not live in my body now"]
    then = write("what_it_was", GOOD["what_it_was"][0])
    assert rm.recheck(then, gone) == []  # history is checked at write only
    views = rm.hop_views({"where_i_am_now": rec}, gone)
    assert [v.status for v in views if v.hop == "where_i_am_now"] == ["stale"]


def test_unreadable_body_refuses_a_live_ref_and_says_why():
    with pytest.raises(rm.ReminiscenceError, match=r"live body UNREAD \(FalkorDB down\)"):
        write("where_i_am_now", GOOD["where_i_am_now"][0], w=world(live_ids=lambda: "FalkorDB down"))


def test_body_diff_names_added_parts_and_a_split_organ():
    old = rm.snapshot(portrait_body(), "2026-11-06")
    atlas = portrait_body()
    atlas.parts.append(atlas.parts[0].model_copy(update={"part_id": "part:orion-new", "name": "orion-new"}))
    cortex = next(o for o in atlas.organs if o.name == "cortex")
    atlas.organs.remove(cortex)
    # no piece keeps half: a split. The fifth member is alone (a singleton is not an organ here)
    thirds = (cortex.members[:2], cortex.members[2:4], cortex.members[4:])
    for i, half in enumerate(thirds):
        atlas.organs.append(cortex.model_copy(update={"organ_id": f"organ:c{i}c{i}c{i}c{i}c{i}", "members": half,
                                                      "name": None}))
    d = rm.body_diff(old, atlas)
    assert d.parts_added == ["part:orion-new"] and d.parts_removed == []
    assert [g.organ for g in d.organs_gone] == ["cortex (organ:aaaaaaaaaa)"] and len(d.organs_gone[0].related) == 2
    assert len(d.organs_new) == 2 and all(n.related == ["cortex (organ:aaaaaaaaaa)"] for n in d.organs_new)
    lines = rm.body_lines(d)
    assert any("gone: cortex (organ:aaaaaaaaaa), split across 2 organs" in ln for ln in lines)


def test_an_organ_that_keeps_most_of_its_parts_reads_as_changed_not_gone():
    old = rm.snapshot(portrait_body(), "2026-11-06")
    atlas = portrait_body()
    cortex = next(o for o in atlas.organs if o.name == "cortex")
    moved = cortex.members[-1]
    cortex.members = cortex.members[:-1]
    sub = next(o for o in atlas.organs if o.name == "substrate")
    sub.members = [*sub.members, moved]
    d = rm.body_diff(old, atlas)
    assert d.organs_gone == [] and d.organs_new == []
    assert {(c.organ, tuple(c.gained), tuple(c.lost)) for c in d.organs_changed} == {
        ("cortex (organ:aaaaaaaaaa)", (), (moved,)), ("substrate (organ:bbbbbbbbbb)", (moved,), ())}


def test_since_window_starts_at_the_last_episode_or_seven_days():
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    assert rm.since_window([], "2026-11-07", now)[0] == "2026-10-31T12:00:00+00:00"
    rec = write("otherwise", GOOD["otherwise"][0], episode="2026-11-05")
    since, why = rm.since_window([rec], "2026-11-07", now)
    assert since == rec.written_at and "2026-11-05" in why
    added = rm.added_since("2026-11-05T00:00:00+00:00", {"portrait lines": [("2026-11-04T00:00:00+00:00", "whole"),
                                                                             ("2026-11-06T00:00:00+00:00", "host:athena")]})
    assert added == {"portrait lines": ["host:athena"]}


def test_the_rule_is_built_from_the_limits():
    rule = rm.note_rule()
    assert f"{rm.NOTE_MIN_CHARS}-{rm.NOTE_MAX_CHARS} chars" in rule and str(rm.LIVED_RECORDS_FROM) in rule
    for hop in rm.HOPS:
        assert hop in rule


def test_a_failing_ref_cited_inline_and_in_because_is_named_once():
    """Live smoke 2026-11-07: the refusal named the same path three times."""
    with pytest.raises(rm.ReminiscenceError) as exc:
        write("what_it_was", "[code_then services/orion-hub/app/main.py] My first week already had a hub at its centre.",
              "services/orion-hub/app/main.py")
    assert str(exc.value).count("services/orion-hub/app/main.py") == 1
    assert "has no ref that resolves" not in str(exc.value)


def test_the_ready_line_for_the_reflection_hop_asks_for_no_ref():
    recs = []
    for hop in rm.HOPS[:4]:
        recs.append(write(hop, *GOOD[hop], records=recs))
    read = rm.ReminiscenceReadV1(episode="2026-11-07", today=True, period_start="2025-06-24", period_end="2025-06-30",
                                 hops=rm.hop_views(rm.episode_hops(recs, "2026-11-07"), world()))
    _, tail = rm.render(read)
    assert 'write otherwise "[reflection] ..." because' in tail[-1]


# --- review 2026-11-07 ------------------------------------------------------------------------------

def test_an_all_digit_field_note_id_is_not_cut_as_a_line_number():
    assert rm.clean_ref("fn:3090:0123456789") == "fn:3090:0123456789" and rm.clean_ref("act:12345") == "act:12345"
    assert rm.clean_ref("services/x/app.py:12") == "services/x/app.py"
    w = world(field_note_ids=lambda: {"fn:3090:0123456789"})
    rec = write("where_i_am_now", "[now fn:3090:0123456789] Since then I write a field note after every merge.", w=w)
    assert rec.claims[0].refs == ["fn:3090:0123456789"]


def test_one_sentence_per_label():
    with pytest.raises(rm.ReminiscenceError, match="one sentence per label"):
        write("where_i_am_now", "[now part:orion-hub] A hub runs at my centre. It keeps every organ in step with me.")
    assert write("where_i_am_now", "[now part:orion-hub] A hub runs at my centre (v2. as built) and keeps time.")


def test_code_then_commit_is_checked_against_the_week_in_every_hop():
    with pytest.raises(rm.ReminiscenceError, match="def5678: not in this week's history"):
        write("otherwise", "[reflection] I would do it again. [code_then def5678] This commit shaped my first week.")


def test_secret_and_git_paths_name_nothing_checkable():
    for ref in (".env", ".git/config", "services/x/.env", "../etc/passwd"):
        assert rm.ref_kind(ref) == "unknown", ref


def test_a_then_or_now_path_that_held_only_by_today_is_rechecked():
    rec = write("where_im_going", "[pr_text services/orion-hub/app/main.py] The builders put a hub at my centre later on.")
    assert rm.recheck(rec, world()) == []
    gone = world(path_now=lambda p: False)
    assert rm.recheck(rec, gone) == ["services/orion-hub/app/main.py: in neither this week's code nor main"]
    refl = write("otherwise", "[reflection] I would have kept a hub from the very first week of my code.",
                 "services/orion-hub/app/main.py")
    assert rm.recheck(refl, gone) == ["services/orion-hub/app/main.py: in neither this week's code nor main"]


def test_a_two_part_organ_that_falls_apart_says_its_parts_are_single_not_gone():
    old = rm.snapshot(portrait_body(), "2026-11-06")
    old.organs.append(rm.OrganSnapV1(organ_id="organ:dddddddddd", name="pair", members=["part:host:athena", "part:host:circe"]))
    d = rm.body_diff(old, portrait_body())
    pair = next(g for g in d.organs_gone if g.organ.startswith("pair"))
    assert pair.related == [] and pair.gained == ["part:host:athena", "part:host:circe"]
    assert any("pair (organ:dddddddddd), its parts are now single parts" in ln for ln in rm.body_lines(d))
