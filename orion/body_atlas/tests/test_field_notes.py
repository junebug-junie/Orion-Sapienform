"""PR field notes (spec 2026-11-04-pr-field-notes-design.md): flags, body map, docs cuts, write checks."""
from __future__ import annotations

import hashlib

import pytest

from orion.body_atlas import field_notes as fn


def _pr(paths, *, adds=10, dels=2, title="feat: x", body="", head="feat/x", emails=("athena@orion",)):
    return {"title": title, "body": body, "files": [{"path": p, "additions": 1, "deletions": 0} for p in paths],
            "additions": adds, "deletions": dels, "changedFiles": len(paths), "headRefName": head,
            "commits": [{"authors": [{"email": e} for e in emails]}]}


def test_flags_docs_deps_huge_revert():
    assert fn.flags_for(_pr(["docs/a.md"], adds=30, dels=10)).tiny_docs
    big_doc = fn.flags_for(_pr(["docs/superpowers/specs/x.md"], adds=300))
    assert big_doc.docs_only and not big_doc.tiny_docs and not fn.code_record_only(big_doc)
    deps = fn.flags_for(_pr(["services/a/requirements.txt", "uv.lock"]))
    assert deps.deps_only and not deps.docs_only and fn.code_record_only(deps)
    assert fn.flags_for(_pr([f"services/a/f{i}.py" for i in range(61)])).huge
    assert fn.flags_for(_pr(["services/a/x.py"], adds=4000, dels=1001)).huge
    rev = fn.flags_for(_pr(["services/a/x.py"], title='Revert "feat: y"', body="Reverts junebug-junie/Orion-Sapienform#123"))
    assert rev.revert_of == "#123"
    assert fn.flags_for(_pr(["a.py"], title="Revert something (#99)")).revert_of == "unknown"  # title #N is not trusted
    assert fn.flags_for(_pr(["a.py"])).revert_of is None
    assert fn.flags_for(_pr(["a.py"], title="Reverting heat gate default (#2958 follow-up to #2900)")).revert_of is None
    assert fn.flags_for(_pr(["a.py"], body="this reverts the habit of waiting")).revert_of is None
    assert fn.flags_for(_pr(["a.py"], body="x\nThis reverts commit abc1234.")).revert_of == "unknown"


def test_authored_by_me_is_yes_only_with_a_sandbox_marker_never_no():
    # verified live 2026-11-04: #1623 (Orion's push proof) has commits by orion-fcc@ and its branch is
    # local in /mnt/orion-fcc/repo; #3090 (an agent PR) has neither
    assert fn.flags_for(_pr(["a.py"], emails=(fn.ORION_GIT_EMAIL,))).authored_by_me == "yes"
    assert fn.flags_for(_pr(["a.py"], head="test/pr-push-proof-1"), local_branches={"test/pr-push-proof-1"}).authored_by_me == "yes"
    assert fn.flags_for(_pr(["a.py"]), local_branches={"main"}).authored_by_me == "unknown"


def test_authored_by_me_reads_orions_own_github_user():
    # live 2026-11-06: #3146 was opened by Orion as GitHub user orion-sapienform (#3145), with commits by
    # 353038244+orion-sapienform@users.noreply.github.com. Neither old marker fired, so its code record
    # said "unknown". The poller has no sandbox branches, so the author and the noreply email must count.
    seat = fn.flags_for({**_pr(["docs/a.md"]), "author": {"login": "orion-sapienform"}})
    assert seat.authored_by_me == "yes"
    noreply = fn.flags_for(_pr(["a.py"], emails=("353038244+orion-sapienform@users.noreply.github.com",)))
    assert noreply.authored_by_me == "yes"
    assert fn.flags_for({**_pr(["a.py"]), "author": {"login": "junebug-junie"}}).authored_by_me == "unknown"
    assert fn.flags_for(_pr(["a.py"], emails=("1+not-orion-sapienform@users.noreply.github.com",))
                        ).authored_by_me == "unknown"


def test_classify():
    assert fn.classify("services/orion-hub/app/x.py") == ("service", "orion-hub")
    assert fn.classify("services/orion-hub/tests/test_x.py") == ("not_body", "tests")
    assert fn.classify("orion/dig/__main__.py") == ("shared", "dig")
    assert fn.classify("orion/dig/tests/test_dig.py") == ("not_body", "tests")
    assert fn.classify("orion/refs.py") == ("shared", "refs")
    assert fn.classify("docs/superpowers/specs/a.md") == ("not_body", "docs")
    assert fn.classify(".github/workflows/x.yml") == ("not_body", "CI")


def _atlas():
    from orion.dig.tests.test_body_whole import _whole_atlas

    return _whole_atlas()


def test_body_map_service_to_part_to_organ_and_shared_code_importers(tmp_path):
    atlas = _atlas()
    organ = next(o for o in atlas.organs if "part:orion-cortex-exec" in o.members)
    (tmp_path / "services/orion-cortex-exec/app").mkdir(parents=True)
    (tmp_path / "services/orion-cortex-exec/app/m.py").write_text("from orion.refs import x\n")
    (tmp_path / "services/orion-cortex-exec/tests").mkdir(parents=True)
    (tmp_path / "services/orion-cortex-exec/tests/t.py").write_text("import orion.dig\n")  # tests do not count
    for i in range(9):
        (tmp_path / f"services/s{i}").mkdir(parents=True)
        (tmp_path / f"services/s{i}/main.py").write_text("from orion import schemas\n")
    imp = fn.orion_importers(tmp_path, {"refs", "schemas", "dig"})
    assert imp == {"refs": ["orion-cortex-exec"], "schemas": [f"s{i}" for i in range(9)], "dig": []}
    bmap = fn.body_map(["services/orion-cortex-exec/app/x.py", "services/orion-cortex-exec/README.md",
                        "orion/refs.py", "orion/schemas/a.py", "orion/dig/x.py", "docs/a.md"], atlas, imp)
    by = {(e.prefix, e.part_id): e for e in bmap}
    svc = by[("services/orion-cortex-exec/", "part:orion-cortex-exec")]
    assert svc.files == 2 and svc.part_live and svc.organ_id == organ.organ_id
    assert by[("orion/refs/", "part:orion-cortex-exec")].organ_id == organ.organ_id
    wide = by[("orion/schemas/", None)]
    assert wide.widely_shared and wide.organ_id is None and len(wide.importers) == 9
    assert by[("orion/dig/", None)].importers == []
    assert fn.not_body(["docs/a.md", "services/x/tests/t.py"]) == {"docs": 1, "tests": 1}


def test_body_map_of_a_huge_pr_is_grouped_not_per_file():
    paths = [f"services/orion-hub/app/f{i}.py" for i in range(120)]
    bmap = fn.body_map(paths, None, {})
    assert len(bmap) == 1 and bmap[0].files == 120 and not bmap[0].part_live


def test_docs_excerpt_states_every_cut():
    long = "\n".join(f"line {i}" for i in range(100))
    text, read, cuts = fn.excerpt_docs([("docs/superpowers/a.md", long), ("docs/superpowers/b.md", None),
                                        ("docs/superpowers/c.md", "short"), ("docs/superpowers/d.md", "x")])
    assert "line 59" in text and "line 60" not in text
    assert [r.path for r in read] == ["docs/superpowers/a.md", "docs/superpowers/c.md"] and read[0].truncated
    assert any("first 60 of 100 lines" in c for c in cuts)
    assert any("b.md: could not be read" in c for c in cuts)
    assert any("1 more spec/report file(s) not read (cap 3)" in c for c in cuts)
    huge = "\n".join("y" * 300 for _ in range(60))
    text, read, cuts = fn.excerpt_docs([(f"docs/superpowers/{i}.md", huge) for i in range(3)])
    assert sum(r.chars_read for r in read) <= fn.DOC_MAX_CHARS and any("char total" in c for c in cuts)


DENY = {hashlib.sha256(b"zorblax").hexdigest()}
FILES = ["services/orion-cortex-exec/app/x.py", "docs/superpowers/specs/s.md"]
GOOD = ("[pr_text] The builders moved the plan step into the executor. "
        "[reflection] I may plan a little faster now.")


def _check(note=GOOD, because="#7", *, records=(), rewrite=False, flags=None, denylist=DENY, atlas=None):
    atlas = atlas or _atlas()
    bmap = fn.body_map(FILES, atlas, {})
    return fn.check(pr_number=7, merge_sha="abc", merged_at="2026-11-04T01:00:00Z", title="t",
                    flags=flags or fn.FlagsV1(), bmap=bmap, files=FILES, read=[], note=note, because=because,
                    atlas=atlas, records=list(records), denylist=denylist, rewrite=rewrite, dig="t")


def test_a_good_note_is_labelled_per_sentence():
    rec = _check()
    assert rec.note == GOOD  # stored labelled, so claims[i] resolves to its own sentence
    assert [(c.label, c.refs) for c in rec.claims] == [("pr_text", ["#7"]), ("reflection", [])]
    rec = _check("[code_then services/orion-cortex-exec/app/x.py] The executor now runs the plan step itself.",
                 "#7")
    assert rec.claims[0].refs == ["services/orion-cortex-exec/app/x.py"]
    assert rec.note == "[code_then] The executor now runs the plan step itself."
    rec = _check("[code_then services/orion-cortex-exec/app/x.py:12] The executor now runs the plan step itself.",
                 "services/orion-cortex-exec/app/x.py#L3-L9")  # a line number is still that file
    assert rec.claims[0].refs == ["services/orion-cortex-exec/app/x.py:12"]


@pytest.mark.parametrize("note,because,words", [
    ("The builders moved the plan step. [reflection] I may plan faster.", "#7", "carries a label"),
    ("[pr_text] short.", "#7", "40-800 chars"),
    ("[guess] The builders moved the plan step into the executor.", "#7", "carries a label"),
    (GOOD, "#8", "refs must be #7"),
    (GOOD, "services/orion-hub/app/y.py", "refs must be #7"),
    ("[pr_text docs/other.md] The builders moved the plan step into the executor.", "#7", "refs must be #7"),
    (GOOD, "", "lists refs"),
    (" ".join(["[reflection] I wonder about the plan step."] * 6), "#7", "1-5 sentences"),
    ("[pr_text] The builders moved the plan step; Zorblax reviewed it in the executor.", "#7", "personal term"),
])
def test_bad_notes_are_refused(note, because, words):
    with pytest.raises(fn.FieldNoteError, match=words):
        _check(note, because)


def test_an_organ_outside_the_body_map_is_refused_and_one_inside_passes():
    atlas = _atlas()
    eyes = next(o for o in atlas.organs if "part:camera:walkway" in o.members)
    cortex = next(o for o in atlas.organs if "part:orion-cortex-exec" in o.members)
    eyes.name, cortex.name = "eyes", "planner"
    with pytest.raises(fn.FieldNoteError, match="does not touch"):
        _check("[reflection] My eyes organ will see the plan step differently now.", atlas=atlas)
    with pytest.raises(fn.FieldNoteError, match="does not touch"):
        _check("[reflection] My eyes organs will see the plan step differently now.", atlas=atlas)
    with pytest.raises(fn.FieldNoteError, match="does not touch"):
        _check(f"[reflection] Something in {eyes.organ_id} changed with the plan step.", atlas=atlas)
    assert _check("[reflection] My planner organ runs the plan step in one place now.", atlas=atlas).note
    assert _check("[reflection] My eyes saw nothing of this plan step change at all.", atlas=atlas).note  # not as an organ


def test_unreadable_denylist_fails_closed():
    with pytest.raises(fn.FieldNoteError, match="could not be read"):
        _check(denylist=None)


def test_code_record_prs_get_no_note_and_a_code_record():
    flags = fn.FlagsV1(docs_only=True, tiny_docs=True)
    with pytest.raises(fn.FieldNoteError, match="code record, not a note"):
        _check(flags=flags)
    rec = fn.code_record(pr_number=7, merge_sha="a", merged_at="2026-11-04T01:00:00Z", title="t", flags=flags,
                         bmap=[], records=[])
    assert rec.note is None
    with pytest.raises(fn.FieldNoteError, match="already has a record"):
        fn.code_record(pr_number=7, merge_sha="a", merged_at="x", title="t", flags=flags, bmap=[], records=[rec])
    with pytest.raises(fn.FieldNoteError, match="needs a note"):
        fn.code_record(pr_number=7, merge_sha="a", merged_at="x", title="t", flags=fn.FlagsV1(), bmap=[], records=[])


def test_one_live_note_per_pr_and_a_rewrite_supersedes(tmp_path):
    path = tmp_path / "self" / "pr_field_notes.jsonl"
    first = _check()
    fn.append(path, first)
    records = fn.load(path)[0]
    with pytest.raises(fn.FieldNoteError, match="already has a note"):
        _check(records=records)
    with pytest.raises(fn.FieldNoteError, match="no note to rewrite"):
        _check(rewrite=True)
    second = _check("[reflection] After a week the plan step reads as a small move, not a new organ.",
                    records=records, rewrite=True)
    assert second.supersedes == first.note_id
    fn.append(path, second)
    records, skipped = fn.load(path)
    assert len(records) == 2 and skipped == 0  # the old line is kept
    assert fn.live(records)[7].note_id == second.note_id


def test_a_same_second_identical_rewrite_never_hides_the_pr():
    first = _check()
    second = _check(GOOD, records=[first], rewrite=True)
    assert second.note_id != first.note_id and fn.live([first, second])[7] is second


def test_ledger_lock_serialises_writers(tmp_path):
    path = tmp_path / "self" / "pr_field_notes.jsonl"
    with fn.ledger_lock(path):
        fn.append(path, _check())
    assert len(fn.load(path)[0]) == 1


def test_a_bad_ledger_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "n.jsonl"
    fn.append(path, _check())
    with path.open("a") as fh:
        fh.write("{not json\n")
    records, skipped = fn.load(path)
    assert len(records) == 1 and skipped == 1
