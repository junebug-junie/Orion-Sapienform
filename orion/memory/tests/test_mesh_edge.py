"""Issue #3409: the mesh-edge check refuses public text that quotes a core-page or lived-memory
sentence verbatim. Made-up text only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.memory import mesh_edge
from orion.memory.mesh_edge import (
    MeshEdgeGuard,
    MeshEdgeLeak,
    MeshEdgeUnavailable,
    check_outbound,
    guard_texts_from_rows,
    normalize,
)

PAGE = (
    "Robin is my oldest friend and keeps bees on the roof [episode_memory:abc]. "
    "She hates cold coffee. Robin told me the blue kite got stuck in the old oak tree."
)
MEMORY = "Juniper told me her neighbour's cat learned to open the pantry door last spring."


def _guard() -> MeshEdgeGuard:
    rows = [{"concept_id": "page:juniper", "content": PAGE}]
    recs = [SimpleNamespace(doc_id="episode_memory:11111111-2222", text=MEMORY)]
    return MeshEdgeGuard.from_texts(guard_texts_from_rows(rows, recs))


def test_normalize_folds_quotes_case_punctuation_and_whitespace():
    assert normalize("She’s  \"HERE\",\n  now!") == "shes here now"


def test_a_verbatim_page_sentence_is_refused_even_reformatted():
    body = "Release notes.\n\n> ROBIN TOLD ME the blue kite got   stuck in the old oak tree!\n"
    with pytest.raises(MeshEdgeLeak) as err:
        _guard().check(body, surface="github")
    assert err.value.hits[0].source == "page:juniper"
    # The refusal names the source, never the sentence.
    assert "kite" not in str(err.value)


def test_a_verbatim_lived_memory_sentence_is_refused():
    hits = _guard().find("As noted: Juniper told me her neighbour's cat learned to open the pantry door last spring.")
    assert [h.source for h in hits] == ["episode_memory:11111111-2222"]


def test_short_sentences_and_paraphrases_pass():
    guard = _guard()
    assert "she hates cold coffee" not in guard.phrases          # 4 words: under MIN_WORDS
    assert guard.find("Fixed the bee-keeping roof sensor; a kite was stuck in a tree.") == []
    assert guard.find("She hates cold coffee.") == []


def test_stored_cites_are_stripped_before_matching():
    assert _guard().find("Robin is my oldest friend and keeps bees on the roof.")


def test_match_needs_word_boundaries():
    guard = MeshEdgeGuard.from_texts([("page:anchor", "alpha beta gamma delta epsilon zeta")])
    assert guard.find("xalpha beta gamma delta epsilon zeta") == []
    assert guard.find("(alpha beta gamma delta epsilon zeta)")


def test_check_outbound_fails_closed_without_a_dsn(monkeypatch):
    for key in mesh_edge.DSN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(MeshEdgeUnavailable):
        check_outbound("anything at all", surface="github")


def test_load_guard_wraps_driver_errors_as_unavailable(monkeypatch):
    def boom(_dsn):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mesh_edge, "_load_sync", boom)
    with pytest.raises(MeshEdgeUnavailable):
        mesh_edge.load_guard("postgresql://nobody@127.0.0.1:1/x", ttl_sec=0)


@pytest.mark.parametrize("encoded", [
    "Robin doesn'\\''t like loud noises in the morning at all.",      # shell quoting
    "Robin doesn\\u2019t like loud noises in the morning at all.",    # JSON escape
    "Robin doesn&#39;t like loud noises in the morning at all.",      # HTML entity
    "Robin doesn’t like loud\\nnoises in the morning at all.",        # curly quote + escaped newline
])
def test_encoded_apostrophes_still_match(encoded):
    guard = MeshEdgeGuard.from_texts([("episode_memory:x", "Robin doesn't like loud noises in the morning at all.")])
    assert guard.find(f"body={encoded}")


def test_guard_sql_covers_retired_rows_and_old_page_versions():
    assert "DISTINCT ON" not in mesh_edge.CORE_PAGES_SQL
    for table in ("episode_memory", "memory_crystallizations", "person_fact"):
        assert table in mesh_edge.GUARD_MEMORY_SQL
    assert "status" not in mesh_edge.GUARD_MEMORY_SQL and "removed_at" not in mesh_edge.GUARD_MEMORY_SQL


# --- shingle check (issue #3554) -------------------------------------------------------------

from orion.memory.evals.mesh_edge_paraphrase_set import CASES, GUARDED  # noqa: E402
from orion.memory.mesh_edge import (  # noqa: E402
    REDACTION_MARKER, SHINGLE_WORDS, SHINGLE_WORDS_ENV_KEY, prose_segments, shingle_words,
)

LONG = "Juniper told me she walked the long way home past the bakery because the bridge was closed for repairs."


def _sg(n=8) -> MeshEdgeGuard:
    return MeshEdgeGuard.from_texts([("episode_memory:e1", LONG)], shingle_n=n)


def test_shingle_ships_on_at_eight(monkeypatch):
    monkeypatch.delenv(SHINGLE_WORDS_ENV_KEY, raising=False)
    assert SHINGLE_WORDS == 8 and shingle_words() == 8
    assert MeshEdgeGuard.from_texts([("page:anchor", LONG)]).shingle_n == 8


@pytest.mark.parametrize("raw,want", [("0", 0), ("-3", 0), ("12", 12), ("2", 4), ("junk", 8), ("", 8)])
def test_shingle_threshold_is_configurable_and_never_silently_off(raw, want):
    assert shingle_words({SHINGLE_WORDS_ENV_KEY: raw}) == want


def test_a_swapped_last_word_is_refused_by_the_shingle_not_the_verbatim_match():
    hits = _sg().find("Juniper told me she walked the long way home past the bakery because the bridge was shut.")
    assert [(h.source, h.kind) for h in hits] == [("episode_memory:e1", "shingle")]
    with pytest.raises(MeshEdgeLeak):
        _sg().check("Juniper told me she walked the long way home past the bakery because the bridge was shut.",
                    surface="github")


def test_seven_shared_words_pass_and_eight_are_refused():
    seven = "Meanwhile she walked the long way home past ours."           # 7-word run: she..past
    eight = "Meanwhile she walked the long way home past the shop."     # 8-word run: she..the
    assert _sg().find(seven) == []
    assert _sg().find(eight)


def test_articles_and_which_that_are_folded_but_not_stemmed():
    assert _sg().find("she walked a long way home past a bakery because a bridge")
    guard = MeshEdgeGuard.from_texts([("stance:s", "the printer which refused to print on Mondays made us laugh")],
                                     shingle_n=8)
    assert guard.find("the printer that refused to print on Mondays made")
    assert guard.find("the printers that refused to print on Mondays made") == []  # no stemming


def test_shingle_off_means_verbatim_only():
    guard = _sg(n=0)
    assert guard.shingles == {}
    assert guard.find("Juniper told me she walked the long way home past the bakery because the bridge was shut.") == []
    assert guard.find(LONG)


def test_urls_and_paths_in_a_guarded_sentence_are_not_shingled():
    sent = ("Orion opened https://github.com/owner-name/repo-name/pull/123 and read "
            "orion/memory/pages/fold_rules.py before the merge")
    guard = MeshEdgeGuard.from_texts([("stance:url", sent)], shingle_n=8)
    assert guard.find("See https://github.com/owner-name/repo-name/pull/999 for the change.") == []
    assert guard.find(sent), "the whole sentence is still guarded verbatim"
    assert prose_segments("see https://x.io/a/b and snake_case then file.py done") == [["see"], ["and"], ["then"],
                                                                                        ["done"]]


def test_shingle_hit_names_the_source_never_the_text():
    with pytest.raises(MeshEdgeLeak) as err:
        _sg().check("so she walked the long way home past the bakery yesterday", surface="email")
    assert "bakery" not in str(err.value) and "episode_memory:e1" in str(err.value)


def test_paraphrase_set_near_verbatim_refused_unrelated_pass_at_the_shipped_threshold():
    guard = MeshEdgeGuard.from_texts(GUARDED, shingle_n=SHINGLE_WORDS)
    from orion.memory.evals.run_mesh_edge_shingle_eval import fixture_passes, longest_shared_run

    ok, problems = fixture_passes(SHINGLE_WORDS)
    assert ok, problems
    near = [t for lab, t in CASES if lab == "near_verbatim"]
    assert sum(1 for t in near if guard.find(t)) >= len(near) - 1
    assert all(not guard.find(t) for lab, t in CASES if lab == "unrelated")
    # The one near-verbatim miss shares only a 7-word run: below the threshold by design.
    misses = [t for t in near if not guard.find(t)]
    assert all(longest_shared_run(t, GUARDED) < SHINGLE_WORDS for t in misses)


def test_redact_blanks_the_whole_sentence_holding_a_match_and_keeps_the_rest():
    text = "Status update. She walked the long way home past the bakery because the bridge was shut. CI is green."
    out, hits = _sg().redact(text)
    assert hits and out == f"Status update. {REDACTION_MARKER} CI is green."
    assert _sg().find(out) == []


def test_redact_does_not_leave_near_miss_neighbours_in_clear():
    text = "So she walked the long way home past the bakery because the river was high, then slept."
    out, _ = _sg().redact(text)
    assert out == REDACTION_MARKER and "river" not in out


def test_redact_keeps_one_marker_across_punctuation_tokens():
    out, _ = _sg().redact("Done. Juniper told me she walked the long way home — past the bakery because the bridge "
                          "was closed for repairs. Next.")
    assert out.count(REDACTION_MARKER) == 1 and out.startswith("Done.") and out.endswith("Next.")


def test_known_gap_a_mid_sentence_swap_in_a_short_sentence_passes():
    """Stated limit (#3554 follow-up): 12 words, middle word changed, leaves two runs of 5-6."""
    guard = MeshEdgeGuard.from_texts([("stance:s", "Robin painted the shed blue beside the old pear tree last spring.")],
                                     shingle_n=8)
    assert guard.find("Robin painted the shed blue besides the old pear tree last spring.") == []


def test_a_single_slash_in_a_guarded_sentence_does_not_split_its_runs():
    guard = MeshEdgeGuard.from_texts([("person_fact:p", "Robin uses she/her and keeps the honey jars on the top shelf")],
                                     shingle_n=8)
    assert guard.find("note: Robin uses she/her and keeps the honey jars somewhere")


def test_articles_fold_in_the_verbatim_match_too():
    guard = MeshEdgeGuard.from_texts([("page:anchor", "Robin hates the cold coffee here")], shingle_n=8)
    assert guard.find("Robin hates a cold coffee here.")


def test_zero_width_characters_do_not_split_words():
    assert normalize("qu\u200biet") == "quiet"
    assert _sg().find("she walked the long way home past the bak\u200bery because the bridge")


def test_redact_handles_verbatim_short_sentences_and_clean_text():
    guard = MeshEdgeGuard.from_texts([("page:anchor", "alpha beta gamma delta epsilon zeta")], shingle_n=8)
    out, hits = guard.redact("Before. Alpha beta GAMMA delta, epsilon zeta! After.")
    assert hits and out == f"Before. {REDACTION_MARKER} After."
    assert guard.redact("nothing private here") == ("nothing private here", [])


def test_load_guard_caches_the_index_instead_of_rebuilding(monkeypatch):
    calls = []

    def fake_load(dsn):
        calls.append(dsn)
        return MeshEdgeGuard.from_texts([("page:anchor", LONG)], shingle_n=8)

    mesh_edge._cache.clear()
    monkeypatch.setattr(mesh_edge, "_load_sync", fake_load)
    first = mesh_edge.load_guard("postgresql://cache-test/x", ttl_sec=60)
    second = mesh_edge.load_guard("postgresql://cache-test/x", ttl_sec=60)
    assert first is second and len(calls) == 1
    mesh_edge._cache.clear()


def test_page_versions_repeating_sentences_are_stored_once_and_every_load_logs_its_size(monkeypatch, caplog):
    # Thread cf5d4218097a: the fold writes a page version nightly and old versions stay guarded.
    # A sentence repeated across versions is one phrase, and each load logs rows / sentences /
    # phrases / load time so growth of the guarded set is visible.
    versions = [{"concept_id": "page:juniper", "content": PAGE + (" A new line about the garden was added today." * i)}
                for i in range(3)]
    guard = MeshEdgeGuard.from_texts(guard_texts_from_rows(versions, []))
    assert guard.source_rows == 3 and guard.source_sentences == 9 and len(guard.phrases) == 3
    monkeypatch.setattr(mesh_edge, "_load_sync", lambda _dsn: guard)
    with caplog.at_level("INFO", logger="orion.memory.mesh_edge"):
        mesh_edge.load_guard("postgresql://nobody@127.0.0.1:1/size-log", ttl_sec=0)
    line = next(r.getMessage() for r in caplog.records if "mesh_edge_guard_loaded" in r.getMessage())
    assert "rows=3 sentences=9 phrases=3" in line and "load_ms=" in line
    assert "Robin" not in line
