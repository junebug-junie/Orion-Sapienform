"""Operator-curated turns kept out of topic-foundry's training corpus.

Context (2026-08-30): AI Town material reaches the main corpus through Orion's
own turns. Every layer of separation is intact -- the AI Town graph, dataset
and source table are all correctly separate -- and none of it helps, because
the polluted rows are ordinary `hub_ws`/`orion_journal` chat with no marker
distinguishing them. Two such rows, 0.7% of a 273-row corpus, produced a
concept that reached the top three by betweenness in the whole atlas and was
re-induced across five separate runs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.pipelines.chat_corpus_builder.exclusions import (  # noqa: E402
    _ENV_VAR,
    load_excluded_turn_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_CONFIG = REPO_ROOT / "config" / "corpus" / "topic_foundry_excluded_turns.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "excluded.yaml"
    target.write_text(body)
    return target


def test_the_checked_in_config_parses_and_is_not_empty():
    """A curated list that silently parses to nothing is the same as no
    filter at all, which is the failure this whole mechanism exists to end."""
    ids = load_excluded_turn_ids(LIVE_CONFIG)
    assert ids, "the shipped exclusion list must not be empty"
    assert "324c0b06-36d6-4362-a547-29711c239857" in ids
    assert "de4a86e2-aaee-4bdc-a859-48b88867d48d" in ids


def test_entries_may_omit_the_reason_field(tmp_path):
    """A hand-edit that drops the metadata must still exclude the turn --
    silently doing nothing would be the worst outcome of a typo."""
    cfg = _write(tmp_path, 'excluded_turn_ids:\n  - id: "abc"\n  - "bare-string-id"\n')
    assert load_excluded_turn_ids(cfg) == frozenset({"abc", "bare-string-id"})


def test_a_missing_config_fails_open(tmp_path):
    """Refusing to train on a config typo turns an operator mistake into a
    silent halt of the whole induction pipeline -- worse than re-learning a
    concept someone wanted dropped."""
    assert load_excluded_turn_ids(tmp_path / "nope.yaml") == frozenset()


@pytest.mark.parametrize(
    "body",
    [
        "excluded_turn_ids: not-a-list\n",
        "excluded_turn_ids:\n",
        "",
        "{{{ not yaml at all",
    ],
)
def test_malformed_config_fails_open_rather_than_raising(tmp_path, body):
    assert load_excluded_turn_ids(_write(tmp_path, body)) == frozenset()


def test_blank_ids_are_dropped(tmp_path):
    cfg = _write(tmp_path, 'excluded_turn_ids:\n  - id: ""\n  - id: "   "\n  - id: "real"\n')
    assert load_excluded_turn_ids(cfg) == frozenset({"real"})


def test_env_var_overrides_the_search(tmp_path, monkeypatch):
    from app.pipelines.chat_corpus_builder import exclusions

    cfg = _write(tmp_path, 'excluded_turn_ids:\n  - id: "from-env"\n')
    monkeypatch.setenv(_ENV_VAR, str(cfg))
    assert exclusions._config_path() == cfg
    assert load_excluded_turn_ids() == frozenset({"from-env"})


def test_config_is_found_from_the_container_layout_too(tmp_path):
    """THE BUG THIS REPLACED. The resolver used a fixed `parents[5]`, which is
    the repo root from
    services/orion-topic-foundry/app/pipelines/chat_corpus_builder/ but clamps
    to `/` from the image's `/app/app/pipelines/chat_corpus_builder/`. In the
    container it therefore resolved to `/config/corpus/...`, found nothing, and
    failed open -- the exclusions would have applied everywhere except the one
    place that matters. Verified against the real image layout, which is
    `/app/app/pipelines/...` (checked live).
    """
    deep = tmp_path / "app" / "app" / "pipelines" / "chat_corpus_builder"
    deep.mkdir(parents=True)
    module_src = Path(
        SERVICE_ROOT / "app" / "pipelines" / "chat_corpus_builder" / "exclusions.py"
    )
    shutil.copy(module_src, deep / "exclusions.py")
    cfg_dir = tmp_path / "app" / "config" / "corpus"
    cfg_dir.mkdir(parents=True)
    shutil.copy(LIVE_CONFIG, cfg_dir / "topic_foundry_excluded_turns.yaml")

    sys.path.insert(0, str(deep))
    try:
        sys.modules.pop("exclusions", None)
        import exclusions as container_module

        resolved = container_module._config_path()
        assert resolved.exists(), f"container layout did not find the config: {resolved}"
        assert container_module.load_excluded_turn_ids(), "must load ids in the image layout"
    finally:
        sys.path.remove(str(deep))
        sys.modules.pop("exclusions", None)


# --- the query must exclude in SQL, not after the fetch ---------------------


def test_the_corpus_query_excludes_in_sql_so_limit_stays_honest():
    """Filtering after the fetch would let excluded turns consume LIMIT slots
    and silently shrink the real training window."""
    import inspect

    from app.pipelines.chat_corpus_builder import repository

    src = inspect.getsource(repository.fetch_chat_turn_rows)
    assert "NOT (id = ANY(%s))" in src
    assert "cached_excluded_turn_ids()" in src
    # the exclusion parameter must be bound BEFORE the limit, matching the
    # placeholder order in the query
    assert src.index("AND NOT (id = ANY(%s))") < src.index("LIMIT %s")
    assert "(start_at, end_at, excluded, limit)" in src


def test_the_query_binds_ids_as_a_parameter_not_interpolated():
    """Turn ids come from a config file an operator edits; interpolating them
    into SQL would make that file an injection surface."""
    import inspect

    from app.pipelines.chat_corpus_builder import repository

    src = inspect.getsource(repository.fetch_chat_turn_rows)
    assert "%s" in src
    assert "f\"\"\"" not in src and ".format(" not in src
