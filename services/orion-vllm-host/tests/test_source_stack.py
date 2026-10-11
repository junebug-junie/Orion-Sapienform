"""Exercise real Git graphs: pinned patch commits must not import their bases."""
import json
from pathlib import Path
import runpy
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_source.py'
assemble = runpy.run_path(str(SCRIPT))['assemble']


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def commit(root, name):
    (root / name).write_text(name)
    git(root, 'add', name)
    git(root, 'commit', '-qm', name)
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv('GIT_AUTHOR_NAME', 'Test')
    monkeypatch.setenv('GIT_AUTHOR_EMAIL', 'test@example.invalid')
    monkeypatch.setenv('GIT_COMMITTER_NAME', 'Test')
    monkeypatch.setenv('GIT_COMMITTER_EMAIL', 'test@example.invalid')
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '-q')
    base = commit(upstream, 'base')
    base_tree = git(upstream, 'rev-parse', 'HEAD^{tree}')
    bad = commit(upstream, 'regression')
    patch = commit(upstream, 'feature')
    git(upstream, 'checkout', '-qb', 'correct', base)
    git(upstream, 'cherry-pick', patch)
    tree = git(upstream, 'rev-parse', 'HEAD^{tree}')
    lock = dict(source=str(upstream), base=base, base_tree=base_tree,
                forbidden_ancestors=[bad], assembled_tree=tree,
                steps=[dict(action='cherry-pick', commit=patch, tree=tree)])
    return tmp_path / 'destination', lock


def test_cherry_pick_excludes_regression_and_resume_is_idempotent(source):
    root, lock = source
    assemble(root, lock)
    assert (root / 'feature').exists()
    assert not (root / 'regression').exists()
    head = git(root, 'rev-parse', 'HEAD')
    assemble(root, lock, resume=True)
    assert git(root, 'rev-parse', 'HEAD') == head


def test_rejects_old_merge_recipe_even_with_matching_tree(source):
    root, lock = source
    git(root.parent, 'clone', '-q', lock['source'], str(root))
    git(root, 'checkout', lock['steps'][0]['commit'])
    tree = git(root, 'rev-parse', 'HEAD^{tree}')
    lock['steps'][0]['tree'] = lock['assembled_tree'] = tree
    with pytest.raises(ValueError, match='forbidden upstream regression'):
        assemble(root, lock, resume=True)


def test_rejects_unreviewed_resolution(source):
    root, lock = source
    assemble(root, lock)
    commit(root, 'extra-change')
    with pytest.raises(ValueError, match='not a reviewed checkpoint'):
        assemble(root, lock, resume=True)


def test_rejects_dirty_resume(source):
    root, lock = source
    assemble(root, lock)
    (root / 'base').write_text('uncommitted')
    with pytest.raises(ValueError, match='pending source changes'):
        assemble(root, lock, resume=True)


def test_rejects_unexpected_step_tree(source):
    root, lock = source
    lock['steps'][0]['tree'] = '0' * 40
    with pytest.raises(ValueError, match='differs from reviewed step'):
        assemble(root, lock)


def test_resume_from_reviewed_base(source):
    root, lock = source
    git(root.parent, 'clone', '-q', lock['source'], str(root))
    git(root, 'checkout', lock['base'])
    assemble(root, lock, resume=True)
    assert git(root, 'rev-parse', 'HEAD^{tree}') == lock['assembled_tree']


def test_production_lock_uses_six_cherry_picks_then_exl3_merge():
    lock = json.loads(SCRIPT.with_name('glm53.lock.json').read_text())
    assert [s['action'] for s in lock['steps']] == ['cherry-pick'] * 6 + ['merge']
    assert lock['steps'][0]['commit'] == '587fac6eb37b8aee1cd42df76f4af284d30f503d'
    assert lock['steps'][-1]['commit'] == 'b96d43e5c8df201992a181836d631198ef827edf'
    assert lock['forbidden_ancestors'] == ['d4ce513991b79a9cac2a4d72fbe8bcac5f566d5e']
    assert lock['assembled_tree'] == lock['steps'][-1]['tree']


def test_mixed_stack_resumes_after_reviewed_merge_resolution(source):
    root, lock = source
    upstream = Path(lock['source'])
    git(upstream, 'checkout', '-qb', 'loader', lock['base'])
    (upstream / 'feature').write_text('loader')
    git(upstream, 'add', 'feature')
    git(upstream, 'commit', '-qm', 'loader')
    loader = git(upstream, 'rev-parse', 'HEAD')
    git(upstream, 'checkout', 'correct')
    with pytest.raises(subprocess.CalledProcessError):
        git(upstream, 'merge', '--no-edit', loader)
    (upstream / 'feature').write_text('feature and loader')
    git(upstream, 'add', 'feature')
    git(upstream, 'commit', '--no-edit')
    tree = git(upstream, 'rev-parse', 'HEAD^{tree}')
    lock['steps'].append(dict(action='merge', commit=loader, tree=tree))
    lock['assembled_tree'] = tree
    with pytest.raises(subprocess.CalledProcessError):
        assemble(root, lock)
    with pytest.raises(ValueError, match='pending source changes'):
        assemble(root, lock, resume=True)
    (root / 'feature').write_text('feature and loader')
    git(root, 'add', 'feature')
    git(root, 'commit', '--no-edit')
    assemble(root, lock, resume=True)
    assert git(root, 'rev-parse', 'HEAD^{tree}') == tree
    assert not (root / 'regression').exists()
