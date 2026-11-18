#!/usr/bin/env python3
"""Assemble the locked source without importing newer, incompatible branch bases.

No compilation, package install, model download, or activation. Conflicts stop
for manual inspection: resolve and commit, then rerun with --resume. Resume
accepts only a clean, reviewed intermediate tree, including the final tree.
"""
import argparse
import json
from pathlib import Path
import subprocess

LOCK = Path(__file__).with_name('glm53.lock.json')


def git(root, *args):
    subprocess.run(['git', '-C', str(root), *args], check=True)


def output(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify_history(root, lock):
    git(root, 'merge-base', '--is-ancestor', lock['base'], 'HEAD')
    for commit in lock['forbidden_ancestors']:
        # An absent forbidden object cannot be an ancestor of a complete checkout.
        exists = subprocess.run(['git', '-C', str(root), 'cat-file', '-e', commit],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if exists.returncode:
            continue
        result = subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', commit, 'HEAD'])
        if result.returncode == 0:
            raise ValueError('source includes a forbidden upstream regression; use a new directory')
        if result.returncode != 1:
            raise ValueError('cannot verify source ancestry')


def assemble(root, lock, resume=False):
    root = root.resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        git(root, 'init')
        git(root, 'remote', 'add', 'origin', lock['source'])
        git(root, 'fetch', '--filter=blob:none', 'origin', lock['base'])
        git(root, 'checkout', '-b', 'orion-glm53-pinned', lock['base'])
    else:
        if (Path(output(root, 'rev-parse', '--show-toplevel')).resolve() != root
                or output(root, 'remote', 'get-url', 'origin') != lock['source']):
            raise ValueError('resume requires the dedicated locked source checkout')
    if output(root, 'status', '--porcelain'):
        raise ValueError('resolve and commit pending source changes before resuming')
    for state in ('MERGE_HEAD', 'CHERRY_PICK_HEAD'):
        pending = Path(output(root, 'rev-parse', '--git-path', state))
        if not pending.is_absolute():
            pending = root / pending
        if pending.exists():
            raise ValueError('finish the pending source operation before resuming')
    verify_history(root, lock)
    trees = [lock['base_tree']] + [step['tree'] for step in lock['steps']]
    tree = output(root, 'rev-parse', 'HEAD^{tree}')
    if tree not in trees:
        raise ValueError('source tree is not a reviewed checkpoint; use a new directory')
    for step in lock['steps'][trees.index(tree):]:
        if step['action'] not in ('cherry-pick', 'merge'):
            raise ValueError('unsupported source operation')
        git(root, 'fetch', '--filter=blob:none', 'origin', step['commit'])
        git(root, step['action'], '--no-edit', step['commit'])
        verify_history(root, lock)
        if output(root, 'rev-parse', 'HEAD^{tree}') != step['tree']:
            raise ValueError('source differs from reviewed step; inspect resolution before resuming')
    if output(root, 'rev-parse', 'HEAD^{tree}') != lock['assembled_tree']:
        raise ValueError('assembled source differs from reviewed lock')
    print(output(root, 'rev-parse', 'HEAD'))
    print('Pinned source stack assembled. CUDA build and model inference remain UNVERIFIED.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        assemble(args.directory, json.loads(LOCK.read_text()), args.resume)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
