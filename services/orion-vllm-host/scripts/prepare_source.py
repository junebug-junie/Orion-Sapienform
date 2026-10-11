#!/usr/bin/env python3
"""Assemble pinned sources in a NEW directory; stop on conflicts for inspection.

No CUDA compilation, package install, model download, or service activation.
On a known additive merge conflict, resolve it and commit in that source directory,
then rerun with --resume. Never strips conflict markers automatically.
"""
import argparse
import json
from pathlib import Path
import subprocess

LOCK = Path(__file__).with_name('glm53.lock.json')


def git(root, *args):
    subprocess.run(['git', '-C', str(root), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    root = args.directory.resolve()
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
        git(root, 'init')
        git(root, 'remote', 'add', 'origin', lock['source'])
        git(root, 'fetch', '--filter=blob:none', 'origin', lock['base'])
        git(root, 'checkout', '-b', 'orion-glm53-pinned', lock['base'])
    else:
        top = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True).strip()
        remote = subprocess.check_output(['git', '-C', str(root), 'remote', 'get-url', 'origin'], text=True).strip()
        if Path(top).resolve() != root or remote != lock['source']:
            parser.error('resume requires the dedicated 1Cat-vLLM source checkout')
        status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
        if status:
            parser.error('resolve and commit the pending source merge before resuming')
        git(root, 'merge-base', '--is-ancestor', lock['base'], 'HEAD')
    for patch in lock['merge_order']:
        git(root, 'fetch', '--filter=blob:none', 'origin', patch['commit'])
        git(root, 'merge', '--no-edit', patch['commit'])
    tree = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    if tree != lock['assembled_tree']:
        parser.error('assembled source tree differs from reviewed lock; inspect merge resolutions')
    git(root, 'log', '-1', '--format=%H')
    print('Pinned source stack assembled. CUDA build and model inference remain UNVERIFIED.')


if __name__ == '__main__':
    main()
