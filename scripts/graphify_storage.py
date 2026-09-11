#!/usr/bin/env python3
"""Local Graphify bundles. Git holds code; warm storage holds graph and recovery data."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from merge_graphify_json import validate

DEFAULT_ROOT = Path('/mnt/storage-warm/orion-graphify')
BUNDLE = ('graph.json', 'manifest.json', 'GRAPH_REPORT.md', 'graph.html',
          '.graphify_labels.json', '.graphify_learning.json')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def storage_root():
    return Path(os.environ.get('ORION_GRAPHIFY_STORAGE_ROOT', DEFAULT_ROOT)).resolve()


def target(root):
    common = Path(git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir'))
    private = Path(git(root, 'rev-parse', '--absolute-git-dir'))
    key = 'main' if private == common else 'worktrees/' + hashlib.sha256(str(private).encode()).hexdigest()[:20]
    return storage_root() / key / 'graphify-out'


@contextmanager
def locked(directory, *, blocking=True):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.orion-storage.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def hashes(directory, *, bundle=False):
    paths = [directory / name for name in BUNDLE] if bundle else sorted(directory.rglob('*'))
    result = {}
    for path in paths:
        if path.is_file():
            with path.open('rb') as handle:
                result[str(path.relative_to(directory))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return result


def validate_bundle(directory):
    doc = json.loads((directory / 'graph.json').read_text())
    validate(doc)
    if not doc['nodes']:
        raise ValueError('refusing an empty graph')
    if not isinstance(json.loads((directory / 'manifest.json').read_text()), dict):
        raise ValueError('manifest must be an object')
    if not (directory / 'GRAPH_REPORT.md').read_text().strip():
        raise ValueError('report is empty')
    return {'nodes': len(doc['nodes']), 'links': len(doc['links']),
            'hyperedges': len(doc.get('hyperedges', []))}


def copy_verified(source, destination, *, bundle=False):
    before = hashes(source, bundle=bundle)
    destination.mkdir(parents=True, exist_ok=False)
    for name in before:
        dest = destination / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, dest)
    if hashes(destination, bundle=bundle) != before or hashes(source, bundle=bundle) != before:
        raise ValueError(f'copy verification failed; originals retained: {source}')


def checkpoint(output):
    """Caller holds the bundle lock. Never discard a previous recovery snapshot."""
    counts = validate_bundle(output)
    snapshot = output.parent / 'checkpoints' / uuid.uuid4().hex
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    copy_verified(output, snapshot, bundle=True)
    pointer = output.parent / ('.checkpoint-' + uuid.uuid4().hex)
    pointer.symlink_to(snapshot)
    pointer.replace(output.parent / 'checkpoint')
    return counts


def initialize(root, seed=None):
    root = Path(root).resolve()
    output = target(root)
    link = root / 'graphify-out'
    # Read-only fast path also permits a CLI child under safe_update's lock.
    if link.is_symlink():
        if link.resolve() != output:
            raise ValueError(f'graphify-out points elsewhere: {link.resolve()}')
        if not output.is_dir():
            raise ValueError(f'storage unavailable: {output}')
        if not (output.parent / 'checkpoint').is_dir():
            with locked(output.parent):
                checkpoint(output)
        return output
    with locked(output.parent):
        if link.is_symlink():
            return initialize(root, seed)
        if not output.exists():
            # Removing formerly tracked artifacts can leave only ignored caches
            # or untracked query notes. Seed missing core data from publication.
            source = link if (link / 'graph.json').is_file() else Path(seed) if seed else storage_root() / 'published/graphify-out'
            validate_bundle(source)
            with locked(source.parent) if source.is_relative_to(storage_root()) else nullcontext():
                staging = output.parent / ('import-' + uuid.uuid4().hex)
                copy_verified(source, staging)
                staging.rename(output)
        if not (output.parent / "checkpoint").is_dir():
            checkpoint(output)
        if link.exists():
            # Preserve residual notes/caches left behind when Git removes its
            # old tracked bundle. Existing local files win; originals below
            # retain any same-name collision without overwriting either copy.
            for path in link.rglob('*'):
                relative = path.relative_to(link)
                if path.is_file() and str(relative) not in BUNDLE:
                    destination = output / relative
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_name('.import-' + uuid.uuid4().hex)
                        try:
                            shutil.copy2(path, temporary)
                            with path.open('rb') as source_file, temporary.open('rb') as copied_file:
                                if hashlib.file_digest(source_file, 'sha256').digest() != hashlib.file_digest(copied_file, 'sha256').digest():
                                    raise ValueError(f'residual copy verification failed: {path}')
                            temporary.replace(destination)
                        finally:
                            temporary.unlink(missing_ok=True)
            # Preserve the entire old output, even after a branch switch.
            local_backup = root / ('.graphify-migrated-' + uuid.uuid4().hex)
            link.rename(local_backup)
            link.symlink_to(output, target_is_directory=True)
            # Moving, not deleting: the original bytes remain on warm storage.
            shutil.move(str(local_backup), str(output.parent / ('original-' + uuid.uuid4().hex)))
        else:
            link.symlink_to(output, target_is_directory=True)
    return output


def check_local(root, *, check_only=False, already_locked=False, allow_escape=True):
    output = target(Path(root))
    result = dict(dirty=[], destructive=False, desync=False, restored=False,
                  escaped=False, backup_dir=None, detail='local checkpoint matches')
    try:
        with nullcontext() if already_locked else locked(output.parent, blocking=False):
            baseline = output.parent / 'checkpoint'
            if not baseline.is_dir():
                raise ValueError('local graph checkpoint missing; run graphify_storage.py init')
            old, new = hashes(baseline, bundle=True), hashes(output, bundle=True)
            dirty = sorted(set(old) | set(new)) if old != new else []
            result['dirty'] = ['graphify-out/' + name for name in dirty]
            if not dirty:
                return result
            result.update(desync=True, detail='bundle differs from last accepted local checkpoint')
            if allow_escape and os.environ.get('ORION_ALLOW_GRAPH_SHRINK') == '1':
                result['escaped'] = True
                return result
            if check_only:
                return result
            backup = output.parent / ('unaccepted-' + uuid.uuid4().hex)
            copy_verified(output, backup, bundle=True)
            # Every overwritten byte is retained above. Restore the entire bundle.
            for name in BUNDLE:
                dest = output / name
                if (baseline / name).exists():
                    shutil.copy2(baseline / name, dest)
                else:
                    dest.unlink(missing_ok=True)
            result.update(restored=True, backup_dir=str(backup))
    except BlockingIOError:
        result['detail'] = 'local graph update in progress; deferred'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'checkpoint', 'publish', 'status'])
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--seed', type=Path)
    args = parser.parse_args()
    if args.action == 'init':
        print(initialize(args.repo, args.seed))
    else:
        output = target(args.repo)
        if args.action == 'checkpoint':
            with locked(output.parent):
                print(json.dumps(checkpoint(output)))
        elif args.action == 'publish':
            # Explicit promotion only: branch updates never overwrite the service graph.
            with locked(output.parent):
                snapshot = (output.parent / 'checkpoint').resolve(strict=True)
                counts = validate_bundle(snapshot)
                published = storage_root() / 'published'
                with locked(published):
                    pointer = published / ('.publish-' + uuid.uuid4().hex)
                    pointer.symlink_to(os.path.relpath(snapshot, published), target_is_directory=True)
                    pointer.replace(published / 'graphify-out')
                print(json.dumps(dict(path=str(published / 'graphify-out'), **counts)))
        else:
            print(json.dumps(dict(path=str(output), **validate_bundle(output),
                                  integrity=check_local(args.repo, check_only=True))))


if __name__ == '__main__':
    main()
