#!/usr/bin/env python3
"""Run installed Graphify with Orion local storage and no graph size ceiling."""
import os
from pathlib import Path
import subprocess
import sys


def configure():
    from graphify_storage import initialize
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    output = initialize(root)
    os.environ['GRAPHIFY_OUT'] = str(output)
    # Upstream 0.9.15 treats 0 as its default 512 MiB, not unlimited.
    # Override only the graph byte ceiling, retaining all other validation.
    from graphify import security
    security.check_graph_file_size_cap = lambda _path: None
    return root, output


def main():
    root, output = configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'merge-driver':
        from merge_graphify_json import main as merge
        return merge(sys.argv[2:])
    from graphify.__main__ import main as upstream
    # safe_graphify_update.sh owns the surrounding transaction. Verify the
    # inherited fd really names this bundle's lock before skipping acquisition.
    inherited = os.environ.get('ORION_GRAPHIFY_LOCK_FD')
    if inherited:
        fd_stat = os.fstat(int(inherited))
        path_stat = (output.parent / '.orion-storage.lock').stat()
        if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError('inherited Graphify lock does not match this bundle')
        return upstream()

    if len(sys.argv) > 1 and sys.argv[1] == 'watch':
        import graphify.watch as watcher
        rebuild = watcher._rebuild_code

        def accepted_rebuild(*args, **kwargs):
            # Upstream recursively calls itself after acquiring its own lock.
            if kwargs.get('acquire_lock') is False:
                return rebuild(*args, **kwargs)
            try:
                transaction(root, output, lambda: 0 if rebuild(*args, **kwargs) else 1)
                return True
            except Exception as exc:
                print(f'Graphify watch rebuild refused: {exc}', file=sys.stderr)
                return False

        watcher._rebuild_code = accepted_rebuild
        return upstream()  # Do not lock for the observer's entire lifetime.
    return transaction(root, output, upstream)


def transaction(root, output, run):
    from graphify_storage import locked, hashes, check_local, checkpoint, validate_bundle
    # Native reflect/label/cluster-only mutate sidecars too. Keep those changes
    # as accepted checkpoints instead of letting the session guard undo them.
    with locked(output.parent):
        check_local(root, already_locked=True)
        before = hashes(output, bundle=True)
        try:
            try:
                result = run()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
                result = 0
            if result not in (None, 0):
                raise RuntimeError(f'Graphify failed: {result}')
            if hashes(output, bundle=True) != before:
                old = validate_bundle(output.parent / 'checkpoint')['nodes']
                new = validate_bundle(output)['nodes']
                threshold = float(os.environ.get('GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT', '10'))
                if (old - new) / old * 100 > threshold and os.environ.get('ORION_ALLOW_GRAPH_SHRINK') != '1':
                    raise ValueError(f'Graphify node loss refused: {old} -> {new}')
                checkpoint(output)
            return result
        except BaseException:
            check_local(root, already_locked=True, allow_escape=False)
            raise


if __name__ == '__main__':
    sys.exit(main())
