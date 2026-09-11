#!/usr/bin/env python3
"""Local integration eval using the installed Graphify interpreter; no network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import graphify_local
import graphify_storage
import merge_graphify_json


def main():
    from graphify import security
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    output = graphify_storage.initialize(root)
    counts = graphify_storage.validate_bundle(output)
    # A genuinely >512 MiB valid JSON input, not a mocked stat or changed threshold.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='size-eval-') as directory:
        large = Path(directory) / 'large.json'
        graph = dict(directed=False, multigraph=False, nodes=[{'id': 'probe'}], links=[])
        with large.open('wb') as handle:
            handle.write(json.dumps(graph).encode())
            padding = b' ' * (1024 * 1024)
            for _ in range(513):
                handle.write(padding)
        os.environ.pop('GRAPHIFY_MAX_GRAPH_BYTES', None)
        try:
            security.check_graph_file_size_cap(large)
        except Exception as exc:
            rejection = type(exc).__name__
        else:
            raise AssertionError('upstream no longer rejects >512 MiB; re-audit adapter')
        graphify_local.configure()
        security.check_graph_file_size_cap(large)
        others = [Path(directory) / name for name in ('current.json', 'other.json')]
        for path in others:
            path.write_text(json.dumps(graph))
        merge_graphify_json.main([str(large), *map(str, others)])
        assert json.loads(others[0].read_text())['nodes'] == graph['nodes']
        print(json.dumps(dict(real_graph=counts, large_input_bytes=large.stat().st_size,
                              upstream_rejected=rejection, local_load_and_merge='PASS')))
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='reflect-eval-') as directory:
        repo = Path(directory) / 'repo'
        repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        fixture = repo / 'graphify-out'
        fixture.mkdir()
        (fixture / 'graph.json').write_text(json.dumps(graph))
        (fixture / 'manifest.json').write_text('{}')
        (fixture / 'GRAPH_REPORT.md').write_text('# Graph Report\n1 node')
        env = dict(os.environ, ORION_GRAPHIFY_STORAGE_ROOT=str(Path(directory) / 'warm'))
        adapter = str(root / 'scripts/graphify_local.py')
        for args in [
            ['save-result', '--question', 'Local storage?', '--answer', 'Preserved.', '--nodes', 'probe', '--outcome', 'useful'],
            ['reflect'],
        ]:
            result = subprocess.run([sys.executable, adapter, *args], cwd=repo, env=env,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
        learning = fixture / '.graphify_learning.json'
        assert learning.is_file(), 'native reflect did not create learning overlay'
        before = learning.read_bytes()
        result = subprocess.run([sys.executable, str(root / 'scripts/check_graph_worktree_integrity.py'), '--json'],
                                cwd=repo, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert learning.read_bytes() == before, 'guard undid native reflect'
        print(json.dumps(dict(native_reflect_then_guard='PASS')))



if __name__ == '__main__':
    main()
