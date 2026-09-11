"""Local bundles survive worktrees, corruption, publication and old graph ceilings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import graphify_storage as storage
import merge_graphify_json as merger


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Test', '-c',
                        'user.email=test@example.com', 'commit', '--allow-empty', '-qm', 'initial'], check=True)
        self.env = patch.dict(os.environ, ORION_GRAPHIFY_STORAGE_ROOT=str(self.base / 'warm'))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.source = self.repo / 'graphify-out'
        self.source.mkdir()
        self.graph = dict(directed=False, multigraph=False, nodes=[{'id': 'a'}], links=[], hyperedges=[])
        (self.source / 'graph.json').write_text(json.dumps(self.graph))
        (self.source / 'manifest.json').write_text('{}')
        (self.source / 'GRAPH_REPORT.md').write_text('# Graph Report\n1 node')
        (self.source / 'memory').mkdir()
        (self.source / 'memory' / 'note.md').write_text('keep this query note')
        (self.source / '2026-07-29').mkdir()
        (self.source / '2026-07-29' / 'graph.json').write_text('historical bytes')

    def test_migration_preserves_all_bytes_and_originals(self):
        before = storage.hashes(self.source)
        output = storage.initialize(self.repo)
        self.assertTrue(self.source.is_symlink())
        self.assertEqual(storage.hashes(output), before)
        original, = output.parent.glob('original-*')
        self.assertEqual(storage.hashes(original), before)
        self.assertEqual(storage.initialize(self.repo), output)
        self.assertFalse(storage.check_local(self.repo)['dirty'])
        # Native atomic graph replacement must replace a file in the store, not the symlink.
        replacement = output / 'next.json'
        replacement.write_text(json.dumps(self.graph))
        replacement.replace(self.source / 'graph.json')
        self.assertEqual(json.loads((output / 'graph.json').read_text()), self.graph)

    def test_worktrees_are_independent_and_missing_seed_fails(self):
        output = storage.initialize(self.repo)
        other = self.base / 'other'
        subprocess.run(['git', '-C', str(self.repo), 'worktree', 'add', '-qb', 'other', str(other)], check=True)
        with self.assertRaises(FileNotFoundError):
            storage.initialize(other)
        private = storage.initialize(other, output)
        self.assertNotEqual(private, output)
        (private / 'graph.json').write_text('corrupt')
        self.assertEqual(json.loads((output / 'graph.json').read_text()), self.graph)
        subprocess.run(['git', '-C', str(self.repo), 'worktree', 'remove', '--force', str(other)], check=True)
        self.assertTrue(private.exists())

    def test_guard_restores_whole_bundle_and_preserves_rejected_bytes(self):
        output = storage.initialize(self.repo)
        baseline = storage.hashes(output, bundle=True)
        (output / 'graph.json').write_text('broken')
        (output / 'GRAPH_REPORT.md').write_text('wrong report')
        (output / 'manifest.json').unlink()
        (output / 'graph.html').write_text('new stale html')
        self.assertTrue(storage.check_local(self.repo, check_only=True)['desync'])
        result = storage.check_local(self.repo)
        self.assertTrue(result['restored'])
        self.assertEqual(storage.hashes(output, bundle=True), baseline)
        self.assertEqual((Path(result['backup_dir']) / 'graph.json').read_text(), 'broken')
        self.assertEqual((output / 'memory/note.md').read_text(), 'keep this query note')

    def test_guard_defers_while_update_lock_is_held(self):
        output = storage.initialize(self.repo)
        with storage.locked(output.parent):
            (output / 'graph.json').write_text('in flight')
            self.assertIn('deferred', storage.check_local(self.repo)['detail'])
            self.assertEqual(storage.initialize(self.repo), output)
        self.assertTrue(storage.check_local(self.repo)['restored'])

    def test_unlimited_nodes_and_bytes_keep_schema_validation(self):
        data = dict(self.graph, nodes=[{'id': str(n)} for n in range(100_001)])
        merger.validate(data)
        self.assertEqual(len(merger.merge_graphs(self.graph, data, self.graph)['nodes']), 100_002)
        data['links'] = [{'source': 'missing', 'target': '0'}]
        with self.assertRaisesRegex(ValueError, 'absent'):
            merger.validate(data)
        # Probe metadata above the old byte ceiling; the integration eval
        # separately exercises a real 513 MiB JSON input.
        paths = [self.base / name for name in ('base.json', 'current.json', 'other.json')]
        for path in paths:
            path.write_text(json.dumps(self.graph))
        with paths[0].open('ab') as handle:
            # JSON permits whitespace; use a real large input in the optional live eval,
            # while this gate verifies the loader has no stat-based ceiling.
            handle.write(b' ')
        with patch.object(Path, 'stat') as mocked:
            mocked.return_value.st_size = 600 * 1024 * 1024
            mocked.return_value.st_mode = 0o644
            self.assertEqual(merger.main([str(p) for p in paths]), 0)

    def test_publication_is_stable_until_explicitly_replaced(self):
        output = storage.initialize(self.repo)
        with patch.object(sys, 'argv', ['storage', 'publish', '--repo', str(self.repo)]):
            storage.main()
        published = storage.storage_root() / 'published/graphify-out'
        self.assertFalse(os.readlink(published).startswith('/'))
        original = (published / 'graph.json').read_bytes()
        (output / 'graph.json').write_text(json.dumps(dict(self.graph, nodes=[{'id': 'a'}, {'id': 'b'}])))
        storage.checkpoint(output)
        self.assertEqual((published / 'graph.json').read_bytes(), original)
        with patch.object(sys, 'argv', ['storage', 'publish', '--repo', str(self.repo)]):
            storage.main()
        self.assertEqual(len(json.loads((published / 'graph.json').read_text())['nodes']), 2)

    def test_interrupted_initial_checkpoint_recovers_on_retry(self):
        with patch.object(storage, 'checkpoint', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                storage.initialize(self.repo)
        self.assertFalse(self.source.is_symlink())
        output = storage.initialize(self.repo)
        self.assertTrue(self.source.is_symlink())
        self.assertFalse(storage.check_local(self.repo)['dirty'])
        self.assertTrue((output.parent / 'checkpoint').is_dir())

    def test_cli_sidecar_changes_checkpoint_and_failed_changes_restore(self):
        import graphify_local
        output = storage.initialize(self.repo)
        package = types.ModuleType('graphify')
        package.__path__ = []
        cli = types.ModuleType('graphify.__main__')
        def reflect():
            (output / '.graphify_learning.json').write_text('{"learned": true}')
            raise SystemExit(0)
        cli.main = reflect
        with patch.dict(sys.modules, {'graphify': package, 'graphify.__main__': cli}), \
             patch.object(graphify_local, 'configure', return_value=(self.repo, output)), \
             patch.object(sys, 'argv', ['graphify', 'reflect']):
            self.assertEqual(graphify_local.main(), 0)
            self.assertFalse(storage.check_local(self.repo)['dirty'])
            def fail():
                (output / 'graph.json').write_text('broken')
                raise SystemExit(1)
            cli.main = fail
            with self.assertRaises(SystemExit):
                graphify_local.main()
        self.assertEqual(json.loads((output / 'graph.json').read_text()), self.graph)
        self.assertTrue((output / '.graphify_learning.json').exists())

    def test_watch_lifetime_unlocked_but_rebuild_checkpoints(self):
        import graphify_local
        output = storage.initialize(self.repo)
        package = types.ModuleType('graphify')
        package.__path__ = []
        cli, watcher = types.ModuleType('graphify.__main__'), types.ModuleType('graphify.watch')
        package.watch = watcher
        def rebuild(*args, **kwargs):
            if kwargs.get('acquire_lock', True):
                return watcher._rebuild_code(*args, acquire_lock=False)
            with self.assertRaises(BlockingIOError), storage.locked(output.parent, blocking=False):
                pass
            (output / '.graphify_learning.json').write_text('{"watch": true}')
            return True
        watcher._rebuild_code = rebuild
        def watch():
            with storage.locked(output.parent, blocking=False):
                pass  # Observer lifetime must not hold the lock.
            self.assertTrue(watcher._rebuild_code(self.repo))
            self.assertFalse(storage.check_local(self.repo)['dirty'])
        cli.main = watch
        with patch.dict(sys.modules, {'graphify': package, 'graphify.__main__': cli, 'graphify.watch': watcher}), \
             patch.object(graphify_local, 'configure', return_value=(self.repo, output)), \
             patch.object(sys, 'argv', ['graphify', 'watch']):
            graphify_local.main()

    def test_pull_untracking_graph_keeps_residual_notes_and_seeds_core(self):
        seed = self.base / 'seed'
        storage.copy_verified(self.source, seed, bundle=True)
        for name in ('graph.json', 'manifest.json', 'GRAPH_REPORT.md'):
            (self.source / name).unlink()  # What the untracking merge removes.
        output = storage.initialize(self.repo, seed)
        self.assertEqual(json.loads((output / 'graph.json').read_text()), self.graph)
        self.assertEqual((output / 'memory/note.md').read_text(), 'keep this query note')
        self.assertEqual((output / '2026-07-29/graph.json').read_text(), 'historical bytes')
        original, = output.parent.glob('original-*')
        self.assertEqual((original / 'memory/note.md').read_text(), 'keep this query note')

    def test_interrupted_residual_copy_retries_complete_note(self):
        seed = self.base / 'seed'
        storage.copy_verified(self.source, seed, bundle=True)
        for name in ('graph.json', 'manifest.json', 'GRAPH_REPORT.md'):
            (self.source / name).unlink()
        original_copy = storage.shutil.copy2
        def interrupted(source, destination):
            if Path(source).name == 'note.md':
                Path(destination).write_text('partial')
                raise OSError('disk full')
            return original_copy(source, destination)
        with patch.object(storage.shutil, 'copy2', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'disk full'):
                storage.initialize(self.repo, seed)
        output = storage.initialize(self.repo, seed)
        self.assertEqual((output / 'memory/note.md').read_text(), 'keep this query note')
        self.assertFalse(storage.check_local(self.repo)['dirty'])

    def test_failed_copy_does_not_move_original(self):
        with patch.object(storage.shutil, 'copy2', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                storage.initialize(self.repo)
        self.assertTrue(self.source.is_dir())
        self.assertFalse(self.source.is_symlink())
        self.assertEqual(json.loads((self.source / 'graph.json').read_text()), self.graph)


if __name__ == '__main__':
    unittest.main()
