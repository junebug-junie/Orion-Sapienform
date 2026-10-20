"""Persistent save states on the host bind mount (/saves).

Layout: <root>/<game_id>/<save_id>.state (raw retro_serialize bytes) plus
<save_id>.json (metadata). The .json is written last, so a listing only ever
sees complete saves. A save is tied to the exact ROM and core that made it:
loading refuses any rom_sha256/core_sha256 mismatch or a damaged state file.

Disk bound: each state is at most `max_bytes`; each game keeps at most
`max_per_game` saves and the oldest are deleted on every new save.
The worker writes and reads states; the API process only lists metadata.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

SAVE_ID = re.compile(r'^[0-9a-f]{32}$')
GAME_ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')
FIELDS = {'save_id', 'game_id', 'rom_sha256', 'core_sha256', 'frame', 'played_frames',
          'created_ns', 'created_utc', 'bytes', 'state_sha256', 'session_id'}


def available(root):
    root = Path(root)
    return root.is_dir() and os.access(root, os.W_OK | os.X_OK)


class SaveStore:
    def __init__(self, root, max_per_game=20, max_bytes=4 * 1024 * 1024):
        self.root = Path(root)
        self.max_per_game, self.max_bytes = max_per_game, max_bytes

    def _dir(self, game_id):
        if not isinstance(game_id, str) or not GAME_ID.match(game_id):
            raise ValueError('invalid_game')
        return self.root / game_id

    def write(self, game_id, identity, state, *, frame, played_frames, session_id):
        if not available(self.root): raise ValueError('saves_unavailable')
        if not 0 < len(state) <= self.max_bytes: raise ValueError('save_too_large')
        directory = self._dir(game_id)
        directory.mkdir(mode=0o700, exist_ok=True)
        now = time.time_ns(); save_id = uuid.uuid4().hex
        meta = dict(save_id=save_id, game_id=game_id, rom_sha256=identity['rom_sha256'],
                    core_sha256=identity['core_sha256'], frame=frame, played_frames=played_frames,
                    created_ns=now,
                    created_utc=datetime.fromtimestamp(now / 1e9, timezone.utc).isoformat(timespec='seconds'),
                    bytes=len(state), state_sha256=hashlib.sha256(state).hexdigest(), session_id=session_id)
        try:
            for suffix, data in (('.state', state), ('.json', (json.dumps(meta) + '\n').encode())):
                tmp = directory / f'.{save_id}{suffix}.tmp'
                # No fsync: the worker's frame loop must never block on disk (monitor()
                # fails a worker silent for 1 s). Page cache survives container restarts.
                tmp.write_bytes(data)
                tmp.replace(directory / f'{save_id}{suffix}')
        except OSError:
            for leftover in directory.glob(f'*{save_id}*'): leftover.unlink(missing_ok=True)
            raise ValueError('save_write_failed')
        self.prune(game_id)
        return meta

    def list(self, game_id):
        directory = self._dir(game_id)
        if not directory.is_dir(): return []
        rows = []
        for path in directory.glob('*.json'):
            try: meta = json.loads(path.read_text())
            except (OSError, ValueError): continue
            if (isinstance(meta, dict) and set(meta) == FIELDS and meta['game_id'] == game_id
                    and path.stem == meta['save_id'] and SAVE_ID.match(meta['save_id'])
                    and (directory / f"{meta['save_id']}.state").is_file()):
                rows.append(meta)
        return sorted(rows, key=lambda m: m['created_ns'], reverse=True)

    def meta(self, game_id, save_id, identity):
        if not isinstance(save_id, str) or not SAVE_ID.match(save_id): raise ValueError('invalid_save')
        meta = next((m for m in self.list(game_id) if m['save_id'] == save_id), None)
        if meta is None: raise ValueError('unknown_save')
        if (meta['rom_sha256'] != identity['rom_sha256']
                or meta['core_sha256'] != identity['core_sha256']):
            raise ValueError('save_identity_mismatch')
        return meta

    def read(self, game_id, save_id, identity):
        meta = self.meta(game_id, save_id, identity)
        path = self._dir(game_id) / f'{save_id}.state'
        if path.stat().st_size != meta['bytes'] or meta['bytes'] > self.max_bytes:
            raise ValueError('save_corrupt')
        state = path.read_bytes()
        if hashlib.sha256(state).hexdigest() != meta['state_sha256']: raise ValueError('save_corrupt')
        return meta, state

    def prune(self, game_id):
        directory = self._dir(game_id)
        keep = self.list(game_id)
        for meta in keep[self.max_per_game:]:
            (directory / f"{meta['save_id']}.json").unlink(missing_ok=True)
            (directory / f"{meta['save_id']}.state").unlink(missing_ok=True)
        live = {m['save_id'] for m in keep[:self.max_per_game]}
        # A crash between the two renames can leave a .state with no metadata.
        for orphan in directory.glob('*.state'):
            if orphan.stem not in live: orphan.unlink(missing_ok=True)
