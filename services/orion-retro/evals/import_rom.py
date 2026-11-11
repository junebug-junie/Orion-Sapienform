"""Add one operator-owned, extracted ROM without replacing existing games or saves.

Run with PYTHONPATH=.:services/orion-retro. No ROM discovery or downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from app.contracts import Start


def import_rom(catalog, rom, game_id, title, players):
    Start(game_id=game_id, idempotency_key='import')
    suffix = rom.suffix.lower()
    system = {'.nes': 'nes', '.sfc': 'snes', '.smc': 'snes'}.get(suffix)
    if system is None: raise ValueError('extract a .nes, .sfc or .smc ROM first')
    if not title.strip() or len(title) > 128: raise ValueError('invalid title')
    if players not in (1, 2): raise ValueError('players must be 1 or 2')
    if not 0 < rom.stat().st_size <= 32 * 1024 * 1024: raise ValueError('invalid ROM size')
    data = rom.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    catalog = catalog.resolve()
    rows = json.loads(catalog.read_text()) if catalog.exists() else []
    if not isinstance(rows, list): raise ValueError('catalog must be a list')
    entry = dict(game_id=game_id, title=title, system=system, file=game_id+suffix,
                 rom_sha256=sha, ports=list(range(players)),
                 provenance={'source': 'operator-provided', 'file': rom.name})
    old = next((row for row in rows if row['game_id'] == game_id), None)
    if old is not None:
        # Re-import is safe; metadata edits or replacing a ROM need a deliberate migration.
        if any(old.get(k, 'nes' if k == 'system' else None) != entry[k]
               for k in ('title', 'system', 'file', 'rom_sha256', 'ports')):
            raise ValueError('refusing to replace existing game')
    elif len(rows) >= 64:
        raise ValueError('catalog is full')
    dest = catalog.parent/entry['file']
    if dest.is_symlink() or (dest.exists() and dest.read_bytes() != data):
        raise ValueError('refusing to replace existing ROM')
    catalog.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not dest.exists():
        with dest.open('xb') as stream: stream.write(data)
        dest.chmod(0o600)
    if old is None:
        rows.append(entry)
        # Atomic publication: a startup sees either complete catalog, never half-written JSON.
        with tempfile.NamedTemporaryFile(mode='w', dir=catalog.parent, delete=False) as stream:
            tmp = Path(stream.name)
            try:
                stream.write(json.dumps(rows, indent=2)+'\n')
                stream.flush()
                tmp.replace(catalog)
            finally:
                tmp.unlink(missing_ok=True)
    return entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--rom', type=Path, required=True)
    parser.add_argument('--game-id', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--players', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    row = import_rom(args.catalog, args.rom, args.game_id, args.title, args.players)
    print(json.dumps({k: row[k] for k in ('game_id', 'system', 'rom_sha256')}, indent=2))
